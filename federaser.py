import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional

from fedavg import fedavg, get_model_weights, set_model_weights
from model import model_init
from data_preprocess import get_partitioned_data
from server import Server
from backdoor import BackdoorAttack, evaluate_backdoor
from utils.metrics_logger import save_metrics_to_excel, plot_metrics

class FedEraser:
    def __init__(self, 
                 model_path: str,
                 history_path: str,
                 config: dict,
                 history_dir: str = './fl_history',
                 save_dir: str = './unlearned_models',
                 device: Optional[torch.device] = None):
        """
        Initialize FedEraser
        
        Args:
            model_path: Path to the federated learning model
            history_path: Path to the complete history of federated learning
            history_dir: Directory for federated learning history
            save_dir: Directory to save models after removing client 
                        contributions
            device: Computation device
        """
        # Initialize timers
        self.timers = {
            'init': 0.0,
            'data_loading': 0.0,
            'aggregation': 0.0,
            'client_training': 0.0,
            'calibration': 0.0,
            'evaluation': 0.0,
            'rounds': {}
        }
        
        # Record initialization start time
        init_start_time = time.time()
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model_path = model_path
        self.history_path = history_path
        self.config = config
        self.history_dir = history_dir
        self.save_dir = save_dir
        
        # Create save directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Load global model
        model_load_start = time.time()
        self.global_model = model_init(self.config).to(self.device)
        self.global_model.load_state_dict(torch.load(model_path, map_location=self.device))
        model_load_time = time.time() - model_load_start
        print(f"Model loading time: {model_load_time:.2f} seconds")
        
        # Load history
        history_load_start = time.time()
        with open(history_path, 'rb') as f:
            complete_data = pickle.load(f)
            
        self.client_updates_history = complete_data['client_updates_history']
        self.aggregation_history = complete_data['aggregation_history']
        self.current_round = complete_data['current_round']
        history_load_time = time.time() - history_load_start
        print(f"History loading time: {history_load_time:.2f} seconds")
        
        # Get client ID list
        self.client_ids = set()
        for round_num in self.client_updates_history:
            self.client_ids.update(self.client_updates_history[round_num].keys())
        self.client_ids = sorted(list(self.client_ids))
        
        # Record total initialization time
        self.timers['init'] = time.time() - init_start_time
        
        print(f"Loaded federated learning history, total rounds: {self.current_round}, number of clients: {len(self.client_ids)}")
        print(f"FedEraser initialization time: {self.timers['init']:.2f} seconds")
        
    def select_remaining_clients(self, client_to_remove: int) -> List[int]:
        """
        Select all clients except the one to be removed
        
        Args:
            client_to_remove: ID of client to remove
            
        Returns:
            List of remaining client IDs
        """
        return [client_id for client_id in self.client_ids if client_id != client_to_remove]
        
    def get_client_model(self, client_id: int, round_num: int) -> Dict[str, np.ndarray]:
        """
        Get client model weights for a specific round
        
        Args:
            client_id: Client ID
            round_num: Round number
            
        Returns:
            Client model weights
        """
        if round_num in self.client_updates_history and client_id in self.client_updates_history[round_num]:
            return self.client_updates_history[round_num][client_id]['weights']
        else:
            raise ValueError(f"Client {client_id} does not exist in round {round_num}")
    
    def get_global_model_at_round(self, round_num: int) -> Dict[str, np.ndarray]:
        """
        Get global model weights for a specific round
        
        Args:
            round_num: Round number
            
        Returns:
            Global model weights
        """
        if round_num in self.aggregation_history:
            return self.aggregation_history[round_num]
        else:
            raise ValueError(f"Global model for round {round_num} does not exist")
    
    def evaluate_model(self, model: torch.nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model performance on test set
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            
        Returns:
            (test_loss, test_accuracy)
        """
        # Record evaluation start time
        eval_start = time.time()
        
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        # Record evaluation time
        eval_time = time.time() - eval_start
        self.timers['evaluation'] += eval_time
        
        print(f"Model evaluation time: {eval_time:.2f} seconds")
        
        return test_loss, accuracy
    
    def aggregate_remaining_clients(self, round_num: int, 
                                   remaining_clients: List[int]) -> Dict[str, np.ndarray]:
        """
        Aggregate weights of remaining clients
        
        Args:
            round_num: Round number
            remaining_clients: List of remaining client IDs
            
        Returns:
            Aggregated global model weights
        """
        # Record aggregation start time
        agg_start = time.time()
        
        weights_list = []
        sample_sizes = []
        
        for client_id in remaining_clients:
            if client_id in self.client_updates_history[round_num]:
                client_data = self.client_updates_history[round_num][client_id]
                weights_list.append(client_data['weights'])
                sample_sizes.append(client_data['sample_size'])
        
        # Use FedAvg for aggregation
        result = fedavg(weights_list, sample_sizes)
        
        # Record aggregation time
        agg_time = time.time() - agg_start
        self.timers['aggregation'] += agg_time
        
        print(f"Remaining clients aggregation time (Round {round_num}): {agg_time:.2f} seconds")
        
        return result
    
    def train_clients_one_step(self, 
                              client_models: List[Dict[str, np.ndarray]], 
                              global_model: Dict[str, np.ndarray], 
                              train_loaders: List[DataLoader],
                              lr: float = 0.01,
                              epochs: int = 1) -> List[Dict[str, np.ndarray]]:
        """
        Train client models for one step
        
        Args:
            client_models: List of client model weights
            global_model: Global model weights
            train_loaders: List of client training data loaders
            lr: Learning rate
            epochs: Number of training epochs
            
        Returns:
            List of trained client model weights
        """
        # Record training start time
        train_start = time.time()
        
        new_client_models = []
        client_times = []
        
        for i, (client_weights, train_loader) in enumerate(zip(client_models, train_loaders)):
            # Record client training start time
            client_start = time.time()
            
            # Initialize model
            model = model_init(self.config).to(self.device)
            
            # Load global model weights
            for name, param in model.named_parameters():
                if name in global_model:
                    param.data = torch.from_numpy(global_model[name]).to(self.device)
            
            # Set optimizer
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
            
            # Train model
            model.train()
            for epoch in range(epochs):
                epoch_start = time.time()
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Extract weights
            new_weights = {}
            for name, param in model.named_parameters():
                new_weights[name] = param.detach().cpu().numpy()
            
            new_client_models.append(new_weights)
            
            # Record client training time
            client_time = time.time() - client_start
            client_times.append(client_time)
            print(f"Client {i} total training time: {client_time:.2f} seconds")
        
        # Record client training statistics
        if client_times:
            print(f"Client training time statistics - Min: {min(client_times):.2f}s, Max: {max(client_times):.2f}s, Average: {sum(client_times)/len(client_times):.2f}s")
        
        # Record total training time
        train_time = time.time() - train_start
        self.timers['client_training'] += train_time
        
        print(f"Total training time for all clients: {train_time:.2f} seconds")
        
        return new_client_models
    
    def unlearning_step_once(self,
                             old_client_models: List[Dict[str, np.ndarray]],
                             new_client_models: List[Dict[str, np.ndarray]],
                             global_model_before_forget: Dict[str, np.ndarray],
                             global_model_after_forget: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Perform one unlearning step using calibration to adjust model weights
        
        Args:
            old_client_models: List of old client model weights
            new_client_models: List of new client model weights
            global_model_before_forget: Global model weights before forgetting
            global_model_after_forget: Global model weights after forgetting
            
        Returns:
            Calibrated global model weights
        """
        # Record calibration start time
        calib_start = time.time()
        
        old_param_update = {}  # oldCM - oldGM_t
        new_param_update = {}  # newCM - newGM_t
        return_model_state = {}  # newGM_t + ||oldCM - oldGM_t|| * (newCM - newGM_t) / ||newCM - newGM_t||
        
        assert len(old_client_models) == len(new_client_models)
        
        for layer in global_model_before_forget.keys():
            # Skip BatchNorm buffers (running_mean, running_var, num_batches_tracked) to avoid mismatches
            if any(bn_key in layer for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']):
                return_model_state[layer] = global_model_after_forget[layer]
                continue
            # Initialize parameter updates
            old_param_update[layer] = np.zeros_like(global_model_before_forget[layer])
            new_param_update[layer] = np.zeros_like(global_model_before_forget[layer])
            
            # Calculate average of client models
            for client_idx in range(len(old_client_models)):
                old_param_update[layer] += old_client_models[client_idx][layer]
                new_param_update[layer] += new_client_models[client_idx][layer]
            
            old_param_update[layer] /= len(old_client_models)  # oldCM
            new_param_update[layer] /= len(new_client_models)  # newCM
            
            # Calculate update direction
            old_param_update[layer] = old_param_update[layer] - global_model_before_forget[layer]  # oldCM - oldGM_t
            new_param_update[layer] = new_param_update[layer] - global_model_after_forget[layer]   # newCM - newGM_t
            
            # Calculate step size and direction
            old_norm = np.linalg.norm(old_param_update[layer])  # ||oldCM - oldGM_t||
            new_norm = np.linalg.norm(new_param_update[layer])  # ||newCM - newGM_t||
            
            # Avoid division by zero
            if new_norm > 1e-10:
                step_direction = new_param_update[layer] / new_norm  # (newCM - newGM_t) / ||newCM - newGM_t||
                return_model_state[layer] = global_model_after_forget[layer] + old_norm * step_direction
            else:
                # If new update is almost zero, don't calibrate
                return_model_state[layer] = global_model_after_forget[layer]
        
        # Record calibration time
        calib_time = time.time() - calib_start
        self.timers['calibration'] += calib_time
        
        print(f"FedEraser calibration time: {calib_time:.2f} seconds")
        
        return return_model_state
    
    def run(self, 
            client_to_remove: int, 
            unlearn_rounds: Optional[List[int]] = None,
            save_model_path: Optional[str] = None,
            batch_size: int = 64,
            lr: float = 0.01,
            epochs: int = 1,
            max_rounds: int = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Run the FedEraser algorithm with a brand new model
        """
        total_start_time = time.time()
        print(f"Starting FedEraser, removing client ID: {client_to_remove}")
        
        if unlearn_rounds is None:
            if max_rounds is not None and 0 < max_rounds < self.current_round:
                unlearn_rounds = list(range(max_rounds))
            else:
                unlearn_rounds = list(range(self.current_round))
        else:
            unlearn_rounds = [r for r in unlearn_rounds if 0 <= r < self.current_round]
        
        remaining_clients = self.select_remaining_clients(client_to_remove)
        
        train_loaders, test_loader = get_partitioned_data(
            dataset_name=self.config['dataset'],
            num_clients=len(remaining_clients),
            batch_size=batch_size,
            partition_type='iid'
        )
        
        history = {
            'round': [],
            'test_loss': [],
            'test_accuracy': [],
            'round_time': [],
            'backdoor_success_rate': []
        }
        
        # Initial evaluation
        initial_model = model_init(self.config).to(self.device)
        initial_weights = {name: param.clone().detach().cpu().numpy() for name, 
                           param in initial_model.named_parameters()}
        
        round_idx = -1  # explicitly set to avoid undefined
        test_loss, test_accuracy = self.evaluate_model(initial_model, test_loader)
        backdoor_sr = evaluate_backdoor(
            initial_model,
            test_loader,
            self.device,
            trigger_pattern='square',
            trigger_size=5,
            target_label=7
        )
        print(f"Round {round_idx}: Test Loss = {test_loss:.4f}, Accuracy = {test_accuracy:.2f}%, Backdoor SR = {backdoor_sr:.2f}%")
        history['round'].append(round_idx)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)
        history['round_time'].append(0.0)
        history['backdoor_success_rate'].append(backdoor_sr)
        
        new_global_models = {r: copy.deepcopy(initial_weights) for r in unlearn_rounds}
        
        for round_idx in unlearn_rounds:
            round_start_time = time.time()
            try:
                print(f"--- FedEraser processing round {round_idx} ---")
                old_global_weights = self.get_global_model_at_round(round_idx)
                old_client_models = [
                    self.get_client_model(cid, round_idx)
                    for cid in remaining_clients if cid in self.client_updates_history[round_idx]
                ]
                if not old_client_models:
                    raise RuntimeError("No valid client models")
                
                client_weights = [copy.deepcopy(initial_weights) for _ in remaining_clients]
                client_trainers = self.train_clients_one_step(
                    client_models=client_weights,
                    global_model=new_global_models[round_idx],
                    train_loaders=train_loaders,
                    lr=lr,
                    epochs=epochs
                )
                
                sample_sizes = [len(t.dataset) for t in train_loaders]
                new_global_weights = fedavg(client_trainers, sample_sizes)
                
                calibrated_weights = self.unlearning_step_once(
                    old_client_models, client_trainers, old_global_weights, new_global_weights
                )
                new_global_models[round_idx] = calibrated_weights
                
                new_model = model_init(self.config).to(self.device)
                for name, param in new_model.named_parameters():
                    if name in calibrated_weights:
                        param.data = torch.from_numpy(calibrated_weights[name]).to(self.device)
                
                test_loss, test_accuracy = self.evaluate_model(new_model, test_loader)
                backdoor_sr = evaluate_backdoor(
                    new_model,
                    test_loader,
                    self.device,
                    trigger_pattern='square',
                    trigger_size=5,
                    target_label=7
                )
            except Exception as e:
                print(f"Error during round {round_idx}: {e}")
                test_loss, test_accuracy, backdoor_sr = 0.0, 0.0, 0.0
            print(f"Round {round_idx}: Test Loss = {test_loss:.4f}, Accuracy = {test_accuracy:.2f}%, Backdoor SR = {backdoor_sr:.2f}%")
            
            round_time = time.time() - round_start_time
            history['round'].append(round_idx)
            history['test_loss'].append(test_loss)
            history['test_accuracy'].append(test_accuracy)
            history['round_time'].append(round_time)
            history['backdoor_success_rate'].append(backdoor_sr)
        
        if new_global_models:
            final_round = max(new_global_models.keys())
            final_weights = new_global_models[final_round]
            final_model = model_init(self.config).to(self.device)
            for name, param in final_model.named_parameters():
                if name in final_weights:
                    param.data = torch.from_numpy(final_weights[name]).to(self.device)
            
            test_loss, test_accuracy = self.evaluate_model(final_model, test_loader)
            if save_model_path:
                torch.save(final_model.state_dict(), save_model_path)
        else:
            final_weights = None
        
        total_time = time.time() - total_start_time
        history['total_time'] = total_time
        
        print("=== History Field Lengths ===")
        for k, v in history.items():
            if isinstance(v, list):
                print(f"{k}: {len(v)}")
        
        save_metrics_to_excel(history, os.path.join(self.save_dir, "federaser_metrics.xlsx"))
        plot_metrics(history, self.save_dir, "FedEraser Unlearning")
        
        return final_weights, history
