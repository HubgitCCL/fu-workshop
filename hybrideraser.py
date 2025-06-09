import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import numpy as np
import time
import os
import pickle
from typing import Dict, List, Tuple, Optional

from fedavg import fedavg, get_model_weights, set_model_weights
from model import model_init
from data_preprocess import get_partitioned_data
from server import Server
from federaser import FedEraser
from kd import subtract_model, knowledge_distillation
from backdoor import evaluate_backdoor
from utils.metrics_logger import save_metrics_to_excel

class HybridEraser:
    """
    HybridEraser: Combine FedEraser unlearning with Knowledge Distillation (KD)
    in a pseudo-parallel way.
    Executes FedEraser rounds and KD epochs asynchronously and performs periodic
    weighted model aggregation.
    """
    def __init__(self,
                 model_path: str,
                 history_path: str,
                 config: dict,
                 device: Optional[torch.device] = None,
                 kd_epochs_per_round: int = 2,
                 aggregation_interval: int = 5,
                 aggregation_alpha: float = 0.5,
                 save_dir: str = './hybrid_results'):
        """
        Initialize HybridEraser with required components and parameters.

        Args:
            model_path: Path to the trained federated learning model (final
                global model).
            history_path: Path to the pickled training history (client updates
                and aggregation history).
            device: Computation device (CPU or CUDA).
            kd_epochs_per_round: Number of KD training epochs to run per
                FedEraser round (0 means no KD).
            aggregation_interval: Frequency (in FedEraser rounds) to perform
                model aggregation.
            aggregation_alpha: Weighting factor for KD model in aggregation
                (FedEraser gets 1 - aggregation_alpha).
            save_dir: Directory to save results (like metrics or intermediate
                models if needed).
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Save paths and parameters
        self.model_path = model_path
        self.history_path = history_path
        self.kd_epochs_per_round = kd_epochs_per_round
        self.aggregation_interval = aggregation_interval
        self.aggregation_alpha = aggregation_alpha
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 保存 config 字典（数据集和模型名称）
        self.config = config

        # Load the original trained model (teacher model for KD) and history
        # Teacher model represents the original model with all clients (before
        # unlearning)
        # self.teacher_model = model_init({'dataset':
        #     self._infer_dataset_from_path(model_path)}).to(self.device)
        self.teacher_model = model_init(self.config).to(self.device)  # 修改：使用 config 初始化教师模型
        self.teacher_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.teacher_model.eval()  # Teacher model won't be trained, so set to eval mode

        # Load training history to assist FedEraser
        with open(history_path, 'rb') as f:
            complete_history = pickle.load(f)
        self.client_updates_history = complete_history['client_updates_history']
        self.aggregation_history = complete_history['aggregation_history']
        # Determine dataset name and number of clients from history or model
        # self.dataset = self._infer_dataset_from_path(model_path)
        self.dataset = config['dataset']  # 使用 config 中的 dataset

        # Use FedEraser to access helper functions and data (like client list)
        self.federaser = FedEraser(model_path=model_path,
                                   history_path=history_path,
                                   config=self.config,  # 传入 config，用于 FedEraser 初始化
                                   history_dir=os.path.join(save_dir, 'history'),
                                   save_dir=os.path.join(save_dir, 'models'),
                                   device=self.device)
        # Determine remaining clients after removal (list of IDs excluding the one to remove)
        self.remaining_clients = None  # will be set in run() after client_to_remove is known

        # Initialize KD student model:
        # Start by removing the target client's contribution from the model weights (if any)
        # This gives an initial student model that has partially unlearned the target client.
        self.student_model = None  # will initialize in run() when client_to_remove is specified

        # Prepare data loaders (to be loaded in run when client_to_remove is known)
        self.train_loaders: List[DataLoader] = []
        self.test_loader: DataLoader = None

        # Metrics history dictionary to record performance each FedEraser round
        self.history = {
            'round': [],
            'test_loss': [],
            'test_accuracy': [],
            'backdoor_success_rate': []
        }

    def _infer_dataset_from_path(self, model_path: str) -> str:
        """Infer dataset name from the model file path."""
        path_lower = model_path.lower()
        if 'mnist' in path_lower and 'fashion' not in path_lower:
            return 'mnist'
        elif 'fashion' in path_lower:
            return 'fashion_mnist'
        elif 'cifar' in path_lower:
            return 'cifar10'
        else:
            # Default fallback (can adjust if needed)
            return 'mnist'

    def _aggregate_weights(self, weights_a: Dict[str, np.ndarray],
                           weights_b: Dict[str, np.ndarray], alpha: float) -> Dict[str, np.ndarray]:
        """Compute weighted average of two model weight dictionaries."""
        aggregated = {}
        bn_keys = ["running_mean", "running_var", "num_batches_tracked"]
        for key in weights_a:
            if any(bn_key in key for bn_key in bn_keys):
                # Preserve BatchNorm buffers from model A (FedEraser model) without mixing
                aggregated[key] = weights_a[key]
            elif key in weights_b:
                aggregated[key] = alpha * weights_a[key] + (1 - alpha) * weights_b[key]
            else:
                # If a key is missing in one of the models, take model A's weight
                aggregated[key] = weights_a[key]
        return aggregated

    def _evaluate_performance(self, model: nn.Module) -> Tuple[float, float, float]:
        """Evaluate the given model on test set for loss, accuracy, and backdoor success rate."""
        model.eval()
        # Standard test loss and accuracy
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        test_loss /= total
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        # Backdoor success rate on test set
        backdoor_sr = evaluate_backdoor(model, self.test_loader, self.device)
        return test_loss, accuracy, backdoor_sr

    def run(self,
            client_to_remove: int,
            max_rounds: Optional[int] = None,
            batch_size: int = 64,
            lr: float = 0.01,
            local_epochs: int = 1,
            kd_temperature: float = 2.0,
            kd_alpha: float = 0.5,
            output_excel_path: Optional[str] = None) -> Tuple[nn.Module, Dict]:
        total_rounds = self.federaser.current_round if hasattr(self.federaser, 'current_round') else len(self.aggregation_history)
        if max_rounds is not None and max_rounds < total_rounds:
            rounds_to_process = list(range(max_rounds))
        else:
            rounds_to_process = list(range(total_rounds))

        self.remaining_clients = self.federaser.select_remaining_clients(client_to_remove)
        num_remaining_clients = len(self.remaining_clients)
        self.train_loaders, self.test_loader = get_partitioned_data(
            dataset_name=self.dataset,
            num_clients=num_remaining_clients,
            batch_size=batch_size,
            partition_type='iid'
        )

        self.student_model = subtract_model(model_path=self.model_path,
                                           client_to_remove=client_to_remove,
                                           history_path=self.history_path,
                                           device=self.device,
                                           config=self.config)  # 使用 config 初始化模型
        self.student_model.to(self.device)
        self.student_model.train()

        current_fed_weights = None
        current_fed_model = None

        for round_idx in rounds_to_process:
            current_round = round_idx
            print(f"\n--- FedEraser Round {current_round} ---")

            try:
                if current_fed_weights is not None:
                    starting_global = current_fed_weights
                else:
                    starting_global = self.federaser.get_global_model_at_round(current_round)

                remaining_client_models = [copy.deepcopy(starting_global) for _ in self.remaining_clients]
                new_client_models = self.federaser.train_clients_one_step(
                    client_models=remaining_client_models,
                    global_model=starting_global,
                    train_loaders=self.train_loaders,
                    lr=lr,
                    epochs=local_epochs
                )

                sample_sizes = [len(loader.dataset) for loader in self.train_loaders]
                aggregated_weights = fedavg(new_client_models, sample_sizes)

                old_global_weights = self.federaser.get_global_model_at_round(current_round)
                old_client_models = []
                for cid in self.remaining_clients:
                    if current_round in self.client_updates_history and cid in self.client_updates_history[current_round]:
                        old_client_models.append(self.client_updates_history[current_round][cid]['weights'])
                if len(old_client_models) > 0:
                    calibrated_weights = self.federaser.unlearning_step_once(
                        old_client_models, new_client_models, old_global_weights, aggregated_weights
                    )
                    # Skip BatchNorm buffer keys: preserve buffers from aggregated (post-unlearning) model
                    bn_keys = ["running_mean", "running_var", "num_batches_tracked"]
                    for key in calibrated_weights:
                        if any(bn_key in key for bn_key in bn_keys):
                            calibrated_weights[key] = aggregated_weights[key]
                    current_fed_weights = calibrated_weights
                else:
                    current_fed_weights = aggregated_weights

                current_fed_model = model_init(self.config).to(self.device)  # 使用 config 初始化模型
                current_fed_model = set_model_weights(current_fed_model, current_fed_weights, self.device)
            except Exception as e:
                print(f"Error in FedEraser round {current_round}: {e}")
                continue

            if self.kd_epochs_per_round > 0:
                self.student_model.train()
                optimizer = optim.SGD(self.student_model.parameters(), lr=lr, momentum=0.9)
                for epoch in range(self.kd_epochs_per_round):
                    for data, target in self.train_loaders[0]:
                        data, target = data.to(self.device), target.to(self.device)
                        optimizer.zero_grad()
                        student_out = self.student_model(data)
                        with torch.no_grad():
                            teacher_out = self.teacher_model(data)
                        soft_loss = F.kl_div(
                            F.log_softmax(student_out / kd_temperature, dim=1),
                            F.softmax(teacher_out / kd_temperature, dim=1),
                            reduction='batchmean'
                        ) * (kd_temperature ** 2)
                        hard_loss = F.cross_entropy(student_out, target)
                        loss = kd_alpha * soft_loss + (1 - kd_alpha) * hard_loss

                        loss.backward()
                        optimizer.step()

                self.student_model.eval()
                student_loss, student_acc, student_sr = self._evaluate_performance(self.student_model)
                print(f"Student Model: Test Loss = {student_loss:.4f}, Test Accuracy = {student_acc:.2f}%, Backdoor SR = {student_sr:.2f}%")

            test_loss, test_acc, backdoor_sr = self._evaluate_performance(current_fed_model)
            print(f"Round {current_round}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.2f}%, Backdoor SR = {backdoor_sr:.2f}%")

            if self.kd_epochs_per_round > 0:
                fed_acc = test_acc
                kd_acc = student_acc
                new_alpha = fed_acc / (fed_acc + kd_acc + 1e-8)
                self.aggregation_alpha = new_alpha
                print(f"[Dynamic Alpha] Updated aggregation alpha to {self.aggregation_alpha:.4f}")

            if self.kd_epochs_per_round > 0 and self.aggregation_interval > 0 and (current_round + 1) % self.aggregation_interval == 0:
                fed_weights = current_fed_weights
                kd_weights = get_model_weights(self.student_model)
                combined_weights = self._aggregate_weights(fed_weights, kd_weights, self.aggregation_alpha)
                current_fed_weights = combined_weights
                current_fed_model = set_model_weights(current_fed_model, combined_weights, self.device)
                self.student_model = set_model_weights(self.student_model, combined_weights, self.device)
                print(f"[Aggregation] Combined FedEraser and KD models at round {current_round} with alpha={self.aggregation_alpha:.2f}")

            self.history['round'].append(current_round)
            self.history['test_loss'].append(test_loss)
            self.history['test_accuracy'].append(test_acc)
            self.history['backdoor_success_rate'].append(backdoor_sr)

        final_model = current_fed_model
        if output_excel_path is None:
            output_excel_path = os.path.join(self.save_dir, 'hybrid_metrics.xlsx')
        save_metrics_to_excel(self.history, output_excel_path)
        print(f"Metrics saved to {output_excel_path}")

        return final_model, self.history