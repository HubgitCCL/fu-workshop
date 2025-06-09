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
from typing import Dict, List, Tuple, Any

# Correct imports for your project structure
from fedavg import fedavg, get_model_weights, set_model_weights
from model import model_init
from data_preprocess import get_partitioned_data
from backdoor import BackdoorAttack, evaluate_backdoor
from utils.metrics_logger import save_metrics_to_excel, plot_metrics

def subtract_model(model_path='./models/mnist_fedavg_model.pth', 
                  client_to_remove=0, 
                  history_path='./fl_history/complete_history_r0-r1.pkl',
                  unlearn_rounds=None,
                  device=None,
                  config=None):
    """
    Subtract specific client contribution from a federated learning model
    
    Args:
        model_path: Path to the federated learning model
        client_to_remove: ID of client to remove
        history_path: Path to the federated learning history
        unlearn_rounds: List of rounds to unlearn, None means all rounds
        device: Computation device
    
    Returns:
        Model after removing client contribution
    """
    # Start timing
    subtract_start_time = time.time()
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Removing contribution from client {client_to_remove}...")
    
    # Load federated learning model
    model_load_start = time.time()
    # Determine dataset from model path (if config not provided)
    if config:
        data_name = config['dataset']
        model = model_init(config).to(device)  # 使用 config 字典初始化模型
    else:
        # Determine dataset from model path
        if 'mnist' in model_path.lower():
            data_name = 'mnist'
        elif 'fashion' in model_path.lower():
            data_name = 'fashion_mnist'
        elif 'cifar' in model_path.lower():
            data_name = 'cifar10'
        else:
            data_name = 'mnist'  # Default to mnist
        model = model_init({'dataset': data_name}).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.2f} seconds")
    
    # Load history
    history_load_start = time.time()
    with open(history_path, 'rb') as f:
        complete_history = pickle.load(f)
    
    client_updates_history = complete_history['client_updates_history']
    aggregation_history = complete_history['aggregation_history']
    history_load_time = time.time() - history_load_start
    print(f"History loading time: {history_load_time:.2f} seconds")
    
    # Determine rounds to remove
    rounds = list(client_updates_history.keys()) if unlearn_rounds is None else unlearn_rounds
    
    # Calculate client contribution weights
    contribution_start = time.time()
    client_contribution = {}
    
    for round_num in rounds:
        if client_to_remove in client_updates_history[round_num]:
            client_data = client_updates_history[round_num][client_to_remove]
            client_weights = client_data['weights']
            client_sample_size = client_data['sample_size']
            
            # Get sample sizes of all clients in this round
            total_samples = sum(client_updates_history[round_num][client_id]['sample_size'] for client_id in client_updates_history[round_num])
            
            # Calculate client contribution ratio in this round
            contribution_ratio = client_sample_size / total_samples
            
            # Calculate client contribution for each parameter
            for key in client_weights:
                # Skip BatchNorm buffer keys
                if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                    continue
                if key not in client_contribution:
                    client_contribution[key] = np.zeros_like(client_weights[key])
                
                # Calculate client contribution based on FedAvg weight aggregation
                client_contribution[key] += client_weights[key] * contribution_ratio
    
    contribution_time = time.time() - contribution_start
    print(f"Client contribution calculation time: {contribution_time:.2f} seconds")
    
    # Subtract client contribution from aggregated model
    remove_start = time.time()
    current_weights = {}
    for name, param in model.named_parameters():
        current_weights[name] = param.detach().cpu().numpy()
    
    # Adjust remaining weights
    for key in current_weights:
        if key in client_contribution:
            # Remove client contribution
            adjusted_weights = current_weights[key] - client_contribution[key]
            
            # Update model parameters
            for name, param in model.named_parameters():
                if name == key:
                    param.data = torch.from_numpy(adjusted_weights).to(device)
    
    remove_time = time.time() - remove_start
    print(f"Client contribution removal time: {remove_time:.2f} seconds")
    
    # Total time
    subtract_total_time = time.time() - subtract_start_time
    print(f"Total model subtraction time: {subtract_total_time:.2f} seconds")
    
    print("Client contribution has been removed from the model")
    return model

def knowledge_distillation(teacher_model, 
                          student_model, 
                          train_loader, 
                          test_loader, 
                          temperature=2.0, 
                          alpha=0.5, 
                          epochs=10, 
                          lr=0.01,
                          device=None):
    """
    Perform knowledge distillation from teacher model to student model
    
    Args:
        teacher_model: Teacher model (original federated learning model)
        student_model: Student model (model after removing client contribution)
        train_loader: Training data loader
        test_loader: Test data loader
        temperature: Distillation temperature parameter
        alpha: Distillation loss weight
        epochs: Number of distillation training epochs
        lr: Learning rate
        device: Computation device
    """
    # Record start time
    kd_start_time = time.time()
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set optimizer
    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.9)
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'epoch_time': []
    }
    
    # Set teacher model to evaluation mode
    teacher_model.eval()
    
    print(f"Starting knowledge distillation training, total {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        student_model.train()
        epoch_loss = 0.0
        backdoor_sr = evaluate_backdoor(student_model, test_loader, device, 
                                trigger_pattern='square', trigger_size=3, 
                                target_label=7)
        history.setdefault('backdoor_success_rate', []).append(backdoor_sr)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            student_output = student_model(data)
            
            with torch.no_grad():
                teacher_output = teacher_model(data)
            
            # Calculate knowledge distillation loss (soft targets)
            soft_target_loss = F.kl_div(
                F.log_softmax(student_output / temperature, dim=1),
                F.softmax(teacher_output / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # Calculate standard classification loss (hard targets)
            hard_target_loss = F.cross_entropy(student_output, target)
            
            # Combined loss
            loss = alpha * soft_target_loss + (1 - alpha) * hard_target_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Record training loss
        avg_loss = epoch_loss / len(train_loader)
        
        # Evaluate on test set
        eval_start = time.time()
        test_loss, test_accuracy = evaluate_model(student_model, test_loader, device)
        eval_time = time.time() - eval_start
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)
        history['epoch_time'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print(f"Epoch time: {epoch_time:.2f} seconds (Training: {epoch_time-eval_time:.2f}s, Evaluation: {eval_time:.2f}s)")
    
    # Calculate total time
    kd_total_time = time.time() - kd_start_time
    print(f"Total knowledge distillation time: {kd_total_time:.2f} seconds")
    print("Knowledge distillation training completed!")
    
    # Add total time to history
    history['total_time'] = kd_total_time
    
    return student_model, history

def evaluate_model(model, test_loader, device):
    """Evaluate model performance on test set"""
    eval_start = time.time()
    
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    eval_time = time.time() - eval_start
    print(f"Model evaluation time: {eval_time:.2f} seconds")
    
    return test_loss, accuracy

def kd_unlearning(model_path='./models/mnist_fedavg_model.pth',
                 history_path='./fl_history/complete_history_r0-r1.pkl',
                 client_to_remove=0,
                 unlearning_rounds=None,
                 save_model_path='./models/unlearned_model.pth',
                 batch_size=64,
                 kd_epochs=10,
                 kd_temperature=2.0,
                 kd_alpha=0.5,
                 kd_lr=0.01,
                 dataset='mnist',
                 config=None):
    """
    Perform federated unlearning using knowledge distillation
    
    Args:
        model_path: Path to the federated learning model
        history_path: Path to the federated learning history
        client_to_remove: ID of client to remove
        unlearning_rounds: List of rounds to unlearn, None means all rounds
        save_model_path: Path to save the final model
        batch_size: Batch size
        kd_epochs: Number of knowledge distillation epochs
        kd_temperature: Distillation temperature parameter
        kd_alpha: Distillation loss weight
        kd_lr: Distillation learning rate
        dataset: Dataset type ('mnist', 'fashion_mnist', or 'cifar10')
    Returns:
        Processed model and training history
    """
    # Record total start time
    kd_unlearn_start = time.time()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_load_start = time.time()
    train_loaders, test_loader = get_partitioned_data(
        dataset_name=dataset,
        num_clients=1,
        batch_size=batch_size,
        partition_type='iid',
        backdoor_client=None,
        backdoor_target_label=None,
        backdoor_poison_ratio=0.0,
        backdoor_trigger_pattern=None,
        backdoor_trigger_size=0
    )
    data_load_time = time.time() - data_load_start
    print(f"Data loading time: {data_load_time:.2f} seconds")
    
    # Use last data loader as distillation training data
    train_loader = train_loaders[0]
    # Load teacher model (original federated learning model)
    teacher_load_start = time.time()
    if config:
        teacher_model = model_init(config).to(device)  # 使用 config 初始化教师模型
    else:
        teacher_model = model_init({'dataset': dataset}).to(device)
    teacher_model.load_state_dict(torch.load(model_path, map_location=device))
    teacher_load_time = time.time() - teacher_load_start
    print(f"Teacher model loading time: {teacher_load_time:.2f} seconds")
    
    # Evaluate teacher model performance
    print("Teacher model (original federated model) performance:")
    test_loss, test_accuracy = evaluate_model(teacher_model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Subtract client contribution from model
    print(f"Removing client {client_to_remove}'s contribution from the model...")
    subtract_start = time.time()
    student_model = subtract_model(
        model_path=model_path,
        client_to_remove=client_to_remove,
        history_path=history_path,
        unlearn_rounds=unlearning_rounds,
        device=device,
        config=config
    )
    subtract_time = time.time() - subtract_start
    print(f"Total client contribution removal time: {subtract_time:.2f} seconds")
    
    # Evaluate model performance after removing client contribution
    print("Model performance after removing client contribution:")
    test_loss, test_accuracy = evaluate_model(student_model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Use knowledge distillation to restore model performance
    print("Starting knowledge distillation to restore model performance...")
    kd_start = time.time()
    student_model, history = knowledge_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        test_loader=test_loader,
        temperature=kd_temperature,
        alpha=kd_alpha,
        epochs=kd_epochs,
        lr=kd_lr,
        device=device
    )
    kd_time = time.time() - kd_start
    print(f"Knowledge distillation training time: {kd_time:.2f} seconds")
    
    # Evaluate model performance after knowledge distillation
    print("Model performance after knowledge distillation:")
    test_loss, test_accuracy = evaluate_model(student_model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Save model
    if save_model_path:
        save_start = time.time()
        torch.save(student_model.state_dict(), save_model_path)
        save_time = time.time() - save_start
        print(f"Model saved to {save_model_path} (Saving time: {save_time:.2f}s)")
    
    # Calculate total time
    total_time = time.time() - kd_unlearn_start
    print(f"Total knowledge distillation unlearning time: {total_time:.2f} seconds")
    
    # Add time information to history
    history['subtract_time'] = subtract_time
    history['kd_time'] = kd_time
    history['total_time'] = total_time
    
    return student_model, history