import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from model import model_init
from data_preprocess import get_partitioned_data
from fedavg import fedavg, get_model_weights, set_model_weights
from backdoor import BackdoorAttack, evaluate_backdoor
from server import Server
import pickle
import sys
import argparse

sys.path.append(".")

# Import unlearning methods
from federaser import FedEraser
from kd import kd_unlearning
from hybrideraser import HybridEraser
from utils.metrics_logger import save_metrics_to_excel, plot_metrics

def train(model, train_loader, optimizer, device, epochs=1):
    """Train a client model"""
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model

def test(model, test_loader, device):
    """Test a model"""
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
    
    print(f"Test set: Average loss: {test_loss:.4f}, "
         f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    
    return test_loss, accuracy

def train_federated(config):
    """Run federated learning training"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize server for history tracking
    dummy_model = model_init(config).to(device)
    model_structure = {name: param.clone().detach().cpu().numpy() for name, 
                       param in dummy_model.named_parameters()}
    server = Server(model_structure, save_dir=config['history_dir'])

    # Load data
    print("\nLoading and partitioning data...")
    train_loaders, test_loader = get_partitioned_data(
        dataset_name=config['dataset'],
        num_clients=config['num_clients'],
        batch_size=config['batch_size'],
        partition_type=config['partition_type'],
        backdoor_client=config['backdoor_client'],
        backdoor_target_label=config['backdoor_target'],
        backdoor_poison_ratio=config['backdoor_ratio'],
        backdoor_trigger_pattern=config['backdoor_pattern'],
        backdoor_trigger_size=config['backdoor_size']
    )
    
    # debug use
    
    backdoor_pattern = config['backdoor_pattern']
    backdoor_size = config['backdoor_size']
    dataset = config['dataset']
    if dataset == 'mnist':
        mean, std = 0.1307, 0.3081
    elif dataset == 'fashion_mnist':
        mean, std = 0.2860, 0.3530
    elif dataset == 'cifar10':
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    else:
        mean, std = 0.5, 0.5  # fallback default

    # Display a sample image from training data for debugging
    
    train_data, train_label = next(iter(train_loaders[0]))
    train_img = train_data[0].cpu().clone()
    if isinstance(mean, list):
        for c in range(train_img.shape[0]):
            train_img[c] = train_img[c] * std[c] + mean[c]
    else:
        train_img = train_img * std + mean
    train_img = train_img.clip(0, 1)

    train_img_np = train_img.permute(1, 2, 0).numpy()
    plt.imshow(train_img_np, cmap='gray' if train_img_np.shape[2] == 1 else None)
    plt.title(f"Train Label: {train_label[0].item()}")
    plt.axis('off')
    os.makedirs("debug_images", exist_ok=True)
    plt.show
    plt.savefig("debug_images/sample_train_image.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved debug image to debug_images/sample_train_image.png")
    
    # Display the train image with trigger
    
    img = train_img.clone()
    c, h, w = img.shape
    color = torch.ones(c, device=img.device) if c == 1 else torch.tensor([1.0, 0, 0], device=img.device)

    if backdoor_pattern == 'cross':
        cx, cy = w // 2, h // 2
        size = backdoor_size // 2
        img[:, cy - size:cy + size + 1, cx] = color.view(c, 1)
        img[:, cy, cx - size:cx + size + 1] = color.view(c, 1)
    elif backdoor_pattern == 'square':
        sx = w - backdoor_size - 2
        sy = h - backdoor_size - 2
        img[:, sy:sy + backdoor_size, sx:sx + backdoor_size] = color.view(c, 1, 1)
        
    poisoned_train_img_np = img.permute(1, 2, 0).numpy()
    plt.imshow(poisoned_train_img_np, cmap='gray' if poisoned_train_img_np.shape[2] == 1 else None)
    plt.title(f"Test Label: {train_label[0].item()}")
    plt.axis('off')
    plt.show
    plt.savefig("debug_images/sample_poisoned_train_image.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved debug image to debug_images/sample_poisoned_train_image.png")
    
    test_data, test_label = next(iter(test_loader))
    test_img = test_data[0].cpu().clone()
    if isinstance(mean, list):
        for c in range(test_img.shape[0]):
            test_img[c] = test_img[c] * std[c] + mean[c]
    else:
        test_img = test_img * std + mean
    test_img = test_img.clip(0, 1)
    test_img_np = test_img.permute(1, 2, 0).numpy()
    plt.imshow(test_img_np, cmap='gray' if test_img_np.shape[2] == 1 else None)
    plt.title(f"Test Label: {test_label[0].item()}")
    plt.axis('off')
    plt.show
    plt.savefig("debug_images/sample_test_image.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved debug image to debug_images/sample_test_image.png")
    
    # Display the test image with trigger
    
        # Display the train image with trigger
    
    img = test_img.clone()
    c, h, w = img.shape
    color = torch.ones(c, device=img.device) if c == 1 else torch.tensor([1.0, 0, 0], device=img.device)

    if backdoor_pattern == 'cross':
        cx, cy = w // 2, h // 2
        size = backdoor_size // 2
        img[:, cy - size:cy + size + 1, cx] = color.view(c, 1)
        img[:, cy, cx - size:cx + size + 1] = color.view(c, 1)
    elif backdoor_pattern == 'square':
        sx = w - backdoor_size - 2
        sy = h - backdoor_size - 2
        img[:, sy:sy + backdoor_size, sx:sx + backdoor_size] = color.view(c, 1, 1)
        
    poisoned_test_img_np = img.permute(1, 2, 0).numpy()
    plt.imshow(poisoned_test_img_np, cmap='gray' if poisoned_test_img_np.shape[2] == 1 else None)
    plt.title(f"Test Label: {test_label[0].item()}")
    plt.axis('off')
    plt.show
    plt.savefig("debug_images/sample_poisoned_test_image.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved debug image to debug_images/sample_poisoned_test_image.png")
           
    # Initialize global model
    global_model = model_init(config).to(device)

    # Start federated learning
    print("\nStarting federated learning...")

    # Initialize metrics dictionary
    fl_metrics = {
        'round': [],
        'accuracy': [],
        'loss': [],
        'backdoor_success_rate': []
    }

    for round_idx in range(config['rounds']):
        print(f"\n--- Round {round_idx+1}/{config['rounds']} ---")

        # Initialize client models from global model
        client_models = []
        for i in range(config['num_clients']):
            client_model = model_init(config).to(device)
            client_model.load_state_dict(global_model.state_dict())
            client_models.append(client_model)

        # Train each client model
        client_weights = []
        sample_sizes = []

        for client_idx, (client_model, train_loader) in enumerate(zip(client_models, train_loaders)):
            print(f"Training client {client_idx+1}/{config['num_clients']}")
            optimizer = optim.SGD(client_model.parameters(), lr=config['lr'], momentum=config['momentum'])

            sample_sizes.append(len(train_loader.dataset))

            client_model = train(client_model, train_loader, optimizer, device, epochs=config['epochs'])

            weights = get_model_weights(client_model)
            client_weights.append(weights)

            server.store_client_updates(client_idx, weights, len(train_loader.dataset))

        # Aggregate client models using FedAvg
        global_weights = fedavg(client_weights, sample_sizes)
        global_model = set_model_weights(global_model, global_weights, device)

        server.store_aggregation_result(global_weights)

        # Evaluate global model
        print(f"\nEvaluating global model after round {round_idx+1}...")
        test_loss, test_accuracy = test(global_model, test_loader, device)

        backdoor_sr = 0
        if config['backdoor_client'] is not None:
            backdoor_sr = evaluate_backdoor(
                global_model, 
                test_loader, 
                device,
                trigger_pattern=config['backdoor_pattern'],
                trigger_size=config['backdoor_size'],
                target_label=config['backdoor_target']
            )

        # Store metrics
        fl_metrics['round'].append(round_idx + 1)
        fl_metrics['accuracy'].append(test_accuracy)
        fl_metrics['loss'].append(test_loss)
        fl_metrics['backdoor_success_rate'].append(backdoor_sr)

        server.save_history()
        server.next_round()

    # Final evaluation
    print("\n--- Final Evaluation ---")
    test_loss, test_accuracy = test(global_model, test_loader, device)

    backdoor_success_rate = None
    if config['backdoor_client'] is not None:
        print("\nEvaluating backdoor attack success rate on final model:")
        backdoor_success_rate = evaluate_backdoor(
            global_model, 
            test_loader, 
            device,
            trigger_pattern=config['backdoor_pattern'],
            trigger_size=config['backdoor_size'],
            target_label=config['backdoor_target']
        )

        if backdoor_success_rate > 50:
            print("\nBackdoor attack successful!")
        else:
            print("\nBackdoor attack unsuccessful.")

    history_path = server.save_complete_history()

    model_path = os.path.join(config['save_dir'], f'{config["dataset"]}_fedavg_model.pth')
    torch.save(global_model.state_dict(), model_path)
    print(f"\nTraining completed and model saved to {model_path}")
    print(f"Complete history saved to {history_path}")

    # Save metrics to Excel and plot
    save_metrics_to_excel(fl_metrics, os.path.join(config['save_dir'], "fed_training_metrics.xlsx"))
    plot_metrics(fl_metrics, config['save_dir'], "Federated Training")

    return global_model, model_path, history_path, test_loader, backdoor_success_rate, train_loaders

def unlearn_client(config, global_model_path, history_path, device, test_loader, original_backdoor_success_rate=None):
    """Unlearn a specific client's contribution"""
    print(f"\n--- Unlearning Client {config['unlearn_client']} Using {config['unlearn_method']} ---")
    
    unlearn_start_time = time.time()
    
    # Load the original model for comparison
    original_model = model_init(config).to(device)
    original_model.load_state_dict(torch.load(global_model_path, map_location=device))
    
    # Evaluate original model's backdoor success rate if it wasn't provided
    if original_backdoor_success_rate is None and config['backdoor_client'] is not None:
        print("\nEvaluating original model's backdoor success rate:")
        original_backdoor_success_rate = evaluate_backdoor(
            original_model,
            test_loader,
            device,
            trigger_pattern=config['backdoor_pattern'],
            trigger_size=config['backdoor_size'],
            target_label=config['backdoor_target']
        )
    
    if config['unlearn_method'] == 'federaser':
        # Create directory for unlearned models
        unlearned_dir = os.path.join(config['save_dir'], 'unlearned')
        os.makedirs(unlearned_dir, exist_ok=True)
        
        # Setup unlearned model path
        unlearned_model_path = os.path.join(unlearned_dir, f"{config['dataset']}_federaser_unlearned_c{config['unlearn_client']}.pth")
        
        # Create FedEraser instance
        federaser = FedEraser(
            model_path=global_model_path,
            history_path=history_path,
            config=config,  # 传入 config 配置
            history_dir=config['history_dir'],
            save_dir=unlearned_dir,
            device=device
        )
        
        # Run FedEraser unlearning
        final_weights, history = federaser.run(
            client_to_remove=config['unlearn_client'],
            save_model_path=unlearned_model_path,
            batch_size=config['batch_size'],
            lr=config['lr'],
            epochs=config['federaser_epochs'],
            unlearn_rounds=config.get('federaser_specific_rounds'),
            max_rounds=config.get('federaser_max_rounds', 10)
        )
        
        # Load unlearned model for evaluation
        unlearned_model = model_init(config).to(device)
        unlearned_model.load_state_dict(torch.load(unlearned_model_path, map_location=device))
        
    elif config['unlearn_method'] == 'kd':
        # Create directory for unlearned models
        unlearned_dir = os.path.join(config['save_dir'], 'unlearned')
        os.makedirs(unlearned_dir, exist_ok=True)
        
        # Setup unlearned model path
        unlearned_model_path = os.path.join(unlearned_dir, f"{config['dataset']}_kd_unlearned_c{config['unlearn_client']}.pth")
        
        # Run Knowledge Distillation unlearning
        unlearned_model, history = kd_unlearning(
            model_path=global_model_path,
            history_path=history_path,
            client_to_remove=config['unlearn_client'],
            save_model_path=unlearned_model_path,
            batch_size=config['batch_size'],
            kd_epochs=config['kd_epochs'],
            kd_temperature=config['kd_temperature'],
            kd_alpha=config['kd_alpha'],
            kd_lr=config['lr'],
            config=config
        )
        # Load unlearned model for evaluation
        unlearned_model = model_init(config).to(device)
        unlearned_model.load_state_dict(torch.load(unlearned_model_path, map_location=device))
        
    elif config['unlearn_method'] == 'hybrid':
        # Create directory for unlearned models
        unlearned_dir = os.path.join(config['save_dir'], 'unlearned')
        os.makedirs(unlearned_dir, exist_ok=True)

        # Setup unlearned model path
        unlearned_model_path = os.path.join(unlearned_dir, f"{config['dataset']}_hybrid_unlearned_c{config['unlearn_client']}.pth")

        # Run HybridEraser unlearning
        hybrid_eraser = HybridEraser(
            model_path=global_model_path,
            history_path=history_path,
            config=config,  # 传入 config 参数
            device=device,
            kd_epochs_per_round=config['kd_epochs_per_round'],
            aggregation_interval=config['aggregation_interval'],
            aggregation_alpha=config['aggregation_alpha'],
            save_dir=os.path.join(unlearned_dir, 'hybrid_results')
        )

        unlearned_model, history = hybrid_eraser.run(
            client_to_remove=config['unlearn_client'],
            max_rounds=config.get('federaser_max_rounds', 10),
            batch_size=config['batch_size'],
            lr=config['lr'],
            local_epochs=config['federaser_epochs'],
            kd_temperature=config['kd_temperature'],
            kd_alpha=config['kd_alpha'],
            output_excel_path=os.path.join(unlearned_dir, 'hybrid_metrics.xlsx')
        )

        # Save model
        torch.save(unlearned_model.state_dict(), unlearned_model_path)
        # Load unlearned model for evaluation
        unlearned_model = model_init(config).to(device)
        unlearned_model.load_state_dict(torch.load(unlearned_model_path, map_location=device))
        
    unlearn_time = time.time() - unlearn_start_time
    print(f"\nUnlearning completed in {unlearn_time:.2f} seconds")
    
    # Evaluate unlearned model
    print("\n--- Unlearned Model Evaluation ---")
    test_loss, test_accuracy = test(unlearned_model, test_loader, device)
    
    # Always evaluate backdoor success rate after unlearning
    backdoor_after_unlearning = None
    if config['backdoor_client'] is not None:
        print("\nEvaluating backdoor success rate AFTER unlearning:")
        backdoor_after_unlearning = evaluate_backdoor(
            unlearned_model, 
            test_loader, 
            device,
            trigger_pattern=config['backdoor_pattern'],
            trigger_size=config['backdoor_size'],
            target_label=config['backdoor_target']
        )
        
        # Compare backdoor success rate before and after unlearning
        if original_backdoor_success_rate is not None:
            backdoor_reduction = original_backdoor_success_rate - backdoor_after_unlearning
            print(f"\nBackdoor success rate reduction: {backdoor_reduction:.2f}%  (from {original_backdoor_success_rate:.2f}% to {backdoor_after_unlearning:.2f}%)")
            
            if backdoor_after_unlearning < 10:
                print("Backdoor was successfully removed by unlearning!")
            elif backdoor_reduction > 50:
                print("Backdoor was significantly reduced by unlearning.")
            elif backdoor_reduction > 20:
                print("Backdoor was moderately reduced by unlearning.")
            elif backdoor_reduction > 0:
                print("Backdoor was slightly reduced by unlearning.")
            else:
                print("Backdoor remains unchanged or worsened after unlearning.")
    
    return unlearned_model, unlearned_model_path, backdoor_after_unlearning

def retrain_without_backdoor(config, global_model_path, train_loaders, test_loader, device, backdoored_clients=None):
    """
    Retrain the model using only non-poisoned clients' data
    
    Args:
        config: Configuration dictionary
        global_model_path: Path to the original model
        train_loaders: List of client training data loaders
        test_loader: Test data loader
        device: Computing device
        backdoored_clients: List of client IDs with backdoors (default: [config['backdoor_client']])
        
    Returns:
        retrained_model, retrained_model_path, backdoor_after_retrain
    """
    print(f"\n--- Retraining Model Without Backdoored Clients ---")
    
    # Set default backdoored clients if not provided
    if backdoored_clients is None and config['backdoor_client'] is not None:
        backdoored_clients = [config['backdoor_client']]
    elif backdoored_clients is None:
        backdoored_clients = []
    
    # Create directory for retrained models
    retrained_dir = os.path.join(config['save_dir'], 'retrained')
    os.makedirs(retrained_dir, exist_ok=True)
    
    # Setup retrained model path
    backdoored_str = '_'.join([f'c{c}' for c in backdoored_clients]) if backdoored_clients else 'none'
    retrained_model_path = os.path.join(retrained_dir, f'{config["dataset"]}_retrained_without_{backdoored_str}.pth')
    
    # Load the original model
    original_model = model_init(config).to(device)
    original_model.load_state_dict(torch.load(global_model_path, map_location=device))
    
    # Evaluate original model's backdoor success rate
    print("\nEvaluating original model's backdoor success rate:")
    original_backdoor_success_rate = None
    if config['backdoor_client'] is not None:
        original_backdoor_success_rate = evaluate_backdoor(
            original_model,
            test_loader,
            device,
            trigger_pattern=config['backdoor_pattern'],
            trigger_size=config['backdoor_size'],
            target_label=config['backdoor_target']
        )
    
    retrain_start_time = time.time()
    
    # Create a new model for retraining
    retrained_model = model_init(config).to(device)
    
    # retrained_model.load_state_dict(original_model.state_dict())
    # Create optimizer
    optimizer = optim.SGD(retrained_model.parameters(), lr=config['lr'], momentum=config['momentum'])
    
    # Select non-backdoored clients
    clean_client_loaders = []
    for client_idx, loader in enumerate(train_loaders):
        if client_idx not in backdoored_clients:
            clean_client_loaders.append(loader)
    
    if not clean_client_loaders:
        print("Error: No clean clients available for retraining.")
        return None, None, None
    
    print(f"Retraining with {len(clean_client_loaders)} clean clients for {config['retrain_rounds']} rounds...")
    
    # Start retraining
    for round_idx in range(config['retrain_rounds']):
        print(f"\n--- Retraining Round {round_idx+1}/{config['retrain_rounds']} ---")
        
        # Initialize client models from current retrained model
        client_models = []
        for _ in range(len(clean_client_loaders)):
            client_model = model_init(config).to(device)
            client_model.load_state_dict(retrained_model.state_dict())
            client_models.append(client_model)
        
        # Train each client model
        client_weights = []
        sample_sizes = []
        
        for client_idx, (client_model, train_loader) in enumerate(zip(client_models, clean_client_loaders)):
            print(f"Training clean client {client_idx+1}/{len(clean_client_loaders)}")
            client_optimizer = optim.SGD(client_model.parameters(), lr=config['lr'], momentum=config['momentum'])
            
            # Record dataset size for weighted averaging
            sample_sizes.append(len(train_loader.dataset))
            
            # Train client model
            client_model = train(client_model, train_loader, client_optimizer, device, epochs=config['epochs'])
            
            # Extract model weights
            weights = get_model_weights(client_model)
            client_weights.append(weights)
        
        # Aggregate client models using FedAvg
        global_weights = fedavg(client_weights, sample_sizes)
        retrained_model = set_model_weights(retrained_model, global_weights, device)
        
        # Evaluate retrained model
        print(f"\nEvaluating retrained model after round {round_idx+1}...")
        test_loss, test_accuracy = test(retrained_model, test_loader, device)
        
        # Evaluate backdoor if specified
        if config['backdoor_client'] is not None:
            backdoor_sr = evaluate_backdoor(
                retrained_model, 
                test_loader, 
                device,
                trigger_pattern=config['backdoor_pattern'],
                trigger_size=config['backdoor_size'],
                target_label=config['backdoor_target']
            )
    
    retrain_time = time.time() - retrain_start_time
    print(f"\nRetraining completed in {retrain_time:.2f} seconds")
    
    # Save retrained model
    torch.save(retrained_model.state_dict(), retrained_model_path)
    print(f"Retrained model saved to {retrained_model_path}")
    
    # Final evaluation
    print("\n--- Final Retrained Model Evaluation ---")
    test_loss, test_accuracy = test(retrained_model, test_loader, device)
    
    # Evaluate backdoor success rate after retraining
    backdoor_after_retrain = None
    if config['backdoor_client'] is not None:
        print("\nEvaluating backdoor success rate AFTER retraining:")
        backdoor_after_retrain = evaluate_backdoor(
            retrained_model, 
            test_loader, 
            device,
            trigger_pattern=config['backdoor_pattern'],
            trigger_size=config['backdoor_size'],
            target_label=config['backdoor_target']
        )
        
        # Compare backdoor success rate before and after retraining
        if original_backdoor_success_rate is not None:
            backdoor_reduction = original_backdoor_success_rate - backdoor_after_retrain
            print(f"\nBackdoor success rate reduction: {backdoor_reduction:.2f}%  (from {original_backdoor_success_rate:.2f}% to {backdoor_after_retrain:.2f}%)")
            
            if backdoor_after_retrain < 10:
                print("Backdoor was successfully removed by retraining!")
            elif backdoor_reduction > 50:
                print("Backdoor was significantly reduced by retraining.")
            elif backdoor_reduction > 20:
                print("Backdoor was moderately reduced by retraining.")
            elif backdoor_reduction > 0:
                print("Backdoor was slightly reduced by retraining.")
            else:
                print("Backdoor remains unchanged or worsened after retraining.")
    
    return retrained_model, retrained_model_path, backdoor_after_retrain

def main():
    # Set configuration parameters directly
    config = {
        # === Basic Federated Learning Configuration ===
        "dataset": "fashion_mnist",           # Options: 'mnist', 'fashion_mnist', 'cifar10'
        "model_name": "net_mnist",       # Model options: 'net_mnist', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg18'
        "num_clients": 5,               # Number of federated clients
        "batch_size": 128,              # Batch size for training
        "partition_type": "iid",        # Data partitioning: 'iid' or 'non-iid'
        "epochs": 10,                   # Local training epochs per client
        "rounds": 50,                   # Total number of federated training rounds
        "lr": 0.01,                    # Learning rate
        "momentum": 0.9,                # SGD momentum
        "save_dir": "./models",         # Directory to save trained models
        "history_dir": "./fl_history",  # Directory to save client updates history (used by FedEraser)

        # === Backdoor Attack Configuration ===
        "backdoor_client": 0,           # ID of the client that will receive poisoned data (set to None to disable)
        "backdoor_target": 7,           # Target label for the backdoor attack
        "backdoor_ratio": 0.3,          # Fraction of the client's data that is poisoned (0 to 1)
        "backdoor_pattern": "square",   # Trigger pattern: 'square', 'cross', etc.
        "backdoor_size": 3,             # Size of the trigger (in pixels, square shape)

        # === Unlearning Method Configuration ===
        "unlearn_method": "federaser",  # Options: 'federaser', 'kd', 'hybrid'
        "unlearn_client": 0,            # ID of the client to be unlearned

        # --- FedEraser Specific ---
        "federaser_epochs": 2,         # Local epochs per correction step
        "federaser_max_rounds": 50,     # Maximum rounds to roll back in FedEraser

        # --- Knowledge Distillation (KD) Specific ---
        "kd_epochs": 200,               # Total epochs for KD retraining
        "kd_temperature": 0.1,          # Temperature for softmax in KD
        "kd_alpha": 1.0,                # Weight for soft loss in KD (1 = only soft labels)

        # --- HybridEraser Specific ---
        "kd_epochs_per_round": 5,       # KD epochs per round in HybridEraser
        "aggregation_interval": 1,      # How often KD and FedEraser alternate (in rounds)
        "aggregation_alpha": 0.5,       # Weight to combine FedEraser and KD (it's unused as dynamic alpha is used)

        # === Retraining Configuration ===
        "retrain_rounds": 50,           # Rounds of retraining with only clean clients
        "perform_retrain": False,       # Set to True to enable retraining phase

        # === Optional Execution Controls ===
        "skip_training": False,         # If True, skip training and directly load model from disk
        "model_path": "./models/mnist_fedavg_model.pth",  # Path to pretrained model (used if skip_training = True)
        "history_path": "./fl_history/complete_history_r0-r19.pkl"  # Path to training history (used by unlearning)
    }
        
    parser = argparse.ArgumentParser()

    for key, value in config.items():
        arg_type = type(value) if value is not None else str
        if arg_type is bool:
            parser.add_argument(f'--{key}', type=str, default=None)
        else:
            parser.add_argument(f'--{key}', type=arg_type, default=None)

    args = parser.parse_args()
    args_dict = vars(args)

    for key, val in args_dict.items():
        if val is not None:
            if isinstance(config.get(key), bool):
                config[key] = val.lower() in ['true', '1', 'yes']
            else:
                config[key] = val
                
    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['history_dir'], exist_ok=True)
    metrics_dir = os.path.join(config['save_dir'], 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Federated learning training
    original_backdoor_success_rate = None
    train_loaders = None
    fl_metrics = None
    
    if not config['skip_training']:
        # Initialize metrics dictionary
        fl_metrics = {
            'round': [],
            'accuracy': [],
            'loss': [],
            'backdoor_success_rate': []
        }
        
        global_model, model_path, history_path, test_loader, original_backdoor_success_rate, train_loaders = train_federated(config)
        
        # Collect metrics during training
        for round_idx in range(config['rounds']):
            # Evaluate model
            test_loss, test_accuracy = test(global_model, test_loader, device)
            
            # Record metrics
            fl_metrics['round'].append(round_idx + 1)
            fl_metrics['accuracy'].append(test_accuracy)
            fl_metrics['loss'].append(test_loss)
            
            # Evaluate backdoor if specified
            backdoor_sr = None
            if config['backdoor_client'] is not None:
                backdoor_sr = evaluate_backdoor(
                    global_model, 
                    test_loader, 
                    device,
                    trigger_pattern=config['backdoor_pattern'],
                    trigger_size=config['backdoor_size'],
                    target_label=config['backdoor_target']
                )
                fl_metrics['backdoor_success_rate'].append(backdoor_sr)
            else:
                fl_metrics['backdoor_success_rate'].append(0)
        
        # Save federated metrics to Excel and plot
        excel_path = os.path.join(metrics_dir, f"{config['dataset']}_federated_metrics.xlsx")
        save_metrics_to_excel(fl_metrics, excel_path)
        plot_metrics(fl_metrics, metrics_dir, f"{config['dataset']} Federated Learning")
    else:
        if config['model_path'] is None or config['history_path'] is None:
            print("Error: If skip-training is True, model-path and history-path must be provided.")
            return
        model_path = config['model_path']
        history_path = config['history_path']
        global_model = model_init(config).to(device)
        global_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded existing model from {model_path}")
        print(f"Using history from {history_path}")
        
        # Load data for evaluation and potential retraining
        train_loaders, test_loader = get_partitioned_data(
            dataset_name=config['dataset'],
            num_clients=config['num_clients'],
            batch_size=config['batch_size'],
            partition_type=config['partition_type'],
            backdoor_client=config['backdoor_client'],
            backdoor_target_label=config['backdoor_target'],
            backdoor_poison_ratio=config['backdoor_ratio'],
            backdoor_trigger_pattern=config['backdoor_pattern'],
            backdoor_trigger_size=config['backdoor_size']
        )
        
        # Evaluate backdoor if backdoor client is specified
        if config['backdoor_client'] is not None:
            print("\nEvaluating backdoor success rate on loaded model:")
            original_backdoor_success_rate = evaluate_backdoor(
                global_model,
                test_loader,
                device,
                trigger_pattern=config['backdoor_pattern'],
                trigger_size=config['backdoor_size'],
                target_label=config['backdoor_target']
            )
    
    # Collect results to compare methods
    backdoor_results = {
        'original': original_backdoor_success_rate,
        'unlearning': None,
        'retrain': None
    }
    
    unlearn_metrics = None
    # Client unlearning
    if config['unlearn_client'] is not None and config['unlearn_method'] != 'none':
        unlearned_model, unlearned_model_path, backdoor_after_unlearning = unlearn_client(
            config, 
            model_path, 
            history_path, 
            device, 
            test_loader,
            original_backdoor_success_rate
        )
        
        # Initialize and collect metrics based on different methods
        if config['unlearn_method'] == 'federaser':
            unlearn_metrics = {
                'round': [],             # Use round as x-axis
                'accuracy': [],
                'loss': [],
                'backdoor_success_rate': []
            }
            
            # Extract data from federaser history
            if isinstance(unlearned_model, tuple) and len(unlearned_model) > 1:
                history = unlearned_model[1]
                if isinstance(history, dict) and 'test_accuracy' in history:
                    for i, acc in enumerate(history['test_accuracy']):
                        unlearn_metrics['round'].append(i + 1)  # round starts from 1
                        unlearn_metrics['accuracy'].append(acc)
                        unlearn_metrics['loss'].append(history.get('test_loss', [0])[i] if i < len(history.get('test_loss', [])) else 0)
                        unlearn_metrics['backdoor_success_rate'].append(0)  # Default value, update last item later
                        
        elif config['unlearn_method'] == 'kd':
            unlearn_metrics = {
                'epoch': [],             # Use epoch as x-axis
                'accuracy': [],
                'loss': [],
                'backdoor_success_rate': []
            }
            
            # Extract data from kd history
            if isinstance(unlearned_model, tuple) and len(unlearned_model) > 1:
                history = unlearned_model[1]
                if isinstance(history, dict) and 'test_accuracy' in history:
                    for i, acc in enumerate(history['test_accuracy']):
                        unlearn_metrics['epoch'].append(i + 1)  # epoch starts from 1
                        unlearn_metrics['accuracy'].append(acc)
                        unlearn_metrics['loss'].append(history.get('test_loss', [0])[i] if i < len(history.get('test_loss', [])) else 0)
                        unlearn_metrics['backdoor_success_rate'].append(0)  # Default value, update last item later
                        
        elif config['unlearn_method'] == 'hybrid':
            unlearn_metrics = {
                'iteration': [],             # Use iteration as x-axis for hybrid
                'accuracy': [],
                'loss': [],
                'backdoor_success_rate': []
            }
            
            # Extract data from hybrid history
            if isinstance(unlearned_model, tuple) and len(unlearned_model) > 1:
                history = unlearned_model[1]
                
                # Debug print
                print("\n--- Hybrid Metrics Structure Debug ---")
                for key in history:
                    if isinstance(history[key], dict):
                        print(f"Main key: {key}")
                        for subkey in history[key]:
                            if isinstance(history[key][subkey], list):
                                print(f"  Subkey: {subkey}, Length: {len(history[key][subkey])}")
                            else:
                                print(f"  Subkey: {subkey}, Type: {type(history[key][subkey])}")
                
                # Try to get data from the hybrid key
                if 'hybrid' in history and isinstance(history['hybrid'], dict):
                    hybrid_data = history['hybrid']
                    
                    # Check which keys are available in hybrid data
                    available_keys = []
                    for key in ['iterations', 'test_accuracy', 'test_loss', 'time']:
                        if key in hybrid_data and isinstance(hybrid_data[key], list) and len(hybrid_data[key]) > 0:
                            available_keys.append(key)
                            print(f"Found valid hybrid key: {key}, Length: {len(hybrid_data[key])}")
                      
                    # If there's enough data, populate metrics
                    if 'test_accuracy' in available_keys:
                        print("Populating metrics with hybrid.test_accuracy")
                        for i, acc in enumerate(hybrid_data['test_accuracy']):
                            # Add iteration number
                            if 'iterations' in available_keys and i < len(hybrid_data['iterations']):
                                unlearn_metrics['iteration'].append(hybrid_data['iterations'][i])
                            else:
                                unlearn_metrics['iteration'].append(i + 1)
                            
                            # Add accuracy
                            unlearn_metrics['accuracy'].append(acc)
                            
                            # Add loss
                            if 'test_loss' in available_keys and i < len(hybrid_data['test_loss']):
                                unlearn_metrics['loss'].append(hybrid_data['test_loss'][i])
                            else:
                                unlearn_metrics['loss'].append(0)
                            
                            # Add backdoor success rate placeholder
                            unlearn_metrics['backdoor_success_rate'].append(0)
                
                # If can't get data from hybrid key, try other method keys
                elif not unlearn_metrics['accuracy'] and 'federaser' in history:

                    print("Trying to get data from federaser key")
                    fed_data = history['federaser']
                    if 'test_accuracy' in fed_data and isinstance(fed_data['test_accuracy'], list):
                        for i, acc in enumerate(fed_data['test_accuracy']):
                            unlearn_metrics['iteration'].append(i + 1)
                            unlearn_metrics['accuracy'].append(acc)
                            
                            if 'test_loss' in fed_data and i < len(fed_data['test_loss']):
                                unlearn_metrics['loss'].append(fed_data['test_loss'][i])
                            else:
                                unlearn_metrics['loss'].append(0)
                            
                            unlearn_metrics['backdoor_success_rate'].append(0)

                elif not unlearn_metrics['accuracy'] and 'kd' in history:
                    print("Trying to get data from kd key")
                    kd_data = history['kd']
                    if 'test_accuracy' in kd_data and isinstance(kd_data['test_accuracy'], list):
                        for i, acc in enumerate(kd_data['test_accuracy']):
                            unlearn_metrics['iteration'].append(i + 1)
                            unlearn_metrics['accuracy'].append(acc)
                            
                            if 'test_loss' in kd_data and i < len(kd_data['test_loss']):
                                unlearn_metrics['loss'].append(kd_data['test_loss'][i])
                            else:
                                unlearn_metrics['loss'].append(0)
                            
                            unlearn_metrics['backdoor_success_rate'].append(0)
                
                # If all attempts fail, create at least one data point
                if not unlearn_metrics['accuracy']:
                    print("Could not find valid metrics data, creating single data point")
                    unlearn_metrics['iteration'].append(1)
                    unlearn_metrics['accuracy'].append(test_accuracy)  # Use final accuracy
                    unlearn_metrics['loss'].append(test_loss)  # Use final loss
                    unlearn_metrics['backdoor_success_rate'].append(0)
        
        # Ensure there's at least one data point that can be saved
        if not unlearn_metrics or not unlearn_metrics.get('accuracy', []):
            print("Creating default metrics data point")
            key_name = 'round' if 'round' in unlearn_metrics else 'epoch' if 'epoch' in unlearn_metrics else 'iteration'
            unlearn_metrics = {
                key_name: [1],
                'accuracy': [0],
                'loss': [0],
                'backdoor_success_rate': [0]
            }
        
        # Update final backdoor success rate
        backdoor_results['unlearning'] = backdoor_after_unlearning
        
        # Update backdoor success rate in the last metrics entry
        if unlearn_metrics and 'backdoor_success_rate' in unlearn_metrics and len(unlearn_metrics['backdoor_success_rate']) > 0:
            unlearn_metrics['backdoor_success_rate'][-1] = backdoor_after_unlearning
        
        if unlearned_model is not None:
            print(f"\nUnlearned model saved to {unlearned_model_path}")
            
            # Save metrics to Excel and plot
            if unlearn_metrics:
                excel_path = os.path.join(metrics_dir, f"{config['dataset']}_{config['unlearn_method']}_unlearn_metrics.xlsx")
                save_metrics_to_excel(unlearn_metrics, excel_path)
                
                plot_metrics(unlearn_metrics, metrics_dir, f"{config['dataset']} {config['unlearn_method'].capitalize()} Unlearning")
    
    retrain_metrics = None
    # Retraining without backdoored clients
    if config['perform_retrain'] and train_loaders is not None:
        # Initialize retrain metrics
        retrain_metrics = {
            'round': [],
            'accuracy': [],
            'loss': [],
            'backdoor_success_rate': []
        }
        
        retrained_model, retrained_model_path, backdoor_after_retrain = retrain_without_backdoor(
            config,
            model_path,
            train_loaders,
            test_loader,
            device
        )
        
        # Collect metrics during retraining
        for round_idx in range(config['retrain_rounds']):
            # Evaluate model
            test_loss, test_accuracy = test(retrained_model, test_loader, device)
            
            # Record metrics
            retrain_metrics['round'].append(round_idx + 1)
            retrain_metrics['accuracy'].append(test_accuracy)
            retrain_metrics['loss'].append(test_loss)
            
            # Evaluate backdoor
            backdoor_sr = None
            if config['backdoor_client'] is not None:
                backdoor_sr = evaluate_backdoor(
                    retrained_model, 
                    test_loader, 
                    device,
                    trigger_pattern=config['backdoor_pattern'],
                    trigger_size=config['backdoor_size'],
                    target_label=config['backdoor_target']
                )
                retrain_metrics['backdoor_success_rate'].append(backdoor_sr)
            else:
                retrain_metrics['backdoor_success_rate'].append(0)
        
        backdoor_results['retrain'] = backdoor_after_retrain
        
        if retrained_model is not None:
            print(f"\nRetrained model saved to {retrained_model_path}")

    # Compare all methods
    if config['backdoor_client'] is not None:
        print("\n--- Backdoor Mitigation Methods Comparison ---")
        print(f"Original model backdoor success rate: {backdoor_results['original']:.2f}%")
        
        if backdoor_results['unlearning'] is not None:
            unlearn_reduction = backdoor_results['original'] - backdoor_results['unlearning']
            print(f"Unlearning ({config['unlearn_method']}) backdoor success rate: {backdoor_results['unlearning']:.2f}% (Reduction: {unlearn_reduction:.2f}%)")
        
        if backdoor_results['retrain'] is not None:
            retrain_reduction = backdoor_results['original'] - backdoor_results['retrain']
            print(f"Retraining backdoor success rate: {backdoor_results['retrain']:.2f}% (Reduction: {retrain_reduction:.2f}%)")
            
        print("\nEffectiveness ranking (lower backdoor success rate is better):")
        
        # Create a list of methods and their effectiveness
        methods = []
        if backdoor_results['original'] is not None:
            methods.append(('Original', backdoor_results['original']))
        if backdoor_results['unlearning'] is not None:
            methods.append((f'Unlearning ({config["unlearn_method"]})', backdoor_results['unlearning']))
        if backdoor_results['retrain'] is not None:
            methods.append(('Retraining', backdoor_results['retrain']))
        
        # Sort by backdoor success rate (lower is better)
        methods.sort(key=lambda x: x[1])
        
        # Print ranking
        for i, (method, rate) in enumerate(methods):
            print(f"{i+1}. {method}: {rate:.2f}%")
        
        # Create comparison bar chart
        plt.figure(figsize=(10, 6))
        method_names = [m[0] for m in methods]
        method_rates = [m[1] for m in methods]
        
        # Set y-axis range to 0-100%
        plt.ylim(0, 100)
        
        # Draw grid lines (before bars)
        plt.grid(True, axis='y')
        
        # Then draw bars
        bars = plt.bar(method_names, method_rates, color=['blue', 'green', 'orange'][:len(methods)], zorder=3)  # zorder=3 ensures bars appear above grid lines
        
        plt.title(f"{config['dataset']} - Backdoor Success Rate Comparison")
        plt.ylabel('Backdoor Success Rate (%)')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.2f}%', ha='center', va='bottom')
        
        # Save comparison chart
        plt.tight_layout()
        plots_dir = os.path.join(metrics_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f"{config['dataset']}_backdoor_comparison.png"))
        plt.close()
        
        # Save comparison to Excel
        comparison_data = {
            'Method': [m[0] for m in methods],
            'Backdoor Success Rate (%)': [m[1] for m in methods]
        }
        
        df = pd.DataFrame(comparison_data)
        excel_path = os.path.join(metrics_dir, f"{config['dataset']}_method_comparison.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"Comparison metrics saved to {excel_path}")
    
    print("\nAll operations completed!")

if __name__ == "__main__":
    main()