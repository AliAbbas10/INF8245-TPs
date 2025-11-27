import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import csv
from fmnist import FashionMNIST, PytorchMLPFashionMNIST, CustomMLPFashionMNIST

np.random.seed(33)
torch.manual_seed(33)
torch.cuda.manual_seed_all(33)

def build_single_hidden_layer_pytorch(hidden_size):
    model = PytorchMLPFashionMNIST()
    model.layers = nn.Sequential(
        nn.Linear(784, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10)
    )
    return model

def build_single_hidden_layer_custom(hidden_size):
    from q1 import Dense, ReLULayer, SoftmaxLayer
    from q2 import MLP
    model = CustomMLPFashionMNIST()
    model.layers = MLP([
        Dense(784, hidden_size),
        ReLULayer(),
        Dense(hidden_size, 10),
        SoftmaxLayer()
    ])
    return model

def part1_validation_accuracy():
    # Part 1: Train
    learning_rates = [0.001, 0.002, 0.1]
    epochs = 10
    batch_size = 128
    dataset = FashionMNIST(batch_size=batch_size, val_perc=0.2)
    results = {}
    
    for lr in learning_rates:
        print(f"\nLR={lr}")
        
        # PyTorch
        pytorch_model = PytorchMLPFashionMNIST()
        pytorch_losses = []
        pytorch_accs = []
        
        for epoch in range(epochs):
            loss, acc = pytorch_model.train(dataset, lr=lr)
            pytorch_losses.append(loss)
            pytorch_accs.append(acc)
        
        # Custom
        custom_model = CustomMLPFashionMNIST()
        custom_losses = []
        custom_accs = []
        
        for epoch in range(epochs):
            loss, acc = custom_model.train(dataset, lr=lr)
            custom_losses.append(loss)
            custom_accs.append(acc)
        
        results[lr] = {
            'pytorch_losses': pytorch_losses,
            'pytorch_accs': pytorch_accs,
            'custom_losses': custom_losses,
            'custom_accs': custom_accs
        }
    
    epochs_range = range(1, epochs + 1)
    
    for lr in learning_rates:
        # Loss plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, results[lr]['pytorch_losses'], 'b-', linewidth=2, label='PyTorch')
        plt.plot(epochs_range, results[lr]['custom_losses'], 'r-', linewidth=2, label='Custom')
        plt.title(f'Training Loss (LR={lr})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'loss_lr_{lr}_pytorch_vs_custom.png')
        plt.close()
        
        # Validation accuracy plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, results[lr]['pytorch_accs'], 'b-', linewidth=2, label='PyTorch')
        plt.plot(epochs_range, results[lr]['custom_accs'], 'r-', linewidth=2, label='Custom')
        plt.title(f'Validation Accuracy (LR={lr})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'valacc_lr_{lr}_pytorch_vs_custom.png')
        plt.close()
    
    return results

def part2_test_performance():
    print("\n\nPart 2 Tests")
    epochs = 10
    
    configs = [
        {'name': 'Original 256,128', 'hidden1': 256, 'hidden2': 128,'remove_second': False, 'lr': 0.002, 'batch': 128},

        {'name': 'Hidden1=8',   'hidden1': 8,   'hidden2': 128,'remove_second': False, 'lr': 0.002, 'batch': 128},
        {'name': 'Hidden1=16',  'hidden1': 16,  'hidden2': 128,'remove_second': False, 'lr': 0.002, 'batch': 128},
        {'name': 'Hidden1=64',  'hidden1': 64,  'hidden2': 128,'remove_second': False, 'lr': 0.002, 'batch': 128},
        {'name': 'Hidden1=1024','hidden1': 1024,'hidden2': 128,'remove_second': False, 'lr': 0.002, 'batch': 128},

        # Hidden2, keep Hidden1 = 256
        {'name': 'Hidden2=8',   'hidden1': 256, 'hidden2': 8,'remove_second': False, 'lr': 0.002, 'batch': 128},
        {'name': 'Hidden2=16',  'hidden1': 256, 'hidden2': 16,'remove_second': False, 'lr': 0.002, 'batch': 128},
        {'name': 'Hidden2=64',  'hidden1': 256, 'hidden2': 64,'remove_second': False, 'lr': 0.002, 'batch': 128},
        {'name': 'Hidden2=1024','hidden1': 256, 'hidden2': 1024,'remove_second': False, 'lr': 0.002, 'batch': 128},

        {'name': 'No Hidden2',  'hidden1': 256, 'hidden2': None,'remove_second': True, 'lr': 0.002, 'batch': 128},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}")
        dataset = FashionMNIST(batch_size=config['batch'], val_perc=0.2)
        
        # PyTorch
        if config['remove_second']:
            pytorch_model = build_single_hidden_layer_pytorch(config['hidden1'])
        else:
            pytorch_model = PytorchMLPFashionMNIST(
                hidden_size1=config['hidden1'],
                hidden_size2=config['hidden2']
            )
        
        start_time = time.time()
        for epoch in range(epochs):
            pytorch_model.train(dataset, lr=config['lr'])
        pytorch_time = time.time() - start_time
        pytorch_test_acc = pytorch_model.test(dataset)
        
        # Custom
        if config['remove_second']:
            custom_model = build_single_hidden_layer_custom(config['hidden1'])
        else:
            custom_model = CustomMLPFashionMNIST(
                hidden_size1=config['hidden1'],
                hidden_size2=config['hidden2']
            )
        
        start_time = time.time()
        for epoch in range(epochs):
            custom_model.train(dataset, lr=config['lr'])
        custom_time = time.time() - start_time
        custom_test_acc = custom_model.test(dataset)
        
        results.append({
            'config': config['name'],
            'lr': config['lr'],
            'batch': config['batch'],
            'epochs': epochs,
            'pytorch_time': pytorch_time,
            'pytorch_acc': pytorch_test_acc,
            'custom_time': custom_time,
            'custom_acc': custom_test_acc
        })
    
    print(f"{'Config':<18} | {'LR':<6} | {'Batch':<6} | {'Epochs':<7} | {'PyTorch Time':<13} | {'PyTorch Acc':<12} | {'Custom Time':<12} | {'Custom Acc'}")
    
    for r in results:
        print(f"{r['config']:<18} | {r['lr']:<6} | {r['batch']:<6} | {r['epochs']:<7} | {r['pytorch_time']:<13.2f} | {r['pytorch_acc']:<12.3f} | {r['custom_time']:<12.2f} | {r['custom_acc']:.3f}")
    
    with open('part2_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['config', 'lr', 'batch', 'epochs', 'pytorch_time', 'pytorch_acc', 'custom_time', 'custom_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    return results

def main():
    print("Assignment 4")
    print("\nPart 1: Learning Rates")
    part1_results = part1_validation_accuracy()
    part2_results = part2_test_performance()

if __name__ == "__main__":
    main()
