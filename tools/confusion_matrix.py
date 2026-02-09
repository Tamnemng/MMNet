"""
Script để tạo Confusion Matrix cho model đã train với recognition_rgb_only.py

Cách sử dụng:
    python tools/confusion_matrix.py --config config/nucla_123/train_rgb.yaml --weights <path_to_weights.pt>

Ví dụ:
    python tools/confusion_matrix.py --config config/nucla_123/train_rgb.yaml --weights work_dir/best_model.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns

# Import model và feeder
from torchlight import import_class


def load_model(config_path, weights_path, device):
    """Load model từ config và weights"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model class
    Model = import_class(config['model'])
    model = Model(**config.get('model_args', {}))
    
    # Load weights
    weights = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in weights:
        model.load_state_dict(weights['model_state_dict'])
    else:
        model.load_state_dict(weights)
    
    model = model.to(device)
    model.eval()
    
    return model, config


def load_data(config, split='test'):
    """Load data feeder"""
    Feeder = import_class(config['feeder'])
    
    if split == 'test':
        feeder_args = config.get('test_feeder_args', {})
    else:
        feeder_args = config.get('train_feeder_args', {})
    
    dataset = Feeder(**feeder_args)
    
    batch_size = config.get('test_batch_size', config.get('batch_size', 32))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set 0 for Windows compatibility
        pin_memory=True
    )
    
    return data_loader


def evaluate_model(model, data_loader, device):
    """Chạy evaluation và thu thập predictions"""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            # Xử lý batch - có thể là (data, label) hoặc (skeleton, rgb, label)
            if len(batch) == 3:
                skeleton, rgb, label = batch
                # Model RGB only chỉ dùng RGB
                data = rgb.float().to(device)
            elif len(batch) == 2:
                data, label = batch
                data = data.float().to(device)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
            label = label.long().to(device)
            
            output = model(data)
            preds = torch.argmax(output, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path='confusion_matrix.png', 
                          normalize=True, figsize=(12, 10)):
    """Vẽ và lưu confusion matrix"""
    
    if normalize:
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        fmt = '.2%'
        title = 'Confusion Matrix (Normalized)'
    else:
        cm = confusion_matrix(y_true, y_pred)
        fmt = 'd'
        title = 'Confusion Matrix'
    
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")
    
    return cm


def main():
    parser = argparse.ArgumentParser(description='Generate Confusion Matrix for trained model')
    parser.add_argument('--config', '-c', required=True, help='Path to config file')
    parser.add_argument('--weights', '-w', required=True, help='Path to model weights (.pt file)')
    parser.add_argument('--split', default='test', choices=['train', 'test'], 
                        help='Data split to evaluate')
    parser.add_argument('--output', '-o', default='confusion_matrix.png', 
                        help='Output path for confusion matrix image')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize confusion matrix')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print(f"Config: {args.config}")
    print(f"Weights: {args.weights}")
    print(f"Device: {args.device}")
    print(f"Split: {args.split}")
    
    # Load model
    print("\nLoading model...")
    model, config = load_model(args.config, args.weights, args.device)
    
    # Load data
    print(f"\nLoading {args.split} data...")
    data_loader = load_data(config, args.split)
    print(f"Total samples: {len(data_loader.dataset)}")
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred, y_true = evaluate_model(model, data_loader, args.device)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # UCLA class names (10 classes)
    ucla_classes = [
        'pick up with one hand',
        'pick up with two hands', 
        'drop trash',
        'walk around',
        'sit down',
        'stand up',
        'donning',
        'doffing',
        'throw',
        'carry'
    ]
    
    # Detect number of classes
    num_classes = config.get('model_args', {}).get('num_class', len(np.unique(y_true)))
    if num_classes == 10:
        class_names = ucla_classes
    else:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, class_names=class_names, 
                          save_path=args.output, normalize=args.normalize)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save detailed results
    results_path = args.output.replace('.png', '_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"Config: {args.config}\n")
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Total samples: {len(y_true)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
