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
    from collections import OrderedDict
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model class
    Model = import_class(config['model'])
    model = Model(**config.get('model_args', {}))
    
    # Load weights - xử lý giống torchlight/io.py
    print(f"Loading weights from: {weights_path}")
    weights = torch.load(weights_path, map_location=device)
    
    # Xử lý nếu weights được wrap trong dict
    if 'model_state_dict' in weights:
        weights = weights['model_state_dict']
    elif 'state_dict' in weights:
        weights = weights['state_dict']
    
    # Xử lý module prefix (từ DataParallel)
    weights = OrderedDict([
        [k.split('module.')[-1], v] for k, v in weights.items()
    ])
    
    # Load với strict=False để xử lý partial loading
    try:
        model.load_state_dict(weights, strict=True)
        print("Loaded weights successfully (strict mode)")
    except RuntimeError as e:
        print(f"Strict loading failed, trying partial load...")
        # Lấy state dict hiện tại của model
        model_state = model.state_dict()
        
        # Tìm keys khớp nhau
        matched_keys = []
        missing_keys = []
        unexpected_keys = []
        
        for k in model_state.keys():
            if k in weights:
                model_state[k] = weights[k]
                matched_keys.append(k)
            else:
                missing_keys.append(k)
        
        for k in weights.keys():
            if k not in model_state:
                unexpected_keys.append(k)
        
        model.load_state_dict(model_state)
        print(f"Loaded {len(matched_keys)} keys")
        if missing_keys:
            print(f"Missing keys (will use pretrained): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")
    
    model = model.to(device)
    model.eval()
    
    return model, config


def load_data(config, split='test', custom_feeder=None, custom_feeder_args=None):
    """Load data feeder
    
    Args:
        config: Config dict từ yaml file
        split: 'train' hoặc 'test'
        custom_feeder: Tên feeder tùy chỉnh (override config)
        custom_feeder_args: Args cho feeder tùy chỉnh
    """
    # Sử dụng custom feeder nếu được chỉ định
    if custom_feeder:
        feeder_name = custom_feeder
    else:
        feeder_name = config['feeder']
    
    Feeder = import_class(feeder_name)
    
    # Lấy feeder args
    if custom_feeder_args:
        feeder_args = custom_feeder_args
    elif split == 'test':
        feeder_args = config.get('test_feeder_args', {})
    else:
        feeder_args = config.get('train_feeder_args', {})
    
    print(f"Using feeder: {feeder_name}")
    print(f"Feeder args: {feeder_args}")
    
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
            # Xử lý batch - có thể là nhiều format khác nhau:
            # - (data, label) - 2 elements
            # - (rgb, label, filename) - 3 elements từ feeder_rgb_only
            # - (skeleton, rgb, label) - 3 elements từ feeder_rgb_fused
            
            if len(batch) == 2:
                data, label = batch
                data = data.float().to(device)
            elif len(batch) == 3:
                # Kiểm tra xem element thứ 3 là label (int/tensor) hay filename (string)
                if isinstance(batch[2], (int, torch.Tensor)) or (isinstance(batch[2], list) and len(batch[2]) > 0 and isinstance(batch[2][0], (int, float))):
                    # Format: (skeleton, rgb, label) - dùng rgb cho RGB-only model
                    _, data, label = batch
                else:
                    # Format: (rgb, label, filename) từ feeder_rgb_only
                    data, label, _ = batch
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
    
    # Custom feeder arguments
    parser.add_argument('--feeder', default=None,
                        help='Custom feeder class to use (e.g., feeder.feeder_rgb_only.Feeder)')
    parser.add_argument('--rgb_path', default=None,
                        help='Path to RGB frames folder (for feeder_rgb_only)')
    parser.add_argument('--label_path', default=None,
                        help='Path to label file (for feeder, use "val" or "test" to trigger val mode)')
    
    args = parser.parse_args()
    
    print(f"Config: {args.config}")
    print(f"Weights: {args.weights}")
    print(f"Device: {args.device}")
    print(f"Split: {args.split}")
    
    # Load model
    print("\nLoading model...")
    model, config = load_model(args.config, args.weights, args.device)
    
    # Prepare custom feeder args if specified
    custom_feeder = args.feeder
    custom_feeder_args = None
    
    if args.feeder:
        custom_feeder_args = {}
        if args.rgb_path:
            custom_feeder_args['rgb_path'] = args.rgb_path
        if args.label_path:
            custom_feeder_args['label_path'] = args.label_path
        else:
            # Use 'val' or 'test' in label_path to trigger val mode
            custom_feeder_args['label_path'] = f'{args.split}_label.pkl'
    
    # Load data
    print(f"\nLoading {args.split} data...")
    data_loader = load_data(config, args.split, custom_feeder, custom_feeder_args)
    print(f"Total samples: {len(data_loader.dataset)}")
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred, y_true = evaluate_model(model, data_loader, args.device)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # UCLA class names (10 classes) - labels 1-10 map to indices 0-9
    ucla_classes = [
        'pick up with one hand',     # label 1 -> index 0
        'pick up with two hands',    # label 2 -> index 1
        'drop trash',                # label 3 -> index 2
        'walk around',               # label 4 -> index 3
        'sit down',                  # label 5 -> index 4
        'stand up',                  # label 6 -> index 5
        'donning',                   # label 7 -> index 6
        'doffing',                   # label 8 -> index 7
        'throw',                     # label 9 -> index 8
        'carry'                      # label 10 -> index 9
    ]
    
    # NTU-RGB+D class names (60 or 120 classes)
    # UCLA 12 classes (in case labels go from 1-12)
    ucla_12_classes = [
        'pick up with one hand',
        'pick up with two hands',
        'drop trash',
        'walk around',
        'sit down',
        'stand up',
        'donning',
        'doffing',
        'throw',
        'carry',
        'drinking',
        'eating'
    ]
    
    # Detect number of classes
    num_classes = config.get('model_args', {}).get('num_class', len(np.unique(y_true)))
    actual_classes = len(np.unique(y_true))
    print(f"Number of classes in config: {num_classes}")
    print(f"Actual unique classes in data: {actual_classes}")
    
    if num_classes == 10 or actual_classes == 10:
        class_names = ucla_classes
    elif num_classes == 12 or actual_classes == 12:
        class_names = ucla_12_classes
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

