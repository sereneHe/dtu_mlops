"""Data exploration and visualization script for corrupted MNIST dataset."""
import matplotlib.pyplot as plt
import torch
import typer
from pathlib import Path
import numpy as np
from collections import Counter

from pic_classification_mnist_v01_xh.data import corrupt_mnist


def explore_data(output_dir: str = "reports/figures") -> None:
    """Explore and visualize the corrupted MNIST dataset.
    
    This script generates multiple visualizations:
    1. Class distribution (train and test)
    2. Sample images from each class
    3. Pixel intensity distribution
    4. Image statistics (mean, std per class)
    
    Args:
        output_dir: Directory to save all visualizations
    """
    print("=" * 60)
    print("Exploring Corrupted MNIST Dataset")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    train_set, test_set = corrupt_mnist()
    
    # Extract data
    train_images = torch.stack([img for img, _ in train_set])
    train_labels = torch.tensor([label for _, label in train_set])
    test_images = torch.stack([img for img, _ in test_set])
    test_labels = torch.tensor([label for _, label in test_set])
    
    print(f"Train set: {len(train_set)} samples")
    print(f"Test set: {len(test_set)} samples")
    print(f"Image shape: {train_images[0].shape}")
    
    # 1. Class Distribution
    print("\n[1/4] Generating class distribution plot...")
    plot_class_distribution(train_labels, test_labels, output_path / "class_distribution.png")
    
    # 2. Sample Images
    print("[2/4] Generating sample images grid...")
    plot_sample_images(train_set, output_path / "sample_images.png")
    
    # 3. Pixel Intensity Distribution
    print("[3/4] Generating pixel intensity distribution...")
    plot_pixel_distribution(train_images, test_images, output_path / "pixel_distribution.png")
    
    # 4. Per-Class Statistics
    print("[4/4] Generating per-class statistics...")
    plot_class_statistics(train_images, train_labels, output_path / "class_statistics.png")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics Summary")
    print("=" * 60)
    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Image dimensions: {train_images[0].shape}")
    print(f"Pixel value range: [{train_images.min():.2f}, {train_images.max():.2f}]")
    print(f"Mean pixel value: {train_images.mean():.4f}")
    print(f"Std pixel value: {train_images.std():.4f}")
    
    train_counts = Counter(train_labels.numpy())
    print(f"\nTrain class distribution:")
    for i in range(10):
        print(f"  Class {i}: {train_counts[i]:5d} samples ({100*train_counts[i]/len(train_set):.1f}%)")
    
    print(f"\nAll visualizations saved to: {output_path}/")
    print("✓ Data exploration complete!")


def plot_class_distribution(train_labels, test_labels, save_path):
    """Plot class distribution for train and test sets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train distribution
    train_counts = Counter(train_labels.numpy())
    classes = sorted(train_counts.keys())
    train_values = [train_counts[c] for c in classes]
    
    ax1.bar(classes, train_values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(train_values):
        ax1.text(i, v + 50, str(v), ha='center', va='bottom', fontsize=10)
    
    # Test distribution
    test_counts = Counter(test_labels.numpy())
    test_values = [test_counts[c] for c in classes]
    
    ax2.bar(classes, test_values, color='coral', alpha=0.7)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(test_values):
        ax2.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {save_path}")


def plot_sample_images(dataset, save_path, samples_per_class=10):
    """Plot sample images from each class."""
    fig, axes = plt.subplots(10, samples_per_class, figsize=(15, 15))
    
    # Organize samples by class
    class_samples = {i: [] for i in range(10)}
    for img, label in dataset:
        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(img)
        if all(len(samples) >= samples_per_class for samples in class_samples.values()):
            break
    
    # Plot samples
    for class_idx in range(10):
        for sample_idx in range(samples_per_class):
            ax = axes[class_idx, sample_idx]
            if sample_idx < len(class_samples[class_idx]):
                img = class_samples[class_idx][sample_idx].squeeze()
                ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            # Add class label on the left
            if sample_idx == 0:
                ax.set_ylabel(f'Class {class_idx}', fontsize=12, fontweight='bold')
    
    plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {save_path}")


def plot_pixel_distribution(train_images, test_images, save_path):
    """Plot pixel intensity distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Flatten images to get all pixel values
    train_pixels = train_images.flatten().numpy()
    test_pixels = test_images.flatten().numpy()
    
    # Train distribution
    ax1.hist(train_pixels, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(train_pixels.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {train_pixels.mean():.2f}')
    ax1.set_xlabel('Pixel Intensity', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Training Set Pixel Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Test distribution
    ax2.hist(test_pixels, bins=100, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(test_pixels.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {test_pixels.mean():.2f}')
    ax2.set_xlabel('Pixel Intensity', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Test Set Pixel Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {save_path}")


def plot_class_statistics(images, labels, save_path):
    """Plot mean and std statistics per class."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    classes = range(10)
    means = []
    stds = []
    
    for c in classes:
        class_images = images[labels == c]
        means.append(class_images.mean().item())
        stds.append(class_images.std().item())
    
    # Mean per class
    ax1.bar(classes, means, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Mean Pixel Value', fontsize=12)
    ax1.set_title('Mean Pixel Value per Class', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add value labels
    for i, v in enumerate(means):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Std per class
    ax2.bar(classes, stds, color='coral', alpha=0.7)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Std Pixel Value', fontsize=12)
    ax2.set_title('Std Pixel Value per Class', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(stds):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {save_path}")


if __name__ == "__main__":
    typer.run(explore_data)
