"""Main script for training the MNIST classification model."""
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import typer
from pathlib import Path

from pic_classification_mnist_v01_xh.model import MyNeuralNet
from pic_classification_mnist_v01_xh.data import corrupt_mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def main(
    lr: float = 0.001,
    batch_size: int = 64,
    epochs: int = 10,
    model_checkpoint: str = "models/model.pth"
) -> None:
    """Train a model on corrupted MNIST dataset.
    
    This script performs the following:
    1. Loads the preprocessed corrupted MNIST data
    2. Trains a CNN model on the training set
    3. Evaluates on the test set after each epoch
    4. Saves the trained model to the models/ folder
    5. Generates and saves training statistics visualization
    
    Args:
        lr: Learning rate for optimizer (default: 0.001)
        batch_size: Batch size for training (default: 64)
        epochs: Number of training epochs (default: 10)
        model_checkpoint: Path to save the trained model (default: models/model.pth)
    """
    print("=" * 60)
    print("Training MNIST Classification Model")
    print("=" * 60)
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_set, test_set = corrupt_mnist()
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print(f"Training samples: {len(train_set)}")
    print(f"Test samples: {len(test_set)}")
    
    # Initialize model, loss, optimizer
    print("\nInitializing model...")
    model = MyNeuralNet().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop with statistics
    print("\nStarting training...")
    statistics = {"train_loss": [], "train_accuracy": []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            
            # Record statistics
            statistics["train_loss"].append(loss.item())
            batch_accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(batch_accuracy)
            
            epoch_loss += loss.item()
            epoch_correct += (y_pred.argmax(dim=1) == target).sum().item()
            epoch_total += target.size(0)

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i}/{len(train_dataloader)}], "
                      f"Loss: {loss.item():.4f}, Acc: {batch_accuracy:.4f}")
        
        # Epoch statistics
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_acc = epoch_correct / epoch_total
        
        # Validation at end of each epoch
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = test_correct / test_total
        
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {avg_epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        print(f"  Test Acc: {test_acc:.4f}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Save model to models/ folder
    model_path = Path(model_checkpoint)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Save training statistics visualization
    print("\nGenerating training statistics...")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Training Loss", fontsize=14, fontweight='bold')
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Training Accuracy", fontsize=14, fontweight='bold')
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Accuracy")
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    stats_path = figures_dir / "training_statistics.png"
    fig.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training statistics saved to: {stats_path}")
    print("\nAll done! 🎉")


if __name__ == "__main__":
    typer.run(main)
