import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import typer
from pathlib import Path

from pic_classification_mnist_v01_xh.model import MyNeuralNet
from pic_classification_mnist_v01_xh.data import corrupt_mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(
    lr: float = 0.001,
    batch_size: int = 64,
    epochs: int = 10,
    model_checkpoint: str = "models/model.pth"
) -> None:
    """Train a model on MNIST.
    
    Args:
        lr: Learning rate for optimizer
        batch_size: Batch size for training
        epochs: Number of training epochs
        model_checkpoint: Path to save the trained model
    """
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")
    print(f"Using device: {DEVICE}")
    
    # Load data
    train_set, test_set = corrupt_mnist()
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, optimizer
    model = MyNeuralNet().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with statistics
    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            
            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")
        
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
        
        test_acc = 100. * test_correct / test_total
        print(f"Epoch {epoch+1}/{epochs} - Test Acc: {test_acc:.2f}%")

    print("Training complete")
    
    # Save model
    Path(model_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_checkpoint)
    print(f"Model saved to {model_checkpoint}")
    
    # Save training statistics visualization
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Accuracy")
    
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "training_statistics.png", dpi=150, bbox_inches='tight')
    print(f"Training statistics saved to {figures_dir / 'training_statistics.png'}")
    plt.close()


if __name__ == "__main__":
    typer.run(train)
