import torch
import typer
from pic_classification_mnist_v01_xh.data import corrupt_mnist
from pic_classification_mnist_v01_xh.model import MyNeuralNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str = "models/model.pth") -> None:
    """Evaluate a trained model.
    
    Args:
        model_checkpoint: Path to the saved model checkpoint
    """
    print("Evaluating like my life depended on it")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Using device: {DEVICE}")

    model = MyNeuralNet().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    
    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    typer.run(evaluate)
