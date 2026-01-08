import torch
import typer
from pathlib import Path


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images to have mean 0 and standard deviation 1."""
    return (images - images.mean()) / images.std()


def preprocess_data(
    raw_dir: str = "data/corruptmnist/raw", 
    processed_dir: str = "data/processed"
) -> None:
    """Process raw corrupted MNIST data and save it to processed directory.
    
    This function:
    1. Loads the 6 training files (train_images_0.pt through train_images_5.pt) and concatenates them
    2. Loads the test files (test_images.pt and test_target.pt)
    3. Adds channel dimension and converts to appropriate dtypes
    4. Normalizes images to mean=0, std=1
    5. Saves processed tensors to the processed directory
    
    Args:
        raw_dir: Directory containing raw .pt files (default: data/corruptmnist/raw)
        processed_dir: Directory to save processed tensors (default: data/processed)
    """
    print(f"Loading data from {raw_dir}...")
    
    # Load and concatenate training data
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # Load test data
    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    print(f"Train images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")
    
    # Add channel dimension and convert to appropriate dtypes
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    # Normalize images to mean=0, std=1
    print("Normalizing images...")
    train_images = normalize(train_images)
    test_images = normalize(test_images)
    
    print(f"Train images - mean: {train_images.mean():.4f}, std: {train_images.std():.4f}")
    print(f"Test images - mean: {test_images.mean():.4f}, std: {test_images.std():.4f}")

    # Create output directory if it doesn't exist
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    
    # Save processed tensors
    print(f"Saving processed data to {processed_dir}...")
    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")
    
    print("Done! Data preprocessing complete.")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
