from torch import nn
import torch


class MyNeuralNet(nn.Module):
    """Simple convolutional neural network for MNIST digit classification.
    
    Architecture:
    - Conv2d(1, 32, 3) + ReLU + MaxPool
    - Conv2d(32, 64, 3) + ReLU + MaxPool
    - Flatten
    - Linear(64*5*5, 128) + ReLU + Dropout
    - Linear(128, 10)
    """
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch, 10)
        """
        x = self.conv1(x)  # (batch, 32, 26, 26)
        x = nn.functional.relu(x)
        x = self.pool(x)  # (batch, 32, 13, 13)
        
        x = self.conv2(x)  # (batch, 64, 11, 11)
        x = nn.functional.relu(x)
        x = self.pool(x)  # (batch, 64, 5, 5)
        
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # (batch, 64*5*5)
        
        x = self.fc1(x)  # (batch, 128)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)  # (batch, 10)
        return x


if __name__ == "__main__":
    model = MyNeuralNet()
    print(model)
    
    # Test with sample input
    x = torch.rand(1, 1, 28, 28)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
