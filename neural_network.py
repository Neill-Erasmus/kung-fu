from torch.nn import functional as F
from typing import Tuple
from torch import nn
import torch

class NeuralNetwork(nn.Module):
    """
    Neural Network class representing the Q-function approximator.

    Args:
        action_size (int): Number of possible actions.

    Attributes:
        conv1 to fc2a (nn.Module): Layers of the neural network for action values.
        fc2s (nn.Module): Layer for state values.
    """

    def __init__(self, action_size : int) -> None:
        super().__init__()
        self.conv1   : nn.Conv2d = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2)
        self.conv2   : nn.Conv2d = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=32, kernel_size=3, stride=2)
        self.conv3   : nn.Conv2d = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=32, kernel_size=3, stride=2)
        self.flatten : nn.Flatten = torch.nn.Flatten()
        self.fc1     : nn.Linear = torch.nn.Linear(in_features=512, out_features=128)
        self.fc2a    : nn.Linear = torch.nn.Linear(in_features=self.fc1.out_features, out_features=action_size)
        self.fc2s    : nn.Linear = torch.nn.Linear(in_features=self.fc1.out_features, out_features=1)

    def forward(self, state : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the neural network.

        Args:
            state (torch.Tensor): Input state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output Q-values for each action and state value.
        """

        x : torch.Tensor = self.conv1(state)
        x : torch.Tensor = F.relu(x)
        x : torch.Tensor = self.conv2(x)
        x : torch.Tensor = F.relu(x)
        x : torch.Tensor = self.conv3(x)
        x : torch.Tensor = self.flatten(x)
        x : torch.Tensor = self.fc1(x)
        x : torch.Tensor = F.relu(x)
        action_values : torch.Tensor = self.fc2a(x)
        state_value   : torch.Tensor = self.fc2s(x)[0]
        return action_values, state_value