from hyperparameters import Hyperparameters
from neural_network import NeuralNetwork
from torch.nn import functional as F
import numpy as np
import torch

parameters = Hyperparameters()

class Agent():
    """
    Proximal Policy Optimization (PPO) Agent.

    Args:
        action_size (int): Number of possible actions.

    Attributes:
        device (torch.device): Device (CPU or GPU) for training the agent.
        action_size (int): Number of possible actions.
        neural_net (NeuralNetwork): Neural network for the agent.
        optimizer (torch.optim): Optimizer for updating the neural network parameters.
    """

    def __init__(self, action_size) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.action_size = action_size
        self.neural_net = NeuralNetwork(action_size=action_size).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.neural_net.parameters(), lr=parameters.learning_rate)

    def action(self, state):
        """
        Select an action using the current policy.

        Args:
            state (np.ndarray): Current state.

        Returns:
            np.ndarray: Selected action.
        """

        if state.ndim == 3:
            state = [state]
        state = torch.tensor(data=np.array(state), dtype=torch.float32, device=self.device) #####
        action_values, _ = self.neural_net(state)
        policy = F.softmax(input=action_values, dim=-1)
        return np.array([np.random.choice(len(p), p=p.detach().cpu().numpy()) for p in policy])

    def step(self, state, action, reward, next_state, done):
        batch_size = state.shape[0]
        state = torch.tensor(data=np.array(state), dtype=torch.float32, device=self.device)
        next_state = torch.tensor(data=next_state, dtype=torch.float32, device=self.device)
        reward = torch.tensor(data=reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(data=done, dtype=torch.bool, device=self.device).to(dtype=torch.float32)
        action_values, state_value = self.neural_net(state)
        _, next_state_value = self.neural_net(next_state)
        target_state_value = reward + parameters.gamma * next_state_value * (1 - done)
        advantage = target_state_value - state_value
        probs = F.softmax(input=action_values, dim=-1)
        logprobs = F.log_softmax(input=action_values, dim=-1)
        entropy = -torch.sum(input=(probs * logprobs),  dim=-1)
        batch_idx = np.arange(batch_size)
        logp_actions = logprobs[batch_idx, action]
        actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
        critic_loss = F.mse_loss(input=target_state_value.detach(), target=state_value)
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()