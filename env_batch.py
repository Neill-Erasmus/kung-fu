from environment import Environment
from typing import List, Tuple
import numpy as np

class EnvBatch:
    def __init__(self, n_envs : int = 10) -> None:
        """
        Initializes a batch of environments.

        Args:
            n_envs (int): Number of environments in the batch.
        """

        self.envs: List[Environment] = [Environment() for _ in range(n_envs)]
        
    def reset(self) -> np.ndarray:
        """
        Resets all environments in the batch.

        Returns:
            np.ndarray: Initial states of all environments in the batch.
        """

        _state : List[np.ndarray] = []
        for env in self.envs:
            _state.append(env.reset()[0])
        return np.array(_state)
    
    def step(self, actions : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Takes a step in all environments in the batch and updates the agent's policy.

        Args:
            actions (np.ndarray): Actions to take in each environment.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, dict]: Tuple containing the next states, rewards, done flags, and additional information.
        """

        next_states, rewards, dones, infos, _ = map(np.array, zip(*[env.step(a) for env, a in zip(self.envs, actions)]))
        for i in range(0, len(self.envs)):
            if dones[i]:
                next_states[i] = self.envs[i].reset()[0]
        return next_states, rewards, dones, infos #type: ignore