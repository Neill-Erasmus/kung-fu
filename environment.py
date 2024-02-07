from preprocess import PreprocessAtari
from typing import Tuple
import gymnasium as gym

class Environment:
    """
    Wrapper class for handling the Atari environment.

    Attributes:
        env (gym.Env): Wrapped gym environment.
        state_shape (Tuple[int, ...]): Shape of the preprocessed state.
        number_actions (int): Number of possible actions in the environment.
    """

    def __init__(self) -> None:
        """
        Initializes the Environment class.

        Initializes the environment, preprocesses it, and sets necessary attributes.
        """

        self.env: gym.Env = gym.make("KungFuMasterDeterministic-v0", render_mode='rgb_array')
        self.env = PreprocessAtari(self.env, height=42, width=42, crop=lambda img: img, dim_order='pytorch', color=False, n_frames=4) #type: ignore
        self.state_shape: Tuple[int, ...] = self.env.observation_space.shape
        self.number_actions: int = self.env.action_space.n #type: ignore
        print(f'State Shape: {self.state_shape}\nNumber Actions: {self.number_actions}\nAction Names: {self.env.env.env.get_action_meanings()}') #type: ignore

    def step(self, action: int) -> Tuple:
        """
        Takes a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: A tuple containing the next state, the reward,
                whether the episode is done, and additional information.
        """

        return self.env.step(action) #type: ignore

    def reset(self):
        """
        Resets the environment.

        Returns:
            np.ndarray: The initial state of the environment.
        """

        return self.env.reset()