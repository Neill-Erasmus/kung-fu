from gymnasium import ObservationWrapper
from typing import Callable, Tuple
from gymnasium.spaces import Box
import numpy as np
import cv2
import gym

class PreprocessAtari(ObservationWrapper):
    """
    Wrapper class for preprocessing Atari observations.

    Args:
        env (gym.Env): Environment to wrap.
        height (int): Height of the output frame.
        width (int): Width of the output frame.
        crop (Callable[[np.ndarray], np.ndarray]): Function for cropping the input frame.
        dim_order (str): Order of dimensions, 'tensorflow' or 'pytorch'.
        color (bool): Flag for color frames.
        n_frames (int): Number of frames to stack.

    Attributes:
        img_size (Tuple[int, int]): Size of the output frame.
        crop (Callable[[np.ndarray], np.ndarray]): Function for cropping the input frame.
        dim_order (str): Order of dimensions.
        color (bool): Flag for color frames.
        frame_stack (int): Number of frames to stack.
        observation_space (gym.spaces.Box): Observation space of the wrapper.
        frames (np.ndarray): Buffer for stacked frames.
    """

    def __init__(
        self,
        env       : gym.Env,
        height    : int = 42,
        width     : int = 42,
        crop      : Callable[[np.ndarray], np.ndarray] = lambda img: img,
        dim_order : str = 'pytorch',
        color     : bool = False,
        n_frames  : int = 4
    ) -> None:
        super().__init__(env) #type: ignore
        self.img_size          : Tuple[int, int] = (height, width)
        self.crop              : Callable[[np.ndarray], np.ndarray] = crop
        self.dim_order         : str = dim_order
        self.color             : bool = color
        self.frame_stack       : int = n_frames
        n_channels             : int = 3 * n_frames if color else n_frames
        obs_shape              : Tuple[int, ...] = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
        self.observation_space : Box = Box(0.0, 1.0, obs_shape)
        self.frames            : np.ndarray = np.zeros(obs_shape, dtype=np.float32)

    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment and the frame buffer.

        Returns:
            Tuple[np.ndarray, dict]: Initial observation and environment information.
        """

        self.frames = np.zeros_like(self.frames)
        obs, info = self.env.reset()
        self.update_buffer(obs)
        return self.frames, info

    def observation(self, img : np.ndarray) -> np.ndarray:
        """
        Preprocess the input frame and update the frame buffer.

        Args:
            img (np.ndarray): Input frame.

        Returns:
            np.ndarray: Preprocessed frame.
        """

        img = self.crop(img)
        img = cv2.resize(img, self.img_size)
        if not self.color:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype('float32') / 255.
        if self.color:
            self.frames = np.roll(self.frames, shift=-3, axis=0)
        else:
            self.frames = np.roll(self.frames, shift=-1, axis=0)
        if self.color:
            self.frames[-3:] = img
        else:
            self.frames[-1] = img
        return self.frames

    def update_buffer(self, obs : np.ndarray) -> None:
        """
        Update the frame buffer with a new observation.

        Args:
            obs: New observation.
        """

        self.frames = self.observation(obs)