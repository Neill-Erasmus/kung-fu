from hyperparameters import Hyperparameters
from environment import Environment
from env_batch import EnvBatch
from typing import List
from agent import Agent
import numpy as np
import tqdm

def main() -> None:
    """
    Main function to run the reinforcement learning training loop.
    """

    def evaluate(agent: Agent, env: Environment, n_episodes: int = 1) -> List[float]:
        """
        Evaluate the agent's performance in the environment.

        Args:
            agent (Agent): The agent to evaluate.
            env (Environment): The environment to evaluate in.
            n_episodes (int): Number of episodes to run.

        Returns:
            List[float]: List of rewards obtained in each episode.
        """

        episodes_rewards = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            while True:
                action = agent.action(state=state)
                state, reward, done, info, _ = env.step(action=action[0])
                total_reward += reward
                if done:
                    break
            episodes_rewards.append(total_reward)
        return episodes_rewards

    env = Environment()
    parameters = Hyperparameters()
    agent = Agent(action_size=env.number_actions)
    env_batch = EnvBatch(n_envs=parameters.number_environments)
    batch_states = env_batch.reset()
    with tqdm.trange(10001) as progress_bar:
        for i in progress_bar:
            batch_actions = agent.action(state=batch_states)
            batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(batch_actions)
            batch_rewards *= 0.01
            agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
            batch_states = batch_next_states
            if i % 1000 == 0:
                print(f'Average Agent Reward: {np.mean(evaluate(agent=agent, env=env, n_episodes=10))}')  # type: ignore

if __name__ == "__main__":
    main()