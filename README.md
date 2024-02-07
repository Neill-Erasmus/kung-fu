# Kung-Fu

A Deep Convolutional Q-Network (DCQN) has been developed and trained with the aim of mastering the kung fu environment offered by OpenAI's Gymnasium. This project focuses on leveraging the Asynchronous Advantage Actor-Critic (A3C) algorithm, empowering an agent to autonomously navigate and excel in the intricacies of the kung fu environment.

## Deep Convolutional Q-Learning

### Q-Learning

Q-learning is a model-free reinforcement learning algorithm used to learn the quality of actions in a given state. It operates by learning a Q-function, which represents the expected cumulative reward of taking a particular action in a particular state and then following a policy to maximize cumulative rewards thereafter.

### Deep Learning

Deep learning involves using neural networks with multiple layers to learn representations of data. Convolutional neural networks (CNNs) are a specific type of neural network commonly used for analyzing visual imagery. They are composed of layers such as convolutional layers, pooling layers, and fully connected layers, which are designed to extract features from images.

### Combining Q-Learning with Deep Learning

In traditional Q-learning, a Q-table is used to store the Q-values for all state-action pairs. However, for complex environments with large state spaces, maintaining such a table becomes infeasible due to memory constraints. Deep Q-Networks address this issue by using a neural network to approximate the Q-function, mapping states to Q-values directly from raw input pixels.

### Deep Convolutional Q-Learning

Deep Convolutional Q-Learning specifically utilizes CNNs as function approximators within the Q-learning framework. It's particularly effective in environments where the state space is represented as visual input, such as playing Atari games from raw pixel inputs.

The general steps involved in training a Deep Convolutional Q-Learning agent are as follows:

#### Observation

The agent observes the environment, typically represented as images or raw sensory data.
Action Selection: Based on the observed state, the agent selects an action according to an exploration strategy, such as ε-greedy.

#### Reward

The agent receives a reward from the environment based on the action taken.

#### Experience Replay

The agent stores its experiences (state, action, reward, next state) in a replay buffer.

#### Training

Periodically, the agent samples experiences from the replay buffer and uses them to update the parameters of the neural network using techniques like stochastic gradient descent (SGD) or variants like RMSprop or Adam.

#### Target Network

To stabilize training, a separate target network may be used to calculate target Q-values during updates. This network is periodically updated with the parameters from the primary network.

#### Iteration

The process of observation, action selection, and training continues iteratively until convergence.
By leveraging deep convolutional neural networks, Deep Convolutional Q-Learning has demonstrated remarkable success in learning effective control policies directly from high-dimensional sensory input, making it a powerful technique for solving complex reinforcement learning problems, especially in the realm of visual tasks like playing video games or robotic control.

## Asynchronous Advantage Actor-Critic (A3C)

The Asynchronous Advantage Actor-Critic (A3C) algorithm is a reinforcement learning technique used to train artificial intelligence agents in decision-making tasks.

### Asynchronous Training

A3C employs asynchronous training, where multiple instances of the agent run simultaneously and independently. Each instance interacts with its own copy of the environment, gathering experience and updating its parameters asynchronously. This parallelization allows for faster and more efficient learning compared to traditional sequential methods.

### Actor-Critic Architecture

A3C combines elements from both the actor-critic and deep learning approaches.

### Actor

The actor component is responsible for selecting actions based on the current policy. It's typically implemented as a neural network that takes the current state as input and outputs a probability distribution over possible actions.

### Critic

The critic evaluates the actions taken by the actor by estimating their value or advantage. This component helps the agent to learn which actions are more favorable in different situations. The critic is also implemented as a neural network, which takes the state as input and outputs a value or advantage estimate.

### Advantage

The advantage function in A3C is a measure of how much better a particular action is compared to the average action in a given state. It helps the agent to understand which actions lead to better long-term outcomes.

### Asynchronous Updates

As each instance of the agent interacts with the environment independently, they generate different experiences. These experiences are used to update the parameters of both the actor and critic networks asynchronously. By leveraging these diverse experiences, the agent can learn more effectively and explore different strategies.

Overall, A3C combines the benefits of deep learning with reinforcement learning, utilizing asynchronous training and actor-critic architecture to train agents to make decisions in complex environments efficiently.

## Overview of Kung Fu Environment

### Description

In the KungFuMaster environment, you assume the role of a skilled Kung-Fu Master navigating through the treacherous temple of the Evil Wizard. Your primary objective is to rescue Princess Victoria while overcoming various adversaries along the journey.

### Action Space

The action space in KungFuMaster is discrete, with 14 possible actions. Each action corresponds to a specific movement or attack maneuver within the game. These actions range from basic movements like UP, DOWN, LEFT, and RIGHT to more specialized attacks such as firing projectiles in different directions.

### Observation Space

The observation space in KungFuMaster varies depending on the chosen observation type. There are three observation types available: "rgb", "grayscale", and "ram". The "rgb" observation type provides observations as color images represented as an array of shape (210, 160, 3) with pixel values ranging from 0 to 255. The "grayscale" observation type provides grayscale versions of the "rgb" images. The "ram" observation type represents observations as a 1D array with 128 elements, capturing the state of the Atari 2600's RAM.

## Architecture of the Neural Network

### Convolutional Layers (Conv2d)

The input to the network is a state tensor representing the environment.
conv1, conv2, and conv3 are convolutional layers with a kernel size of 3x3 and a stride of 2. These layers are responsible for extracting features from the input state.
The number of output channels (out_channels) for each convolutional layer gradually increases, starting from 32 in conv1.
ReLU activation functions (F.relu) are applied after each convolutional layer to introduce non-linearity into the network.

### Flatten Layer

After the convolutional layers, the output tensor is flattened into a 1-dimensional tensor using flatten. This prepares the feature representation for fully connected layers.

### Fully Connected Layers (Linear)

fc1 is a fully connected layer with 512 input features and 128 output features.
ReLU activation function is applied after fc1.
fc2a is a fully connected layer responsible for estimating the Q-values for each action. It takes the output of fc1 as input and produces an output with dimensions corresponding to the number of possible actions in the environment (specified by action_size).
fc2s is a fully connected layer responsible for estimating the state value. It also takes the output of fc1 as input and produces a single output representing the state value.

### Forward Method

The forward method defines the forward pass through the network.
The input state is processed sequentially through the convolutional layers followed by ReLU activations.
After flattening, the flattened tensor is passed through the fully connected layers (fc1) followed by ReLU activation.
The output of fc1 is then used to compute both the action values (fc2a) and the state value (fc2s).
The action values and state value are returned as a tuple.

Overall, this architecture combines convolutional and fully connected layers to process the input state and estimate both the Q-values for each action and the state value, which are essential for reinforcement learning tasks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.