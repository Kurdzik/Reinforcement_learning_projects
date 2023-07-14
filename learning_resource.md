
# Different algorithms and their use cases

| Algorithm | Description | Application |
| --- | --- | --- |
| A2C | A2C stands for **Advantage Actor-Critic**. It is an on-policy algorithm that uses two neural networks: an actor and a critic. The actor learns a policy that maps states to actions, while the critic learns a value function that estimates the expected return from each state. The advantage is the difference between the value function and the actual return, which measures how good an action is compared to the average. A2C updates both networks using the advantage as a signal to improve the policy and reduce the value error¹. | A2C can be used for tasks that require continuous actions, such as robotics, locomotion, or vehicle control². |
| DDPG | DDPG stands for **Deep Deterministic Policy Gradient**. It is an off-policy algorithm that combines the ideas of DQN and policy gradient methods. It also uses two neural networks: an actor and a critic. The actor learns a deterministic policy that outputs the best action for each state, while the critic learns a value function that estimates the Q-value of each state-action pair. DDPG uses a replay buffer to store and sample transitions, and a target network to stabilize the learning process. DDPG updates the actor network using the policy gradient derived from the critic network³. | DDPG can be used for tasks that require continuous and high-dimensional actions, such as robotics, manipulation, or navigation⁴. |
| DQN | DQN stands for **Deep Q-Network**. It is an off-policy algorithm that uses a single neural network to learn a value function that estimates the Q-value of each state-action pair. The Q-value is the expected return from taking an action in a state and following a certain policy afterwards. DQN uses a replay buffer to store and sample transitions, and a target network to stabilize the learning process. DQN updates the network using the temporal difference error between the target Q-value and the current Q-value⁵. | DQN can be used for tasks that require discrete actions, such as Atari games, board games, or gridworlds. |
| HER | HER stands for **Hindsight Experience Replay**. It is an extension of DQN that enables learning from sparse and binary rewards. It works by storing not only the actual transitions in the replay buffer, but also some fictitious transitions where the goal is replaced by the achieved outcome. This way, the agent can learn from its own failures and successes, and assign positive rewards to previously unrewarded transitions. HER can be combined with any off-policy algorithm that uses a replay buffer. | HER can be used for tasks that require goal-oriented behaviors, such as robotics, manipulation, or navigation. |
| PPO | PPO stands for **Proximal Policy Optimization**. It is an on-policy algorithm that uses a single or multiple neural networks to learn a policy and/or a value function. The policy is updated by optimizing a surrogate objective function that encourages exploration and avoids large policy changes that can harm the performance. The value function is updated by minimizing the clipped value error between the target value and the current value. | PPO can be used for tasks that require continuous or discrete actions, such as robotics, locomotion, or Atari games. |
| SAC | SAC stands for **Soft Actor-Critic**. It is an off-policy algorithm that uses two neural networks: an actor and a critic. The actor learns a stochastic policy that outputs a probability distribution over actions for each state, while the critic learns two value functions that estimate the Q-values of each state-action pair. SAC uses a replay buffer to store and sample transitions, and a target network to stabilize the learning process. SAC updates the actor network using the policy gradient derived from the critic network and an entropy regularization term that encourages exploration. SAC updates the critic network using the temporal difference error between the target Q-value and the current Q-value. | SAC can be used for tasks that require continuous actions, such as robotics, locomotion, or manipulation. |
| TD3 | TD3 stands for **Twin Delayed Deep Deterministic Policy Gradient**. It is an extension of DDPG that addresses some of its limitations, such as overestimation bias and policy oscillation. It also uses two neural networks: an actor and a critic. However, it uses two critic networks instead of one, and takes the minimum of their Q-values to reduce overestimation. It also delays the update of the actor network and the target networks to reduce oscillation. TD3 updates the actor network using the policy gradient derived from the critic network, and updates the critic network using the temporal difference error between the target Q-value and the current Q-value. | TD3 can be used for tasks that require continuous actions, such as robotics, locomotion, or manipulation. |


Source: Conversation with Bing, 7/14/2023
(1) RL — Reinforcement Learning Algorithms Comparison - Medium. https://bing.com/search?q=reinforcement+learning+algorithms+comparison.
(2) Frontiers | Comparing Deep Reinforcement Learning Algorithms’ Ability .... https://www.frontiersin.org/articles/10.3389/frobt.2021.738113/full.
(3) Reinforcement Learning Algorithms: An Overview and Classiﬁcation. https://arxiv.org/pdf/2209.14940.pdf.
(4) [2209.14940] Reinforcement Learning Algorithms: An Overview and .... https://arxiv.org/abs/2209.14940.
(5) undefined. https://doi.org/10.3389/frobt.2021.738113.



# On and Off policy algirithms differeneces

- On-policy algorithms optimize the value of the current policy, while off-policy algorithms optimize the value of a different target policy¹.
- On-policy algorithms require the agent to explore the environment and learn from its own actions, while off-policy algorithms can learn from any data source, such as a fixed dataset or another agent's behavior¹.
- On-policy algorithms are more sample-efficient, meaning they need less data to learn, but they are also more sensitive to noise and variance, meaning they can be unstable or inaccurate². Off-policy algorithms are more robust and flexible, but they also suffer from bias and overestimation, meaning they can be optimistic or unrealistic³.

Some examples of on-policy algorithms are SARSA, REINFORCE, and A2C. Some examples of off-policy algorithms are Q-learning, DQN, and DDPG. 

I hope this helps you understand the difference between off-policy and on-policy reinforcement learning. If you want to learn more, you can check out these links:

- [Off-policy vs. On-policy Reinforcement Learning](^2^)
- [On-Policy v. Off-Policy Reinforcement Learning Explained](^3^)
- [On-Policy VS Off-Policy Reinforcement Learning](^4^)

Source: Conversation with Bing, 7/14/2023
(1) What is the relation between online (or offline) learning and on-policy .... https://ai.stackexchange.com/questions/10474/what-is-the-relation-between-online-or-offline-learning-and-on-policy-or-off.
(2) Off-policy vs. On-policy Reinforcement Learning - Baeldung. https://www.baeldung.com/cs/off-policy-vs-on-policy.
(3) On-Policy v. Off-Policy Reinforcement Learning Explained. https://medium.com/mlearning-ai/on-policy-v-off-policy-reinforcement-learning-explained-89054a6cc6.
(4) Off-policy vs. On-policy Reinforcement Learning - Baeldung. https://www.baeldung.com/cs/off-policy-vs-on-policy.
(5) On-Policy v. Off-Policy Reinforcement Learning Explained. https://medium.com/mlearning-ai/on-policy-v-off-policy-reinforcement-learning-explained-89054a6cc6.
(6) On-Policy VS Off-Policy Reinforcement Learning - Analytics India Magazine. https://analyticsindiamag.com/reinforcement-learning-policy/.
(7) Bootcamp Summer 2020 Week 4 – On-Policy vs Off-Policy Reinforcement .... https://core-robotics.gatech.edu/2022/02/28/bootcamp-summer-2020-week-4-on-policy-vs-off-policy-reinforcement-learning/.

# Different policies


| Policy | Description | Application |
| --- | --- | --- |
| Epsilon-greedy | Epsilon-greedy is a simple policy that balances exploration and exploitation. It chooses a random action with probability epsilon, and the best action according to the current value function with probability 1-epsilon. Epsilon can be fixed or decay over time¹. | Epsilon-greedy can be used for tasks that require discrete actions, such as bandits, gridworlds, or Atari games¹. |
| Boltzmann | Boltzmann is a stochastic policy that chooses actions according to a softmax function of the Q-values. It assigns higher probabilities to actions with higher Q-values, but also allows for some exploration. The temperature parameter controls the degree of exploration: higher temperatures lead to more random actions, while lower temperatures lead to more greedy actions². | Boltzmann can be used for tasks that require discrete actions, such as bandits, gridworlds, or Atari games². |
| Gaussian | Gaussian is a stochastic policy that outputs a normal distribution over continuous actions for each state. The mean of the distribution is determined by a neural network that takes the state as input, while the standard deviation can be fixed or learned. The agent samples an action from the distribution and executes it in the environment³. | Gaussian can be used for tasks that require continuous actions, such as robotics, locomotion, or manipulation³. |
| Beta | Beta is a stochastic policy that outputs a beta distribution over continuous actions for each state. The beta distribution is bounded between 0 and 1, so it can be useful for actions that have natural limits. The parameters of the distribution are determined by two neural networks that take the state as input. The agent samples an action from the distribution and executes it in the environment⁴. | Beta can be used for tasks that require continuous and bounded actions, such as steering angles, throttle values, or joint torques⁴. |
| Categorical | Categorical is a stochastic policy that outputs a categorical distribution over discrete actions for each state. The categorical distribution is parameterized by a vector of probabilities that sum to one. The probabilities are determined by a neural network that takes the state as input. The agent samples an action from the distribution and executes it in the environment⁵. | Categorical can be used for tasks that require discrete actions, such as Atari games, board games, or card games⁵. |
| Mixture of Experts | Mixture of Experts is a stochastic policy that combines multiple policies into one. Each policy is called an expert and has its own neural network. The agent also has a gating network that takes the state as input and outputs a weight for each expert. The agent samples an action from a weighted mixture of the experts' distributions and executes it in the environment. | Mixture of Experts can be used for tasks that require complex and diverse behaviors, such as StarCraft II, Dota 2, or autonomous driving. |
| Hierarchical | Hierarchical is a policy that consists of multiple levels of abstraction. Each level has its own policy and operates at a different time scale. The higher-level policy sets subgoals for the lower-level policy, while the lower-level policy executes actions to achieve those subgoals. The policies can communicate with each other through observations, rewards, or latent variables. | Hierarchical can be used for tasks that require long-term planning, temporal abstraction, or skill reuse, such as robotics, navigation, or strategy games. |
| Meta | Meta is a policy that can adapt to new tasks quickly and efficiently. It learns from a distribution of tasks and updates its parameters based on a few samples from a new task. The meta-policy can be initialized by pre-training on related tasks, or learned by optimizing an objective function that measures the performance on new tasks. | Meta can be used for tasks that require transfer learning, few-shot learning, or multi-task learning, such as robotics, vision, or natural language processing. |
| Imitation | Imitation is a policy that learns from demonstrations provided by an expert. It tries to mimic the expert's behavior by minimizing the discrepancy between its actions and the expert's actions. The imitation policy can be learned by supervised learning, inverse reinforcement learning, or generative adversarial learning. | Imitation can be used for tasks that are hard to specify with rewards, require human-like behavior, or have access to expert data, such as robotics, manipulation, or driving. |
| Curiosity | Curiosity is a policy that learns from intrinsic motivation rather than extrinsic rewards. It seeks novel and informative states by maximizing a measure of curiosity, such as prediction error, information gain, or empowerment. The curiosity policy can be combined with an extrinsic policy to enhance exploration and learning. | Curiosity can be used for tasks that have sparse, delayed, or deceptive rewards, require exploration, or have diverse and complex environments, such as Atari games, maze navigation, or Montezuma's Revenge. |


- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](^1^)
- [Boltzmann Exploration Done Right](^2^)
- [Continuous control with deep reinforcement learning](^3^)
- [Beta Policy for Continuous Control Deep Reinforcement Learning](^4^)
- [A Distributional Perspective on Reinforcement Learning](^5^)

Source: Conversation with Bing, 7/14/2023
(1) What Is a Policy in Reinforcement Learning? - Baeldung. https://www.baeldung.com/cs/ml-policy-reinforcement-learning.
(2) Verified Probabilistic Policies for Deep Reinforcement Learning. https://arxiv.org/abs/2201.03698.
(3) reinforcement learning - Why are policy gradients popular in RL when .... https://ai.stackexchange.com/questions/37406/why-are-policy-gradients-popular-in-rl-when-there-exists-a-dual-lp-formulation-i.
(4) Accelerating deep reinforcement learning via knowledge-guided policy .... https://link.springer.com/article/10.1007/s10458-023-09600-1.
(5) undefined. https://doi.org/10.48550/arXiv.2201.03698.

# Custom training Env



| Method | Description | Sample Code |
| --- | --- | --- |
| __init__ | This method initializes the environment and sets up some attributes, such as the observation space, the action space, and any parameters or variables that are needed for the environment. | def __init__(self, size): self.size = size # Size of the square grid self.observation_space = spaces.Dict({ \"agent\": spaces.MultiDiscrete([self.size, self.size]), \"target\": spaces.MultiDiscrete([self.size, self.size]) }) self.action_space = spaces.Discrete(4) # Four possible actions: right, up, left, down self._reset() # Reset the environment |
| reset | This method resets the environment to its initial state and returns the initial observation. It is called at the beginning of each episode or when the environment is done. | def reset(self): self.agent_pos = np.array([0, 0]) # Agent starts at the top-left corner self.target_pos = np.random.randint(0, self.size, size=2) # Target is placed randomly on the grid return { \"agent\": self.agent_pos, \"target\": self.target_pos } |
| step | This method takes an action as input and returns a tuple of (observation, reward, done, info). The observation is the state of the environment after the action. The reward is the scalar reward for taking the action. The done is a boolean flag that indicates whether the episode has ended. The info is a dictionary that contains any additional information about the environment or the transition. | def step(self, action): assert self.action_space.contains(action) # Check if the action is valid # Move the agent according to the action if action == 0: # right self.agent_pos[1] = min(self.agent_pos[1] + 1, self.size - 1) elif action == 1: # up self.agent_pos[0] = max(self.agent_pos[0] - 1, 0) elif action == 2: # left self.agent_pos[1] = max(self.agent_pos[1] - 1, 0) elif action == 3: # down self.agent_pos[0] = min(self.agent_pos[0] + 1, self.size - 1) # Check if the agent has reached the target done = np.array_equal(self.agent_pos, self.target_pos) # Reward is 1 if done, 0 otherwise reward = int(done) # Observation is the same as before observation = { \"agent\": self.agent_pos, \"target\": self.target_pos } # Info is an empty dictionary info = {} return observation, reward, done, info |
| render | This method renders the environment to a display or returns an image of the environment. It takes a mode argument that specifies how to render the environment. The mode can be one of the values in the metadata['render.modes'] attribute of the class. The method should return None if mode is None. | def render(self, mode='human'): if mode == 'human': # Render to the current display grid = np.zeros((self.size, self.size, 3)) # Create a black grid grid[self.target_pos[0], self.target_pos[1], :] = np.array([255, 0, 0]) # Paint the target cell red grid[self.agent_pos[0], self.agent_pos[1], :] = np.array([0, 0, 255]) # Paint the agent cell blue img = Image.fromarray(grid.astype('uint8'), 'RGB') # Convert the grid to an image img = img.resize((256, 256)) # Resize the image img.show() # Show the image elif mode == 'rgb_array': # Return an image of the environment grid = np.zeros((self.size, self.size, 3)) grid[self.target_pos[0], self.target_pos[1], :] = np.array([255, 0, 0]) grid[self.agent_pos[0], self.agent_pos[1], :] = np.array([0, 0, 255]) img = Image.fromarray(grid.astype('uint8'), 'RGB') return np.array(img) else: raise NotImplementedError |


- [Make your own custom environment](^1^)
- [Gym Documentation](^2^)
- [How to create a new gym environment in OpenAI?](^3^)

If you have any questions or want to try out your custom environment, please feel free to ask me. 😊

Source: Conversation with Bing, 7/14/2023
(1) Make your own custom environment - Gym Documentation. https://www.gymlibrary.dev/content/environment_creation/.
(2) Gym Documentation. https://www.gymlibrary.dev/.
(3) How to create a new gym environment in OpenAI?. https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai.
(4) undefined. https://github.com/Farama-Foundation/gym-examples.
(5) undefined. https://github.com/openai/gym/blob/master/docs/creating_environments.md.

# Scoring methods


| Environment | Scoring Method | Sample Code |
| --- | --- | --- |
| CartPole-v1 | The score is the number of steps taken before the pole falls over or the cart moves more than 2.4 units from the center². | def cartpole_score(env, agent): observation = env.reset() done = False score = 0 while not done: action = agent.act(observation) observation, reward, done, info = env.step(action) score += reward return score |
| MountainCar-v0 | The score is the negative of the position of the car at the end of the episode. The goal is to reach the flag at position 0.5³. | def mountaincar_score(env, agent): observation = env.reset() done = False while not done: action = agent.act(observation) observation, reward, done, info = env.step(action) score = -observation[0] return score |
| LunarLander-v2 | The score is the total reward accumulated during the episode. The reward is based on landing pad contact, fuel consumption, speed, and angle⁴. | def lunarlander_score(env, agent): observation = env.reset() done = False score = 0 while not done: action = agent.act(observation) observation, reward, done, info = env.step(action) score += reward return score |

I hope this table helps you understand how to create a custom scoring method in Gym. If you want to learn more, you can check out these links:

- [Creating an Automatic and Custom Scoring System for Any Sport](^1^)
- [Make your own custom environment](^2^)
- [Building a Reinforcement Learning Environment using OpenAI Gym](^3^)


Source: Conversation with Bing, 7/14/2023
(1) Make your own custom environment - Gym Documentation. https://www.gymlibrary.dev/content/environment_creation/.
(2) Creating an Automatic and Custom Scoring System for Any Sport. https://www.themeboy.com/blog/creating-automatic-custom-scoring-system-sport/.
(3) undefined. https://github.com/Farama-Foundation/gym-examples.