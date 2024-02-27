# Approaching controlling problem (CartPole-v1) with Reinforcement Learning

### Background

The purpose of this project is to practice my data science and researching skill. In order to concentrate myself more on the algorithmic side and less on the data side, I chose inverted pendulum problem which is simple in its nature but still pose enough of challenge to be a non-trivial for Deep-Reinforcement-Learning.

There are two versions of `CartPole`, namely the `v0` and `v1` versions. According to my best knowledge, the only difference between the two versions is their threshold of when a solution is satisfactory [[1]](https://stackoverflow.com/a/56926451). In my target version `CartPole-v1`, a solution is good if it achieves in average $475/500$ points over $100$ random initial states. A point is rewarded for each step the pendulum's angle is in the range $(-0.2095, 0.2095)$ rad and the pivot point's distance to center is in the range $[0, 2.4)$ [[2]](https://www.gymlibrary.dev/environments/classic_control/cart_pole/).  

<p align="center">
<img src="images/cartpole.png" width=400/>
</p>

### Goals of the project

To consider this project to be a success, I aim to achieve the gold-standard score with Deep-Q-Learning without too much computing resource wasted on training. Since I lack experience in bootstrapping ML projects with the correct hyperparameter, resource spent on hyper-parameter optimization will not be considered. 

### Related practical projects

Below are some other projects that provide performance baselines for my own implementation. Since some projects are of tutorial nature, this listing is not meant for performance comparison; I consider it a survey of methodology.

<table>
  <tr>
    <th>Link</th>
    <th>Method</th>
    <th>Performance</th>
  </tr>
  <tr>
    <td><a src="https://medium.com/analytics-vidhya/q-learning-is-the-most-basic-form-of-reinforcement-learning-which-doesnt-take-advantage-of-any-8944e02570c5">Solving Open AI’s CartPole Using Reinforcement Learning Part-1
</a></td>
    <td>Q-Learning</td>
    <td>Rewards 195 after 28,000 episodes.</td>
  </tr>
  <tr>
    <td><a src="https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html">PyTorch - Reinforcement Learning (DQN) Tutorial</a></td>
    <td>Deep-Q-Learning with one hidden layer consists of 128 neurons.</td>
    <td>Best rewards ~175 after ~350 episodes.</td>
  </tr>
  <tr>
    <td><a src="https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic">TensorFlow - Playing CartPole with the Actor-Critic method</a></td>
    <td>Actor-Critic with one hidden layer consists of 128 neurons.</td>
    <td>Best rewards 475 after 237 episodes.</td>
  </tr>
</table>

An episode is a round, beginning with a random initial state of the pendulum. The episode is over when the pendulum's state is outside of permitted range or when the score is over $200$ (`v0` version) or $500$ (`v1` version).

### Related research projects