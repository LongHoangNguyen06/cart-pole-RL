# Approaching controlling problem (CartPole-v1) with Reinforcement Learning

### Background

The purpose of this project is to practice my data science and researching skill. In order to concentrate myself more on the algorithmic side and less on the data side, I chose inverted pendulum problem which is simple in its nature but still pose enough of challenge to be a non-trivial for Deep-Reinforcement-Learning.

There are two versions of `CartPole`, namely the `v0` and `v1` versions. According to my best knowledge, the only difference between the two versions are their threshold of when a solution is satisfactory [[1]](https://stackoverflow.com/a/56926451). In my target version `CartPole-v1`, a solution is good if it achieves in average $475/500$ points over $100$ random initial states. A point is rewarded for each step the pendulum's angle is in the range $(-.2095, .2095)$ rad and the pivot point's distance to center is in the range $[0, 2.4)$.  

<p align="center">
<img src="images/cartpole.png"/>
</p>

### Goals of the project

To consider this project to be a success, I aim to implement popular approaches in Deep-RL to achieve a score 

### Related works

 Below are some other projects that provide performance baselines for my own implementation