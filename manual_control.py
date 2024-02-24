import gym
import my_parameters

# Initialize the environment
env = gym.make('CartPole-v1', render_mode="human")
env.action_space.seed(seed=my_parameters.MAP_SEED)
# Reset the environment to start a new episode
observation, info = env.reset(seed=my_parameters.MAP_SEED)

for _ in range(1000):  # Run for 1000 steps, you can adjust this as needed
    env.render()  # Render the environment to visualize the state
    
    # Select a random action (either 0 or 1)
    action = env.action_space.sample()
    
    # Take the action and receive the new state, reward, done (if the episode is over), and info
    #step = env.step(action=action)
    observation, reward, done, truncated, info = env.step(action)
    
    if done:
        observation = env.reset()  # Reset the environment if the episode is over

env.close()  # Close the environment
