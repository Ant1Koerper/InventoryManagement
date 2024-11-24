import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from inventory_env import InventoryManagementEnv
from plots import plot_agent_value_coverage, plot_value_sellout_and_return_rates

# Define number of articles
num_articles = 10

# Create the environment
env = InventoryManagementEnv(num_articles=num_articles)

# Check the environment
check_env(env)

# Instantiate the PPO agent
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1
)

# Train the agent
model.learn(total_timesteps=7e5)

# Save the model (optional)
# model.save("ppo_inventory_management")
# model.load("ppo_inventory_management")

# Evaluate the agent
options = dict(
    demand_std_dev =  np.full(num_articles, 25)
)
plot_agent_value_coverage(env, agent=model, steps=30, options=options)
plot_value_sellout_and_return_rates(env, agent=model, steps=30, options=options)

options = dict(
    demand_std_dev =  np.full(num_articles, 1)
)
plot_agent_value_coverage(env, agent=model, steps=30, options=options)
plot_value_sellout_and_return_rates(env, agent=model, steps=30, options=options)
