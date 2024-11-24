import gymnasium as gym
from gymnasium import spaces
import numpy as np

class InventoryManagementEnv(gym.Env):
    """
    Inventory Management Environment with Multiple Articles and Normalized Spaces.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, num_articles=1):
        super(InventoryManagementEnv, self).__init__()
        
        self.num_articles = num_articles  # Number of articles
        
        # Environment constants
        self.max_demand = 100
        self.max_delivery = 200
        self.delivery_increment = 10
        self.max_stock = 400  # Maximum possible stock (to define observation space)
        
        # Action space: Normalized delivery quantities for each article in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_articles,),
            dtype=np.float32
        )
        
        # Observation space: Normalized to [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5 * self.num_articles + 1,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initial stock levels for each article
        self.stock_day1 = np.zeros(self.num_articles, dtype=np.int32)  # Stock that can be sold today
        self.stock_day2 = np.zeros(self.num_articles, dtype=np.int32)  # Stock that will expire after today

        # Simulate predicted demand
        self.predicted_demand_day1 = np.maximum(0, np.random.uniform(-25, self.max_demand, size=self.num_articles))
        self.predicted_demand_day2 = np.maximum(0, np.random.uniform(-25, self.max_demand, size=self.num_articles))

        # Standard deviation of demand error for each article
        # Set demand_std_dev from options if provided, else use random values.
        if options is not None and 'demand_std_dev' in options:
            self.demand_std_dev = options['demand_std_dev']
        else:
            self.demand_std_dev = np.random.uniform(0, 50, size=self.num_articles)

        # Initialize selling price
        if options is not None and 'unit_selling_price' in options:
            self.unit_selling_price = options['unit_selling_price']
        else:
            self.unit_selling_price = np.random.uniform(1, 15, size=self.num_articles)
        
        self.current_step = 0
        self.max_steps = 30
        return self._get_observation(), {}
    
    def step(self, action):
        # Map normalized action [-1, 1] to delivery quantity [0, max_delivery]
        # First, scale action from [-1, 1] to [0, 1]
        scaled_action = (action + 1) / 2  # Now in [0, 1]
        # Then scale to delivery quantity
        delivery_quantity = (scaled_action * self.max_delivery).astype(np.int32)
        # Round to nearest delivery increment
        delivery_quantity = (np.round(delivery_quantity / self.delivery_increment) * self.delivery_increment).astype(np.int32)
        delivery_quantity = np.clip(delivery_quantity, 0, self.max_delivery)
        
        # Update stock levels with new delivery
        # Delivered stock becomes available for sale today (Day 1 stock)
        self.stock_day1 += delivery_quantity
        
        # Demand realization for each article
        actual_demand, adjusted_actual_demand = self._get_actual_demand()
        
        # Sales happen, oldest stock sold first (FIFO)
        total_available_stock = self.stock_day1 + self.stock_day2
        units_sold = np.minimum(total_available_stock, adjusted_actual_demand)

        # Some of the demand cannot be met.
        sellout_value = sum(np.maximum(0, (adjusted_actual_demand - total_available_stock) * self.unit_selling_price))

        # Any remaining stock is unsold stock from Day 2 (expires)
        unsellable_stock = np.maximum(0, self.stock_day2 - units_sold)
        
        # Update stock levels for the next day
        self.stock_day2 = np.maximum(0, self.stock_day1 - np.maximum(0, units_sold - self.stock_day2))
        self.stock_day1 = np.zeros(self.num_articles, dtype=np.int32)  # New deliveries will be added in the next step
        
        # Turnover calculation
        turnover = units_sold * self.unit_selling_price  # Array of turnovers per article
        total_turnover = np.sum(turnover)
        
        # Adjust turnover for unsellable stock (deduct value of expired stock)
        lost_value = unsellable_stock * self.unit_selling_price
        total_lost_value = np.sum(lost_value)
        
        adjusted_turnover = total_turnover - total_lost_value

        # Normalize the reward
        discounting_factor = sum(self.max_delivery * self.unit_selling_price)
        # discounting_factor = np.maximum(sum((total_available_stock + units_sold) * self.unit_selling_price) / 2, 1)
        normalized_reward = adjusted_turnover / discounting_factor
        
        # Set the info dictionary (can include diagnostic information)
        info = {
            'units_sold': units_sold,
            'turnover': total_turnover,
            'actual_demand_value': sum(actual_demand * self.unit_selling_price),
            'adjusted_actual_demand_value': sum(adjusted_actual_demand * self.unit_selling_price),
            'unsellable_stock': unsellable_stock,
            'returned_stock_value': total_lost_value,
            'remaining_stock': self.stock_day2,
            'total_available_stock_after_deliveries_value': sum(total_available_stock * self.unit_selling_price),
            'sellout_value': sellout_value,
            'predicted_value': sum(self.predicted_demand_day1 * self.unit_selling_price)
        }

        # Prepare for next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Get new prediction for next day
        self.predicted_demand_day1 = self.predicted_demand_day2.copy()
        self.predicted_demand_day2 = np.maximum(0, np.random.uniform(-25, self.max_demand, size=self.num_articles))
        
        # Get new observation
        observation = self._get_observation()
        
        return observation, normalized_reward, done, truncated, info
    
    def _get_observation(self):
        # Normalize stock levels to [-1, 1]
        normalized_stock_day2 = (self.stock_day2 / self.max_stock) * 2 - 1  # Scale to [-1, 1]

        # Normalize prediction to max stock to [-1, 1]
        normalized_demand_day1 = (self.predicted_demand_day1 / self.max_stock) * 2 - 1
        normalized_demand_day2 = (self.predicted_demand_day2 / self.max_stock) * 2 - 1

        normalized_std_dev = (self.demand_std_dev / 50) * 2 - 1

        # Normalize price to [-1, 1]
        normalized_price = (self.unit_selling_price / 15) * 2 - 1

        # Normalize time left to [-1, 1]
        time_left = np.array([ (self.max_steps - self.current_step) / self.max_steps ]) * 2 - 1

        # Combine everything
        observation = np.concatenate([
            normalized_stock_day2, 
            normalized_demand_day1, normalized_demand_day2,
            normalized_std_dev,
            normalized_price,
            time_left])

        return observation.astype(np.float32)
    
    def _get_actual_demand(self):
        # Actual demand is predicted demand plus normally distributed error for each article
        demand_error = np.random.normal(0, self.demand_std_dev)
        actual_demand = self.predicted_demand_day1 + demand_error
        actual_demand = np.maximum(0, actual_demand).astype(np.int32)  # Demand cannot be negative

        # If demand is higher than stock levels, we assume substitution to other articles.
        # However, we always assume that 30 % of sales are not realized.
        substitution_amount = sum(np.maximum(0, actual_demand - (self.stock_day1 + self.stock_day2))) * 0.7 / self.num_articles

        # Calculate the number of sellouts
        difference = np.maximum(0, actual_demand - (self.stock_day1 + self.stock_day2))
        sellout_occurred = difference > 0
        # Max is used to ensure that no devision by 0 occurs.
        num_sellouts = np.maximum(np.sum(sellout_occurred), 1)

        # Adjust actual_demand based on sellouts and substitution
        adjusted_actual_demand = np.where(
            sellout_occurred,
            actual_demand - difference * 0.7,
            actual_demand + substitution_amount / num_sellouts
        )

        return actual_demand, adjusted_actual_demand
    
    def render(self):
        print(f"Day {self.current_step}")
        for i in range(self.num_articles):
            print(f"Article {i+1}:")
            print(f"  Stock for Day 1: {self.stock_day1[i]}")
            print(f"  Stock for Day 2: {self.stock_day2[i]}")
        
    def close(self):
        pass

