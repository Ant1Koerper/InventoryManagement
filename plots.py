import matplotlib.pyplot as plt
import numpy as np

def plot_agent_value_coverage(env, agent, steps, options=None):
    # Initialize lists to store aggregated monetary values
    predicted_demand_value_history = []
    actual_demand_value_history = []
    sales_revenue_history = []
    total_stock_value_history = []
    unmet_demand_value_history = []
    unsold_stock_value_history = []
    
    obs, _ = env.reset(options=options)
    for _ in range(steps):
        action, _ = agent.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        # Store aggregated monetary values
        predicted_demand_value_history.append(info['predicted_value'])
        actual_demand_value_history.append(info['actual_demand_value'])
        sales_revenue_history.append(info['turnover'])
        total_stock_value_history.append(info['total_available_stock_after_deliveries_value'])
        unmet_demand_value_history.append(info['sellout_value'])
        unsold_stock_value_history.append(info['returned_stock_value'])
        
        if done:
            break
    
    # Convert lists to arrays
    predicted_demand_value_history = np.array(predicted_demand_value_history)
    actual_demand_value_history = np.array(actual_demand_value_history)
    sales_revenue_history = np.array(sales_revenue_history)
    total_stock_value_history = np.array(total_stock_value_history)
    unmet_demand_value_history = np.array(unmet_demand_value_history)
    unsold_stock_value_history = np.array(unsold_stock_value_history)
    
    # Plot the aggregated monetary values
    plt.figure(figsize=(14, 8))
    time_steps = np.arange(len(predicted_demand_value_history))
    
    # Plot sales revenue
    plt.plot(time_steps, sales_revenue_history, label='Sales Revenue', linestyle='-.')
    
    # Plot total stock value
    plt.plot(time_steps, total_stock_value_history, label='Total Stock Value', linestyle=':')
    
    # Highlight unmet demand value (lost revenue)
    plt.bar(time_steps, unmet_demand_value_history, bottom=sales_revenue_history, color='red', alpha=0.3, label='Unmet Demand Value (Lost Sales)')
    
    # Highlight unsold stock value (returns)
    plt.bar(time_steps, unsold_stock_value_history, bottom=0, color='orange', alpha=0.3, label='Unsold Stock Value (Returns)')
    
    plt.title('Aggregate Demand Coverage in Monetary Value Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Monetary Value')
    
    # Fix the legend position
    plt.legend(loc='upper right')
    
    # Set y-axis ticks to increase in increments of 2000
    y_min = 0
    # Determine the maximum y-value among all plotted data
    y_max = max(
        max(sales_revenue_history + unmet_demand_value_history),
        max(total_stock_value_history),
        max(unsold_stock_value_history)
    )
    # Round up to the next multiple of 2000
    y_max = int((y_max // 2000 + 1) * 2000)
    plt.yticks(np.arange(y_min, y_max + 1, 2000))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_value_sellout_and_return_rates(env, agent, steps, options=None):
    # Initialize lists to store aggregated monetary values
    sales_revenue_history = []
    total_stock_value_available_history = []
    unsold_stock_value_history = []
    sellout_value = []
    
    obs, _ = env.reset(options=options)
    for _ in range(steps):        
        action, _ = agent.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        # Store aggregated monetary values
        sales_revenue_history.append(info['turnover'])
        total_stock_value_available_history.append(info['total_available_stock_after_deliveries_value'])
        unsold_stock_value_history.append(info['returned_stock_value'])
        sellout_value.append(info['sellout_value'])
        
        if done:
            break
    
    # Convert lists to arrays
    sales_revenue_history = np.array(sales_revenue_history)
    total_stock_value_available_history = np.array(total_stock_value_available_history)
    unsold_stock_value_history = np.array(unsold_stock_value_history)
    sellout_value = np.array(sellout_value)
    
    # Calculate Sellout Rate and Return Rate based on value
    sellout_rate = sellout_value / np.maximum(sales_revenue_history + sellout_value, 1)  # Avoid division by zero
    return_rate = unsold_stock_value_history / np.maximum(total_stock_value_available_history, 1)
    
    # Plot Sellout Rate and Return Rate over time
    plt.figure(figsize=(14, 6))
    time_steps = np.arange(len(sales_revenue_history))
    
    # Plot Sellout Rate
    plt.plot(time_steps, sellout_rate, label='Sellout Rate (Value)', color='green', marker='o')
    
    # Plot Return Rate
    plt.plot(time_steps, return_rate, label='Return Rate (Value)', color='red', marker='x')
    
    plt.title('Sellout Rate and Return Rate Based on Value Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Rate')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()


