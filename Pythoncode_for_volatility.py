import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go


def calculate_implied_volatility(stock_symbol, expiration_date, strike_price, option_type='call'):
    try:
       
        option_chain = yf.Ticker(stock_symbol).option_chain(expiration_date)
        
       
        if option_type == 'call':
            option_data = option_chain.calls
        else:
            option_data = option_chain.puts

       
        option_data_filtered = option_data[option_data.strike == strike_price]
        
        
        if option_data_filtered.empty:
            print(f"Error: No options data found for strike price {strike_price} on {expiration_date}.")
            return None

        
        option_price = option_data_filtered['lastPrice'].iloc[0]
        
       
        stock_data = yf.Ticker(stock_symbol).history(start="2015-01-01", end="2015-12-31")
        
        if stock_data.empty:
            print(f"Error: No stock data found for {stock_symbol}.")
            return None
        
        S = stock_data['Close'].iloc[-1]

        
        expiration_date = pd.to_datetime(expiration_date).normalize()  # Normalize to remove any time component
        stock_data_date = stock_data.index[-1].normalize()  # Normalize the stock data date as well
        
       
        expiration_date = expiration_date.tz_localize(None)
        stock_data_date = stock_data_date.tz_localize(None)
        
       
        T = (expiration_date - stock_data_date).days / 365.0
        
        
        r = 0.01
        
        
        daily_volatility = np.std(np.diff(np.log(stock_data['Close'])))  # Using log returns
        sigma = daily_volatility * np.sqrt(252)
        
       
        d1 = (np.log(S / strike_price) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        
        theoretical_price = S * norm.cdf(d1) - strike_price * np.exp(-r * T) * norm.cdf(d2)
        
        
        implied_vol = np.abs(option_price - theoretical_price)
        
        return implied_vol

    except ValueError as e:
        print(f"Error: {e}")
        return None


def calculate_moving_average_volatility(stock_symbol, window=30):
    stock_data = yf.Ticker(stock_symbol).history(start="2015-01-01", end="2015-12-31")  # Get stock data for 2015
    if stock_data.empty:
        print(f"Error: No stock data found for {stock_symbol}.")
        return None
    
    stock_data['Log Return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    stock_data['Volatility'] = stock_data['Log Return'].rolling(window=window).std() * np.sqrt(252)
    return stock_data['Volatility']


def simulate_volatility_heston(stock_symbol, num_simulations=1000, days=252, kappa=2.0, theta=0.02, sigma=0.05, rho=-0.5, V0=0.02):
    stock_data = yf.Ticker(stock_symbol).history(start="2015-01-01", end="2015-12-31")
    if stock_data.empty:
        print(f"Error: No stock data found for {stock_symbol}.")
        return None
    
    S0 = stock_data['Close'].iloc[-1]  # Current stock price
    mu = stock_data['Close'].pct_change().mean()  # Mean return
    
    dt = 1/252  # daily time step
    simulations = np.zeros((num_simulations, days))

    for i in range(num_simulations):
        V = np.full(days, V0)  # Volatility path, starting with initial volatility V0
        S = np.full(days, S0)  # Stock price path
        for t in range(1, days):
            
            dW_v = np.random.normal(0, np.sqrt(dt))  # Brownian motion for volatility
            V[t] = V[t-1] + kappa*(theta - V[t-1])*dt + sigma*np.sqrt(V[t-1])*dW_v
            
            
            V[t] = max(V[t], 0)
            
           
            dW_s = np.random.normal(0, np.sqrt(dt))  # Brownian motion for stock
            S[t] = S[t-1] * np.exp((mu - 0.5*V[t-1]) * dt + np.sqrt(V[t-1]) * dW_s)
        
        
        simulated_volatility = np.std(np.diff(np.log(S))) * np.sqrt(252)
        simulations[i] = simulated_volatility

    return simulations


def plot_volatility_comparison(stock_symbol, moving_avg_volatility, simulated_volatility):
    fig = go.Figure()

 
    if moving_avg_volatility is not None:
        fig.add_trace(go.Scatter(x=moving_avg_volatility.index, y=moving_avg_volatility, mode='lines', name='Moving Average Volatility'))

  
    if simulated_volatility is not None:
        fig.add_trace(go.Histogram(x=simulated_volatility.flatten(), name='Simulated Volatility', opacity=0.75))

    fig.update_layout(title=f"Volatility Comparison for {stock_symbol}",
                      xaxis_title="Date",
                      yaxis_title="Volatility",
                      barmode='overlay')

    fig.show()


def calculate_rmse(moving_avg_volatility, simulated_volatility):
   
    min_length = min(len(moving_avg_volatility), len(simulated_volatility))
    moving_avg_volatility = moving_avg_volatility[-min_length:]
    simulated_volatility = simulated_volatility.flatten()[-min_length:]
    

    rmse = np.sqrt(np.mean((moving_avg_volatility - simulated_volatility) ** 2))
    
    return rmse


if __name__ == "__main__":
    stock_symbol = 'AAPL'  # Apple stock symbol
    expiration_date = '2025-01-17'  # Valid expiration date (change as per data availability)
    strike_price = 120  # Example strike price for the option


    implied_volatility = calculate_implied_volatility(stock_symbol, expiration_date, strike_price)
    
    if implied_volatility:
        print(f"Implied Volatility for strike price {strike_price} on {expiration_date}: {implied_volatility:.4f}")
    else:
        print("Implied Volatility calculation failed.")

   
    moving_avg_volatility = calculate_moving_average_volatility(stock_symbol, window=30)
    
    if moving_avg_volatility is not None:
        print(f"Moving Average Volatility: {moving_avg_volatility.tail()}")
    else:
        print("Moving Average Volatility calculation failed.")

  
    simulated_volatility = simulate_volatility_heston(stock_symbol, num_simulations=1000, days=252)

    if simulated_volatility is not None:
        print(f"Simulated Volatility (first 5 samples): {simulated_volatility[:5]}")
    else:
        print("Volatility simulation failed.")

   
    plot_volatility_comparison(stock_symbol, moving_avg_volatility, simulated_volatility)

    if moving_avg_volatility is not None and simulated_volatility is not None:
        rmse = calculate_rmse(moving_avg_volatility, simulated_volatility)
        print(f"RMSE between moving average and simulated volatility: {rmse:.4f}")

