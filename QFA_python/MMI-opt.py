import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from concurrent.futures import ThreadPoolExecutor

# Constants
STOCKS = ["AAPL", "WMT", "TSLA", "GE", "AMZN", "DB"]
NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000
START_DATE = '2012-01-01'
END_DATE = '2017-01-01'

def download_data():
    """Download stock data using yfinance's batch download capability"""
    data = yf.download(STOCKS, start=START_DATE, end=END_DATE, progress=False)['Close']
    # Make sure the data is a proper DataFrame with the right column names
    return pd.DataFrame(data)

def calculate_return(data):
    """Calculate log returns"""
    log_return = np.log(data/data.shift(1))
    return log_return.dropna()  # More explicit than [1:]

def portfolio_statistics(weights, returns):
    """Calculate portfolio statistics in one function"""
    mean_returns = returns.mean()
    cov_matrix = returns.cov() * NUM_TRADING_DAYS
    
    portfolio_return = np.sum(mean_returns * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])

def generate_portfolios_chunk(chunk_size, n_stocks, mean_returns, cov_matrix):
    """Generate a chunk of random portfolios"""
    # Use Dirichlet distribution for weights that sum to 1
    weights = np.random.dirichlet(np.ones(n_stocks), chunk_size)
    
    # Calculate portfolio means (vectorized)
    portfolio_means = np.sum(weights * mean_returns.values.reshape(1, -1), axis=1) * NUM_TRADING_DAYS
    
    # Calculate portfolio risks
    portfolio_risks = np.zeros(chunk_size)
    for i in range(chunk_size):
        portfolio_risks[i] = np.sqrt(np.dot(weights[i].T, np.dot(cov_matrix.values, weights[i])))
    
    return weights, portfolio_means, portfolio_risks

def generate_portfolios(returns, num_workers=4):
    """Generate portfolios using parallel processing"""
    n_stocks = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov() * NUM_TRADING_DAYS
    
    chunk_size = NUM_PORTFOLIOS // num_workers
    chunks = [chunk_size] * num_workers
    
    # Add remainder to last chunk
    chunks[-1] += NUM_PORTFOLIOS % num_workers
    
    # Use ThreadPoolExecutor for parallel processing
    all_weights = []
    all_means = []
    all_risks = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for size in chunks:
            futures.append(executor.submit(
                generate_portfolios_chunk, 
                size, n_stocks, mean_returns, cov_matrix
            ))
        
        for future in futures:
            weights, means, risks = future.result()
            all_weights.append(weights)
            all_means.append(means)
            all_risks.append(risks)
    
    return (
        np.vstack(all_weights),
        np.concatenate(all_means),
        np.concatenate(all_risks)
    )

def optimize_portfolio(returns):
    """Find optimal portfolio weights to maximize Sharpe ratio"""
    n_stocks = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov() * NUM_TRADING_DAYS
    
    # Define the negative Sharpe ratio function (to minimize)
    def min_function_sharpe(weights):
        portfolio_return = np.sum(mean_returns * weights) * NUM_TRADING_DAYS
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility
    
    # Use equal weights as initial guess
    initial_weights = np.ones(n_stocks) / n_stocks
    
    # Define constraints and bounds
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_stocks))
    
    # Run optimization
    return optimization.minimize(
        fun=min_function_sharpe, 
        x0=initial_weights,
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )

def display_results(optimum, returns, means, risks):
    """Display results with enhanced visualization"""
    if not optimum.success:
        print("Optimization failed to converge")
        return
    
    # Calculate statistics for optimal portfolio
    opt_stats = portfolio_statistics(optimum.x, returns)
    
    # Print results
    print("\nOptimal Portfolio Weights:")
    for i, stock in enumerate(returns.columns):
        print(f"{stock}: {optimum.x[i]:.4f}")
    
    print("\nPortfolio Statistics:")
    print(f"Expected Annual Return: {opt_stats[0]:.4f}")
    print(f"Expected Annual Volatility: {opt_stats[1]:.4f}")
    print(f"Sharpe Ratio: {opt_stats[2]:.4f}")
    
    # Calculate Sharpe ratios for all portfolios
    sharpe = means / risks
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(risks, means, c=sharpe, cmap='viridis', 
                         alpha=0.5, marker='o')
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.scatter(opt_stats[1], opt_stats[0], marker='*', color='r', s=300, 
               label='Optimal Portfolio')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title('Portfolio Optimization - Efficient Frontier')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    """Main function with error handling"""
    try:
        # Download data
        print("Downloading stock data...")
        stock_data = download_data()
        
        # Calculate returns
        print("Calculating returns...")
        log_returns = calculate_return(stock_data)
        
        # Generate random portfolios
        print(f"Generating {NUM_PORTFOLIOS} random portfolios...")
        weights, means, risks = generate_portfolios(log_returns)
        
        # Find optimal portfolio
        print("Finding optimal portfolio...")
        optimum = optimize_portfolio(log_returns)
        
        # Display results
        display_results(optimum, log_returns, means, risks)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
