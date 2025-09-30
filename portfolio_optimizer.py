Advanced Portfolio Optimization Pipeline

This script implements an end-to-end pipeline for constructing and analyzing
an optimal financial portfolio based on Modern Portfolio Theory (MPT).

Key features include:
- Hybrid forecasting of returns using ARIMA and LSTM models.
- Dynamic volatility forecasting with GARCH.
- Markowitz optimization with constraints using cvxpy.
- Advanced risk analysis (VaR, CVaR, Monte Carlo simulations).
- Parallel processing for efficient forecasting across multiple assets.

Author: Imad LOTFI
Date: May 2024
Context: Personal project for the 'Analytic Edge' module at EMINES - UM6P.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import cvxpy as cp

import warnings
import logging
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from scipy.stats import norm
import concurrent.futures

# Optional: Import TensorFlow for LSTM modeling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# Utility for display in Jupyter notebooks or terminals
try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging to a file
logging.basicConfig(
    filename='portfolio_optimization.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

######################################################################
# 1. General Parameters
######################################################################

tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JNJ', 'V', 'WMT',
    'JPM', 'UNH', 'NVDA', 'HD', 'PG', 'BAC', 'DIS', 'MA', 'PYPL', 'CMCSA',
    'ADBE', 'NFLX', 'PFE', 'INTC', 'T', 'VZ', 'KO', 'CSCO', 'PEP', 'XOM',
    'CVX', 'MRK', 'ABT', 'TMO', 'ACN', 'AVGO', 'COST', 'DHR', 'LIN', 'MCD',
    'NKE', 'ORCL', 'QCOM', 'TXN', 'UPS', 'WFC', 'ABBV', 'AMD', 'BMY', 'CAT',
    'CRM', 'DE', 'F', 'GE', 'GM', 'GS', 'HON', 'IBM', 'INTU', 'JCI', 'KMB',
    'LLY', 'LMT', 'LOW', 'MMM', 'MO', 'MRVL', 'MS', 'NEE', 'NOC', 'PGR', 'PLD',
    'PM', 'RTX', 'SBUX', 'SCHW', 'SO', 'SPG', 'TGT', 'TJX', 'USB', 'VLO', 'WM'
]

# Macroeconomic factors (e.g., S&P 500, VIX, 10-Year US Treasury)
macro_tickers = ['^GSPC', '^VIX', '^TNX']

START_DATE = '2015-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

# Daily risk-free rate (approx). Annualized = daily * 252 trading days
RISK_FREE_RATE_DAILY = 0.0001

# Portfolio weight constraints
MAX_WEIGHT = 0.15
MIN_WEIGHT = 0.0

# Monte Carlo simulation parameters
MC_SIMULATIONS = 5000
MC_HORIZON_DAYS = 252  # Approx. 1 trading year

######################################################################
# 2. Data Download & Cleaning
######################################################################

def download_data(tickers, start_date, end_date):
    """Downloads adjusted closing prices from Yahoo Finance."""
    try:
        logging.info("Downloading market data from yfinance.")
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker')

        adj_close = pd.DataFrame()
        for tk in tickers:
            if tk in data and not data[tk].empty:
                adj_close[tk] = data[tk].get('Adj Close', data[tk].get('Close'))

        adj_close.index = adj_close.index.tz_localize(None)

        missing = [tk for tk in tickers if tk not in adj_close.columns]
        if missing:
            logging.warning(f"Could not download data for tickers: {missing}")

        return adj_close

    except Exception as e:
        logging.error(f"Error during data download: {e}")
        return None

def download_macro_data(macro_tickers, start_date, end_date):
    """Downloads macroeconomic factor data from Yahoo Finance."""
    try:
        logging.info("Downloading macro factor data from yfinance.")
        macro_data = yf.download(macro_tickers, start=start_date, end=end_date, progress=False, group_by='ticker')

        macro_close = pd.DataFrame()
        for tk in macro_tickers:
            if tk in macro_data and not macro_data[tk].empty:
                macro_close[tk] = macro_data[tk].get('Adj Close', macro_data[tk].get('Close'))

        macro_close.index = macro_close.index.tz_localize(None)
        return macro_close

    except Exception as e:
        logging.error(f"Error during macro data download: {e}")
        return None

def clean_data(data):
    """
    Cleans the data by dropping empty columns and imputing missing values using KNN.
    Note: KNN imputation on time series can introduce lookahead bias.
    A rolling window imputation would be a more robust approach in a production system.
    """
    try:
        logging.info("Cleaning data (KNN Imputer).")
        
        # Drop columns that are entirely NaN
        empty_cols = data.columns[data.isna().all()]
        if not empty_cols.empty:
            logging.warning(f"Dropping fully empty columns: {list(empty_cols)}")
            data = data.drop(columns=empty_cols)

        # Impute remaining missing values
        imputer = KNNImputer(n_neighbors=3)
        clean_array = imputer.fit_transform(data)
        df_clean = pd.DataFrame(clean_array, index=data.index, columns=data.columns)
        return df_clean

    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        return None

######################################################################
# 3. Feature Engineering
######################################################################

def create_technical_features(prices, window=14):
    """Creates basic technical features (SMA, EMA, RSI, Volatility)."""
    
    def rsi(series, window):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    feats = {}
    for col in prices.columns:
        series = prices[col]
        feats[f"{col}_SMA"] = series.rolling(window).mean()
        feats[f"{col}_EMA"] = series.ewm(span=window).mean()
        feats[f"{col}_RSI"] = rsi(series, window)
        feats[f"{col}_Vol"] = series.pct_change().rolling(window).std()

    return pd.DataFrame(feats, index=prices.index)

def merge_prices_and_features(prices, macro_data, tech_feats):
    """Merges all dataframes into a single one for modeling."""
    return pd.concat([prices, macro_data, tech_feats], axis=1, join='inner').dropna()

######################################################################
# 4. Returns Calculation
######################################################################

def calculate_log_returns(prices):
    """Calculates logarithmic returns from a price series."""
    return np.log(prices / prices.shift(1)).dropna()

######################################################################
# 5. Forecasting Models (ARIMA, GARCH, LSTM)
######################################################################

def forecast_arima(ts, steps=1):
    """
    Performs an ARIMA forecast after a small grid search for the best order (p,d,q).
    """
    if len(ts) < 30: # Fallback to mean for short series
        return np.full(steps, ts.mean())
    
    best_aic, best_model = np.inf, None
    for p in [0, 1, 2]:
        for d in [0, 1]:
            for q in [0, 1, 2]:
                try:
                    model = ARIMA(ts, order=(p, d, q)).fit(method_kwargs={"warn_convergence": False})
                    if model.aic < best_aic:
                        best_aic, best_model = model.aic, model
                except:
                    continue
    
    return best_model.forecast(steps=steps).values if best_model else np.full(steps, ts.mean())


def forecast_garch_vol(ts, steps=1):
    """Forecasts future volatility (standard deviation) using a GARCH model."""
    if len(ts) < 30: # Fallback to historical standard deviation
        return np.full(steps, ts.std())

    try:
        garch = arch_model(ts, p=1, q=1, vol='GARCH', dist='normal').fit(disp='off')
        forecast = garch.forecast(horizon=steps)
        var_forecast = forecast.variance.iloc[-1].values
        return np.sqrt(var_forecast)
    except:
        return np.full(steps, ts.std())


def create_supervised_dataset(data, window=30):
    """Creates a supervised learning dataset (X, y) for LSTM."""
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:(i + window), :-1])
        y.append(data[i + window, -1])
    return np.array(X), np.array(y)


def forecast_lstm_features(df, window=30, epochs=10, steps=1):
    """
    Forecasts a target series using an LSTM model with multiple features.
    The last column of the input dataframe is assumed to be the target.
    """
    if len(df) < 2 * window:
        return np.full(steps, df.iloc[:, -1].mean())

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = create_supervised_dataset(df_scaled, window=window)
    if X.shape[0] < 10:
        return np.full(steps, df.iloc[:, -1].mean())

    model = Sequential([
        LSTM(32, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    
    # Iterative forecasting
    last_window = df_scaled[-window:, :-1]
    forecast_scaled = []
    for _ in range(steps):
        pred = model.predict(np.expand_dims(last_window, axis=0), verbose=0)[0, 0]
        forecast_scaled.append(pred)
        # Append prediction and shift window
        new_row = np.append(last_window[-1, 1:], pred)
        last_window = np.vstack([last_window[1:], new_row])
    
    # Inverse transform the forecast
    dummy_array = np.zeros((len(forecast_scaled), df.shape[1]))
    dummy_array[:, -1] = forecast_scaled
    return scaler.inverse_transform(dummy_array)[:, -1]


def forecast_ensemble(ts, feats_df, steps=1, alpha=0.5):
    """Combines ARIMA and LSTM forecasts into a weighted ensemble."""
    fc_arima = forecast_arima(ts, steps=steps)

    if feats_df.empty or not LSTM_AVAILABLE:
        fc_lstm = np.full(steps, ts.mean())
    else:
        df_aligned = feats_df.copy()
        df_aligned['target'] = ts
        df_aligned.dropna(inplace=True)
        fc_lstm = forecast_lstm_features(df_aligned, steps=steps)

    return alpha * fc_lstm + (1 - alpha) * fc_arima

######################################################################
# 6. Parallel Forecasting Execution
######################################################################

def forecast_for_ticker_parallel(ticker, returns, full_df, macro_tickers, alpha_ensemble):
    """
    A single-ticker forecasting function designed for parallel execution.
    Returns a dictionary with the forecast return and volatility.
    """
    try:
        ts = returns[ticker].dropna()
        if ts.empty:
            return {'ticker': ticker, 'forecast_1d': 0.0, 'garch_vol': 0.0}

        # Prepare features for LSTM
        feature_cols = [c for c in full_df.columns if c.startswith(f"{ticker}_") or c in macro_tickers]
        sub_df = full_df[feature_cols].copy()

        # Forecast return with ensemble model
        fc_1d = forecast_ensemble(ts, sub_df, steps=1, alpha=alpha_ensemble)[0]
        
        # Forecast volatility with GARCH
        vol_1d = forecast_garch_vol(ts, steps=1)[0]

        return {'ticker': ticker, 'forecast_1d': fc_1d, 'garch_vol': vol_1d}
        
    except Exception as e:
        logging.error(f"Error forecasting for {ticker}: {e}")
        return {'ticker': ticker, 'forecast_1d': 0.0, 'garch_vol': returns[ticker].std()}


def run_all_forecasts_in_parallel(returns, full_df, macro_tickers, alpha_ensemble=0.5):
    """
    Executes forecasts for all tickers in parallel using a process pool.
    """
    forecasts = {}
    vols = {}
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(forecast_for_ticker_parallel, tk, returns, full_df, macro_tickers, alpha_ensemble): tk for tk in returns.columns}
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            ticker = result['ticker']
            forecasts[ticker] = result['forecast_1d']
            vols[ticker] = result['garch_vol']

    return pd.Series(forecasts), pd.Series(vols)

######################################################################
# 7. Portfolio Construction & Optimization
######################################################################

def calculate_covariance_matrix(returns, daily_vol_forecasts):
    """
    Constructs a forward-looking covariance matrix using historical correlations
    and GARCH-forecasted volatilities.
    """
    corr_matrix = returns.corr()
    vol_series = daily_vol_forecasts.reindex(index=returns.columns).fillna(returns.std())
    
    # Sigma = D * Corr * D, where D is the diagonal matrix of standard deviations
    Sigma = corr_matrix.multiply(vol_series, axis=0).multiply(vol_series, axis=1)
    return Sigma


def optimize_portfolio(mu, Sigma, tickers):
    """
    Performs Markowitz portfolio optimization to find the portfolio with the
    maximum Sharpe ratio.
    """
    n = len(tickers)
    w = cp.Variable(n)
    
    # Objective: Maximize Sharpe Ratio (equivalent to minimizing risk for a given return)
    risk = cp.quad_form(w, Sigma)
    ret = mu @ w
    
    # We scan through a range of target returns to trace the efficient frontier
    max_sharpe = -np.inf
    optimal_weights = None
    
    mu_mean = mu.mean()
    for target_ret in np.linspace(0.8 * mu_mean, 1.2 * mu_mean, 20):
        constraints = [
            cp.sum(w) == 1,
            w >= MIN_WEIGHT,
            w <= MAX_WEIGHT,
            ret >= target_ret
        ]
        prob = cp.Problem(cp.Minimize(risk), constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        
        if w.value is not None:
            er = ret.value
            vol = np.sqrt(risk.value)
            sharpe = (er - (RISK_FREE_RATE_DAILY * 252)) / vol if vol > 0 else 0
            
            if sharpe > max_sharpe:
                max_sharpe = sharpe
                optimal_weights = w.value

    # Normalize weights to sum to 1
    if optimal_weights is not None:
        optimal_weights = np.maximum(optimal_weights, 0)
        optimal_weights /= optimal_weights.sum()

    return optimal_weights, max_sharpe


######################################################################
# 8. Risk Analysis (VaR & CVaR)
######################################################################

def calculate_var_cvar(weights, Sigma, mu, portfolio_value=1_000_000, confidence_level=0.95):
    """Calculates parametric and Monte Carlo-based VaR and CVaR."""
    
    portfolio_return = mu @ weights
    portfolio_vol = np.sqrt(weights.T @ Sigma @ weights)
    
    # Parametric VaR (Delta-Normal)
    z_score = norm.ppf(1 - confidence_level)
    var_param = (portfolio_return - z_score * portfolio_vol) * portfolio_value
    print(f"\n[Parametric VaR at {int(confidence_level*100)}%] : {abs(var_param):,.2f} USD (potential loss)")

    # Monte Carlo VaR and CVaR
    sim_returns = np.random.multivariate_normal(mu, Sigma, MC_SIMULATIONS)
    portfolio_sim_returns = sim_returns @ weights
    portfolio_sim_values = portfolio_value * (1 + portfolio_sim_returns)
    
    var_mc = np.percentile(portfolio_sim_values - portfolio_value, (1 - confidence_level) * 100)
    cvar_mc = np.mean(portfolio_sim_values[portfolio_sim_values - portfolio_value <= var_mc]) - portfolio_value

    print(f"[Monte Carlo VaR at {int(confidence_level*100)}%] : {abs(var_mc):,.2f} USD")
    print(f"[Monte Carlo CVaR at {int(confidence_level*100)}%] : {abs(cvar_mc):,.2f} USD (expected shortfall)")
    
    return var_param, var_mc, cvar_mc

######################################################################
# 9. Visualizations
######################################################################

def plot_efficient_frontier(mu_annual, Sigma_annual, optimal_weights, risk_free_rate):
    """
    Plots the Markowitz bullet, the efficient frontier, and the optimal portfolio.
    """
    n = len(mu_annual)
    
    # 1. Generate random portfolios for the feasible region
    random_weights = np.random.rand(3000, n)
    random_weights = (random_weights.T / random_weights.sum(axis=1)).T
    random_returns = random_weights @ mu_annual
    random_risks = np.array([np.sqrt(w @ Sigma_annual @ w) for w in random_weights])

    # 2. Trace the efficient frontier
    frontier_returns = np.linspace(mu_annual.min(), mu_annual.max(), 100)
    frontier_risks = []
    
    for r in frontier_returns:
        w = cp.Variable(n)
        risk = cp.quad_form(w, Sigma_annual)
        constraints = [cp.sum(w) == 1, w >= 0, mu_annual @ w == r]
        prob = cp.Problem(cp.Minimize(risk), constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        if prob.status == 'optimal':
            frontier_risks.append(np.sqrt(prob.value))
    
    # 3. Optimal portfolio stats
    opt_return = optimal_weights @ mu_annual
    opt_risk = np.sqrt(optimal_weights @ Sigma_annual @ optimal_weights)
    
    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(random_risks, random_returns, c='lightgray', alpha=0.6, label='Random Portfolios')
    plt.plot(frontier_risks, frontier_returns, 'b--', label='Efficient Frontier')
    plt.scatter(opt_risk, opt_return, c='red', marker='*', s=200, zorder=5, label='Optimal Portfolio (Max Sharpe)')
    
    # Capital Market Line (CML)
    cml_x = np.linspace(0, opt_risk * 1.5, 100)
    cml_y = risk_free_rate + (opt_return - risk_free_rate) / opt_risk * cml_x
    plt.plot(cml_x, cml_y, 'orange', label='Capital Market Line')
    
    plt.title("Efficient Frontier and Optimal Portfolio")
    plt.xlabel("Annualized Risk (Standard Deviation)")
    plt.ylabel("Annualized Expected Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_weights(weights, tickers):
    """Plots the asset allocation of the optimal portfolio."""
    w_series = pd.Series(weights, index=tickers)
    w_series = w_series[w_series > 1e-4].sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    w_series.plot(kind='bar')
    plt.title("Optimal Portfolio Asset Allocation")
    plt.xlabel("Assets")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.show()

def plot_monte_carlo_simulation(mu_annual, Sigma_annual, weights):
    """Simulates and plots the future value distribution of the portfolio."""
    
    sim_daily_returns = np.random.multivariate_normal(mu_annual / 252, Sigma_annual / 252, (MC_SIMULATIONS, MC_HORIZON_DAYS))
    portfolio_daily_returns = sim_daily_returns @ weights
    
    # Calculate cumulative growth paths
    paths = 1_000_000 * np.cumprod(1 + portfolio_daily_returns, axis=1)
    final_values = paths[:, -1]

    mean_val = np.mean(final_values)
    p5 = np.percentile(final_values, 5)
    p95 = np.percentile(final_values, 95)

    print(f"\n--- Monte Carlo Simulation ({MC_HORIZON_DAYS} days) ---")
    print(f"Initial Value: $1,000,000.00")
    print(f"Average Final Value: ${mean_val:,.2f}")
    print(f"5th Percentile: ${p5:,.2f}")
    print(f"95th Percentile: ${p95:,.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(final_values, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(p5, color='red', linestyle='--', label=f'5th Percentile (${p5:,.0f})')
    plt.axvline(p95, color='green', linestyle='--', label=f'95th Percentile (${p95:,.0f})')
    plt.title(f"Portfolio Value Distribution after {MC_HORIZON_DAYS} Days (MC Simulation)")
    plt.xlabel("Final Portfolio Value (USD)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
######################################################################
# 10. Main Execution Pipeline
######################################################################

def run_portfolio_optimization_pipeline():
    """Executes the entire portfolio optimization pipeline from start to finish."""
    try:
        logging.info("Starting the advanced portfolio optimization pipeline.")
        print(f"--- Pipeline started at {datetime.now()} ---")

        # 1. Data Loading and Cleaning
        prices_raw = download_data(tickers, START_DATE, END_DATE)
        macro_raw = download_macro_data(macro_tickers, START_DATE, END_DATE)
        if prices_raw is None or prices_raw.empty:
            raise ValueError("Failed to download market data.")
            
        prices = clean_data(prices_raw)
        returns = calculate_log_returns(prices)

        # 2. Feature Engineering
        tech_feats = create_technical_features(prices)
        full_df = merge_prices_and_features(prices, macro_raw, tech_feats)

        # 3. Parallel Forecasting
        print("Running forecasts for all assets in parallel...")
        logging.info("Running parallel forecasts for returns and volatility.")
        fc_1d_series, garch_vols_series = run_all_forecasts_in_parallel(returns, full_df, macro_tickers)
        print("Forecasts complete.")

        # 4. Covariance and Expected Returns
        mu_annual = fc_1d_series * 252
        Sigma_daily = calculate_covariance_matrix(returns, garch_vols_series)
        Sigma_annual = Sigma_daily * 252

        # Ensure alignment
        common_tickers = returns.columns.intersection(mu_annual.index).intersection(Sigma_annual.index)
        mu_annual = mu_annual.loc[common_tickers]
        Sigma_annual = Sigma_annual.loc[common_tickers, common_tickers]
        
        # 5. Portfolio Optimization
        print("Optimizing portfolio for maximum Sharpe ratio...")
        logging.info("Performing Markowitz optimization.")
        w_opt, max_sharpe = optimize_portfolio(mu_annual.values, Sigma_annual.values, common_tickers)
        if w_opt is None:
            raise RuntimeError("Optimization failed to find a solution.")
        print("Optimization complete.")

        # 6. Results and Analysis
        opt_return = w_opt @ mu_annual
        opt_risk = np.sqrt(w_opt @ Sigma_annual @ w_opt)
        
        print("\n--- Optimal Portfolio Results ---")
        print(f"Expected Annual Return: {opt_return:.2%}")
        print(f"Annualized Volatility (Risk): {opt_risk:.2%}")
        print(f"Maximum Sharpe Ratio: {max_sharpe:.2f}")
        
        w_series = pd.Series(w_opt, index=common_tickers)
        print("\nTop 10 Asset Allocations:")
        print(w_series[w_series > 0.001].sort_values(ascending=False).head(10))

        # 7. Risk Analysis
        calculate_var_cvar(w_opt, Sigma_annual.values, mu_annual.values)
        
        # 8. Visualizations
        risk_free_annual = RISK_FREE_RATE_DAILY * 252
        plot_efficient_frontier(mu_annual, Sigma_annual, w_opt, risk_free_annual)
        plot_weights(w_opt, common_tickers)
        plot_monte_carlo_simulation(mu_annual.values, Sigma_annual.values, w_opt)
        
        # 9. Save Results
        w_series.to_csv(f"optimal_weights_{datetime.now().strftime('%Y%m%d')}.csv")
        logging.info("Optimal weights saved to CSV.")

        print(f"\n--- Pipeline finished successfully at {datetime.now()} ---")
        logging.info("Pipeline finished successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred in the main pipeline: {e}")
        print(f"ERROR: A critical error occurred. Check 'portfolio_optimization.log' for details.")

######################################################################
# 11. Entry Point
######################################################################

if __name__ == "__main__":
    # Run the optimization pipeline once
    run_portfolio_optimization_pipeline()
