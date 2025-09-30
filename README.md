# Quantitative Portfolio Optimization Pipeline

This repository contains a comprehensive Python pipeline for advanced financial portfolio optimization. The project leverages a hybrid forecasting model (ARIMA + LSTM), GARCH for volatility, parallel processing for efficiency, and advanced risk analysis techniques like Monte Carlo simulations, VaR, and CVaR.

## My Personal Journey with this Project

This project was born from a desire to bridge academic theory with the practical, high-impact skills demanded by the competitive world of quantitative finance. As part of my "Analytic Edge" module, I recognized the need to go beyond standard models. I undertook a self-directed learning journey, completing an intensive quantitative finance course from **IMT on YouTube**, which provided a deep dive into advanced modeling techniques.

This pipeline is the direct application of that knowledge: a tool that is not just academically sound but also robust, efficient, and built with real-world performance in mind.

## 🚀 Key Features

-   **Hybrid Forecasting Model (ARIMA + LSTM):** Combines the linear modeling strength of ARIMA with the ability of LSTMs to capture non-linear patterns in financial data, using macroeconomic factors and technical indicators as features.
-   **Dynamic Volatility with GARCH:** Moves beyond simple historical volatility by forecasting future volatility for each asset using GARCH models, leading to a more forward-looking covariance matrix.
-   **Efficient Parallel Processing:** Leverages `concurrent.futures` to run forecasting tasks for all tickers in parallel, drastically reducing computation time.
-   **Markowitz Optimization with Constraints:** Uses `cvxpy` to find the optimal portfolio on the efficient frontier, incorporating practical constraints like maximum and minimum asset weights.
-   **Advanced Risk Analysis:** Calculates parametric and Monte Carlo-based Value at Risk (VaR) and Conditional Value at Risk (CVaR) to provide a comprehensive view of portfolio risk.
-   **Automated Pipeline & Scheduling:** The entire process is wrapped in a main function, with an optional `APScheduler` to run the analysis automatically (e.g., on the first of every month).

## 🛠️ Project Pipeline

The script follows a structured, end-to-end pipeline:

1.  **Data Acquisition:** Downloads historical price data for a large universe of stocks and key macroeconomic indicators (`^GSPC`, `^VIX`, `^TNX`) from Yahoo Finance.
2.  **Data Cleaning & Feature Engineering:** Imputes missing values using KNNImputer and engineers a rich feature set including technical indicators (SMA, EMA, RSI) for each asset.
3.  **Parallel Forecasting Engine:**
    -   For each asset, a hybrid **ARIMA + LSTM** model forecasts the next day's expected return.
    -   Simultaneously, a **GARCH** model is fitted to forecast the next day's volatility.
    -   These tasks are executed in parallel for maximum speed.
4.  **Covariance Matrix Construction:** Constructs a forward-looking covariance matrix by combining the historical correlation matrix with the *forecasted* daily volatilities from the GARCH models.
5.  **Portfolio Optimization:** Solves the Markowitz optimization problem to find the portfolio with the maximum Sharpe ratio, given the forecasted returns and the constructed covariance matrix.
6.  **Risk Assessment & Simulation:**
    -   Performs a **Monte Carlo simulation** over a one-year horizon to project the distribution of the portfolio's final value.
    -   Calculates daily **VaR and CVaR** to quantify potential losses.
7.  **Visualization & Reporting:** Generates key visualizations and saves the optimal portfolio weights to a CSV file.

## 💻 Technologies & Libraries

-   **Core Libraries:** Pandas, NumPy, Scikit-learn
-   **Financial Data:** `yfinance`
-   **Time Series Modeling:** `statsmodels` (ARIMA), `arch` (GARCH)
-   **Deep Learning:** `TensorFlow` / `Keras` (for LSTM)
-   **Optimization:** `cvxpy`
-   **Parallel Processing:** `concurrent.futures`
-   **Visualization:** Matplotlib, Seaborn
-   **Scheduling:** `APScheduler`

## 📊 Visualizations

### 1. The Efficient Frontier

The "Markowitz Bullet" visualizes the feasible region of portfolios. The script identifies the Global Minimum Variance (GMV) portfolio and the optimal portfolio (Maximum Sharpe Ratio), plotting the Capital Market Line (CML).
<img width="867" height="553" alt="Unknown-4" src="https://github.com/user-attachments/assets/22ab6d23-cfa1-46b1-9cd0-f110881d32f2" />


### 2. Optimal Portfolio Allocation

A bar chart displaying the weight distribution of the assets in the maximum Sharpe ratio portfolio. Only assets with non-zero weights are shown.
<img width="1189" height="590" alt="Repartition" src="https://github.com/user-attachments/assets/c59a1b9d-5f5d-4a7b-879a-123defa64e7f" />



### 3. Monte Carlo Simulation Results

This histogram shows the distribution of the portfolio's final value after a one-year simulation, highlighting the 5th and 95th percentiles to illustrate the range of potential outcomes.
<img width="1010" height="553" alt="Montecarlo" src="https://github.com/user-attachments/assets/471b3809-effb-4bbf-b60b-6d5e6ae23174" />

