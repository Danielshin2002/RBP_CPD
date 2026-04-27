This repository looks at how changes in **Relevance Based Prediction** methods compared to **Change Point Detection** methods for understanding regime change. 
For an intro into RBP, check out [this link](https://github.com/Danielshin2002/RBP_CPD/blob/main/Papers)


# Core Research Questions

Can RBP-derived signals detect regime shifts:
   - earlier than CPD?
   - with fewer false positives?
   - with better interpretability?

Altneratively, does combining RBP diagnostics with CPD improve detection performance?

# Defining Regime Change

A regime change is defined as a structural shift in:
- volatility level
- correlation structure
- return distribution

Operational definitions:
- exogenous (known crisis periods)
- endogenous (future volatility exceeding threshold)


# Data

The data frequency that we will use is monthly data.
Our target variables consiste of 

21-day (1 month) forward realized volatility

## Predictors
The predictors we will use are quite similar to the ones in the paper, with a few additions. 

<sub>Note : Derived means derived from existing variables</sub>
| Predictive Variable | Proxy | Source |
| --- | --- | --- |
| Market Volatility|
| Trailing 7d volatility | Std Dev of daily returns (7d) | openbb-yfinance|
| Trailing 21 volatility | Std Dev of daily returns (21d) | oppenbb-yfinance|
| Trailing 63 realized vol | Std dev of daily returns (63d) | openbb-yfinance |
| Vol ratio | 7d vol / 63d vol | derived from existing|
| Rolling Drawdown | % drop from rolling max | openbb-yfinance|
| Market Returns|
| 7d return | % return over 7 days | openbb-yfinance|
| 21d return | % return over 21 days | openbb-yfinance|
| 63d return | % return over 63 days | openbb-yfinance|
| Volatility Surface|
| Vix | Implied Volatility Index | openbb-cboe|
| IV - RV spread | Vix - 21d vol | derived|
| Rates / Curve |
| 1M Yield | 1-month Treasury Yield | openbb-fred (DGS1MO)|
| 10Y Yield | 10-year Treasury Yield | openbb-fred (DGS10) |
| Slope (10y-1M) | 10Y-1M Yield | Dervied|
| Credit spreads | BAA-10Y Treasury| openbb-fred (BAA10Y)|
| Macro |
| Growth | INdustrial Production YoY | openbb-fred(INDPRO)|
| Inflation | CPI YoY | openbb-fred (CPIAUCSL)|
| Payrolls | Nonfarm Payrolls YoY | openbb-fred(PAYEMS)|
| Money Supply | M2 YoY | openbb-fred(M2SL)|

# Methodology

For each prediction date : 
Construct training sample as an expanding window
Compute relevance for past observations
Apply observation censoring thresholds
Generate prediction weights
Compute prediction as weighted average
Compute reliability metrics (fit and ffit)
Aggregate across the grid