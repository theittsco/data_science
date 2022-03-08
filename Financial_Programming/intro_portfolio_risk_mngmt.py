'''
Intro to portfolio risk management in python

1. Univariate investment risk
2. Portfolio Investing
3. Factor Investing
4. Forecasting and Reducing Risk

Risk: Measure of uncertainty of future returns- variance of returns
Returns:
    -Discrete returns
    -Log returns
'''
# Import pandas as pd
import pandas as pd

# Read in the csv file and parse dates
StockPrices = pd.read_csv(fpath_csv, parse_dates=['Date'])

# Ensure the prices are sorted by Date
StockPrices = StockPrices.sort_values(by='Date')

# Print only the first five rows of StockPrices
print(StockPrices.head())

# Calculate the daily returns of the adjusted close price
StockPrices['Returns'] = StockPrices['Adjusted'].pct_change()

# Check the first five rows of StockPrices
print(StockPrices.head())

# Plot the returns column over time
StockPrices['Returns'].plot()
plt.show()

# Convert the decimal returns into percentage returns
percent_return = StockPrices['Returns']*100

# Drop the missing values
returns_plot = percent_return.dropna()

# Plot the returns histogram
plt.hist(returns_plot, bins=75)
plt.show()

###############################################################################
'''
Mean, variance, normal distributions

Moments of Distributions
1. Mean - Average of data
2. Variance - Width of data
3. Skewness - Tilt of the data
4. Kurtosis - Thickness of the tails of data

Standard normals have mean 0, variance 1, skewness near 0, kurtosis near 3
Financial returns are not typically normally distributed; positive skewness,
kurtosis > 3

- # trading days in a year is 252
- Standard deviation is the volatility, dispersion of returns, scales with the square root
of time.

var_annual = var_daily * np.sqrt(252)
var_monthly = var_daily * np.sqrt(21)
'''
# Import numpy as np
import numpy as np

# Calculate the average daily return of the stock
mean_return_daily = np.mean(StockPrices['Returns'])
print(mean_return_daily)

# Calculate the implied annualized average return
mean_return_annualized = ((1+mean_return_daily)**252)-1
print(mean_return_annualized)

# Calculate the standard deviation of daily return of the stock
sigma_daily = np.std(StockPrices['Returns'])
print(sigma_daily)

# Calculate the daily variance
variance_daily = sigma_daily**2
print(variance_daily)

# Annualize the standard deviation
sigma_annualized = sigma_daily*np.sqrt(252)
print(sigma_annualized)

# Calculate the annualized variance
variance_annualized = sigma_annualized**2
print(variance_annualized)

'''
Daily volatility and mean give a good indication of daily risk and return

Skewness, Kurtosis, and Scaled volatility

Skewness: How much a distribution leans to the left or right
-In finance you want POSITIVE skewness

Kurtosis: Thickness of tails of a distribution, use as a proxy for probability
of outliers.
-In finance, returns are leptokurtic; when a distribution has a positive excess
kurtosis > 3.

Excess Kurtosis: Kurtosis - 3

kurtosis() in scipy.stats calculates the excess kurtosis, BE AWARE
-High excess kurtosis indicates high risk.

Testing for data normality: Shapiro-Wilk test

scipy.stats.shapiro(), null hypothesis is the data is normal, p-value criterion
is p-value < 0.05
'''# Import skew from scipy.stats
from scipy.stats import skew

# Drop the missing values
clean_returns = StockPrices.Returns.dropna()

# Calculate the third moment (skewness) of the returns distribution
returns_skewness = skew(clean_returns)
print(returns_skewness)

# Import kurtosis from scipy.stats
from scipy.stats import kurtosis

# Calculate the excess kurtosis of the returns distribution
excess_kurtosis = kurtosis(clean_returns)
print(excess_kurtosis)

# Derive the true fourth moment of the returns distribution
fourth_moment = excess_kurtosis + 3
print(fourth_moment)

# Import shapiro from scipy.stats
from scipy.stats import shapiro

# Run the Shapiro-Wilk test on the stock returns
shapiro_results = shapiro(clean_returns)
print("Shapiro results:", shapiro_results)

# Extract the p-value from the shapiro_results
p_value = shapiro_results[1]
print("P-value: ", p_value)

###############################################################################
###############################################################################
'''
Portfolio Composition and Backtesting

Portfolio return, R_p:
Return for asset n, R_(a_n):
Weight for asset n, w_(a_n):

Weights being the percentage of your portfolio an asset is

R_p = R_(a_1) + ... + R_(a_n) (assuming linear returns)

Returns are noisy, plot the cumulative returns instead

Market Capitalization: Value of a company's publicly traded shares
'''
# Finish defining the portfolio weights as a numpy array
portfolio_weights = np.array([0.12, 0.15, 0.08, 0.05, 0.09, 0.10, 0.11, 0.14, 0.16])

# Calculate the weighted stock returns
WeightedReturns = StockReturns.mul(portfolio_weights, axis=1)

# Calculate the portfolio returns
StockReturns['Portfolio'] = WeightedReturns.sum(axis=1)

# Plot the cumulative portfolio returns over time
CumulativeReturns = ((1+StockReturns["Portfolio"]).cumprod()-1)
CumulativeReturns.plot()
plt.show()

###############################################################################
'''
Market cap weighted portfolios

The S&P 500 is market cap weighted!
'''
# Create an array of market capitalizations (in billions)
market_capitalizations = np.array([601.51, 469.25, 349.5, 310.48, 299.77, 356.94, 268.88, 331.57, 246.09])

# Calculate the market cap weights
mcap_weights = market_capitalizations/market_capitalizations.sum()

# Calculate the market cap weighted portfolio returns
StockReturns['Portfolio_MCap'] = StockReturns.iloc[:, 0:9].mul(mcap_weights, axis=1).sum(axis=1)
cumulative_returns_plot(['Portfolio', 'Portfolio_EW', 'Portfolio_MCap'])

###############################################################################
'''
Correlation and covariance

Pearson correlation coefficient
In finance, the covariance matrix is used for optimization and risk management
purposes.
'''
# Calculate the correlation matrix
correlation_matrix = StockReturns.corr()

# Print the correlation matrix
print(correlation_matrix)

# Import seaborn as sns
import seaborn as sns

# Create a heatmap
sns.heatmap(correlation_matrix,
            annot=True,
            cmap="YlGnBu",
            linewidths=0.3,
            annot_kws={"size": 8})

# Plot aesthetics
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Calculate the covariance matrix
cov_mat = StockReturns.cov()

# Annualize the co-variance matrix
cov_mat_annual = cov_mat * 252

# Print the annualized co-variance matrix
print(cov_mat_annual)

# Import numpy as np
import numpy as np

# Calculate the portfolio standard deviation
portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
print(portfolio_volatility)

###############################################################################
'''
Markowitz portfolios

Risk adjusted returns

Sharpe ratio: Measures how return an investor can expect for each incremental
unit of risk.

Used to compare different portfolios with different amounts of risk

Any point on the efficient frontier is an optimum portfolio

Markowitz portfolios
    MSR: Max Sharpe ratio -> max((Asset_return - risk_free_return_rate)/Asset_volatility)
    GMV: Global minimum volatility

Higher return than Markowitz portfolios means higher risk
Past performance is not a guarantee of future returns, Sharpe ratios can change
quite dramatically over time.

MSR can be inconsistent, GMV tends to be more reliable.
'''
# Risk free rate
risk_free = 0

# Calculate the Sharpe Ratio for each asset
RandomPortfolios['Sharpe'] = ( RandomPortfolios.Returns - risk_free)/RandomPortfolios.Volatility

# Print the range of Sharpe ratios
print(RandomPortfolios['Sharpe'].describe()[['min', 'max']])

# Sort the portfolios by Sharpe ratio
sorted_portfolios = RandomPortfolios.sort_values(by=['Sharpe'], ascending=False)

# Extract the corresponding weights
MSR_weights = sorted_portfolios.iloc[0, 0:numstocks]

# Cast the MSR weights as a numpy array
MSR_weights_array = np.array(MSR_weights)

# Calculate the MSR portfolio returns
StockReturns['Portfolio_MSR'] = StockReturns.iloc[:, 0:numstocks].mul(MSR_weights_array, axis=1).sum(axis=1)

# Plot the cumulative returns
cumulative_returns_plot(['Portfolio_EW', 'Portfolio_MCap', 'Portfolio_MSR'])

# Sort the portfolios by volatility
sorted_portfolios = RandomPortfolios.sort_values(by=['Volatility'], ascending=True)

# Extract the corresponding weights
GMV_weights = sorted_portfolios.iloc[0, 0:numstocks]

# Cast the GMV weights as a numpy array
GMV_weights_array = np.array(GMV_weights)

# Calculate the GMV portfolio returns
StockReturns['Portfolio_GMV'] = StockReturns.iloc[:, 0:numstocks].mul(GMV_weights_array, axis=1).sum(axis=1)

# Plot the cumulative returns
cumulative_returns_plot(['Portfolio_EW', 'Portfolio_MCap', 'Portfolio_MSR', 'Portfolio_GMV'])

###############################################################################
###############################################################################
'''
Factor Investing

Capital Asset Pricing Model (CAPM):

Excess return: Return - Risk free return

CAPM: E(R_P) - RF = beta_P(E(R_M) - RF)
    E(R_P) - RF: The excess expected return of a stock or portfolio P
    E(R_M) - RF: The excess expected return of a broad market portfolio B
    RF: Regional risk-free rate
    beta_P: Exposure to the broad market portfolio B, beta_P = cov(R_P,R_B)/var(R_B)

'''
# Calculate excess portfolio returns
FamaFrenchData['Portfolio_Excess'] = FamaFrenchData.Portfolio - FamaFrenchData.RF

# Plot returns vs excess returns
CumulativeReturns = ((1+FamaFrenchData[['Portfolio','Portfolio_Excess']]).cumprod()-1)
CumulativeReturns.plot()
plt.show()

# Calculate the co-variance matrix between Portfolio_Excess and Market_Excess
covariance_matrix = FamaFrenchData[['Portfolio_Excess', 'Market_Excess']].cov()

# Extract the co-variance co-efficient
covariance_coefficient = covariance_matrix.iloc[0, 1]
print(covariance_coefficient)

# Calculate the benchmark variance
benchmark_variance = FamaFrenchData['Market_Excess'].var()
print(benchmark_variance)

# Calculating the portfolio market beta
portfolio_beta = covariance_coefficient/benchmark_variance
print(portfolio_beta)

'''
Your portfolio beta is 0.9738. You can think of market beta as a measure of your
exposure to the broad stock market. For every 1.0% rise (or fall) in the market,
you can expect your portfolio to rise (fall) roughly 0.97%.
'''
# Import statsmodels.formula.api
import statsmodels.formula.api as smf

# Define the regression formula
CAPM_model = smf.ols(formula='Portfolio_Excess ~ Market_Excess', data=FamaFrenchData)

# Print adjusted r-squared of the fitted regression
CAPM_fit = CAPM_model.fit()
print(CAPM_fit.rsquared_adj)

# Extract the beta
regression_beta = CAPM_fit.params['Market_Excess']
print(regression_beta)
'''
A high adjusted r-squared (close to 1) means that the majority of your
portfolio's movements can be explained by the factors in your model.
'''

###############################################################################
'''
Alpha and multi-factor models

Fama-French 3 Factor Model: R_P = RF + beta_M(R_M-RF) + b_SMB*SMB + b_HML*HML + alpha

SMB: Small minus big factor
b_SMB: Exposure to the SMB factor
HML: High minus low factor
b_HML: Exposure to the HML factor
alpha: Performance unexplained by other factors
beta_M: Beta to the broad market portfolio B

Fama outperforms CAPM typically, says investors will be rewarded more by value
stocks rather than growth stocks.

pvalues < 0.05 are considered statistically significant.

SMB < 0: When small stocks rise, portfolio falls
HML > 0: When momentum stocks rise, portfolio rises

alpha: Error term, positive alpha -> outperformance skill, luck, timing
Weighted sum of all alpha must be zero
'''
# Import statsmodels.formula.api
import statsmodels.formula.api as smf

# Define the regression formula
FamaFrench_model = smf.ols(formula='Portfolio_Excess ~ Market_Excess + SMB + HML', data=FamaFrenchData)

# Fit the regression
FamaFrench_fit = FamaFrench_model.fit()

# Extract the adjusted r-squared
regression_adj_rsq = FamaFrench_fit.rsquared_adj
print(regression_adj_rsq)

# Extract the p-value of the SMB factor
smb_pval = FamaFrench_fit.pvalues['SMB']

# If the p-value is significant, print significant
if smb_pval < 0.05:
    significant_msg = 'significant'
else:
    significant_msg = 'not significant'

# Print the SMB coefficient
smb_coeff = FamaFrench_fit.params['SMB']
print("The SMB coefficient is ", smb_coeff, " and is ", significant_msg)

'''
Your portfolio has a statistically significant negative exposure (-0.2621) to
small-cap stocks - in other words - positive exposure to large caps!

Finance is all about risk and return. Higher risk tends to lead to higher
returns over time, and lower risk portfolios tend to lead to lower returns over
time.

In the Fama-French factor model:

The HML factor is constructed by calculating the return of growth stocks, or
stocks with high valuations, versus the return of value stocks.
The SMB factor is constructed by calculating the return of small-cap stocks, or
stocks with small market capitalizations, versus the return of large-cap stocks.

Small cap value stocks have the highest return and risk.
'''
# Calculate your portfolio alpha
portfolio_alpha = FamaFrench_fit.params['Intercept']
print(portfolio_alpha)

# Annualize your portfolio alpha
portfolio_alpha_annualized = ((1 + portfolio_alpha)**252) - 1
print(portfolio_alpha_annualized)

'''
The alpha left over by the regression is unexplained performance due to
unknown factors. In a regression model, this is simply the coefficient of the
intercept.

There are two general schools of thought as to why:

The model simply needs to be expanded. When you have found all of the missing
economic factors, you can explain all stock and portfolio returns. This is known
as the Efficient Market Hypothesis.
There is a degree of unexplainable performance that no model will ever capture
reliably. Perhaps it is due to skill, timing, intuition or luck, but investors
should seek to maximize their alpha.
'''

###############################################################################
'''
Expanding the 3-factor model

Fama-French 5 factor model, adds two more parameters, RMW and CMA

RMW: Profitability
CMA: Investment

The RMW factor represents the returns of companies with high operating
profitability versus those with low operating profitability, and the CMA factor
represents the returns of companies with aggressive investments versus those who
are more conservative.
'''
# Import statsmodels.formula.api
import statsmodels.formula.api as smf

# Define the regression formula
FamaFrench5_model = smf.ols(formula='Portfolio_Excess ~ Market_Excess + SMB + HML + RMW + CMA ', data=FamaFrenchData)

# Fit the regression
FamaFrench5_fit = FamaFrench5_model.fit()

# Extract the adjusted r-squared
regression_adj_rsq = FamaFrench5_fit.rsquared_adj
print(regression_adj_rsq)

###############################################################################
###############################################################################
'''
Value at Risk

Estimating tail risk: Risk of extreme outcomes, notably on the negative (tail)
side of a distribution.

Historical drawdowns: Percentage loss from the highest cumulative historical point.

Values at risk: VaR, a threshold with a given confidence level that losses will
not historically exceed a certain level. Commonly quoted as 95, 99, 99.9

Conditional value at risk: CVaR, estimate of expected losses sustained in the
worst 1-x% of scenarios. Also quoted as percentiles. (Expected Shortfall)

CVaR is always worse than the VaR of the same quantile.

Monte Carlos
'''
# Calculate the running maximum
running_max = np.maximum.accumulate(cum_rets)

# Ensure the value never drops below 1
running_max[running_max < 1] = 1

# Calculate the percentage drawdown
drawdown = (cum_rets)/running_max - 1

# Plot the results
drawdown.plot()
plt.show()

###############################################################################
# Calculate historical VaR(95)
var_95 = np.percentile(StockReturns_perc, 100 - 95)
print(var_95)

# Sort the returns for plotting
sorted_rets = StockReturns_perc.sort_values()

# Plot the probability of each sorted return quantile
plt.hist(sorted_rets, normed=True)

# Denote the VaR 95 quantile
plt.axvline(x=var_95, color='r', linestyle='-', label="VaR 95: {0:.2f}%".format(var_95))
plt.show()

###############################################################################
# Historical CVaR 95
cvar_95 = StockReturns_perc[StockReturns_perc <= var_95].mean()
print(cvar_95)

# Sort the returns for plotting
sorted_rets = sorted(StockReturns_perc)

# Plot the probability of each return quantile
plt.hist(sorted_rets, normed=True)

# Denote the VaR 95 and CVaR 95 quantiles
plt.axvline(x=var_95, color="r", linestyle="-", label='VaR 95: {0:.2f}%'.format(var_95))
plt.axvline(x=cvar_95, color='b', linestyle='-', label='CVaR 95: {0:.2f}%'.format(cvar_95))
plt.show()

###############################################################################
'''
Var extensions

Analysts typically uses 99 or 99.9 percentile. Don't go too high for risk
estimation. You could overestimate the risk and miss out on potential returns.

Empirical assumptions: Values that have actually occurred. How do you simulate
the probability of a value that has never occurred? Sample from probability
distribution.

Remember to covert from one day forecast to multi day forecast multiply by
the square root of time.
'''
# Historical VaR(90) quantiles
var_90 = np.percentile(StockReturns_perc, 100 - 90)
print(var_90)

# Historical CVaR(90) quantiles
cvar_90 = StockReturns_perc[StockReturns_perc <= var_90].mean()
print(cvar_90)

# Plot to compare
plot_hist()

'''
Value at Risk can also be computed parametrically using a method known as
variance/co-variance VaR. This method allows you to simulate a range of
possibilities based on historical return distribution properties rather than
actual return values.
'''
# Import norm from scipy.stats
from scipy.stats import norm

# Estimate the average daily return
mu = np.mean(StockReturns)

# Estimate the daily volatility
vol = np.std(StockReturns)

# Set the VaR confidence level
confidence_level = 0.05

# Calculate Parametric VaR
var_95 = norm.ppf(confidence_level, mu, vol)
print('Mean: ', str(mu), '\nVolatility: ', str(vol), '\nVaR(95): ', str(var_95))

# Aggregate forecasted VaR
forecasted_values = np.empty([100, 2])

# Loop through each forecast period
for i in range(0,100):
    # Save the time horizon i
    forecasted_values[i, 0] = i
    # Save the forecasted VaR 95
    forecasted_values[i, 1] = var_95 * np.sqrt(i+1)

# Plot the results
plot_var_scale()

###############################################################################
'''
Random Walks

Random == Stochastic
'''
# Set the simulation parameters
mu = np.mean(StockReturns)
vol = np.std(StockReturns)
T = 252
S0 = 10

# Add one to the random returns
rand_rets = np.random.normal(mu,vol, T) + 1

# Forecasted random walk
forecasted_values = rand_rets.cumprod() * S0

# Plot the random walk
plt.plot(range(0, T), forecasted_values)
plt.show()

'''
Monte-Carlo simulations are used to model a wide range of possibilities.

Monte-Carlos can be constructed in many different ways, but all of them involve
generating a large number of random variants of a given model, allowing a wide
distribution of possible paths to be analyzed. This can allow you to build a
comprehensive forecast of possibilities to sample from without a large amount
of historical data.

Generate 100 Monte-Carlo simulations for the USO oil ETF.
'''
# Loop through 100 simulations
for i in range(0,100):

    # Generate the random returns
    rand_rets = np.random.normal(mu, vol, T) + 1

    # Create the Monte carlo path
    forecasted_values = S0*(rand_rets).cumprod()

    # Plot the Monte Carlo path
    plt.plot(range(T), forecasted_values)

# Show the simulations
plt.show()


'''
Both the return values and the Monte-Carlo paths can be used for analysis of
everything ranging from option pricing models and hedging to portfolio
optimization and trading strategies.

Aggregate the returns data at each iteration, and use the resulting values to
forecast parametric VaR(99).
'''
# Aggregate the returns
sim_returns = []

# Loop through 100 simulations
for i in range(100):

    # Generate the Random Walk
    rand_rets = np.random.normal(mu, vol, T)

    # Save the results
    sim_returns.append(rand_rets)

# Calculate the VaR(99)
var_99 = np.percentile(sim_returns, 100 - 99)
print("Parametric VaR(99): ", round(100*var_99, 2),"%")
