# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 20:27:53 2022

@author: Scott
"""
import yfinance as yf
from datetime import datetime 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

start = datetime(2020, 1, 1)
end = datetime(2021, 7, 24)

#%%
symbols = np.array(['BTC-USD', 'ETH-USD', 'ADA-USD', 'LINK-USD', 'XLM-USD'])
weights = np.array([0.2,0.2,0.2,0.2,0.2])

df = pd.concat([yf.download(crypto[0], start=start, end=end) for crypto in symbols], axis=1)

#%%

dataframes = pd.DataFrame()

for i in range(4,df.shape[1], 6):
     dataframes[i] = df.iloc[:,i].pct_change()
    
dataframes.columns = symbols
dataframes = dataframes.iloc[1:,:]

#%%
dummy = np.ones((1000,5))
rw_df = pd.DataFrame(data=dummy, columns=symbols)

for i in range(0,5):
    for j in range(dataframes.shape[0]):
        new_weights = np.random.dirichlet(np.ones(len(symbols)), size=1)
        rw_df.iloc[j,i] = new_weights[0,i]
        
#%%

mu = dataframes.mean().values
cov = dataframes.cov().values

# Compute the expected return of the portfolio for each random set.
random_portfolio_df = rw_df.dot(mu).to_frame("R_P")
# Compute the variance of the portfolio for each random set.
random_portfolio_df["Var"] = np.sum(rw_df.dot(cov.dot(rw_df.T) ), axis=1)
random_portfolio_df["SD"] = np.sqrt(random_portfolio_df['Var'])

#%%

fig, ax = plt.subplots(figsize=(9, 6))

colors = ["red", "blue", "green", "m", "c"]

# Plot the different portfolios as a scatter plot.
random_portfolio_df.plot.scatter(x="SD", y="R_P", s=10, color="gray", ax=ax)

for (crypto, color) in zip(symbols, colors):
   ax.scatter(x=np.sqrt(dataframes.cov().loc[crypto, crypto]), y=dataframes.mean().loc[crypto],
               label=crypto, s=70, color=color, marker="o")

ax.set_xlabel("Monthly Standard Deviation")
ax.set_ylabel("Monthly Expected Return")
ax.legend()

#%%

# Function to find the minimum variance given a desired return.
def min_variance(mu, cov, desired_ret):

    # Compute the variance
    def variance(weights):
        return np.dot(weights, np.dot(cov, weights))

    # Check that the weights sum up to 1
    def check_sum(weights):
        return np.sum(weights) - 1

    # Check that the return of the portfolio is the desired return.
    def check_return(weights):
        return desired_ret - np.dot(mu.T,weights)

    # Write down the constraints as equality constraints.
    cons = ({"type": "eq", "fun": check_sum},
            {"type": "eq", "fun": check_return})

    # We are guessing a purposely wrong initial guess.
    init_guess = [-10 for _ in range(len(mu))]

    # Run the minimiziation
    results = minimize(variance, x0=init_guess, constraints=cons)
    

    w = results.x

    # Assert that the optimization converged. Otherwise, throw an error.
    assert results.success, "Optimization did not converge for ." + \
        str(desired_ret)

    # Return the desired return and variance of the appropriate portfolio.
    return [desired_ret, variance(w), w]

#%%

desired_rets = np.linspace(0.01/12, 0.18/12, 171)

frontier = [min_variance(mu, cov, desired_ret)
            for desired_ret in desired_rets]

frontier_df = pd.DataFrame(data=frontier, columns=["ER", "Var", "Weights"])
frontier_df['SD'] = np.sqrt(frontier_df.Var)

# 0.0213 is the last interest rate I checked in July 2022
frontier_df['SR'] = (frontier_df.ER - 0.0213) / frontier_df.SD

max_sharpe = frontier_df.loc[frontier_df["SR"] == np.max(frontier_df["SR"])]

print(max_sharpe.Weights)