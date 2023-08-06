'''
Monte Carlo Simulations in Python

1. Intro to Monte Carlo Simulations
2. Foundations for Monte Carlo
3. Principled Monte Carlo Simulation
4. Model Checking and Results Interpretation

MC Simulation: Used to predict the probability of different outcomes impacted by the presence of random variable
- Rely on repeated random sampling to obtain numerical results
- Results are stochastic since the model relies on random inputs

'''
# The year 2022 saw record high inflation. There are many ways to think about what inflation in the future will be. In this exercise, you'll attempt to estimate future inflation using a deterministic model.

# Recall that deterministic models do not include any randomness and allow you to calculate the outcome of a future event exactly. In the next exercise, you'll use a Monte Carlo simulation, which is stochastic. Throughout these exercises, consider which simulation—deterministic or stochastic—is more appropriate for estimating inflation.

# For this example, assume the inflation rate was 8.6% in 2022 and that there is a steady increase of 2% for inflation in each year following. Using these assumptions, what will the inflation rate be in 2050?

def deterministic_inflation(year, yearly_increase_percent):
    inflation_rate = 8.6
    inflation_rate = inflation_rate*((100+yearly_increase_percent)/100)**(year-2022)
    return(inflation_rate)

# Print the deterministic simulation results
print(deterministic_inflation(2050,2))

# Congratulations! You've forecasted inflation using a deterministic simulation. You get exact and deterministic results from deterministic simulations. In the next exercise, you'll use a Monte Carlo simulation to take some randomness into consideration!!

##-------------------------------------------------------------------------------------------------------
# In the previous exercise, you modeled information deterministically. You'll now attempt to estimate future inflation with a stochastic model, using a Monte Carlo simulation.

# Recall that stochastic models simulate randomness in variables by using sampling. This randomness means that each simulation will likely arrive at a different expected outcome, even if the inputs are the same. We saw this in the video by running Monte Carlo simulations with different seeds.

# In this exercise, assume 8.6% inflation in 2022 and a stochastic increase of 1%, 2%, or 3% each year over the previous year (with equal probabilities of 1%, 2%, or 3%) for the following years. What will the inflation rate look like in 2050 under these assumptions?

def monte_carlo_inflation(year, seed):
    random.seed(seed)
    inflation_rate = 8.6
    yearly_increase = random.randint(1, 3)
    for i in range(year - 2022):
        inflation_rate = inflation_rate*((100 + yearly_increase)/100)
    return(inflation_rate)
  
# Simulate the inflation rate for the year 2050 with a seed of 1234
print(monte_carlo_inflation(2050,1234))

# Simulate the inflation rate for the year 2050 with a seed of 34228
print(monte_carlo_inflation(2050,34228))

# Congratulations on successfully running the simulation. You saw a difference in inflation of 5% by varying the seed, due to the stochastic nature of Monte Carlo simulations. What if you ran the inflation many times and took the average? In the next exercise, you'll see the Law of Large Numbers in action!

##-------------------------------------------------------------------------------------------------------
# You learned in the previous exercise that due to the stochastic nature of Monte Carlo simulations, each simulation result can be very different. In this exercise, you'll leverage the Law of Large Numbers to simulate inflation in 2050 based on the average of a large number of simulations.

# The monte_carlo_inflation() function you wrote in the previous exercise is available for use. As a reminder, this is the function code:

# Calculate the average of 1,000 simulation results with a seed between 0 and 20000
rates_1 = []
for i in range(1000):
    seed = random.randint(0, 20000)
    rates_1.append(monte_carlo_inflation(2050, seed))
print(np.mean(rates_1))

# Calculate the average of 10,000 simulation results with a seed between 0 and 20000
rates_2 = []
for i in range(10000):
    seed = random.randint(0, 20000)
    rates_2.append(monte_carlo_inflation(2050, seed))
print(np.mean(rates_2))

# Well done! With a large number of simulations, you can see the average inflation rate is now quite consistent. Contrary to what you saw in the last exercise, on average, the difference between the two simulations is now less than 1%!

#########################################################################################################
'''
Resampling 
'''
# Bootstrapping is great for calculating confidence intervals for means; you'll now practice doing just that!

# nba_weights contains the weights of a group of NBA players in kilograms:

# nba_weights = [96.7, 101.1, 97.9, 98.1, 98.1, 
#                100.3, 101.0, 98.0, 97.4]
# You are interested in calculating the 95% confidence interval of the mean weight of NBA players using this list.

simu_weights = []

# Sample nine values from nba_weights with replacement 1000 times
for i in range(1000):
    bootstrap_sample = random.choices(nba_weights,k=9)
    simu_weights.append(np.mean(bootstrap_sample))

# Calculate the mean and 95% confidence interval of the mean for your results
mean_weight = np.mean(simu_weights)
upper = np.quantile(simu_weights,0.975)
lower = np.quantile(simu_weights,0.025)
print(mean_weight, lower, upper)

##-------------------------------------------------------------------------------------------------------
# Now you'll visualize the results of your simulation from the previous exercise! You'll continue working with nba_weights, which contains the weights of a group of NBA players in kilograms:

# Plot the distribution of the simulated weights
sns.displot(simu_weights)

# Plot vertical lines for the 95% confidence intervals and mean
plt.axvline(lower, color="red")
plt.axvline(upper, color="red")
plt.axvline(mean_weight, color="green")
plt.show()

##-------------------------------------------------------------------------------------------------------
# Are NBA players heavier than US adult males? You are now interested in calculating the 95% confidence interval of the mean difference (in kilograms) between NBA players and US adult males. You'll use the two lists provided.

# Permutation is great when testing for difference, so that's the resampling method you'll use here!

# Define all_weights
all_weights = nba_weights + us_adult_weights
simu_diff = []

for i in range(1000):
	# Perform the permutation on all_weights
    perm_sample = np.random.permutation(all_weights)
    # Assign the permutated samples to perm_nba and perm_adult
    perm_nba, perm_adult = perm_sample[0:13], perm_sample[13:]
    perm_diff = np.mean(perm_nba) - np.mean(perm_adult)
    simu_diff.append(perm_diff)
mean_diff = np.mean(nba_weights) - np.mean(us_adult_weights) 
upper = np.quantile(simu_diff, 0.975)
lower = np.quantile(simu_diff, 0.025)
print(mean_diff, lower, upper)

# Permutating perfection! You can see that the mean difference lies outside the 95% confidence interval, suggesting that given these two lists of samples, the NBA players' mean weight is significantly different from the average US adult males'!

#########################################################################################################
'''
Leveraging Monte Carlo Simulations

Wide applicability
- Finance and business
    - Stock price estimation
    - Risk management
- Engineering
    - Reliability analysis
- Physical sciences
    - Binding site ID (Biology)

Benefits of MC sims
- Take into consideration a rnage of values for various inputs
- Show not only what could happen, but how likely each outcome is
- Make it easy to visualize the range of possible outcomes
- Can examine what would have happened under different circumstances

Limitations of MC sims
- Model output is only as good as its input
- Probability of extreme is often underestimated
'''

# Similar to the example in the lesson, you will roll two dice from two bags, and each bag contains three biased dice.

# bag1 = [[1, 2, 3, 6, 6, 6], [1, 2, 3, 4, 4, 6], [1, 2, 3, 3, 3, 5]]
# bag2 = [[2, 2, 3, 4, 5, 6], [3, 3, 3, 4, 4, 5], [1, 1, 2, 4, 5, 5]]
# The difference is that the dice in the two bags are paired: if you pick the second die in bag1, you will also pick the second die in bag2. In each trial:

# You pick one pair of dice from the two bags randomly and roll them
# Success occurs if the points on dice1 and dice2 add up to eight; otherwise, failure
# Your task is to complete the for-loop in the roll_paired_biased_dice() function and to use this function to calculate the probabilities of success for each unique combination of points on dice1 and dice2.

def roll_paired_biased_dice(n, seed=1231):
    results = {}
    random.seed(seed)
    for i in range(n):
        bag_index = random.randint(0, 1)
        dice_index1 = random.randint(0, 5)
        dice_index2 = random.randint(0, 5)
        point1 = bag1[bag_index][dice_index1]
        point2 = bag2[bag_index][dice_index2]
        key = "%s_%s" % (point1, point2)
        if point1 + point2 == 8: 
            if key not in results:
                results[key] = 1
            else:
                results[key] += 1
    return(pd.DataFrame.from_dict({"dice1_dice2":results.keys(),
		"probability_of_success":np.array(list(results.values()))*100.0/n}))

# Run the simulation 10,000 times and assign the result to df_results
df_results = roll_paired_biased_dice(10000,1231)
sns.barplot(x="dice1_dice2", y="probability_of_success", data=df_results)
plt.show()

# Great work! This is an example of sampling from correlated inputs in simple distributions. What about more complicated probability distributions? Can we sample from them? What if we are interested in more than one variable? Great questions—we are going to learn how to do these things in the next chapter!

#########################################################################################################
#########################################################################################################
'''
Foundations for Monte Carlo

Steps for MC Sim:
1. Define input variables and pick probability distributions for them
2. Generate inputs by sampling from these distributions
3. Perform deterministic calculation of simulated inputs
4. Summarize results
'''

# Perfect! Although the sampling steps in Monte Carlo simulations are stochastic, there are deterministic calculations in the overall Monte Carlo simulation, such as the example of determining whether each point is located in the circle or not.

##-------------------------------------------------------------------------------------------------------
# In this exercise and the next, you'll play around with the pi calculations from the video to further understand the importance of each step in the simulation process.

# Recall that the simulation to find pi generates random points (x,y) where x and y are between -1 and 1, as shown in the graph below.

# What if you incorrectly changed the deterministic calculation where you check whether a point should be added to circle_points? How will this affect the final result? You'll see from the wacky value you get for pi that correctly specifying deterministic calculations is essential for Monte Carlo simulations!

n = 10000
circle_points = 0 
square_points = 0 
for i in range(n):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    dist_from_origin = x**2 + y**2
    # Increment circle_points for any point with a distance from origin of less than .75
    if dist_from_origin < 0.75:
        circle_points += 1
    square_points += 1
pi = 4 * circle_points / square_points
print(pi)

# Correct! Because the determination of circle points is wrongfully defined as the ones with a distance from the origin less than 0.75 rather than 1, the estimated pi value is far below the approximate value of 3.14 because many points that should be counted as circle points are not!

##-------------------------------------------------------------------------------------------------------
# What happens if you change the input probability distribution from the continuous uniform distribution(random.uniform()) to the discrete uniform distribution(random.randint())? Your results will not be reliable, because random.randint() will sample discrete integers, while random.uniform() samples continuous float numbers.

# Pay attention to the estimated pi value that this simulation generates. Because the incorrect probability distribution has been selected, it will not be very accurate! Choosing the correct probability distributions is essential for Monte Carlo simulations, and we will go into more detail on different distributions in later lessons so that you feel confident you are choosing the correct one going forward.

n = 10000
circle_points = 0 
square_points = 0 
for i in range(n):
    # Sample the x and y coordinates from -1 to 1 using random.randint()
    x = random.randint(-1,1)
    y = random.randint(-1,1)
    dist_from_origin = x**2 + y**2
    if dist_from_origin <= 1:
        circle_points += 1
    square_points += 1
pi = 4 * circle_points / square_points
print(pi)

# You've followed the instructions correctly to generate ...a bad simulation! random.randint()is a discrete uniform distribution. When it is wrongfully picked as the input distribution, only the integers -1, 0, or 1 are picked. This illustrates that random.uniform(), the continuous uniform distribution, is more appropriate for sampling the x-y coordinates __continuously__. In the following lessons, we will go over some commonly used distributions to get a better understanding of them!

#########################################################################################################
'''
Generating Discrete Random Variables

Geometric Distribution: The probability distribution of the number of trials X, needed to get one success, given the success probability p.
'''
# Tom has a regular six-sided die showing the numbers one through six. In this exercise, you'll use the discrete uniform distribution, which is perfectly suited for sampling integer values with uniform distributions, to simulate rolling Tom's die 1,000 times. You'll then visualize the results!

# The following have been imported for you: seaborn as sns, scipy.stats as st and matplotlib.pyplot as plt.

# Define low and high for use in rvs sampling below
low = 1
high = 7
# Sample 1,000 times from the discrete uniform distribution
samples = st.randint.rvs(low,high,size=1000)

samples_dict = {'nums':samples}
sns.histplot(x='nums', data=samples_dict, bins=6, binwidth=0.3)
plt.show()

# Well done—your histplot makes it easy to see that if Tom rolls his die many times, he gets roughly uniformly distributed results between one and six!

##-------------------------------------------------------------------------------------------------------
# Eva has a biased coin that has a probability of turning heads only 20% of the time. Eva flips her coin and records the number of flips needed to get a result of heads.

# The geometric distribution is perfectly suited to model the number of flips needed to reach a result of heads, with the success rate p defined as the probability of turning heads each time.

# Your task is to use the geometric distribution to simulate Eva's coin flips to reach heads 10,000 times, recording the number of flips needed to reach heads each time. Then, you'll visualize the results!

# Set p to the appropriate probability of success
p = 0.2

# Sample from the geometric distribution 10,000 times
samples = st.geom.rvs(p,size=10000)
samples_dict = {"nums":samples}
sns.histplot(x="nums", data=samples_dict)  
plt.show()

##-------------------------------------------------------------------------------------------------------
# It's time to play a game between Tom and Eva!

# Recall that Tom has a regular six-faced die and the results of rolling it follow a discrete uniform distribution in the interval of one and six. Eva has a biased coin that has a probability p of turning heads. The distribution of the number of flips Eva needs to land heads is geometric.

# Here are the rules of the game:

# Tom's score: the point of the rolled die
# Eva's score: the number of flips needed to land heads
# The person with the highest score wins
# Your task is to simulate this game! For the list of possible p values [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9] representing the probability of Eva's coin flipping heads, who do you expect to win?

for p in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]: 
    low = 1
    high = 7
	# Simulate rolling Tom's die 10,000 times
    die_samples = st.randint.rvs(low,high,size=10000)
	# Simulate Eva's coin flips to land heads 10,000 times
    coin_samples = st.geom.rvs(p,size=10000)
    diff = np.mean(die_samples - coin_samples)
    print(diff)

#########################################################################################################
'''
Generating Continuous Random Variables

Mean: Loc
Std: Scale
'''
# In this exercise, you'll use sampling to calculate the 95% confidence interval of the average height of American adult males. Recall from the lesson that the heights of American adult males are normally distributed with a mean of 177 centimeters and a standard deviation of eight centimeters.

# After sampling from the distribution with the above sample statistics, you'll change the mean of the heights to 185 centimeters without changing the standard deviation to explore what happens to the mean and confidence interval of the average height after sampling again.

random.seed(1222)

# Sample 1,000 times from the normal distribution where the mean is 177
heights_177_8 = st.norm.rvs(loc=177,scale=8,size=1000)
print(np.mean(heights_177_8))
upper = np.quantile(heights_177_8, 0.975)
lower = np.quantile(heights_177_8, 0.025)
print([lower, upper])

# Sample 1,000 times from the normal distribution where the mean is 185
heights_185_8 = st.norm.rvs(loc=185,scale=8,size=1000)
print(np.mean(heights_185_8))
upper = np.quantile(heights_185_8, 0.975)
lower = np.quantile(heights_185_8, 0.025)
print([lower, upper])

##-------------------------------------------------------------------------------------------------------
# You'll continue exploring the heights of American adult males, which you now know are normally distributed with a mean of 177 centimeters and a standard deviation of eight centimeters.

# In this exercise, you'll also sample from a normal distribution and calculate the 95% confidence interval of the average height. But this time, you'll change the standard deviation to 15 without changing the mean of the heights. You'll explore what will happen to the mean and confidence interval of the average height if you perform the sampling again!

random.seed(1231)
heights_177_8 = st.norm.rvs(loc=177, scale=8, size=1000)
print(np.mean(heights_177_8))
upper = np.quantile(heights_177_8, 0.975)
lower = np.quantile(heights_177_8, 0.025)
print([lower, upper])

# Sample 1,000 times from the normal distribution where the standard deviation is 15
heights_177_15 = st.norm.rvs(loc=177,scale=15,size=1000)
print(np.mean(heights_177_15))
upper = np.quantile(heights_177_15, 0.975)
lower = np.quantile(heights_177_15, 0.025)
print([lower, upper])

##-------------------------------------------------------------------------------------------------------
# Rohit has two freelance jobs. The pay for each job follows two independent normal distributions:

# income1 from Rohit's first job has a mean of $500 and a standard deviation of $50
# income2 from Rohit's second job has a mean of $1,000 and a standard deviation of $200
# Rohit has asked for your help simulating his income so that he can budget his expenses properly. You'll use sampling to find the 95% confidence interval of Rohit's total income from both jobs.

# You are going to perform simulations using normal distributions, which are probably the most important probability distribution used in Monte Carlo simulation.

# Sample from the normal distribution
income1 = st.norm.rvs(loc=500,scale=50,size=1000)
income2 = st.norm.rvs(loc=1000,scale=200,size=1000)

# Define total_income
total_income = income1 + income2
upper = np.quantile(total_income, 0.975)
lower = np.quantile(total_income, 0.025)
print([lower, upper])

# Excellent work—you've found a confidence interval of Rohit's total income by sampling independently from two normal distributions! What if these two incomes are _not_ independent of each other? How could you sample under dependent conditions? That's the topic of the next lesson on multivariate random variable sampling!

#########################################################################################################
'''
Generating Multivariate Random Variables


'''
# Imagine a small town in Colorado, USA, which has on average 300 sunny days, 35 cloudy days, and 30 rainy days each year. As a scientist studying climate change, you are interested in the distributions of sunny, cloudy, and rainy days in a random span of 50 years if these averages are assumed to remain the same.

p_sunny = 300/365
p_cloudy = 35/365
p_rainy = 30/365
num_of_days_in_a_year = 365
number_of_years = 50

# Simulate results
days = st.multinomial.rvs(num_of_days_in_a_year,
    [p_sunny, p_cloudy, p_rainy], size=number_of_years)

# Complete the definition of df_days
df_days = pd.DataFrame({"sunny": days[:,0],
     "cloudy": days[:,1],
     "rainy":  days[:,2]})
sns.pairplot(df_days)
plt.show()

# You have successfully sampled from a multinomial distribution! Note in this case, you are dealing with three probabilities, while in the video you saw only two: the probabilities of heads and tails. As long as the probabilities add up to one, this distribution can handle any number of probabilities!

##-------------------------------------------------------------------------------------------------------
# In this exercise, you'll be working with a DataFrame called house_price_size, which contains two columns called price and size, representing the price and size of houses, respectively.

# You'll first explore the data to gain some understanding of the distribution and relationship of the two variables price and size, and then you will obtain a covariance matrix for these two variables.

sns.pairplot(house_price_size)
plt.show()

# # Estimate the variance and covariances of house prices and sizes
print(house_price_size.cov())

# Great work. You probably noticed a fairly strong positive covariance based on the scatterplots generated in the first step of this exercise. House prices and sizes seem to rise together. Your covariance matrix in the second step confirmed this! Now that you have explored the data, let's see how you can use sampling to obtain simulated results!

##-------------------------------------------------------------------------------------------------------
# In this exercise, you'll continue working with the house_price_size DataFrame, which has been loaded for you. As a reminder, house_price_size contains two columns called price and size representing the price and size of houses in that order.

# Having explored the house_price_size DataFrame, you suspect that this is a multivariate normal distribution because price and size each seem to follow a normal distribution. Based on the covariance matrix that you calculated in the previous exercise, you can now perform multivariate normal distribution sampling with a defined covariance structure!

# To perform multivariate normal distribution sampling with defined covariance, you'll need the following information:

# price has a mean of 20 and size has a mean of 500
# price has a variance of 19 and size has a variance of 50,000
# The covariance for price and size is 950
# # You'll sample 5,000 times

mean_value = [20, 500]
sample_size_value = 5000
cov_mat = np.array([[19, 950], [950, 50000]])

# Simulate the results using sampling
simulated_results = st.multivariate_normal.rvs(mean=mean_value,size=sample_size_value,cov=cov_mat)
simulated_house_price_size = pd.DataFrame({"price":simulated_results[:,0],
                         				   "size":simulated_results[:,1]})

# Visualize the results 
sns.pairplot(simulated_house_price_size)
plt.show()

# Triumph! You've successfully performed multivariate normal sampling with a defined covariance structure: you can see the correlation in the scatter plots. You have proven that you've got a great understanding of foundational probability distributions and now you're ready to move onto a principled Monte Carlo simulation!

#########################################################################################################
#########################################################################################################
'''
Principled Monte Carlo Simulation


'''
# The diabetes dataset, dia, will be used as the real-world example for both this chapter and the next. Proper data exploration is a foundation for performing effective Monte Carlo simulations, so you'll continue exploring the data in the exercises!

# In this exercise, you'll focus on three variables: tc, ldl, and hdl. The dia DataFrame has been loaded for you.

# Create a pairplot of tc, ldl, and hdl
sns.pairplot(dia[['tc','ldl','hdl']])
plt.show()

# Calculate correlation coefficients
print(dia[['tc','ldl','hdl']].corr())

##-------------------------------------------------------------------------------------------------------
# Now, you'll focus on tc, hdl, and y, to understand the relationship between a few independent variables and the dependent one, disease progression. The diabetes dataset has been loaded as the DataFrame dia.

# Later in the course, you'll use a simulation to measure the impact of predictors on responses, so having an initial understanding of relationships between variables will help you understand your simulation results later on!

# Create a pairplot of tc, hdl, and y
sns.pairplot(dia[['tc','hdl','y']])
plt.show()

# Calculate correlation coefficients
print(dia[['tc','hdl','y']].corr())

##-------------------------------------------------------------------------------------------------------
# You are right—this statement is wrong! Choosing the proper probability distribution is key to the success of a Monte Carlo simulation, and we'll explore this importance more in the next lesson!

#########################################################################################################
'''
Choosing probability distributions

Use MLE
- Used to select a probability by measuring fit
- Distribution yielding highest likelihood given the data is considered optimal
- SciPy's nnlf function for negative max likelihood
'''
# Proper choice of input probability distributions is key for performing Monte Carlo simulations. In the video, three distributions were evaluated to determine which was the best fit for the age variable. Those distributions were the Laplace, normal, and exponential distributions. The normal distribution was the best fit.

# In this exercise, you'll see if you can find a distribution that improves upon the fit of the normal distribution! You'll evaluate the fitting of uniform, normal, and exponential distributions. The diabetes dataset has been loaded as a DataFrame, dia. Will the normal distribution still be the best?

distributions = [st.uniform, st.norm, st.expon]
mles = []
for distribution in distributions:
    # Fit the distribution and obtain the MLE value
    pars = distribution.fit(dia['age'])
    mle = distribution.nnlf(pars, dia['age'])
    mles.append(mle)
print(mles)

# Great work! You can see that normal distribution still gives the best fit with the lowest MLE value, while the uniform distribution yields an MLE value that is between that of the normal and exponential distributions.

##-------------------------------------------------------------------------------------------------------
# You've been given a list called clerk_data which contains the time a post office clerk spends with customers (in minutes). Your task is to explore what probability distribution could be a good fit for this data. First, you'll graph the data to gain intuition for choosing candidate probability distributions. Then, you'll evaluate candidate distributions to see if your intuition is correct!

sns.histplot(clerk_data)
plt.show()

# Define a list of distributions to evaluate: uniform, normal and exponential
distributions = [st.uniform, st.norm, st.expon]
mles = []
for distribution in distributions:
    # Fit each distribution, extract the MLE value, and append it to mles
    pars = distribution.fit(clerk_data)
    mle = distribution.nnlf(pars,clerk_data)
    mles.append(mle)
print(mles)

#########################################################################################################
'''
Inputs with Correlations

If variables in a dataset are known to be correlated, use a multivariate distribution for MC simulations

For multivariate normal, need:
1. Mean for each variable
2. Covariance matrix

Use the mean and covariance from the data as inputs when simulating
'''
# A good simulation should have similar results to the historical data. Was that true for the simulation in the video? In this exercise, you'll explore one way to examine the simulation results and find out!

# First, you'll perform a simulation using the multivariate normal distribution and the mean and covariance matrix of dia. Then, you'll check the means of both the historical and simulated data. Are they similar?

cov_dia = dia[["age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]].cov()
mean_dia = dia[["age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]].mean()

# Complete the code to perform the simulation
simulation_results = st.multivariate_normal.rvs(mean=mean_dia, size=10000, cov=cov_dia)

df_results = pd.DataFrame(simulation_results,columns=["age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"])

# Calculate bmi and tc means for the historical and simulated results
print(dia[["bmi","tc"]].mean())
print(df_results[["bmi","tc"]].mean())
      
# Calculate bmi and tc covariances for the historical and simulated results
print(dia[["bmi","tc"]].cov())
print(df_results[["bmi","tc"]].cov())

##-------------------------------------------------------------------------------------------------------
# Previously in the course, you used .cov() to obtain the covariance matrix and .corr() to obtain the correlation matrix. It's easy to confuse the two with each other and use them wrongly in simulations. Let's clarify!

# A correlation matrix is a standardized covariance matrix, where the correlation coefficients in the correlation matrix contain values from 0 to 1.

# corr(x,y) = cov(x,y) / std(x)std(y)

# The equation above tells us that cov(x,y), the covariance value, can be calculated by multiplying the correlation coefficient corr(x,y) with standard deviation of x, std(x), and the standard deviation of y, std(y). You'll test out this relationship in this exercise!

# Calculate the covariance matrix of bmi and tc
cov_dia2 = dia[["bmi","tc"]].cov()

# Calculate the correlation matrix of bmi and tc
corr_dia2 = dia[["bmi","tc"]].corr()
std_dia2 = dia[["bmi","tc"]].std()

print(f'Covariance of bmi and tc from covariance matrix :{cov_dia2.iloc[0,1]}')
print(f'Covariance of bmi and tc from correlation matrix :{corr_dia2.iloc[0,1] * std_dia2[0] * std_dia2[1]}')

#########################################################################################################
'''
Summary Statistics
'''

# In the last lesson, you performed a multivariate normal distribution using the mean and covariance matrix of dia. Now, you'll answer questions of interest using the simulated results!

# You may ask: why do we perform simulations when we have historical data? Can't we just use the data itself to answer questions of interest?

# This is a great question. Monte Carlo simulations are based on modeling using probability distributions, which yield the whole probability distribution for inspection (a large number of samples), rather than the limited number of data points available in the historical data.

# For example, you can ask questions like what is the 0.1st quantile of the age variable for the diabetes patients in our simulation? We can't answer this question with the historical data dia itself: because it only has 442 records, we can't calculate what the one-thousandth value is. Instead, you can leverage the results of a Monte Carlo simulation, which you'll do now!

cov_dia = dia[["age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]].cov()
mean_dia = dia[["age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]].mean()

simulation_results = st.multivariate_normal.rvs(mean=mean_dia, size=10000, cov=cov_dia)

df_results = pd.DataFrame(simulation_results, columns=["age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"])

# Calculate the 0.1st quantile of the tc variable
print(np.quantile(df_results['tc'], 0.001))

# Great! You've successfully calculated the 0.1st percentile of the tc variable using simulated data. We can't do this with historical data dia alone because it only has 442 records. That is the power of modeling using probability distributions. We are going to answer more questions like this in later exercises!

##-------------------------------------------------------------------------------------------------------
# What is the difference in the predicted disease progression (the response, y) for patients who are in the top 10% of BMI compared to the lowest 10% of BMI? You'll use the results of a simulation sampling the multivariate normal distribution to answer this question!

# The simulation has already been performed for you: your task is to evaluate the simulation results in df_results.

simulation_results = st.multivariate_normal.rvs(mean=mean_dia, size=20000, cov=cov_dia)
df_results = pd.DataFrame(simulation_results,columns=["age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"])
predicted_y = regr_model.predict(df_results)
df_y = pd.DataFrame(predicted_y, columns=["predicted_y"])
df_summary = pd.concat([df_results,df_y], axis=1)

# Calculate the 10th and 90th quantile of bmi in the simulated results
bmi_q10 = np.quantile(df_summary["bmi"], 0.1)
bmi_q90 = np.quantile(df_summary["bmi"], 0.9)

# Use bmi_q10 and bmi_q90 to filter df_summary and obtain predicted y values
mean_bmi_q90_outcome = np.mean(df_summary[df_summary['bmi'] > bmi_q90]["predicted_y"]) 
mean_bmi_q10_outcome = np.mean(df_summary[df_summary['bmi'] < bmi_q10]["predicted_y"])
y_diff = mean_bmi_q90_outcome - mean_bmi_q10_outcome
print(y_diff)

# Awesome outcomes! You can probably see that the difference in predicted y between patients in the top 10% and bottom 10% of BMI is about 150 to 160! Remember the stochastic nature of Monte Carlo Simulation: if you want to measure the uncertainty, you can perform even more simulations!

##-------------------------------------------------------------------------------------------------------
# What is the difference in the predicted disease progression (the response y) for patients who are in both the top 10% of BMI and the top 25% of HDL compared to those in both the lowest 10% of BMI and the lowest 25% of HDL? Again, a simulation has already been performed for you: your task is to evaluate the simulation results in df_results to find an answer to this question!

simulation_results = st.multivariate_normal.rvs(mean=mean_dia, size=20000, cov=cov_dia)
df_results = pd.DataFrame(simulation_results,columns=["age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"])
predicted_y = regr_model.predict(df_results)
df_y = pd.DataFrame(predicted_y, columns=["predicted_y"])
df_summary = pd.concat([df_results,df_y], axis=1)
hdl_q25 = np.quantile(df_summary["hdl"], 0.25)
hdl_q75 = np.quantile(df_summary["hdl"], 0.75)
bmi_q10 = np.quantile(df_summary["bmi"], 0.10)
bmi_q90 = np.quantile(df_summary["bmi"], 0.90)

# Complete the mean outcome definitions
bmi_q90_hdl_q75_outcome = np.mean(df_summary[(df_summary["bmi"] > bmi_q90) & (df_summary["hdl"] > hdl_q75)]["predicted_y"]) 
bmi_q10_hdl_q15_outcome = np.mean(df_summary[(df_summary["bmi"] < bmi_q10) & (df_summary["hdl"] < hdl_q25)]['predicted_y']) 
y_diff = bmi_q90_hdl_q75_outcome - bmi_q10_hdl_q15_outcome
print(y_diff)

# Congrats! You can probably see that the difference in predicted y between patients in the top 10% of BMI and top 25% of HDL and those in both the bottom 10% of BMI and bottom 25% of HDL is about 90 to 100! This result is lower than the results fpcus only on BMI in the previous exercise because of the negative correlation between HDL and the predicted y values you saw earlier, during data exploration.

#########################################################################################################
#########################################################################################################
'''
Model Checking and Results Interpretation

Evaluating distribution choices

Choosing variable probability distributions
1. Gain an intuitive understanding of data and available probability distributions
2. Use MLE to compare candidate distributions
3. Use Kolmogorov-Smirnov test to evaluate goodness of fit of probability distributions
- Quantifies distance between the empirical distributions of the data and the theoretical candidate probability distribution
- Use scipy.stats.kstest()

Calculating MLE and running ks-tests have similar but slightly different purposes: MLE yields the best candidate among a set of candidate distributions, while ks-tests provide information about whether a given probability distribution fits the data well.

'''
# In this exercise, you'll focus on one variable of the diabetes dataset dia: the ldl blood serum. You'll determine whether the normal distribution is a still good choice for ldl based on the additional information provided by a Kolmogorov-Smirnov test.

# List candidate distributions to evaluate
list_of_dists = ['laplace', 'norm', 'expon']
for i in list_of_dists:
    dist = getattr(st, i)
    # Fit the data to the probability distribution
    param = dist.fit(dia['ldl'])
    # Perform the ks test to evaluate goodness-of-fit
    result = st.kstest(dia['ldl'], i, args=param)
    print(result)

# <script.py> output:
    # KstestResult(statistic=0.06416228045269268, pvalue=0.05026538465299435)
    # KstestResult(statistic=0.04977872283458512, pvalue=0.21612589008862504)
    # KstestResult(statistic=0.30715826921004363, pvalue=1.6118927860147745e-37)

# Well done! Both the Laplace and normal distributions have p-values above 0.05. The normal distribution appears to be the best choice for the ldl variable, with a p-value around 0.22!

#########################################################################################################
'''
Visualizing Simulation Results
'''
# In this exercise, you'll explore simulation results for two variables: bmi and hdl. The simulated results for differences in predicted y values for people in the fourth quantile compared to the first quantile for each predictor (one at a time) have been generated and loaded as df_diffs for you.

# Create a pairplot of bmi and hdl
sns.pairplot(df_diffs[['bmi','hdl']])
plt.show()

# Plot a cluster map of the correlation between bmi and hdl
sns.clustermap(df_diffs[['bmi','hdl']].corr())
plt.show()

##-------------------------------------------------------------------------------------------------------
# Two common formats of DataFrames are the wide format and long format. The wide format shows different variables represented in different columns, while the long format displays different variables represented by two columns together (one for the variable name and the other for the corresponding values).

# Long versions of DataFrames can be useful for easily creating different visualizations, including the boxplot that you will create in this exercise after converting df_diffs (loaded for you) from wide to long format.

# Convert the hdl and bmi columns of df_diffs from wide to long format, naming the values column "y_diff"
hdl_bmi_long = df_diffs.melt(value_name='y_diff', value_vars=['bmi','hdl'])
print(hdl_bmi_long.head())

# Use a boxplot to visualize the results
sns.boxplot(x='variable',y='y_diff', data=hdl_bmi_long)
plt.show()

#########################################################################################################
'''
Sensitivity Analysis

- Helps us understand the impact of the range of inputs
- Illustrates the patterns or trends when summarized in tables or plots
'''

# You work for a company that manufactures industrial equipment. The sales price of each piece of equipment is $100,000. You also know that there is a strong negative correlation between the inflation_rate and sales volume. This relationship is captured by the covariance matrix cov_matrix, which is available in the console for you.

# The function profit_next_year_mc() performs a Monte Carlo simulation returning expected profit (in thousands of dollars), given the mean inflation rate and mean sales volume as arguments. You'll also need to pass n, the number of time the simulation should be run. The function has been loaded for you, and the definition is below.

def profit_next_year_mc(mean_inflation, mean_volume, n):
  profits = []
  for i in range(n):
    # Generate inputs by sampling from the multivariate normal distribution
    rate_sales_volume = st.multivariate_normal.rvs(mean=[mean_inflation,mean_volume], cov=cov_matrix,size=1000)
    # Deterministic calculation of company profit
    price = 100 * (100 + rate_sales_volume[:,0])/100
    volume = rate_sales_volume[:,1]
    loan_and_cost = 50 * volume + 45 * (100 + 3 * rate_sales_volume[:,0]) * (volume/100)
    profit = (np.mean(price * volume - loan_and_cost))
    profits.append(profit)
  return profits

# Run a Monte Carlo simulation 500 times using a mean_inflation of 2 and a mean_volume of 500
profits = profit_next_year_mc(2, 500, 500)

# Create a displot of the results
sns.displot(profits)
plt.show()

##-------------------------------------------------------------------------------------------------------
# You'll now examine what would happen to profits for the company from the previous exercise at a variety of mean_inflation and mean_volume values. This will help the company plan for several levels of inflation and sales volumes since no company can ever be certain what inflation or sales volumes they will have in the future.

# The mean inflation percentages you'd like to explore are 0, 1, 2, 5, 10, 15, 20, 50, while the sales values for use as the mean volume value are 100, 200, 500, 800, 1000. As a reminder, here is the profit_next_year_mc() function definition, which has already been loaded for you.

x1 = []
x2 = []
y = []
for infl in [0, 1, 2, 5, 10, 15, 20, 50]:
    for vol in [100, 200, 500, 800, 1000]:
		# Run profit_next_year_mc so that it samples 100 times for each infl and vol combination
        avg_prof = np.mean(profit_next_year_mc(infl, vol, 100))
        x1.append(infl)
        x2.append(vol)
        y.append(avg_prof)
df_sa = pd.concat([pd.Series(x1), pd.Series(x2), pd.Series(y)], axis=1)
df_sa.columns = ["Inflation", "Volume", "Profit"]
# Create a displot of the simulation results for "Profit"
sns.displot(df_sa['Profit'])
plt.show()

# Nice work! You can see that there is a wide range of possible values for profits, which is expected due to the wide range of input inflation rates and volumes. You might have noticed that there are also negative values for profits, indicating a loss for the company. Let's use the hexbin plot to perform sensitivity analysis!

##-------------------------------------------------------------------------------------------------------
# The simulation results you generated in the previous exercise are saved in the DataFrame df_sa which has been loaded for you.

# Recall that df_sa has three columns: Inflation contains the mean inflation rates used in the simulations, Volume contains the mean sales volumes used in the simulations, and Profit contains forecasted profits based on your simulation.

# You'll now use a hexbin plot to perform sensitivity analysis and understand the impact of these parameters!

# Complete the hexbin to visualize sensitivity analysis results
df_sa.plot.hexbin(x='Inflation',
     y='Volume',
     C='Profit',
     reduce_C_function=np.mean,
     gridsize=10,
     cmap="viridis",
     sharex=False)
plt.show()

# Congrats! You have successfully performed a sensitivity analysis! With increasing inflation, given the same sales volume, the simulated profits will decrease. Given the same inflation rate at a lower value, with increasing volumes, the profits will increase. However, profits will decrease with high inflation over 10%, even given increasing volumes.