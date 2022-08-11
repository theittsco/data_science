import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#If you want all of the plots to look like seaborn
sns.set()

'''
Note that there are methods for the mean, standard deviation, and variance.
'''
mean = np.mean(data)
max = np.max(data)
min = np.min(data)
std = np.std(data)
variance = np.var(data)

'''
Here is the definition and usage of the cumluative distribution function.
For the point on the CDF curve (n,p), the probability that x < n is p, while
the probability that x > n is 1 - p.
'''
#Defining the ECDF
def ecdf(data):
    """Compute ecdf for a 1D array of measurements. """
    #Number of data points: n
    n = len(data)

    # x-data for the ECDF (It needs to be in numerical order): x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1,n+1) / n

    return x, y

#Computing the ECDF for a dataset
x_graph, y_graph = ecdf(data)

#Generating a plot for the ECDF
plt.plot(x_graph, y_graph, marker = '.', linestyle = 'none')

#labeling the axes
plt.xlabel('x-variable')
plt.ylabel('ECDF')

#Display Plot
plt.show()

'''
Here is the definition and usage of percentiles. You specify an array of the
percentiles you want to know about. Then using the np.percentiles function,
you pass in a 1D dataset that find which values in the dataset match the desired
percentile compared to the rest of the dataset.
'''
percentiles = np.array([2.5,25,50,75,97.5])

ptiles_variable = np.percentiles(data, percentiles)

print(ptiles_variable)

"""Here we overlay the percentiles onto a ECDF plot."""
#Plot the ECDF
_ = plt.plot(x_graph, y_graph, '.')
_ = plt.xlabel('x-variable')
_ = plt.ylabel('ECDF')

#Overlay percentiles as red diamonds
_ = plt.plot(ptiles_variable, percentiles/100, marker = 'D', color = 'red',
linestyle = 'none')

#Show the plot
plt.show()

'''
If you want to compare to properties of the same variable, scatterplots are the
way to do it. Hence correlation.
'''
#Making a scatter plot
plt.plot(x_1, x_2, marker = '.', linestyle = 'none')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()

'''
Covariance definition and usage. It ranges from [-1,1]. If x1 and x2 have a
coefficient of -1, then they are negatively correlated. If +1, positively
correlated. If 0, they are uncorrelated. The covariance is sometimes called a
measure of "linear dependence" between the two random variables. When the
covariance is normalized, one obtains the Pearson correlation coefficient,
which gives the goodness of the fit for the best possible linear function
describing the relation between the variables. In this sense covariance is a
linear gauge of dependence.
'''

#Compute the covariance matrix
covariance_matrix = np.cov(x_1,x_2)

print(covariance_matrix)

x_cov = covariance_matrix[0,1]

'''
Pearson correlation coefficient definition and usage.
'''

def pearson_r(x,y):
    """Compute the Pearson correlation coefficient between two arrays."""
    #Compute correlation matrix
    corr_mat = np.corrcoef(x,y)

    #Return entry [0,1]
    return corr_mat[0,1]

#Computing Pearson correlation coefficient for data
r = pearson_r(x_data, y_data)
print(r)

'''
Random Variables, using numpy
'''
#Seeding a random number generator
np.random.seed(42)

#Initialize random numbers
random_numbers = np.empty(10000)

#Generate random numbers by looping
for i in range(10000):
    random_numbers[i] = np.random.random()

#Plot the result
_ = plt.hist(random_numbers)
plt.show()

"""Here we show the definition of Bernoulli trials, a form of a binomial
distribution."""

def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0

    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success+=1

    return n_success

"""For many Bernoulli Trials, the following shows a probability
distribution."""

# Compute the number of occurances out of 100 events, prob for one is p = 0.05
for i in range(1000):
    n_occurances[i] = perform_bernoulli_trials(100,0.05)


# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of occurnaces out of 100 events')
_ = plt.ylabel('probability')

# Show the plot
plt.show()

"""If A happens when 10 or more events occur, what is the probability of A?"""

# Compute ECDF: x, y
x, y = ecdf(n_occurances)

# Plot the ECDF with labeled axes
plt.plot(x,y,marker='.',linestyle='none')
plt.xlabel('number of occurances out of 100')
plt.ylabel('CDF')

# Show the plot
plt.show()

# Compute the number of 100-loan simulations with 10 or more defaults: n_occurances
n_A = np.sum(n_occurances >= 10)

# Compute and print probability of losing money
print('Probability of A =', n_A / len(n_occurances))

'''
Binomial Distribution: If you want to skip Bernoulli Trial simulations, then
just use the np.random.binomial(events, probability, number of trials)
'''

# Take 10,000 samples out of the binomial distribution: n_occurances
n_occurances = np.random.binomial(100,0.05,size=10000)

# Compute CDF: x, y
x, y = ecdf(n_occurances)

# Plot the CDF with axis labels
plt.plot(x,y,marker='.',linestyle='none')
plt.xlabel('number of occurances out of 100 events')
plt.ylabel('CDF')

# Show the plot
plt.show()

#To plot the PMF (discrete version of a PDF), we need to center the bin edges on integers.
bins = np.arange(0, max(n_occurances) + 1.5) - 0.5

'''
For dstributions that have a high number of events and low proability, use the
Poisson distribution. np.random.poisson(mean, number of trials)
'''
# Draw 10,000 samples out of Poisson distribution: n_A
n_A = np.random.poisson(251/115,size=10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_A>=7)

# Compute probability of getting seven or more: p_large
p_large = n_large/len(n_A)

# Print the result
print('Probability of seven or more A:', p_large)

'''
For Gaussian distributions, the form is np.random.normal(mean, std, size)
'''
sample = np.random.normal(mean, std, size)
plt.hist(sample, bins=np.sqrt(size), normed=True, histtype='step')

"""To check if a dataset is normally distributed, take the mean and std of the
dataset and make a normal simulation and compare the edcfs."""

# Compute mean and standard deviation: mu, sigma
mu = np.mean(data)
sigma = np.std(data)

# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu,sigma,size=10000)

# Get the CDF of the samples and of the data
x_theor, y_theor = ecdf(samples)
x, y = ecdf(data)

# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Data')
_ = plt.ylabel('CDF')
plt.show()

'''
The waiting time between Poisson processes is exponentially distributed. The
form is np.random.exponential(mean, size). It's a process in which events occur
continuously and independently at a constant average rate. It is a particular
case of the gamma distribution.
'''
# Compute mean no-hitter time: tau
tau = np.mean(data_times)

# Draw out of an exponential distribution with parameter tau: inter_nohitter_time
inter_data_time = np.random.exponential(tau, 100000)

# Plot the PDF and label axes
_ = plt.hist(inter_data_time,
             bins=50, normed=True, histtype='step')
_ = plt.xlabel('events between occurances')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# Create an ECDF from real data: x, y
x, y = ecdf(data_times)

# Create a CDF from theoretical samples: x_theor, y_theor
x_theor, y_theor = ecdf(inter_data_time)

# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')

# Margins and axis labels
plt.margins(0.02)
plt.xlabel('events between occurances')
plt.ylabel('CDF')

# Show the plot
plt.show()

"""To find the optimal parameter, take a 2*tau and 0.5*tau cdf and compare
with data."""
# Plot the theoretical CDFs
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('events between occurances')
plt.ylabel('CDF')

# Take samples with half tau: samples_half
samples_half = np.random.exponential(tau/2,size=10000)

# Take samples with double tau: samples_double
samples_double = np.random.exponential(2*tau,size=10000)

# Generate CDFs from these samples
x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)

# Plot these CDFs as lines
_ = plt.plot(x_half, y_half)
_ = plt.plot(x_double, y_double)

# Show the plot
plt.show()

'''
Linear regression. Using the fit parameters given by np.polyfit, and assuming
a linear relationship, we construct a regression line to fit the data.
'''
# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(x,y,1)

# Print the results to the screen
print('slope =', a, 'y / x')
print('intercept =', b, 'x')

# Make theoretical line to plot
x = np.array([min,max])
y = a * x + b

# Add regression line to your plot
_ = plt.plot(x, y)

# Draw the plot
plt.show()

"""To perform a parameter estimation on multiple sets of data, use the
following."""

# Iterate through x,y pairs
for x, y in zip(set_x,set_y):
    # Compute the slope and intercept: a, b
    a, b = np.polyfit(x,y,1)

    # Print the result
    print('slope:', a, 'intercept:', b)

'''
Bootstrap Replicates: These are resampling simulations based on the values
and a property (such as the mean) of the available dataset.
'''

for i in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(data, size=len(data))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    i = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(data)
i = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
i = plt.xlabel('Data')
i = plt.ylabel('ECDF')

# Show the plot
plt.show()

"""Drawing/plotting many bootstrap replicates can come in handy."""

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data,func,size)

    return bs_replicates

"""And using that method here."""
# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(data,func=np.mean,size=10000)

# Compute and print Standard error of the mean
sem = np.std(data) / np.sqrt(len(data))
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean of data')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

"""To give the 95% confidence interval of the simulations, use the
folllowing."""

print(np.percentile(bs_replicates,percentiles))

'''
Pairs bootstrap involves resampling pairs of data. Each collection of pairs
fit with a line, in this case using np.polyfit(). We do this again and again,
getting bootstrap replicates of the parameter values.
'''
def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(0,len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x,bs_y,1)

    return bs_slope_reps, bs_intercept_reps

# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(x,y,size=1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5,97.5]))

# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, normed=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()

"""And by plotting the boot strap regressions, we can see how they vary
visually."""
# Generate array of x-values for bootstrap lines: x
x = np.array([0,100])

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x,
                 bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.plot(x,y,marker='.',linestyle='none')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('x')
_ = plt.ylabel('y')
plt.margins(0.02)
plt.show()

'''
Permutation sampling takes two sets of data and resamples them by mixing
them together.
'''

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1,data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

#And graphing the permutation samples
for _ in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(data_1,data_2)


    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(data_1)
x_2, y_2 = ecdf(data_2)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('data')
_ = plt.ylabel('ECDF')
plt.show()

"""Here is a draw permutation sample function."""

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1,data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1,perm_sample_2)

    return perm_replicates

'''
p-Values: Test a well known statistic, like the mean or median. The p-Value is
the probability of observing a test statistic equally or more extreme than the
one you observed, given that the null hypothesis is true.
'''
#Here's an example of the p-value in use.

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(data_a,data_b)

# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(data_a, data_b,
                                 diff_of_means, size=10000)

# Compute p-value: p
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

# Print the result
print('p-value =', p)
