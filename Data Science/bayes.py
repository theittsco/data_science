'''
Bayesian Data Analysis in Python

1. The Bayesian Way
2. Bayesian Estimation
3. Bayesian Inference
4. Bayesian Linear Regression with pyMC3

Bayesian inference means updating one's belief about something as the new information becomes available

                Frequentist             Bayesian
Probability     Proportion of outcomes  Degree of belief
Parameters      Fixed Values            Random Variables

-Natural handling of uncertainty (because parameters have distributions)
-Possibility to include expert opinion or domain knowledge in the model 
-No need to rely on fixed constants such as p-values

Frequentist
- Probability means proportion of outcomes
- Models' parameters are fixed values
- Requires large enough data sample for the inference to be valid
- Inference is based on fixed constants, such as p-values
'''

# Well done on the previous exercise! Now you have the general idea of what the Bayesian approach is all about. Among other things, you know that for a Bayesian, parameters of statistical models are random variables which can be described by probability distributions.

# This exercise will test your ability to visualize and interpret probability distributions. You have been given a long list of draws from a distribution of the heights of plants in centimeters, contained in the variable draws.

# Print the list of draws
print(draws)

# Print the length of draws
print(len(draws))

# Plot the density of draws
sns.kdeplot(draws, shade=True)
plt.show()

# Based on the density plot you have just drawn, which of the following statements is false?

# Well done! Values larger than 30 are very unlikely, but they might occur with a non-zero probability. Let's now take a look at the core of Bayesian statistics: Bayes' Theorem!

#####################################################################################
'''
Bayes Theorem 

Probability theory:
- Statement of uncertainty
- Expressed as a number between 0 and 1

Sum rule: 
- Probability A or B
- OR = addition
- P(2 or 4) = 1/6 + 1/6 = 1/3 (with a die)

Product rule:
- Probability A and B
- AND = multiplication
- P(2 and 4) = 1/6 * 1/6 = 1/36 

Conditional probability:
- Probability of an event happening given another event having already occurred
- P(A|B)

Bayes' Theorem: P(A|B)P(B) = P(B|A)P(A)
'''
# You have a regular deck of 52 well-shuffled playing cards. The deck consists of 4 suits, and there are 13 cards in each suite: ranks 2 through 10, a jack, a queen, a king, and an ace. This means that in the whole deck of 52, there are four of each distinct rank: four aces, four kings, four tens, four fives, etc.

# Since there are 52 distinct cards, the probability of drawing any one particular card is 1/52. Using the two rules of probability you've learned about in the last video, calculate the probabilities of drawing some specific combinations of cards, as described in the instructions.

# Calculate probability of drawing a king or queen
p_king_or_queen = 4/52 + 4/52
print(p_king_or_queen)

# Calculate probability of drawing <= 5
p_five_or_less = 4 * (4/52)
print(p_five_or_less)

# Calculate probability of drawing four aces
p_all_four_aces = (1/52)**4
print(p_all_four_aces)

##--------------------------------------------------------------------------------------
# Well done on the previous exercise! Let's now tackle the famous Bayes' Theorem and use it for a simple but important task: spam detection.

# While browsing your inbox, you have figured out that quite a few of the emails you would rather not waste your time on reading contain exclamatory statements, such as "BUY NOW!!!". You start thinking that the presence of three exclamation marks next to each other might be a good spam predictor! Hence you've prepared a DataFrame called emails with two variables: spam, whether the email was spam, and contains_3_exlc, whether it contains the string "!!!". The head of the data looks like this:

#      spam    contains_3_excl
# 0    False             False
# 1    False             False
# 2    True              False
# 3    False             False
# 4    False             False
# Your job is to calculate the probability of the email being spam given that it contains three exclamation marks.

# Calculate and print the unconditional probability of spam
p_spam = emails["spam"].mean()
print(p_spam)

# Calculate and print the unconditional probability of "!!!"
p_3_excl = emails["contains_3_excl"].mean()
print(p_3_excl)

# Calculate and print the probability of "!!!" given spam
p_3_excl_given_spam = emails.loc[emails["spam"]]["contains_3_excl"].mean()
print(p_3_excl_given_spam)

# Calculate and print the probability of spam given "!!!"
p_spam_given_3_excl = p_3_excl_given_spam * p_spam / p_3_excl
print(p_spam_given_3_excl)

##--------------------------------------------------------------------------------------
# A doctor suspects a disease in their patient, so they run a medical test. The test's manufacturer claims that 99% of sick patients test positive, while the doctor has observed that the test comes back positive in 2% of all cases. The suspected disease is quite rare: only 1 in 1000 people suffer from it.

# The test result came back positive. What is the probability that the patient is indeed sick? You can use Bayes' Theorem to answer this question. Here is what you should calculate:

# Correct! Not very intuitive, is it? Even though the test is great at discovering the disease and raises false alarms rather seldom, a positive result means only a 5% probability that the patient is sick! Let's move forward to the final lesson of Chapter 1, where you will see a Bayesian statistical model in action!

#####################################################################################
'''
Tasting the Bayes

Binomial Dist: success (1), failure (0)
- One parameter: Probability of success
'''
# In the video, you have seen our custom get_heads_prob() function that estimates the probability of success of a binomial distribution. In this exercise, you will use it yourself and verify whether it does its job well in a coin-flipping experiment.

# Watch out for the confusion: there are two different probability distributions involved! One is the binomial, which we use to model the coin-flipping. It's a discrete distribution with two possible values (heads or tails) parametrized with the probability of success (tossing heads). The Bayesian estimate of this parameter is another, continuous probability distribution. We don't know what kind of distribution it is, but we can estimate it with get_heads_prob() and visualize it.

# Generate 1000 coin tosses
tosses = np.random.binomial(1, 0.5, size=1000)

# Estimate the heads probability
heads_prob = get_heads_prob(tosses)

# Plot the distribution of heads probability
sns.kdeplot(heads_prob, shade=True, label="heads probabilty")
plt.show()

##--------------------------------------------------------------------------------------
# Imagine you are a frequentist (just for a day), and you've been tasked with estimating the probability of tossing heads with a (possibly biased) coin, but without observing any tosses. What would you say? It's impossible, there is no data! Then, you are allowed to flip the coin once. You get tails. What do you say now? Well, if that's all your data, you'd say the heads probability is 0%.

# You can probably feel deep inside that these answers are not the best ones. But what would be better? What would a Bayesian say? Let's find out!

# Estimate and plot heads probability based on no data
heads_prob_nodata = get_heads_prob([])
sns.kdeplot(heads_prob_nodata, shade=True, label="no data")
plt.show()

# Estimate and plot heads probability based on a single tails
heads_prob_onetails = get_heads_prob([0])
sns.kdeplot(heads_prob_onetails, shade=True, label="single tails")
plt.show()

# Estimate and plot heads probability based on 1000 tosses with a biased coin
biased_tosses = np.random.binomial(1,0.05,size=1000)
heads_prob_biased = get_heads_prob(biased_tosses)
sns.kdeplot(heads_prob_biased, shade=True, label="biased coin")
plt.show()

# That's interesting! With no data, each possible value of the heads probabilty is equally likely! That's the Bayesian way of saying 'we don't know'. Having seen a single tails, the model suspects that tails is more likely than heads, but since there is so little data, it is not very sure about it, so other values are possible, too. Having seen 1000 tosses, 5% of them heads, the model is certain: the heads probability is around 5%. You have just witnessed the Bayesian approach at its core: as more data come in, we update our belief about the parameter, and with more data we become more certain about our estimate!

##--------------------------------------------------------------------------------------
# In the last two exercises, you have examined the get_heads_prob() function to discover how the model estimates the probability of tossing heads and how it updates its estimate as more data comes in.

# Now, let's get down to some serious stuff: would you like to play coin flipping against your friend? She is willing to play, as long as you use her special lucky coin. The tosses variable contains a list of 1000 results of tossing her coin. Will you play?

# Assign first 10 and 100 tosses to separate variables
tosses_first_10 = tosses[:10]
tosses_first_100 = tosses[:100]

# Get head probabilities for first 10, first 100, and all tossses
heads_prob_first_10 = get_heads_prob(tosses_first_10)
heads_prob_first_100 = get_heads_prob(tosses_first_100)
heads_prob_all = get_heads_prob(tosses)

# Plot density of head probability for each subset of tosses
sns.kdeplot(heads_prob_first_10, shade=True, label='first_10')
sns.kdeplot(heads_prob_first_100, shade=True, label='first_100')
sns.kdeplot(heads_prob_all, shade=True, label='all')
plt.show()

#####################################################################################
#####################################################################################
'''
Bayesian Estimation

P(par|data) = P(data|par)P(par)/P(data)

P(par|data): Posterior distribution, what we know about parameters after having seen the data
P(data|par): Prior distribution, what we know about the parameters before seeing any data
P(par): Likelihood of the data according to our statistical model
P(data): Scaling factor
'''
# Congratulations! You have just been hired as a data analyst at your government's Department of Health. The cabinet is considering the purchase of a brand-new drug against a deadly and contagious virus. There are some doubts, however, regarding how effective the new drug is against the virus. You have been tasked with estimating the drug's efficacy rate, i.e. the percentage of patients cured by the drug.

# An experiment was quickly set up in which 10 sick patients have been treated with the drug. Once you know how many of them are cured, you can use the binomial distribution with a cured patient being a "success" and the efficacy rate being the "probability of success". While you are waiting for the experiment's results, you decide to prepare the parameter grid.

# Create cured patients array from 1 to 10
num_patients_cured = np.arange(0,11)

# Create efficacy rate array from 0 to 1 by 0.01
efficacy_rate = np.arange(0,1.01,0.01)

# Combine the two arrays in one DataFrame
df = pd.DataFrame([(x, y) for x in num_patients_cured for y in efficacy_rate])

# Name the columns
df.columns = ['num_patients_cured','efficacy_rate']

# Print df
print(df)

# Well done! You have prepared a fine grid of possible values for the parameter you want to estimate (the efficacy rate), for all possible results of the experiment (the number of patients cured by the drug). Uh-oh, it looks like the results have just arrived!

##--------------------------------------------------------------------------------------
# According to the experiment's outcomes, out of 10 sick patients treated with the drug, 9 have been cured. What can you say about the drug's efficacy rate based on such a small sample? Assume you have no prior knowledge whatsoever regarding how good the drug is.

# A DataFrame df with all possible combinations of the number of patients cured and the efficacy rate which you created in the previous exercise is available in the workspace.

# Calculate the prior efficacy rate and the likelihood
df["prior"] = uniform.pdf(df["efficacy_rate"])
df["likelihood"] = binom.pmf(df["num_patients_cured"], 10, df["efficacy_rate"])

# Calculate the posterior efficacy rate and scale it to sum up to one
df["posterior_prob"] = df["prior"] * df["likelihood"]
df["posterior_prob"] /= df["posterior_prob"].sum()

# Compute the posterior probability of observing 9 cured patients
df_9_of_10_cured = df.loc[df["num_patients_cured"] == 9]
df_9_of_10_cured["posterior_prob"] /= df_9_of_10_cured["posterior_prob"].sum()

# Plot the drug's posterior efficacy rate
sns.lineplot(df_9_of_10_cured['efficacy_rate'], df_9_of_10_cured['posterior_prob'])
plt.show()

# Good job! As we might have expected, observing 9 out of 10 patients cured results in the posterior efficacy rate of 90% being very likely. Notice, however, how much uncertainty there is in the posterior distribution: even the efficacy of 50% is plausible. This is the result of a very small data sample and a great example of how Bayesian parameter estimates incorporate uncertainty!

##--------------------------------------------------------------------------------------
# Well done on estimating the posterior distribution of the efficacy rate in the previous exercise! Unfortunately, due to a small data sample, this distribution is quite wide, indicating much uncertainty regarding the drug's quality. Luckily, testing of the drug continues, and a group of another 12 sick patients have been treated, 10 of whom were cured. We need to update our posterior distribution with these new data!

# This is easy to do with the Bayesian approach. We simply need to run the grid approximation similarly as before, but with a different prior. We can use all our knowledge about the efficacy rate (embodied by the posterior distribution from the previous exercise) as a new prior! Then, we recompute the likelihood for the new data, and get the new posterior!

# Assign old posterior to new prior and calculate likelihood
df["new_prior"] = df["posterior_prob"]
df["new_likelihood"] = binom.pmf(df["num_patients_cured"], 12, df["efficacy_rate"])

# Calculate new posterior and scale it
df["new_posterior_prob"] = df["new_prior"] * df["new_likelihood"]
df["new_posterior_prob"] /= df["new_posterior_prob"].sum()

# Compute the posterior probability of observing 10 cured patients
df_10_of_12_cured = df.loc[df['num_patients_cured'] == 10]
df_10_of_12_cured["new_posterior_prob"] /= df_10_of_12_cured['new_posterior_prob'].sum()

# We have two posterior distributions for the efficacy rate now:

# 1. The one from the previous exercise (without prior knowledge, after seeing 9 out of 10 patients cured) which you have used as a new prior in this exercise.
# 2. The updated one you have just calculated (after seeing another 10 out of 12 patients cured).
# You can plot them on top of each other using the following code chunk:

# sns.lineplot(df_10_of_12_cured["efficacy_rate"], 
#              df_10_of_12_cured["new_posterior_prob"], 
#              label="new posterior")
# sns.lineplot(df_9_of_10_cured["efficacy_rate"], 
#              df_9_of_10_cured["posterior_prob"], 
#              label="old posterior = new prior")
# plt.show()
# Based on the plot, which of the following statements is false?

# Yes, this one's false! The difference between these two distributions actually reflects what we have learned only from the data on the 12 new patients. The knowledge about the first 10 patients is already encoded in the old posterior, which became the new prior.

# Instead of using the old posterior as a new prior, we could have come up with our own prior belief about the efficacy rate in the first place. Let's take a look at it in the upcoming video!

#####################################################################################
'''
Prior Belief

Prior distribution chosen before we see the data
Prior choice can impact posterior results
To avoid cherry picking, prior choices should be:
- Clearly stated
- Explainable, based on previous research,, sensible assumptions, expert opinion, etc.

def get_heads_prob(tosses):
    num_heads = np.sum(tosses)
    # prior: Beta(1,1)
    return np.random.beta(num_heads + 1, len(tosses) - num_heads + 1, 1000)
'''

'''
Choosing the prior distribution is a very important step in Bayesian data analysis. Can you classify the following statements about the prior distribution as either true or false?

True:

If you don't explain your prior choice, you might be accused of cherry picking a prior that results in the posterior you want.
The prior distribution should be chosen before one sees any data
The prior distribution is what makes Bayesian inference possible even with little data

False:

The prior choice has no impact if one has little data
Usually, the prior distribution is narrower and higher than the posterior distribution
'''

'''
You continue working on your task to estimate the new drug's efficacy, but with the small data sample you had, you know there is a lot of uncertainty in your estimate. Luckily, a couple of neighboring countries managed to conduct more extensive experiments and have just published their results. You can use them as priors in your analysis!

Having browsed all the publications, you conclude that the reported efficacy rates are mostly between 70% and 90%. A couple of results below 50% were recorded too, but not many.

Which of the following distributions captures this prior information best?

Answer: Beta(5,2)
'''

##--------------------------------------------------------------------------------------
# You have just decided to use a Beta(5, 2) prior for the efficacy rate. You are also using the binomial distribution to model the data (curing a sick patient is a "success", remember?). Since the beta distribution is a conjugate prior for the binomial likelihood, you can simply simulate the posterior!

# For Beta(a,b), the posterior is Beta(x,y), where:

# x = NumSuccesses + a
# y = NumObservations - NumSuccesses + b 

# Can you simulate the posterior distribution? 22 patients, 19 of whom have been cured. Prior is Beta(5,2)

# Define the number of patients treated and cured
num_patients_treated = 22
num_patients_cured = 19

# Simulate 10000 draws from the posterior distribuition
posterior_draws = np.random.beta(num_patients_cured + 5, num_patients_treated - num_patients_cured + 2, 10000)

# Plot the posterior distribution
sns.kdeplot(posterior_draws, shade=True)
plt.show()

# Well done! Notice that the posterior distribuion has a slightly longer tail on the left-hand side, allowing for efficacy rates as small as 50%, even though in your data you observe 86% (19 out of 22). This is the impact of the prior: you learn not only from your own small data, but also from other countries' experience! Now that you know how obtain posterior distributions of the parameters, let's talk about how to report these results!

#####################################################################################
'''
Reporting Bayesian Results

- Report the prior and posterior of each parameter.
- Plot both

With many parameters, summarize. Use the point estimate
- Could be the mean, median, percentile
Also use credible interval: Such an interval that the probability that the parameter falls inside it is x%

Highest Posterior Density (HPD)
'''
# You continue working at your government's Department of Health. You have been tasked with filling the following memo with numbers, before it is sent to the secretary.

# Based on the experiments carried out by ourselves and neighboring countries, should we distribute the drug, we can expect ___ infected people to be cured. There is a 50% probability the number of cured infections will amount to at least ___, and with 90% probability it will not be less than ___.

# The array of posterior draws of the drug's efficacy rate you have estimated before is available to you as drug_efficacy_posterior_draws.

# Calculate the three numbers needed to fill in the memo, knowing there are 100,000 infections at the moment. 

# Calculate the expected number of people cured
cured_expected = np.mean(drug_efficacy_posterior_draws) * 100_000

# Calculate the minimum number of people cured with 50% probability
min_cured_50_perc = np.median(drug_efficacy_posterior_draws) * 100_000

# Calculate the minimum number of people cured with 90% probability
min_cured_90_perc = np.percentile(drug_efficacy_posterior_draws, 10) * 1e5

# Print the filled-in memo
print(f"Based on the experiments carried out by ourselves and neighboring countries, \nshould we distribute the drug, we can expect {int(cured_expected)} infected people to be cured. \nThere is a 50% probability the number of cured infections \nwill amount to at least {int(min_cured_50_perc)}, and with 90% probability \nit will not be less than {int(min_cured_90_perc)}.")

# Great! Your memo compresses the posterior distribution of the drug's efficacy to a couple of useful numbers, certainly helping the secretary decide whether to buy the new drug. However, these numbers convey no information as to how uncertain the estimation is. Let's try quantifying this uncertainty next!

##--------------------------------------------------------------------------------------
# You know that reporting bare point estimates is not enough. It would be great to provide a measure of uncertainty in the drug's efficacy rate estimate, and you have all the means to do so. You decide to add the following to the memo.

# The experimental results indicate that with a 90% probability the new drug's efficacy rate is between ___ and ___, and with a 95% probability it is between ___ and ___.

# You will need to calculate two credible intervals: one of 90% and another of 95% probability.

# Import arviz as az
import arviz as az

# Calculate HPD credible interval of 90%
ci_90 = az.hdi(drug_efficacy_posterior_draws, hdi_prob=0.9)

# Calculate HPD credible interval of 95%
ci_95 = az.hdi(drug_efficacy_posterior_draws, hdi_prob=0.95)

# Print the memo
print(f"The experimental results indicate that with a 90% probability \nthe new drug's efficacy rate is between {np.round(ci_90[0], 2)} and {np.round(ci_90[1], 2)}, \nand with a 95% probability it is between {np.round(ci_95[0], 2)} and {np.round(ci_95[1], 2)}.")

# Well done on calculating the Highest Posterior Density credible interval in the last exercise! Here is the 90% interval you have obtained:

# (0.72, 0.94)

# What is the proper Bayesian interpretation of this credible interval in the context of the drug's efficacy?

# There is one correct answer. The other two answers provide two equivalent but differently phrased frequentist interpretations of a confidence interval.

# The probability that the drug's true efficacy rate lies in the interval (0.72, 0.94) is 90%.

# Perfect, even though this was a hard one! That's the Bayesian interpretation of a credible interval. Since the drug's true efficacy rate is considered a random variable, we can make probabilistic statements about it, as in: "the probability that it takes a particular value or that it lies in a particular interval is X%". Great job on finishing Chapter 2. Next, in Chapter 3, you will apply all you've learned about the Bayesian approach to practical problems: A/B testing, decision analysis, and regression modeling. See you there!

#####################################################################################
#####################################################################################
'''
Bayesian Inference

A/B Testing
- Method of assessing user experience based on a randomized experiment
- Divide users into two groups
- Expose each group to a different version of something
- Compare which group scores better on some metric

A/B Testing: Frequentist
- Based on hypothesis testing
- Check whether A and B perform the same or not
- DOes not say how much better A is than B

A/B Testing: Bayesian
- Calculate posterior distribution for A and B and compare them
- Directly calculate the probability that A is better than B
- Quantify how much better
- Calculate the expected loss in case a wrong decision is made 
'''
# In the upcoming few exercises, you will be using the simulate_beta_posterior() function you saw defined in the last video. In this exercise, you will get a feel for what the function is doing by carrying out the computations it performs.

# You are given a list of ten coin tosses, called tosses, in which 1 stands for heads, 0 for tails, and we define heads as a "success". To simulate the posterior probability of tossing heads, you will use a beta prior.

# Set prior parameters and calculate number of successes
beta_prior_a = 1
beta_prior_b = 1
num_successes = np.sum(tosses)

# Generate 10000 posterior draws
posterior_draws = np.random.beta(
  num_successes + beta_prior_a, 
  len(tosses) - num_successes + beta_prior_b, 
  10000)  

# Plot density of posterior_draws
sns.kdeplot(posterior_draws, shade=True)
plt.show()

# Set prior parameters and calculate number of successes
beta_prior_a = 1
beta_prior_b = 10
num_successes = np.sum(tosses)

# Generate 10000 posterior draws
posterior_draws = np.random.beta(
  num_successes + beta_prior_a, 
  len(tosses) - num_successes + beta_prior_b, 
  10000)  

# Plot density of posterior_draws
sns.kdeplot(posterior_draws, shade=True)
plt.show()

# Cool! Now you see what simulate_beta_posterior() is doing: based on the binomial data and the prior, it samples posterior draws. Notice how using the Beta(1, 10) prior shifts the posterior to the left compared to Beta(1, 1). This effect is quite strong, as there is little data: just 10 coin flips. Let's move on to A/B testing!

##--------------------------------------------------------------------------------------
# After a successful career episode at the Department for Health, you switch to marketing. Your new company has just run two pilot advertising campaigns: one for sneakers, and one for clothes. Your job is to find out which one was more effective as measured by the click-through rate and should be rolled out to a larger audience.

# You decide to run A/B testing, modeling the data using the binomial likelihood. You found out that a typical click-through rate for the previous ads has been around 15% recently, with results varying between 5% and 30%. Based on this, you conclude that Beta(10,50) would be a good prior for the click-through rate.

# Generate prior draws
prior_draws = np.random.beta(10, 50, 100000)

# Plot the prior
sns.kdeplot(prior_draws, shade=True, label="prior")
plt.show()

# Extract the banner_clicked column for each product
clothes_clicked = ads.loc[ads["product"] == "clothes"]["banner_clicked"]
sneakers_clicked = ads.loc[ads["product"] == "sneakers"]["banner_clicked"]

# Simulate posterior draws for each product
clothes_posterior = simulate_beta_posterior(clothes_clicked, 10, 50)
sneakers_posterior = simulate_beta_posterior(sneakers_clicked, 10, 50)

sns.kdeplot(clothes_posterior, shade=True, label="clothes")
sns.kdeplot(sneakers_posterior, shade=True, label="sneakers")
plt.show()

##--------------------------------------------------------------------------------------
# You have just discovered that clothes ads are likely to have a higher click ratio than sneakers ads. But what is the exact probability that this is the case? To find out, you will have to calculate the posterior difference between clothes and sneakers click rates. Then, you will calculate a credible interval for the difference to measure the uncertainty in the estimate. Finally, you will calculate the percentage of cases where this difference is positive, which corresponds to clothes click rate being higher. Let's get on with it!

# Calculate posterior difference and plot it
diff = clothes_posterior - sneakers_posterior
sns.kdeplot(diff, shade=True, label="diff")
plt.show()

# Calculate and print 90% credible interval of posterior difference
interval = az.hdi(diff, hdi_prob=0.9)
print(interval)

# Calculate and print probability of clothes ad being better
clothes_better_prob = (diff > 0).mean()
print(clothes_better_prob)

# Well done! Take a look at the posterior density plot of the difference in click rates: it is very likely positive, indicating that clothes are likely better. The credible interaval indicates that with 90% probability, the clothes ads click rate is up to 2.4 percentage points higher than the one for sneakers. Finally, the probability that the clothes click rate is higher is 98%. Great! But there is a 2% chance that actually sneakers ads are better! How great is that risk? Let's find out!

##--------------------------------------------------------------------------------------
# You have concluded that with 98% probability, clothes ads have a higher click-through ratio than sneakers ads. This suggests rolling out the clothes campaign to a larger audience. However, there is a 2% risk that it's the sneakers ads that are actually better. If that's the case, how many clicks do we lose if we roll out the clothes campaign?

# The answer to this is the expected loss: the average posterior difference between the two click-through ratios given that sneakers ads do better. To calculate it, you only need to take the entries in the posterior difference where the sneakers click-through rate is higher and compute their average.

# Slice diff to take only cases where it is negative
loss = diff[diff < 0]

# Compute and print expected loss
expected_loss = loss.mean()
print(expected_loss)

# Terrific job! You can sefely roll out the clothes campaign to a larger audience. You are 98% sure it has a higher click rare, and even if the 2% risk of this being a wrong decision materializes, you will only lose 0.2 percentage points in the click rate, which is a very small risk!

#####################################################################################
'''
Decision Analysis
'''
# Your journey in marketing continues. You have already calculated the posterior click rates for clothes and sneakers ads, available in your workspace as clothes_posterior and sneakers_posteriors, respectively. Your boss, however, is not interested in the distributions of click rates. They would like to know what would be the cost of rolling out an ad campaign to 10'000 users. The company's advertising partner charges $2.5 per click on a mobile device and $2 on a desktop device. Your boss is interested in the cost of the campaign for each product (clothes and sneakers) on each platform (mobile and desktop): four quantities in total.

# Let's compare these four posterior costs using the forest plot from pymc3, which has been imported for you as pm.

# Calculate distributions of the numbers of clicks for clothes and sneakers
clothes_num_clicks = clothes_posterior * 10_000
sneakers_num_clicks = sneakers_posterior * 10_000

# Calculate cost distributions for each product and platform
ads_costs = {
    "clothes_mobile": clothes_num_clicks * 2.5,
    "sneakers_mobile": sneakers_num_clicks * 2.5,
    "clothes_desktop": clothes_num_clicks * 2,
    "sneakers_desktop": sneakers_num_clicks * 2,
}

# Draw a forest plot of ads_costs
pm.forestplot(ads_costs, hdi_prob=0.99, textsize=15)
plt.show()

# Yup, that's false! The ends of the whiskers mark the 99% credible interval, so there is a 1% chance the cost will fall outside of it. It's very, very unlikely, but there is a slim chance that the clothes-mobile cost will turn out lower. It's important to stay cautious when communicating possible scenarios -- that's the thing with probability, it's rarely the case that something is 'completely impossible'!

##--------------------------------------------------------------------------------------
# Good job translating the posterior click rates into cost distributions! In the meantime, a new company policy has been released. From now on, the goal of the marketing department is not to minimize the costs of campaigns, which was quite ineffective, but rather to maximize the profit. Can you adjust your findings accordingly, knowing that the expected revenue per click from a mobile ad is $3.4, and the one from a desktop ad is $3? To calculate the profit, you need to calculate the revenue from all clicks, then subtract the corresponding cost from it.

# Calculate profit distributions for each product and platform
ads_profit = {
    "clothes_mobile": clothes_num_clicks * 3.4 - ads_costs["clothes_mobile"],
    "sneakers_mobile": sneakers_num_clicks * 3.4 - ads_costs["sneakers_mobile"],
    "clothes_desktop": clothes_num_clicks * 3 - ads_costs["clothes_desktop"],
    "sneakers_desktop": sneakers_num_clicks * 3 - ads_costs["sneakers_desktop"],
}

# Draw a forest plot of ads_profit
pm.forestplot(ads_profit, hdi_prob=0.99)
plt.show()

# Well done! Notice how shifting focus from costs to profit has changed the optimal decision. The sneakers-desktop campaign which minimizes the cost is not the best choice when you care about the profit. Based on these results, you would be more likely to invest in the clothes-desktop campaign, wouldn't you? Let's continue to the final lesson of this chapter, where we look at regression and forecasting, the Bayesian way!

#####################################################################################
'''
Regression and Forecasting
'''
# Your linear regression model has four parameters: the intercept, the impact of clothes ads, the impact of sneakers ads, and the variance. The draws from their respective posterior distributions have been sampled for you and are available as intercept_draws, clothes_draws, sneakers_draws, and sd_draws, respectively.

# Before you make predictions with your model, it's a good practice to analyze the posterior draws visually. In this exercise, you will first take a look at the descriptive statistics for each parameter's draws, and then you will visualize the posterior distribution for one of them as an example.

# Collect parameter draws in a DataFrame
posterior_draws_df = pd.DataFrame({
    "intercept_draws": intercept_draws,
    "clothes_draws": clothes_draws,
  	"sneakers_draws": sneakers_draws,
    "sd_draws": sd_draws,
})

# Describe parameter posteriors
draws_stats = posterior_draws_df.describe()
print(draws_stats)

# Plot clothes parameter posterior
pm.plot_posterior(clothes_draws, hdi_prob=0.99)
plt.show()

# Well done! Take a look at the output in the console and at the plot. The impact parameters of both clothes and sneakers look okay: they are positive, most likely around 0.1, indicating 1 additional click from 10 ad impressions, which makes sense. Let's now use the model to make predictions!

##--------------------------------------------------------------------------------------
# Good job analyzing the parameter draws! Let's now use the linear regression model to make predictions. How many clicks can we expect if we decide to show 10 clothes ads and 10 sneaker ads? To find out, you will have to draw from the predictive distribution: a normal distribution with the mean defined by the linear regression formula and standard deviation estimated by the model.

# First, you will summarize each parameter's posterior with its mean. Then, you will calculate the mean of the predictive distribution according to the regression equation. Next, you will draw a sample from the predictive distribution and finally, you will plot its density. Here is the regression formula for your convenience:

# numClicks ~ N(beta0 + beta1*clotheAdsShown + beta2*sneakerAdsShown, sigma)

# Aggregate posteriors of the parameters to point estimates
intercept_coef = np.mean(intercept_draws)
sneakers_coef = np.mean(sneakers_draws)
clothes_coef = np.mean(clothes_draws)
sd_coef = np.mean(sd_draws)

# Calculate the mean of the predictive distribution
pred_mean = intercept_coef + sneakers_coef * 10 + clothes_coef * 10

# Sample 1000 draws from the predictive distribution
pred_draws = np.random.normal(pred_mean, sd_coef, size=1000)

# Plot the density of the predictive distribution
pm.plot_posterior(pred_draws, hdi_prob=0.99)
plt.show()

# Great job! It looks like you can expect more or less three or four clicks if you show 10 clothes and 10 sneaker ads. Head off to the final chapter of the course where you will be using the pymc3 package to carry out a full-fledged Bayesian linear regression analysis - see you there!

#####################################################################################
#####################################################################################
'''
Bayesian Linear Regression

Markov Monte Carlo
- Grid approximation: Inconvenient with many parameters
- Sampling from known posterior, requires conjugate pairs
- Markov Chain Monte Carlo (MCMC): sampling from an unknown posterior!

Markov Chains: 
- Models a sequence of states, between which one transitions with given probabilities
- After many time periods, transition probabilities become the same no matter where they started
'''
'''
Markov Chain Monte Carlo, or MCMC, combines the concepts of Monte Carlo sampling with Markov Chains' property of converging to a steady state. This allows sampling draws from any, even unknown, posterior distribution. Let's check your intuition about MCMC!

True:
- Using MCMC to get posterior draws will give the same results as using grid approximation or sampling from the posterior directly (if it's known)
- Numbers generated by MCMC become draws from the posterior after many samples thanks to Markov Chains' convergence property
- MC is a way to approximate some quantity by generating random numbers

False:
- You can use all the draws generated in the MCMC process as draws from the parameter's posterior distribution
- MCMC can only be used with the right conjugate priors
'''
# Tired of working for the central government and for the marketing company, you take a new job as a data analyst for your city's local authorities. The city operates a bike-sharing system in the city and they ask you to predict the number of bikes rented per day to plan staff and repairs accordingly.

# You have been given some data on the number of rented vehicles per day, temperature, humidity, wind speed, and whether the day was a working day:

# Try building a regression model to predict num_bikes using the bikes DataFrame and pymc3 (aliased as pm).

formula = "num_bikes ~ temp + work_day"

with pm.Model() as model_1:
    pm.GLM.from_formula(formula, data=bikes)
    trace_1 = pm.sample(draws=1000, tune=500)

#####################################################################################
'''
Interpreting Results and Comparing Models


WAIC: Widely Applicable Information Criterion
'''
# You continue working on your task to predict the number of bikes rented per day in a bike-sharing system. The posterior draws from your regression model which you sampled before are available in your workspace as trace_1.

# You know that after obtaining the posteriors, it is best practice to take a look at them to see if they make sense and if the MCMC process has converged successfully. In this exercise, you will create two plots visualizing posterior draws and summarize them in a table. Let's inspect our posteriors!

# Import pymc3
import pymc3 as pm

# Draw a trace plot of trace_1
pm.traceplot(trace_1)
plt.show()

# Draw a forest plot of trace_1
pm.forestplot(trace_1)
plt.show()

##--------------------------------------------------------------------------------------
# Now that you have successfully built the first, basic model, you take another look at the data at your disposal. You notice a variable called wind_speed. This could be a great predictor of the numbers of bikes rented! Cycling against the wind is not that much fun, is it?

# You fit another model with this additional predictor:

# formula = "num_bikes ~ temp + work_day + wind_speed"

with pm.Model() as model_2:
    pm.GLM.from_formula(formula, data=bikes)
    trace_2 = pm.sample(draws=1000, tune=500)
# Is your new model_2 better than model_1, the one without wind speed? Compare the two models using Widely Applicable Information Criterion, or WAIC, to find out!

# Gather trace_1 and trace_2 into a dictionary
traces_dict = {"trace_1": trace_1, "trace_2": trace_2}

# Create a comparison table based on WAIC
comparison = pm.compare(traces_dict, ic="waic")

# Draw a comparison plot
pm.compareplot(comparison, textsize=20)
plt.show()

#####################################################################################
'''
Making Predictions
'''
# Finally! Your job is to predict the number of bikes rented per day, and you are almost there. You have fitted the model and verified the quality of parameter draws. You have also chosen the better of the two competing models based on the WAIC. Now, it's time to use your best model to make predictions!

# A couple of new observations, not seen by the model, have been collected in a DataFrame named bikes_test. For each of them, we know the true number of bikes rented, which will allow us to evaluate model performance. In this exercise, you will get familiar with the test data and generate predictive draws for every test observation. The trace of your model which you have generated before is available as trace_2, and pymc3 has been imported as pm. Let's make predictions!

# Print bikes_test head
print(bikes_test.head())

# Define the formula
formula = "num_bikes ~ temp + work_day + wind_speed"

# Generate predictive draws
with pm.Model() as model:
    pm.GLM.from_formula(formula, data=bikes_test)
    posterior_predictive = pm.fast_sample_posterior_predictive(trace_2)

##--------------------------------------------------------------------------------------
# Now that you have your posterior_predictive (available to you in your workspace), you can evaluate model performance on new data. To do this, you will need to loop over the test observations, and for each of them, compute the prediction error as the difference between the predictive distribution for this observation and the actual, true value. This will give you the distribution of your model's error, which you can then visualize.

# Initialize errors
errors = []

# Iterate over rows of bikes_test to compute error per row
for index, test_example in bikes_test.iterrows():
    error = posterior_predictive['y'][:, index] - test_example['num_bikes']
    errors.append(error)

# Reshape errors
error_distribution = np.array(errors).reshape(-1)

# Plot the error distribution
pm.plot_posterior(error_distribution)
plt.show()

# Outstanding job! This was a tough one! In practice, you might want to compute the error estimate based on more than just 10 observations, but you can already see some patterns. For example, the error is more often positive than negative, which means that the model tends to overpredict the number of bikes rented!

#####################################################################################
'''
How Much is An Avocado?
'''
# You can use a linear regression model to estimate the avocado price elasticity. The regression formula should be:

# volume ~ N(beta0 + beta1*price + beta2*typeOrganic, sigma)

# Here, beta1 will be the price elasticity, that is the impact of price on sales. You will assume that the elasticity is the same for regular and organic avocados. You also expect it to be negative: the higher the price, the lower the sales, that's the case for most goods. To incorporate this prior knowledge into the model, you decide to use a normal distribution with mean -80 as the prior for price. How would you build such a model?

formula = 'volume ~ price + type_organic'
with pm.Model() as model:
    priors = {'price': pm.Normal.dist(mu=-80)}
    pm.GLM.from_formula(formula, data=avocado, priors=priors)
    trace = pm.Sample(draws=1000, tune=500)

##--------------------------------------------------------------------------------------
# Well done getting the model-building right! The trace is available in your workspace and, following the best practices, you will now inspect the posterior draws to see if there are any convergence issues. Next, you will extract each model parameter from the trace and summarize it with its posterior mean. These posterior means will come in handy later, when you will be making predictions with the model. Let's take a look at the parameter draws!

# Draw a trace plot of trace
pm.traceplot(trace)
plt.show()

# Print a summary of trace
summary = pm.summary(trace)
print(summary)

# Get each parameter's posterior mean
intercept_mean = np.mean(trace.get_values("Intercept")) 
organic_mean = np.mean(trace.get_values("type_organic")) 
price_mean = np.mean(trace.get_values("price")) 
sd_mean = np.mean(trace.get_values("sd")) 

# Well done! Have you noticed something unusual when it comes it MCMC convergence? Look at the left part of the trace plot for price: the density of one of the chains is slightly wobbly. Luckily, it's only one chain and its density is still quite close to the densities of other chains. So, all in all, we don't need to worry about it and we can safely use the model to optimize the price!

##--------------------------------------------------------------------------------------
# Great job on fitting and inspecting the model! Now, down to business: your boss asks you to provide the avocado price that would yield the largest profit, and to state what profit can be expected. Also, they want the price to be divisible by $0.25 so that the customers can easily pay with quarters.

# In this exercise, you will use your model to predict the volume and the profit for a couple of sensible prices. Next, you will visualize the predictive distributions to pick the optimal price. Finally, you will compute the credible interval for your profit prediction. Now go and optimize!

# For each price, predict volume and use it to predict profit
predicted_profit_per_price = {}
for price in [0.5, 0.75, 1, 1.25]:
    pred_mean = (intercept_mean + price_mean * price + organic_mean)
    volume_pred = np.random.normal(pred_mean, sd_mean, size=1000)
    profit_pred = price * volume_pred
    predicted_profit_per_price.update({price: profit_pred})
    
# Draw a forest plot of predicted profit for all prices
pm.forestplot(predicted_profit_per_price)
plt.show()

# Calculate and print HPD of predicted profit for the optimal price
opt_hpd = az.hdi(predicted_profit_per_price[0.75], credible_interval=0.99)
print(opt_hpd)

# Terrfic work! With a higher or lower price, your company would lose profit, but thanks to your modeling skills, they were able to set the best possible price. More than that, knowing the uncertainty in the profit prediction, they can prepare for the worst-case scenario (in which the profit is negative)!

# Congratulations! You did it! You’re now able to perform A/B testing, decision analysis, and regression modeling the Bayesian way in Python.