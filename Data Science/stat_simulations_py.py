'''
Intro to Random Variables

Continuous random variables
-Infinitely many possible values
-Height, weight

Discrete random variables
-Finite set of possible values
-Dice roll

Probability distributions
How likely you are to observe a given outcome out of a set of outcomes

Continuous: Probability Density Function PDF
Discrete: Probability Mass Function PMF
'''
np.random.choice()
#generates a random sample from a 1d array

###############################################################################
# Initialize seed and parameters
np.random.seed(123)
lam, size_1, size_2 = 5, 3, 1000

# Draw samples & calculate absolute difference between lambda and sample mean
samples_1 = np.random.poisson(lam, size_1)
samples_2 = np.random.poisson(lam, size_2)
answer_1 = abs(np.mean(samples_1) - lam)
answer_2 = abs(np.mean(samples_2) - lam)

print("|Lambda - sample mean| with {} samples is {} and with {} samples
is {}. ".format(size_1, answer_1, size_2, answer_2))

###############################################################################
# Shuffle the deck
np.random.shuffle(deck_of_cards)

# Print out the top three cards
card_choices_after_shuffle = deck_of_cards[0:3]
print(card_choices_after_shuffle)

###############################################################################
'''
Simulation Basics

Characterized by repeated sampling
Gives an approximate solution

1. Define possible outcomes for random variables
2. Assign probabilities to each outcome (prob dist)
3. Define relationship between random variables
4. Get multiple outcomes by repeated sampling
5. Analyze outcomes
'''
# Define die outcomes and probabilities
die, probabilities, throws = [1,2,3,4,5,6], [1/6,1/6,1/6,1/6,1/6,1/6], 1

# Use np.random.choice to throw the die once and record the outcome
outcome = np.random.choice(die, size=1, p=probabilities)
print("Outcome of the throw: {}".format(outcome[0]))

###############################################################################
# Initialize number of dice, simulate & record outcome
die, probabilities, num_dice = [1,2,3,4,5,6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], 2
outcomes = np.random.choice(die, size=num_dice, p=probabilities)

# Win if the two dice show the same number
if outcomes[0] == outcomes[1]:
    answer = 'win'
else:
    answer = 'lose'

print("The dice show {} and {}. You {}!".format(outcomes[0], outcomes[1], answer))

###############################################################################
# Initialize model parameters & simulate dice throw
die, probabilities, num_dice = [1,2,3,4,5,6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], 2
sims, wins = 100, 0

for i in range(sims):
    outcomes = np.random.choice(die,p=probabilities,size=num_dice)
    # Increment `wins` by 1 if the dice show same number
    if outcomes[0]==outcomes[1]:
        wins = wins + 1

print("In {} games, you win {} times".format(sims, wins))

###############################################################################
'''
Using simulation for decision making
'''
# Pre-defined constant variables
lottery_ticket_cost, num_tickets, grand_prize = 10, 1000, 10000

# Probability of winning
chance_of_winning = 1/num_tickets

# Simulate a single drawing of the lottery
gains = [-lottery_ticket_cost, grand_prize-lottery_ticket_cost]
probability = [1-chance_of_winning, chance_of_winning]
outcome = np.random.choice(a=gains, size=1, p=probability, replace=True)

print("Outcome of one drawing of the lottery is {}".format(outcome))

###############################################################################
# Initialize size and simulate outcome
lottery_ticket_cost, num_tickets, grand_prize = 10, 1000, 10000
chance_of_winning = 1/num_tickets
size = 2000
payoffs = [-lottery_ticket_cost,grand_prize-lottery_ticket_cost]
probs = [1-chance_of_winning,chance_of_winning]

outcomes = np.random.choice(a=payoffs, size=size, p=probs, replace=True)

# Mean of outcomes.
answer = np.mean(outcomes)
print("Average payoff from {} simulations = {}".format(size, answer))

###############################################################################
# Initialize simulations and cost of ticket
sims, lottery_ticket_cost = 3000, 0

# Use a while loop to increment `lottery_ticket_cost` till average value of outcomes falls below zero
while 1:
    outcomes = np.random.choice([-lottery_ticket_cost, grand_prize-lottery_ticket_cost],
                 size=sims, p=[1-chance_of_winning, chance_of_winning], replace=True)
    if outcomes.mean() < 0:
        break
    else:
        lottery_ticket_cost += 1
answer = lottery_ticket_cost - 1

print("The highest price at which it makes sense to buy the ticket
is {}".format(answer))

###############################################################################
###############################################################################
'''
Probability basics

Sample space: Set of all possible outcomes
Probability: Likelihood of event A
    0 <= P(A) <= 1
    Mutually exclusive events are events that connot occur simultaneously.

Steps for estimating probability
1. Construct sample space or population
2. Determine how to simulate one outcome
3. Determine rule for success
4. Sample repeatedly and count successes
5. Calculate frequency of successes as an estimate of probability
'''
# Shuffle deck & count card occurrences in the hand
n_sims, two_kind = 10000, 0
for i in range(n_sims):
    np.random.shuffle(deck_of_cards)
    hand, cards_in_hand = deck_of_cards[0:5], {}
    for [suite, numeric_value] in hand:
        # Count occurrences of each numeric value
        cards_in_hand[numeric_value] = cards_in_hand.get(numeric_value, 0) + 1

    # Condition for getting at least 2 of a kind
    if max(cards_in_hand.values()) >=2:
        two_kind += 1

print("Probability of seeing at least two of a kind = {} ".format(two_kind/n_sims))

###############################################################################
# Pre-set constant variables
deck, sims, coincidences = np.arange(1, 14), 10000, 0

for _ in range(sims):
    # Draw all the cards without replacement to simulate one game
    draw = np.random.choice(deck, size=13, replace=False)
    # Check if there are any coincidences
    coincidence = (draw == list(np.arange(1, 14))).any()
    if coincidence == True:
        coincidences += 1

# Calculate probability of winning
prob_of_winning = (sims-coincidences)/sims
print("Probability of winning = {}".format(prob_of_winning))

###############################################################################
'''
Conditional Probability:
P(A|B) = Probability of A given that B has already occurred

Bayes' Rule: P(A|B) = P(B|A)P(A)/P(B)

Independent Events: P(A intersection B) = P(A)P(B)
'''
# Initialize success, sims and urn
success, sims = 0, 5000
urn = ['w','w','w','w','w','w','w','b','b','b','b','b','b']

for _ in range(sims):
    # Draw 4 balls without replacement
    draw = np.random.choice(urn, replace=False, size=4)
    # Count the number of successes
    if np.array([k==v for k,v in zip(draw,['w','b','w','b'])]).all():
        success +=1

print("Probability of success = {}".format(success/sims))

###############################################################################
# Draw a sample of birthdays & check if each birthday is unique
days = np.arange(1,366,1)
people = 2

def birthday_sim(people):
    sims, unique_birthdays = 2000, 0
    for _ in range(sims):
        draw = np.random.choice(days, size=people, replace=True)
        if len(draw) == len(set(draw)):
            unique_birthdays += 1
    out = 1 - unique_birthdays / sims
    return out

# Break out of the loop if probability greater than 0.5
while (people > 0):
    prop_bds = birthday_sim(people)
    if prop_bds > 0.5:
        break
    people += 1

print("With {} people, there's a 50% chance that two share a birthday.".format(people))

###############################################################################
#Shuffle deck & count card occurrences in the hand
n_sims, full_house, deck_of_cards = 50000, 0, deck.copy()
for i in range(n_sims):
    np.random.shuffle(deck_of_cards)
    hand, cards_in_hand = deck_of_cards[0:5], {}
    for card in hand:
        # Use .get() method to count occurrences of each card
        cards_in_hand[card[1]] = cards_in_hand.get(card[1], 0) + 1

    # Condition for getting full house
    condition = (max(cards_in_hand.values()) ==3) & (min(cards_in_hand.values())==2)
    if condition:
        full_house += 1
print("Probability of seeing a full house = {}".format(full_house/n_sims))

###############################################################################
'''
Data Generating Process

1. Factors influencing data
2. Sources of uncertainty
3. Relationship
'''
sims, outcomes, p_rain, p_pass = 1000, [], 0.40, {'sun':0.9, 'rain':0.3}

def test_outcome(p_rain):
    # Simulate whether it will rain or not
    weather = np.random.choice(['rain', 'sun'], p=[p_rain,1-p_rain])
    # Simulate and return whether you will pass or fail
    test_result = np.random.choice(['pass', 'fail'], p=[p_pass[weather],1-p_pass[weather]])
    return test_result

for _ in range(sims):
    outcomes.append(test_outcome(p_rain))

# Calculate fraction of outcomes where you pass
pass_outcomes_frac = sum([outcome == 'pass' for outcome in outcomes])/len(outcomes)
print("Probability of Passing the driving test = {}".format(pass_outcomes_frac))

outcomes, sims, probs = [], 1000, p

for _ in range(sims):
    # Simulate elections in the 50 states
    election = np.random.binomial(p=p,n=1)
    # Get average of Red wins and add to `outcomes`
    outcomes.append(np.mean(election))

# Calculate probability of Red winning in less than 45% of the states
prob_red_wins = sum([outcome < 0.45 for outcome in outcomes])/len(outcomes)
print("Probability of Red winning in less than 45% of the states = {}".format(prob_red_wins))

###############################################################################
# Simulate steps & choose prob
for _ in range(sims):
    w = []
    for i in range(days):
        lam = np.random.choice([5000, 15000], p=[0.6, 0.4], size=1)
        steps = np.random.poisson(lam)
        if steps > 10000:
            prob = [0.2,0.8]
        elif steps < 8000:
            prob = [0.8,0.2]
        else:
            prob = [0.5, 0.5]
        w.append(np.random.choice([1, -1], p=prob))
    outcomes.append(sum(w))

# Calculate fraction of outcomes where there was a weight loss
weight_loss_outcomes_frac = sum([outcome < 0 for outcome in outcomes])/len(outcomes)
print("Probability of Weight Loss = {}".format(weight_loss_outcomes_frac))

###############################################################################
'''
Ad Simulation

We will now model the DGP of an eCommerce ad flow starting with sign-ups.

On any day, we get many ad impressions, which can be modeled as Poisson random
variables (RV). You are told that  is normally distributed with a mean of 100k
visitors and standard deviation 2000.

During the signup journey, the customer sees an ad, decides whether or not to
click, and then whether or not to signup. Thus both clicks and signups are
binary, modeled using binomial RVs. What about probability  of success? Our
current low-cost option gives us a click-through rate of 1% and a sign-up rate
of 20%. A higher cost option could increase the clickthrough and signup rate by
up to 20%, but we are unsure of the level of improvement, so we model it as a
uniform RV.
'''
# Initialize click-through rate and signup rate dictionaries
ct_rate = {'low':0.01, 'high':np.random.uniform(low=0.01, high=1.2*0.01)}
su_rate = {'low':0.2, 'high':np.random.uniform(low=0.2, high=1.2*0.2)}

def get_signups(cost, ct_rate, su_rate, sims):
    lam = np.random.normal(loc=100000, scale=2000, size=sims)
    # Simulate impressions(poisson), clicks(binomial) and signups(binomial)
    impressions = np.random.poisson(lam)
    clicks = np.random.binomial(n=impressions,p=ct_rate[cost])
    signups = np.random.binomial(n=clicks,p=su_rate[cost])
    return signups

print("Simulated Signups = {}".format(get_signups('high', ct_rate, su_rate, 1)))

'''
After signups, let's model the revenue generation process. Once the customer has
signed up, they decide whether or not to purchase - a natural candidate for a
binomial RV. Let's assume that 10% of signups result in a purchase.

Although customers can make many purchases, let's assume one purchase. The
purchase value could be modeled by any continuous RV, but one nice candidate is
the exponential RV. Suppose we know that purchase value per customer has
averaged around $1000. We use this information to create the purchase_values RV.
The revenue, then, is simply the sum of all purchase values.

The variables ct_rate, su_rate and the function get_signups() from the last
exercise are pre-loaded for you.
'''
def get_revenue(signups):
    rev = []
    np.random.seed(123)
    for s in signups:
        # Model purchases as binomial, purchase_values as exponential
        purchases = np.random.binomial(s, p=0.1)
        purchase_values = np.random.exponential(scale=1000,size=purchases)

        # Append to revenue the sum of all purchase values.
        rev.append(sum(purchase_values))
    return rev

print("Simulated Revenue = ${}".format(get_revenue(get_signups('low', ct_rate, su_rate, 1))[0]))

'''
In this exercise, we will use the DGP model to estimate probability.

As seen earlier, this company has the option of spending extra money, let's say
$3000, to redesign the ad. This could potentially get them higher clickthrough
and signup rates, but this is not guaranteed. We would like to know whether or
not to spend this extra $3000 by calculating the probability of losing money.
In other words, the probability that the revenue from the high-cost option minus
the revenue from the low-cost option is lesser than the cost.

Once we have simulated revenue outcomes, we can ask a rich set of questions that
might not have been accessible using traditional analytical methods.

This simple yet powerful framework forms the basis of Bayesian methods for
getting probabilities.
'''
# Initialize cost_diff
sims, cost_diff = 10000, 3000

# Get revenue when the cost is 'low' and when the cost is 'high'
rev_low = get_revenue(get_signups('low', ct_rate, su_rate, sims))
rev_high = get_revenue(get_signups('high', ct_rate, su_rate, sims))

# calculate fraction of times rev_high - rev_low is less than cost_diff
frac = (np.array(rev_high) - np.array(rev_low) < cost_diff).mean()
print("Probability of losing money = {}".format(frac))

###############################################################################
###############################################################################
'''
Introduction to Resampling Methods

Dataset -> Resample -> New dataset -> Data analysis -> Estimator -> Repeat

Advantages
-Simple implementation procedure
-Applicable to complex estimators
-No strict assumptions regarding the distribution of the data

Drawbacks
-Computationally expensive

1. Bootstrapping: Sampling with replacement (random)
2. Jackknife: Leave out one or more data points (Finds bias and variance of
estimators) Linear approximation of Bootstrapping
3. Permutation testing: Switching labels in the dataset
'''

###############################################################################
'''
Probability example
In this exercise, we will review the difference between sampling with and
without replacement. We will calculate the probability of an event using
simulation, but vary our sampling method to see how it impacts probability.

Consider a bowl filled with colored candies - three blue, two green, and five
yellow. Draw three candies, one at a time, with replacement and without
replacement. You want to calculate the probability that all three candies are
yellow.
'''

# Set up the bowl
success_rep, success_no_rep, sims = 0, 0, 10000
bowl = list('b'*3 + 'g'*2 + 'y'*5)

for i in range(sims):
    # Sample with and without replacement & increment success counters
    sample_rep = np.random.choice(bowl, size=3, replace=True)
    sample_no_rep = np.random.choice(bowl, size=3, replace=False)
    if ('b' not in sample_rep) & ('g' not in sample_rep) :
        success_rep += 1
    if ('b' not in sample_no_rep) & ('g' not in sample_no_rep) :
        success_no_rep += 1

# Calculate probabilities
prob_with_replacement = success_rep/sims
prob_without_replacement = success_no_rep/sims
print("Probability with replacement = {}, without replacement =
{}".format(prob_with_replacement, prob_without_replacement))

###############################################################################
'''
Bootstrapping

-Run at least 5-10k iterations
-Expect an approximate answer
-Consider bias correction: Dispersion-like variables
'''
# Draw some random sample with replacement and append mean to mean_lengths.
mean_lengths, sims = [], 1000
for i in range(sims):
    temp_sample = np.random.choice(wrench_lengths, replace=True, size=len(wrench_lengths))
    sample_mean = temp_sample.mean()
    mean_lengths.append(sample_mean)

# Calculate bootstrapped mean and 95% confidence interval.
boot_mean = np.mean(mean_lengths)
boot_95_ci = np.percentile(mean_lengths, [2.5, 97.5])
print("Bootstrapped Mean Length = {}, 95% CI = {}".format(boot_mean, boot_95_ci))

###############################################################################
# Sample with replacement and calculate quantities of interest
sims, data_size, height_medians, hw_corr = 1000, df.shape[0], [], []
for i in range(sims):
    tmp_df = df.sample(n=data_size, replace=True)
    height_medians.append(tmp_df.heights.median())
    hw_corr.append(tmp_df.corr()['heights']['weights'])

# Calculate confidence intervals
height_median_ci = np.percentile(height_medians,[2.5,97.5])
height_weight_corr_ci = np.percentile(hw_corr,[2.5,97.5])
print("Height Median CI = {} \nHeight Weight Correlation CI = {}".format( height_median_ci, height_weight_corr_ci))

###############################################################################
rsquared_boot, coefs_boot, sims = [], [], 1000
reg_fit = sm.OLS(df['y'], df.iloc[:,1:]).fit()

# Run 1K iterations
for i in range(sims):
    # First create a bootstrap sample with replacement with n=df.shape[0]
    bootstrap = df.sample(n=df.shape[0],replace=True)
    # Fit the regression and append the r square to rsquared_boot
    rsquared_boot.append(sm.OLS(bootstrap['y'],bootstrap.iloc[:,1:]).fit().rsquared)

# Calculate 95% CI on rsquared_boot
r_sq_95_ci = np.percentile(rsquared_boot,[2.5,97.5])
print("R Squared 95% CI = {}".format(r_sq_95_ci))

###############################################################################
'''
Jackknife Resampling: Useful when the data distribution is unknown

Mean of Jackknife: x_{Jackknife} 1/n sum(x)_{i=1}^{n}

Variance of Jackknife: (n-1)/n sum(x_i - x_{Jackknife})^2
'''
# Leave one observation out from wrench_lengths to get the jackknife sample and store the mean length
mean_lengths, n = [], len(wrench_lengths)
index = np.arange(n)

for i in range(n):
    jk_sample = wrench_lengths[index != i]
    mean_lengths.append(jk_sample.mean())

# The jackknife estimate is the mean of the mean lengths from each sample
mean_lengths_jk = np.mean(np.array(mean_lengths))
print("Jackknife estimate of the mean = {}".format(mean_lengths_jk))

###############################################################################
# Leave one observation out to get the jackknife sample and store the median length
median_lengths = []
for i in range(n):
    jk_sample = wrench_lengths[index != i]
    median_lengths.append(np.median(jk_sample))

median_lengths = np.array(median_lengths)

# Calculate jackknife estimate and it's variance
jk_median_length = np.mean(np.array(median_lengths))
jk_var = (n-1)*np.var(median_lengths)

# Assuming normality, calculate lower and upper 95% confidence intervals
jk_lower_ci = jk_median_length - 1.96*np.sqrt(jk_var)
jk_upper_ci = jk_median_length + 1.96*np.sqrt(jk_var)
print("Jackknife 95% CI lower = {}, upper = {}".format(jk_lower_ci, jk_upper_ci))

###############################################################################
'''
Permutation testing: Obtain the distribution of the test statistic of the null
without any underlying assumptions.

1. Determine test statistic, ex difference of means

Advantages: Very flexible, no strict assumptions, widely applicable
Drawbacks: Computationally expensive, custom coding required


'''
# Concatenate the two arrays donations_A and donations_B into data
len_A, len_B = len(donations_A), len(donations_B)
data = np.concatenate([donations_A, donations_B])

# Get a single permutation of the concatenated length
perm = np.random.permutation(len(donations_A) + len(donations_B))

# Calculate the permutated datasets and difference in means
permuted_A = data[perm[:len(donations_A)]]
permuted_B = data[perm[len(donations_A):]]
diff_in_means = permuted_A.mean() - permuted_B.mean()
print("Difference in the permuted mean values = {}.".format(diff_in_means))

###############################################################################
'''
Hypothesis testing - Difference of means
We want to test the hypothesis that there is a difference in the average
donations received from A and B. Previously, you learned how to generate one
permutation of the data. Now, we will generate a null distribution of the
difference in means and then calculate the p-value.

For the null distribution, we first generate multiple permuted datasets and
store the difference in means for each case. We then calculate the test
statistic as the difference in means with the original dataset. Finally, we
approximate the p-value by calculating twice the fraction of cases where the
difference is greater than or equal to the absolute value of the test statistic
(2-sided hypothesis). A p-value of less than say 0.05 could then determine
statistical significance.
'''
# Generate permutations equal to the number of repetitions
perm = np.array([np.random.permutation(len(donations_A) + len(donations_B)) for i in range(reps)])
permuted_A_datasets = data[perm[:, :len(donations_A)]]
permuted_B_datasets = data[perm[:, len(donations_A):]]

# Calculate the difference in means for each of the datasets
samples = np.mean(permuted_A_datasets, axis=1) - np.mean(permuted_B_datasets, axis=1)

# Calculate the test statistic and p-value
test_stat = donations_A.mean() - donations_B.mean()
p_val = 2*np.sum(samples >= np.abs(test_stat))/reps
print("p-value = {}".format(p_val))

###############################################################################
'''
In the previous two exercises, we ran a permutation test for the difference in
mean values. Now let's look at non-standard statistics.

Suppose that you're interested in understanding the distribution of the
donations received from websites A and B. For this, you want to see if there's
a statistically significant difference in the median and the 80th percentile of
the donations. Permutation testing gives you a wonderfully flexible framework
for attacking such problems.

Let's go through running a test to see if there's a difference in the median
and the 80th percentile of the distribution of donations. As before, you're
given the donations from the websites A and B in the variables donations_A and
donations_B respectively.
'''
# Calculate the difference in 80th percentile and median for each of the permuted datasets (A and B)
samples_percentile = np.percentile(permuted_A_datasets, 80, axis=1) - np.percentile(permuted_B_datasets, 80, axis=1)
samples_median = np.median(permuted_A_datasets, axis=1) - np.median(permuted_B_datasets, axis=1)

# Calculate the test statistic from the original dataset and corresponding p-values
test_stat_percentile = np.percentile(donations_A, 80) - np.percentile(donations_B, 80)
test_stat_median = np.median(donations_A) - np.median(donations_B)
p_val_percentile = 2*np.sum(samples_percentile >= np.abs(test_stat_percentile))/reps
p_val_median = 2*np.sum(samples_median >= np.abs(test_stat_median))/reps

print("80th Percentile: test statistic = {}, p-value = {}".format(test_stat_percentile, p_val_percentile))
print("Median: test statistic = {}, p-value = {}".format(test_stat_median, p_val_median))

###############################################################################
###############################################################################
'''
Advanced applications of Simulation

Modeling Corn Production
Suppose that you manage a small corn farm and are interested in optimizing your
costs. In this illustrative exercise, we will model the production of corn.
We'll abstract away from details like units and focus on the process.

For simplicity, let's assume that corn production depends on only two factors:
rain, which you don't control, and cost, which you control. Rain is normally
distributed with mean 50 and standard deviation 15. For now, let's fix cost at
5,000. Let's assume that corn produced in any season is a Poisson random
variable and that the average corn production is governed by the equation:

100 * (cost)^0.1 * (rain)^0.2

Let's model this production function and simulate one outcome.
'''
# Initialize variables
cost = 5000
rain = np.random.normal(50,15)

# Corn Production Model
def corn_produced(rain, cost):
  mean_corn = 100*(cost**0.1)*(rain**0.2)
  corn = np.random.poisson(mean_corn)
  return corn

# Simulate and print corn production
corn_result = corn_produced(rain,cost)
print("Simulated Corn Production = {}".format(corn_result))

'''
In the previous exercise, you built a model of corn production. For a small
farm, you typically have no control over the price or demand for corn. Suppose
that price is normally distributed with mean 40 and standard deviation 10. You
are given a function corn_demanded(), which takes the price and determines the
demand for corn. This is reasonable because demand is usually determined by the
market and is not in your control.

In this exercise, you will work on a function to calculate the profit by
pulling together all the other simulated variables. The only input to this
function will be the fixed cost of production. Upon completion, you'll have a
function that gives one simulated profit outcome for a given cost. This function
can then be used for planning your costs.
'''
# Function to calculate profits
def profits(cost):
    rain = np.random.normal(50, 15)
    price = np.random.normal(40,10)
    supply = corn_produced(rain,cost)
    demand = corn_demanded(price)
    equil_short = supply <= demand
    if equil_short == True:
        tmp = supply*price - cost
        return tmp
    else:
        tmp2 = demand*price - cost
        return tmp2
result = profits(cost)
print("Simulated profit = {}".format(result))

'''
Now we will use the functions you've built to optimize our cost of production.
We are interested in maximizing average profits. However, our profits depend on
a number of factors, while we only control cost. Thus, we can simulate the
uncertainty in the other factors and vary cost to see how our profits are
impacted.

Since you manage the small corn farm, you have the ability to choose your
cost - from $100 to $5,000. You want to choose the cost that gives you the
maximum average profit. In this exercise, we will simulate multiple outcomes for
each cost level and calculate an average. We will then choose the cost that
gives us the maximum mean profit. Upon completion, you will have a framework for
selecting optimal inputs for business decisions.
'''
# Initialize results and cost_levels variables
sims, results = 1000, {}
cost_levels = np.arange(100, 5100, 100)

# For each cost level, simulate profits and store mean profit
for cost in cost_levels:
    tmp_profits = []
    for i in range(sims):
        tmp_profits.append(profits(cost))
    results[cost] = np.mean(tmp_profits)

# Get the cost that maximizes average profit
cost_max = [x for x in results.keys() if results[x] == max(results.values())][0]
print("Average profit is maximized when cost = {}".format(cost_max))

###############################################################################
'''
Monte Carlo Integration

Integration technique using random variables. Useful for when number of
dimensions is large.

1. Calculate overall area by looking at limits of function
2. Randomly sample points in area
3. Multiply the fraction of the points below the curve by overall area.
'''
# Define the sim_integrate function
def sim_integrate(func, xmin, xmax, sims):
    x = np.random.uniform(xmin, xmax, sims)
    y = np.random.uniform(min(min(func(x)),0), max(func(x)), sims)
    area = (max(y) - min(y))*(xmax-xmin)
    result = area * sum(abs(y) < abs(func(x)))/sims
    return result

# Call the sim_integrate function and print results
result = sim_integrate(func = lambda x: x*np.exp(x), xmin = 0, xmax = 1, sims = 50)
print("Simulated answer = {}, Actual Answer = 1".format(result))

###############################################################################
'''
Calculating the value of pi
Now we work through a classic example - estimating the value of pi.

Imagine a square of side 2 with the origin (0,0) as its center and the four corners
having coordinates (1,1),(1,-1),(-1,1),(-1,-1). The area of this square is 2x2=4.
Now imagine a circle of radius 1 with its center at the origin fitting perfectly
inside this square. The area of the circle will be pi x radius^2 = pi.

To estimate pi, we randomly sample multiple points in this square & get the
fraction of points inside the circle (x^2 + y^2 = 1). The area of the circle
then is 4 times this fraction, which gives us our estimate of pi.

After this exercise, you'll have a grasp of how to use simulation for computation.
'''
# Initialize sims and circle_points
sims, circle_points = 10000, 0

for i in range(sims):
    # Generate the two coordinates of a point
    point = np.random.uniform(-1,1,size=2)
    # if the point lies within the unit circle, increment counter
    within_circle = point[0]**2 + point[1]**2 <= 1
    if within_circle == True:
        circle_points +=1

# Estimate pi as 4 times the avg number of points in the circle.
pi_sim = (4 * np.mean(circle_points))/sims
print("Simulated value of pi = {}".format(pi_sim))

###############################################################################
'''
Power Analysis

Power = P(rejecting Null | true alternative)

Probability of detecting an effect if it exists
Depends on sample size alpha, and effect size, typically 80% power for alpha = 0.05
Computed before running a simulation, used to determine sample size
'''

'''
Now we turn to power analysis. You typically want to ensure that any experiment
or A/B test you run has at least 80% power. One way to ensure this is to
calculate the sample size required to achieve 80% power.

Suppose that you are in charge of a news media website and you are interested in
increasing the amount of time users spend on your website. Currently, the time
users spend on your website is normally distributed with a mean of 1 minute and
a standard deviation of 0.5 minutes. Suppose that you are introducing a feature
that loads pages faster and want to know the sample size required to measure a
5% increase in time spent on the website.

In this exercise, we will set up the framework to run one simulation, run a
t-test, & calculate the p-value.
'''
# Initialize effect_size, control_mean, control_sd
effect_size, sample_size, control_mean, control_sd = 0.05, 50, 1, 0.5

# Simulate control_time_spent and treatment_time_spent, assuming equal variance
control_time_spent = np.random.normal(loc=control_mean, scale=control_sd, size=sample_size)
treatment_time_spent = np.random.normal(loc=control_mean*(1+effect_size), scale=control_sd, size=sample_size)

# Run the t-test and get the p_value
t_stat, p_value = st.ttest_ind(treatment_time_spent, control_time_spent)
stat_sig = p_value < 0.05
print("P-value: {}, Statistically Significant? {}".format(p_value, stat_sig))

'''
Previously, we simulated one instance of the experiment & generated a p-value.
We will now use this framework to calculate statistical power. Power of an
experiment is the experiment's ability to detect a difference between treatment
& control if the difference really exists. It's good statistical hygiene to
strive for 80% power.

For our website, suppose we want to know how many people need to visit each
variant, such that we can detect a 10% increase in time spent with 80% power.
For this, we start with a small sample (50), simulate multiple instances of this
experiment & check power. If 80% power is reached, we stop. If not, we increase
the sample size & try again.
'''
sample_size = 50

# Keep incrementing sample size by 10 till we reach required power
while 1:
    control_time_spent = np.random.normal(loc=control_mean, scale=control_sd, size=(sample_size,sims))
    treatment_time_spent = np.random.normal(loc=control_mean*(1+effect_size), scale=control_sd, size=(sample_size,sims))
    t, p = st.ttest_ind(treatment_time_spent, control_time_spent)

    # Power is the fraction of times in the simulation when the p-value was less than 0.05
    power = (p < 0.05).sum()/sims
    if power >= 0.8:
        break
    else:
        sample_size += 10
print("For 80% power, sample size required = {}".format(sample_size))

###############################################################################
'''
Applications in Finance

In the next few exercises, you will calculate the expected returns of a stock portfolio & characterize its uncertainty.

Suppose you have invested $10,000 in your portfolio comprising of multiple stocks. You want to evaluate the portfolio's performance over 10 years. You can tweak your overall expected rate of return and volatility (standard deviation of the rate of return). Assume the rate of return follows a normal distribution.

First, let's write a function that takes the principal (initial investment), number of years, expected rate of return and volatility as inputs and returns the portfolio's total value after 10 years.

Upon completion of this exercise, you will have a function you can call to determine portfolio performance.
'''
# rates is a Normal random variable and has size equal to number of years
def portfolio_return(yrs,avg_return,sd_of_return,principal):
    np.random.seed(123)
    rates = np.random.normal(loc=avg_return, scale=sd_of_return, size=yrs)
    # Calculate the return at the end of the period
    end_return = principal
    for x in rates:
        end_return = end_return*(1+x)
    return end_return

result = portfolio_return(yrs = 5, avg_return = 0.07, sd_of_return = 0.15, principal = 1000)
print("Portfolio return after 5 years = {}".format(result))

'''
Now we will use the simulation function you built to evaluate 10-year returns.

Your stock-heavy portfolio has an initial investment of $10,000, an expected return of 7% and a volatility of 30%. You want to get a 95% confidence interval of what your investment will be worth in 10 years. We will simulate multiple samples of 10-year returns and calculate the confidence intervals on the distribution of returns.

By the end of this exercise, you will have run a complete portfolio simulation.

The function portfolio_return() from the previous exercise is already initialized in the environment.
'''
# Run 1,000 iterations and store the results
sims, rets = 1000, []

for i in range(sims):
    rets.append(portfolio_return(yrs = 10, avg_return = 0.07,
                                 volatility = 0.3, principal = 10000))

# Calculate the 95% CI
lower_ci = np.percentile(rets,2.5)
upper_ci = np.percentile(rets,97.5)
print("95% CI of Returns: Lower = {}, Upper = {}".format(lower_ci, upper_ci))

'''
Previously, we ran a complete simulation to get a distribution for 10-year returns. Now we will use simulation for decision making.

Let's go back to your stock-heavy portfolio with an expected return of 7% and a volatility of 30%. You have the choice of rebalancing your portfolio with some bonds such that the expected return is 4% & volatility is 10%. You have a principal of $10,000. You want to select a strategy based on how much your portfolio will be worth in 10 years. Let's simulate returns for both the portfolios and choose based on the least amount you can expect with 75% probability (25th percentile).

Upon completion, you will know how to use a portfolio simulation for investment decisions.

The portfolio_return() function is again pre-loaded in the environment.
'''
for i in range(sims):
    rets_stock.append(portfolio_return(yrs = 10, avg_return = 0.07, volatility = 0.3, principal = 10000))
    rets_bond.append(portfolio_return(yrs = 10, avg_return = 0.04, volatility = 0.1, principal = 10000))

# Calculate the 25th percentile of the distributions and the amount you'd lose or gain
rets_stock_perc = np.percentile(rets_stock,25)
rets_bond_perc = np.percentile(rets_bond,25)
additional_returns = rets_stock_perc - rets_bond_perc
print("Sticking to stocks gets you an additional return of {}".format(additional_returns))
