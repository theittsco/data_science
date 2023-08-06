'''
Hypothesis Testing in Python

1. Introduction to hypothesis testing 
2. Two-sample and ANOVA tests
3. Proportion tests
4. Non-parametric tests

Hypothesis tests and z-score

Choose a test statistic, determine if result is meaningfully different from estimate/data
Use z-scores:
- Standardized value: (value - mean)/std
- z-score: (sample stat - hypoth.param.value)/standard error

Hypothesis testing:
- Determine whether sample statistics are close to or far away from an expected/hypothesized values
'''
# In the video, you saw how Electronic Arts used A/B testing on their website when launching SimCity 5. One version of the page showed an advertisement for a discount, and one version did not. Half the users saw one version of the page, and the other half saw the second version of the page.

# What is the main reason to use an A/B test?

# Ace A/B testing! A/B testing lets you compare scenarios to see which best achieves some goal.

##---------------------------------------------------------------------------------------------------
# The late_shipments dataset contains supply chain data on the delivery of medical supplies. Each row represents one delivery of a part. The late columns denotes whether or not the part was delivered late. A value of "Yes" means that the part was delivered late, and a value of "No" means the part was delivered on time.

# You'll begin your analysis by calculating a point estimate (or sample statistic), namely the proportion of late shipments.

# Print the late_shipments dataset
print(late_shipments)

# Calculate the proportion of late shipments
late_prop_samp = (late_shipments['late'] == "Yes").mean()

# Print the results
print(late_prop_samp)

##---------------------------------------------------------------------------------------------------
# Since variables have arbitrary ranges and units, we need to standardize them. For example, a hypothesis test that gave different answers if the variables were in Euros instead of US dollars would be of little value. Standardization avoids that.

# One standardized value of interest in a hypothesis test is called a z-score. To calculate it, you need three numbers: the sample statistic (point estimate), the hypothesized statistic, and the standard error of the statistic (estimated from the bootstrap distribution).

# Hypothesize that the proportion is 6%
late_prop_hyp = 0.06

# Calculate the standard error
std_error = np.std(late_shipments_boot_distn)

# Find z-score of late_prop_samp
z_score = (late_prop_samp - late_prop_hyp) / std_error

# Print z_score
print(z_score)

# Zesty z-scoring! The z-score is a standardized measure of the difference between the sample statistic and the hypothesized statistic.

####################################################################################################
'''
p-Values

Hypothesis: A statement about an unknown population parameter

Hypothesis test: A test of two competing hypotheses
- Null hypothesis: H_0, an existing idea
- Alternative hypothesis: H_A, new challenging idea

Initially H_0 is assumed to be true. If the evidence is significant enough that H_A is true, reject H_0, else choose H_0
Significance level is the "beyond a reasonable doubt" for hypothesis testing

Hypothesis tests check if the sample statistics lie in the tails of the null distribution

Three types:
1. Alternative different from null - two tailed
2. Alternative greater than null - right tailed
3. Alternative less than null - left tailed

P-Values: Probability of obtaining a result given the null hypothesis is true
- Large p-value says a statistic does not lie in the tail of the null distribution -> Likely chance
'''
# In the video, you saw how hypothesis testing follows a similar process to criminal trials.

# Which of the following correctly matches up a criminal trial with properties of a hypothesis test?

# I pronounce you not guilty! It's sometimes helpful to think of your hypothesis test as being a trial of the statistic.

##---------------------------------------------------------------------------------------------------
# Top tail choices! The tails of the distribution that are relevant depend on whether the alternative hypothesis refers to "greater than", "less than", or "differences between."

##---------------------------------------------------------------------------------------------------
# In order to determine whether to choose the null hypothesis or the alternative hypothesis, you need to calculate a p-value from the z-score.

# You'll now return to the late shipments dataset and the proportion of late shipments.

# The null hypothesis, H_0, is that the proportion of late shipments is six percent.

# The alternative hypothesis, H_A, is that the proportion of late shipments is greater than six percent.

# Using a RIGHT-TAILED test

# Calculate the z-score of late_prop_samp
z_score = (late_prop_samp - late_prop_hyp) / std_error

# Calculate the p-value
p_value = 1 - norm.cdf(z_score, loc=0, scale=1)
                 
# Print the p-value
print(p_value) 

####################################################################################################
'''
Statistical Significance

p-Values quantify the evidence for the null hypothesis

Common significance levels (alpha): 0.2, 0.1, 0.05, 0.01

For a significance level alpha, it is common to choose a confidence interval 1-alpha

Types of Errors:
Type I: False Positive
Type II: False Negative
'''
# The p-value, denoted here as p, is a measure of the amount of evidence to reject the null hypothesis or not. By comparing the p-value to the significance level, alpha, you can make a decision about which hypothesis to support.

# Which of the following is the correct conclusion from the decision rule for a significance level alpha?

# Delightful decision-making. If the p-value is less than or equal to the significance level, you reject the null hypothesis.

##---------------------------------------------------------------------------------------------------
# If you give a single estimate of a sample statistic, you are bound to be wrong by some amount. For example, the hypothesized proportion of late shipments was 6%. Even if evidence suggests the null hypothesis that the proportion of late shipments is equal to this, for any new sample of shipments, the proportion is likely to be a little different due to sampling variability. Consequently, it's a good idea to state a confidence interval. That is, you say, "we are 95% 'confident' that the proportion of late shipments is between A and B" (for some value of A and B).

# Calculate 95% confidence interval using quantile method
lower = np.quantile(late_shipments_boot_distn,0.025)
upper = np.quantile(late_shipments_boot_distn,0.975)

# Print the confidence interval
print((lower, upper))

# <script.py> output:
    # (0.047, 0.076)

# Does the confidence interval match up with the conclusion to stick with the original assumption that 6% is a reasonable value for the unknown population parameter? YES

# Cool and confident! When you have a confidence interval width equal to one minus the significance level, if the hypothesized population parameter is within the confidence interval, you should fail to reject the null hypothesis.

####################################################################################################
####################################################################################################
'''
Two-Sample Tests and ANOVA

Two-sample problems: Compare sample statistics across a group of variables

Correct order of workflow is:
1. Identify population parameter that is hypothesized about
2. Specify null and alternative hypotheses
3. Determine (standardized) test statistic and corresponding null distribution
4. Conduct hypothesis testing in Python
5. Measure evidence against the null hypothesis
6. Make a decision comparing evidence to significance level
7. Interpret results in context of original problem
'''
# The hypothesis test for determining if there is a difference between the means of two populations uses a different type of test statistic to the z-scores you saw in Chapter 1. It's called "t", and it can be calculated from three values from each sample using this equation.

# While trying to determine why some shipments are late, you may wonder if the weight of the shipments that were on time is less than the weight of the shipments that were late. The late_shipments dataset has been split into a "yes" group, where late == "Yes" and a "no" group where late == "No". The weight of the shipment is given in the weight_kilograms variable.

# Calculate the numerator of the test statistic
numerator = (xbar_no - xbar_yes)

# Calculate the denominator of the test statistic
denominator = np.sqrt(s_no**2/n_no + s_yes**2/n_yes)

# Calculate the test statistic
t_stat = numerator / denominator

# Print the test statistic
print(t_stat)

# t-rrific! When testing for differences between means, the test statistic is called 't' rather than 'z', and can be calculated using six numbers from the samples. Here, the value is about -2.39 or 2.39, depending on the order you calculated the numerator.

####################################################################################################
'''
Calculating p-Values from a t-test

t-Distributions: Have a parameter called 'degrees of freedom, df, dof'

For low dof, the t-dist has fatter tails than the normal dist
- As dof->Inf, the t-dist converges to the normal dist

Degrees of freedom: Maximum number of logically independent values in a data sample

z-statistic: Needed when using one sample statistic to estimate a population parameter
t-statistic: Needed when using multiple sample statistics to estimate a population parameter
'''
# The process for calculating p-values is to start with the sample statistic, standardize it to get a test statistic, then transform it via a cumulative distribution function. In Chapter 1, that final transformation was denoted z, and the CDF transformation used the (standard normal) z-distribution. In the last video, the test statistic was denoted t, and the transformation used the t-distribution.

# In which hypothesis testing scenario is a t-distribution needed instead of the z-distribution?

# Terrific t! Using a sample standard deviation to estimate the standard error is computationally easier than using bootstrapping. However, to correct for the approximation, you need to use a t-distribution when transforming the test statistic to get the p-value.

##---------------------------------------------------------------------------------------------------
# Previously, you calculated the test statistic for the two-sample problem of whether the mean weight of shipments is smaller for shipments that weren't late (late == "No") compared to shipments that were late (late == "Yes"). In order to make decisions about it, you need to transform the test statistic with a cumulative distribution function to get a p-value.

# Recall the hypotheses:

# H_0: The mean weight of shipments that weren't late is the same as the mean weight of shipments that were late. 

# H_A: The mean weight of shipments that weren't late is less than the mean weight of shipments that were late.

# We need a LEFT-tailed hypothesis

# Calculate the degrees of freedom
degrees_of_freedom = n_no + n_yes - 2

# Calculate the p-value from the test stat
p_value = t.cdf(t_stat, df=degrees_of_freedom)

# Print the p_value
print(p_value)

# <script.py> output:
    # 0.008432382146249523

# What decision should you make based on the results of the hypothesis test? Reject the null hypothesis!

# Perspicacious p-value predictions! When the standard error is estimated from the sample standard deviation and sample size, the test statistic is transformed into a p-value using the t-distribution.

####################################################################################################
'''
Paired t-Test

Use pingouin package for paired t-tests
'''
# t-tests are used to compare two sample means. However, the test involves different calculations depending upon whether the two samples are paired or not. To make sure you use the correct version of the t-test, you need to be able to identify pairing.

# Daring pairing! If you have repeated observations of something, then those observations form pairs.

##---------------------------------------------------------------------------------------------------
# Before you start running hypothesis tests, it's a great idea to perform some exploratory data analysis; that is, calculating summary statistics and visualizing distributions.

# Here, you'll look at the proportion of county-level votes for the Democratic candidate in 2012 and 2016, sample_dem_data. Since the counties are the same in both years, these samples are paired. The columns containing the samples are dem_percent_12 and dem_percent_16.

# Calculate the differences from 2012 to 2016
sample_dem_data['diff'] = sample_dem_data['dem_percent_12'] - sample_dem_data['dem_percent_16']

# Find the mean of the diff column
xbar_diff = sample_dem_data['diff'].mean()

# Find the standard deviation of the diff column
s_diff = sample_dem_data['diff'].std()

# Plot a histogram of diff with 20 bins
sample_dem_data['diff'].hist(bins=20)
plt.show()

# Delightful difference discovery! Notice that the majority of the histogram lies to the right of zero.

##---------------------------------------------------------------------------------------------------
# Manually calculating test statistics and transforming them with a CDF to get a p-value is a lot of effort to compare two sample means. The comparison of two sample means is called a t-test, and the pingouin Python package has a .ttest() method to accomplish it. This method provides some flexibility in how you perform the test.

# As in the previous exercise, you'll explore the difference between the proportion of county-level votes for the Democratic candidate in 2012 and 2016 to identify if the difference is significant. The hypotheses are as follows:

# H_0: The proportion of democratic votes in 2012 and 2016 were the same. 
# H_A: The proportion of democratic votes in 2012 and 2016 were different.

# Conduct a t-test on diff
test_results = pingouin.ttest(x=sample_dem_data['diff'],
                              y=0,
                              alternative='two-sided')
                         
# Print the test results
print(test_results)

# p-Value was near zero. Reject the null hypothesis!

# Conduct a t-test on diff
test_results = pingouin.ttest(x=sample_dem_data['diff'], 
                              y=0, 
                              alternative="two-sided")

# Conduct a paired t-test on dem_percent_12 and dem_percent_16
paired_test_results = pingouin.ttest(x=sample_dem_data['dem_percent_12'],
                                     y=sample_dem_data['dem_percent_16'],
                                     paired=True,
                                     alternative='two-sided')
                
# Print the paired test results
print(paired_test_results)

# Paired t-test party! Using .ttest() lets you avoid manual calculation to run your test. When you have paired data, a paired t-test is preferable to the unpaired version because it reduces the chance of a false negative error.

####################################################################################################
'''
ANOVA Tests

What happens if there are more than two variables?
ANOVA tests for differences between groups

Usually has larger significance, e.g. alpha=0.2

Will tell if any of the groups have a significant difference, but not which one
Need to run pairwise t-tests
More tests run, higher chance of a false positive
Use Bonferroni correction to correct p-value
'''
# So far in this chapter, we've only considered the case of differences in a numeric variable between two categories. Of course, many datasets contain more categories. Before you get to conducting tests on many categories, it's often helpful to perform exploratory data analysis (EDA), calculating summary statistics for each group and visualizing the distributions of the numeric variable for each category using box plots.

# Here, we'll return to the late shipments data, and how the price of each package (pack_price) varies between the three shipment modes (shipment_mode): "Air", "Air Charter", and "Ocean".

# Calculate the mean pack_price for each shipment_mode
xbar_pack_by_mode = late_shipments.groupby("shipment_mode")['pack_price'].mean()

# Calculate the standard deviation of the pack_price for each shipment_mode
s_pack_by_mode = late_shipments.groupby("shipment_mode")['pack_price'].std()

# Boxplot of shipment_mode vs. pack_price
sns.boxplot(x='pack_price',y='shipment_mode',data=late_shipments)
plt.show()

# Beautiful boxplotting! There certainly looks to be a difference in the pack price between each of the three shipment modes. Do you think the differences are statistically significant?

##---------------------------------------------------------------------------------------------------
# The box plots made it look like the distribution of pack price was different for each of the three shipment modes. However, it didn't tell us whether the mean pack price was different in each category. To determine that, we can use an ANOVA test. The null and alternative hypotheses can be written as follows.

# H_0: Pack prices for every category of shipment mode are the same.

# H_A: Pack prices for some categories of shipment mode are different.

# Use a significance level of 0.1.

# Run an ANOVA for pack_price across shipment_mode
anova_results = pingouin.anova(data=late_shipments,
                               dv='pack_price',
                               between='shipment_mode')



# Print anova_results
print(anova_results)

# Amazing ANOVA! There is a significant difference in pack prices between the shipment modes. However, we don't know which shipment modes this applies to.

##---------------------------------------------------------------------------------------------------
# The ANOVA test didn't tell you which categories of shipment mode had significant differences in pack prices. To pinpoint which categories had differences, you could instead use pairwise t-tests.

# Modify the pairwise t-tests to use Bonferroni p-value adjustment
pairwise_results = pingouin.pairwise_tests(data=late_shipments, 
                                           dv="pack_price",
                                           between="shipment_mode",
                                           padjust="bonf")

# Print pairwise_results
print(pairwise_results)

# Pairwise perfection! After applying the Bonferroni adjustment, the p-values for the t-tests between each of the three groups are all less than 0.1.

####################################################################################################
####################################################################################################
'''
Proportion Tests

One-sample proportion tests

Standardized test statistic for proportions
- p: population proportion (unknown)
- p_hat: sample proportion (sample statistic)
- p_0: hypothesized population proportion

z-Score: (p_hat - mean(p_hat))/SE(p_hat) = (p_hat - p)/SE(p_hat)

-Assuming H_0 is true, p = p_0 so z=(p_hat-p_0)/SE(p_hat) 

And SE(p_hat) = sqrt((p_0*(1-p_0))/n), depends on p_0 and sample size n

The standard deviation of the sample, s, is calculated from the sample mean, x-bar. That means that x-bar is used in the numerator to estimate the population mean, and in the denominator to estimate the population standard deviation. This dual usage increases the uncertainty in our estimate of the population parameter. Since t-distributions are effectively a normal distribution with fatter tails, we can use them to account for this extra uncertainty. In effect, the t-distribution provides extra caution against mistakenly rejecting the null hypothesis. For proportions, we only use p-hat in the numerator, thus avoiding the problem with uncertainty, and a z-distribution is fine.

'''
# Some of the hypothesis tests in this course have used a z-test statistic and some have used a t-test statistic. To get the correct p-value, you need to use the right type of test statistic.

# Do tests of proportion(s) use a z or a t test statistic and why? 

# z-Test: Zipadeedoodah for z-scores! The t-test is needed for tests of mean(s) since you are estimating two unknown quantities, which leads to more variability.

##---------------------------------------------------------------------------------------------------
# In Chapter 1, you calculated a p-value for a test hypothesizing that the proportion of late shipments was greater than 6%. In that chapter, you used a bootstrap distribution to estimate the standard error of the statistic. An alternative is to use an equation for the standard error based on the sample proportion, hypothesized proportion, and sample size.

# Hypothesize that the proportion of late shipments is 6%
p_0 = 0.06

# Calculate the sample proportion of late shipments
p_hat = (late_shipments['late'] == "Yes").mean()

# Calculate the sample size
n = len(late_shipments)

# Calculate the numerator and denominator of the test statistic
numerator = p_hat - p_0
denominator = np.sqrt(p_0 * (1 - p_0) / n)

# Calculate the test statistic
z_score = numerator / denominator

# Calculate the p-value from the z-score
p_value = 1 - norm.cdf(z_score)

# Print the p-value
print(p_value)

# Well proportioned! While bootstrapping can be used to estimate the standard error of any statistic, it is computationally intensive. For proportions, using a simple equation of the hypothesized proportion and sample size is easier to compute.

####################################################################################################
'''
Two-sample proportion test
'''
# You may wonder if the amount paid for freight affects whether or not the shipment was late. Recall that in the late_shipments dataset, whether or not the shipment was late is stored in the late column. Freight costs are stored in the freight_cost_group column, and the categories are "expensive" and "reasonable".

# The hypotheses to test, with "late" corresponding to the proportion of late shipments for that group, are

# H_0: late_expensive - late_reasonable = 0

# H_A: late_expensive - late_reasonable > 0

# p_hats contains the estimates of population proportions (sample proportions) for each freight_cost_group:

# freight_cost_group  late
# expensive           Yes     0.082569
# reasonable          Yes     0.035165
# Name: late, dtype: float64
# ns contains the sample sizes for these groups:

# freight_cost_group
# expensive     545
# reasonable    455
# Name: late, dtype: int64

# Calculate the pooled estimate of the population proportion
p_hat = (p_hats["reasonable"] * ns["reasonable"] + p_hats["expensive"] * ns["expensive"]) / (ns["reasonable"] + ns["expensive"])

# Calculate p_hat one minus p_hat
p_hat_times_not_p_hat = p_hat * (1 - p_hat)

# Divide this by each of the sample sizes and then sum
p_hat_times_not_p_hat_over_ns = p_hat_times_not_p_hat / ns["expensive"] + p_hat_times_not_p_hat / ns["reasonable"]

# Calculate the standard error
std_error = np.sqrt(p_hat_times_not_p_hat_over_ns)

# Calculate the z-score
z_score = (p_hats["expensive"] - p_hats["reasonable"]) / std_error

# Calculate the p-value from the z-score
p_value = (1-norm.cdf(z_score))

# Print p_value
print(p_value)

# Mad props! You can calculate a p-value for a two sample proportion test using (a rather exhausting amount of) arithmetic. This tiny p-value leads us to suspect there is a larger proportion of late shipments for expensive freight compared to reasonable freight.

##---------------------------------------------------------------------------------------------------
# That took a lot of effort to calculate the p-value, so while it is useful to see how the calculations work, it isn't practical to do in real-world analyses. For daily usage, it's better to use the statsmodels package.

# Count the late column values for each freight_cost_group
late_by_freight_cost_group = late_shipments.groupby("freight_cost_group")['late'].value_counts()

# Create an array of the "Yes" counts for each freight_cost_group
success_counts = np.array([late_by_freight_cost_group['expensive'].Yes, late_by_freight_cost_group['reasonable'].Yes])

# Create an array of the total number of rows in each freight_cost_group
n = np.array([np.sum(late_by_freight_cost_group.expensive), np.sum(late_by_freight_cost_group.reasonable)])

# Run a z-test on the two proportions
stat, p_value = proportions_ztest(count=success_counts, nobs=n, alternative='larger')

# Print the results
print(stat, p_value)

####################################################################################################
'''
Chi-squared test of independence

Statistical independence: Proportion of successes in the response variable is the same across all categories of the explanatory variable

Left-tailed chi-squared tests are used in statistical forensics to detect if a fit is suspiciously good because the data was fabriacted. Chi-squared tests of variance can be two-tailed
'''
# Chi-square hypothesis tests rely on the chi-square distribution. Like the t-distribution, it has degrees of freedom and non-centrality parameters.

# The plots show the PDF and CDF for a chi-square distribution (solid black line), and for comparison show a normal distribution with the same mean and variance (gray dotted line).

# Which statement about the chi-square distribution is true?

# Très chic chi-square! Like the t-distribution, the chi-square distribution has degrees of freedom and non-centrality parameters. When these numbers are large, the chi-square distribution can be approximated by a normal distribution.

##---------------------------------------------------------------------------------------------------
# Unlike pingouin.ttest() and statsmodels.stats.proportion.proportions_ztest(), pingouin.chi2_independence() does not have an alternative argument to specify which tails are considered by the alternative hypothesis.

# Which tail is almost always considered in chi-square tests?

# Right on! The chi-square test statistic is a square number, so it is always non-negative, so only the right tail tends to be of interest.

##---------------------------------------------------------------------------------------------------
# The chi-square independence test compares proportions of successes of one categorical variable across the categories of another categorical variable.

# Trade deals often use a form of business shorthand in order to specify the exact details of their contract. These are International Chamber of Commerce (ICC) international commercial terms, or incoterms for short.

# The late_shipments dataset includes a vendor_inco_term that describes the incoterms that applied to a given shipment. The choices are:

# EXW: "Ex works". The buyer pays for transportation of the goods.
# CIP: "Carriage and insurance paid to". The seller pays for freight and insurance until the goods board a ship.
# DDP: "Delivered duty paid". The seller pays for transportation of the goods until they reach a destination port.
# FCA: "Free carrier". The seller pays for transportation of the goods.
# Perhaps the incoterms affect whether or not the freight costs are expensive. Test these hypotheses with a significance level of 0.01.

# H_0: vendor_inco_term and freight_cost_group are independent.

# H_A: vendor_inco_term and freight_cost_group are associated.

# Proportion of freight_cost_group grouped by vendor_inco_term
props = late_shipments.groupby('vendor_inco_term')['freight_cost_group'].value_counts(normalize=True)

# Convert props to wide format
wide_props = props.unstack()

# Proportional stacked bar plot of freight_cost_group vs. vendor_inco_term
wide_props.plot(kind="bar", stacked=True)
plt.show()

# Determine if freight_cost_group and vendor_inco_term are independent
expected, observed, stats = pingouin.chi2_independence(data=late_shipments,x='freight_cost_group',y='vendor_inco_term')

# Print results
print(stats[stats['test'] == 'pearson']) 

# Independence insight! The test to compare proportions of successes in a categorical variable across groups of another categorical variable is called a chi-square test of independence.

####################################################################################################
'''
Chi-squared Goodness of Fit Test
'''
# The chi-square goodness of fit test compares proportions of each level of a categorical variable to hypothesized values. Before running such a test, it can be helpful to visually compare the distribution in the sample to the hypothesized distribution.

# Recall the vendor incoterms in the late_shipments dataset. You hypothesize that the four values occur with these frequencies in the population of shipments.

# CIP: 0.05
# DDP: 0.1
# EXW: 0.75
# FCA: 0.1
# These frequencies are stored in the hypothesized DataFrame.

# Find the number of rows in late_shipments
n_total = len(late_shipments)

# Create n column that is prop column * n_total
hypothesized["n"] = hypothesized["prop"] * n_total

# Plot a red bar graph of n vs. vendor_inco_term for incoterm_counts
plt.bar(incoterm_counts['vendor_inco_term'], incoterm_counts['n'], color="red", label="Observed")

# Add a blue bar plot for the hypothesized counts
plt.bar(hypothesized['vendor_inco_term'],hypothesized.n, alpha=0.5, color='blue',label="Hypothesized")
plt.legend()
plt.show()

# Beautiful bars! Two of the bars in the sample are very close to the hypothesized values: one is a little high and one is a little low. Head on over to the next exercise to test if these differences are statistically significant.

##---------------------------------------------------------------------------------------------------
# The bar plot of vendor_inco_term suggests that the distribution across the four categories was quite close to the hypothesized distribution. You'll need to perform a chi-square goodness of fit test to see whether the differences are statistically significant.

# Recall the hypotheses for this type of test:

# H_0: The sample matches with the hypothesized distribution.

# H_A: The sample does not match with the hypothesized distribution.

# To decide which hypothesis to choose, we'll set a significance level of 0.1

# Perform a goodness of fit test on the incoterm counts n
gof_test = chisquare(incoterm_counts.n,hypothesized.n)


# Print gof_test results
print(gof_test)

# What a good goodness of fit! The test to compare the proportions of a categorical variable to a hypothesized distribution is called a chi-square goodness of fit test.

####################################################################################################
####################################################################################################
'''
Non-parametric Tests

Assumptions in Hypothesis Testing

Randomness
1. The samples are random subsets of larger populations
- If not, sample is not representative of population
-- Understand how your data was collected
-- Speak to data collector/domain expert

Independence
2. Each (observation) row in the dataset is independent 
- Increased chance of false negative/positive errors
-- Undersrand how your data was collected

Sample Size
3. The sample is big enough to mitigate uncertainty, so that the central limit theorem applies
- Wider confidence intervals
- Increased chance of false negative/positive errors
-- Checks depends on the test

How big is big enough?
- 1-samp t-test: >= 30 observations
- 2-samp t-test, ANOVA: >= 30 observations for each group 
- Paired t-test: >= 30 observations across both groups

Need null distribution to appear normal 

- 1-samp prop-test: At least 10 successes, 10 failures
-- If prob(success or failure) ~ 1 or 0, need a bigger sample
- 2-samp prop-test: At least 10 successes and 10 failures per group

- Chi2: Need 5 successes and 5 failures at least 

If the bootstrap distribution doesn't look normal, then the assumptions aren't valid.
'''
# Hypothesis tests make assumptions about the dataset that they are testing, and the conclusions you draw from the test results are only valid if those assumptions hold. While some assumptions differ between types of test, others are common to all hypothesis tests.

# Which of the following statements is a common assumption of hypothesis tests?

# Sample Independence

##---------------------------------------------------------------------------------------------------
# In order to conduct a hypothesis test and be sure that the result is fair, a sample must meet three requirements: it is a random sample of the population, the observations are independent, and there are enough observations. Of these, only the last condition is easily testable with code.

# The minimum sample size depends on the type of hypothesis tests you want to perform. You'll now test some scenarios on the late_shipments dataset.

# Count the freight_cost_group values
counts = late_shipments['freight_cost_group'].value_counts()

# Print the result
print(counts)

# Inspect whether the counts are big enough
print((counts >= 30).all())

# Count the late values
counts = late_shipments['late'].value_counts()

# Print the result
print(counts)

# Inspect whether the counts are big enough
print((counts >= 10).all())

# Count the values of freight_cost_group grouped by vendor_inco_term
counts = late_shipments.groupby('vendor_inco_term')['freight_cost_group'].value_counts()

# Print the result
print(counts)

# Inspect whether the counts are big enough
print((counts >= 5).all())

# Count the shipment_mode values
counts = late_shipments['shipment_mode'].value_counts()

# Print the result
print(counts)

# Inspect whether the counts are big enough
print((counts >= 30).all())

# Setting a great example for an ample sample! While randomness and independence of observations can't easily be tested programmatically, you can test that your sample sizes are big enough to make a hypothesis test appropriate. Based on the last result, we should be a little cautious of the ANOVA test results given the small sample size for Air Charter.

####################################################################################################
'''
Non-parametric Tests

All previous tests assume data is normally distributed
Require a sufficiently large sample size

Non-parametric tests are more reliable than parametric tests for non-normal data and data with small sample sizes

Paired t-test   -> Wilcoxon Signed Rank Test
Unpaired t-test -> Wilcoxon-Mann-Whitney Test
'''
# That's right! The Wilcoxon signed-rank test works well when the assumptions of a paired t-test aren't met.

##---------------------------------------------------------------------------------------------------
# You'll explore the difference between the proportion of county-level votes for the Democratic candidate in 2012 and 2016 to identify if the difference is significant.

# Conduct a paired t-test on dem_percent_12 and dem_percent_16
paired_test_results = pingouin.ttest(x=sample_dem_data['dem_percent_12'],
                                     y=sample_dem_data['dem_percent_16'],
                                     paired=True,
                                     alternative='two-sided') 

# Print paired t-test results
print(paired_test_results)

# Conduct a Wilcoxon test on dem_percent_12 and dem_percent_16
wilcoxon_test_results = pingouin.wilcoxon(x=sample_dem_data['dem_percent_12'],
                                          y=sample_dem_data['dem_percent_16'],
                                          alternative='two-sided')

# Print Wilcoxon test results
print(wilcoxon_test_results)

####################################################################################################
'''
Wilcoxon-Mann-Whitney Test

For unpaired t-tests

Kruskal-Wallis Test

For ANOVA tests
'''
# Another class of non-parametric hypothesis tests are called rank sum tests. Ranks are the positions of numeric values from smallest to largest. Think of them as positions in running events: whoever has the fastest (smallest) time is rank 1, second fastest is rank 2, and so on.

# By calculating on the ranks of data instead of the actual values, you can avoid making assumptions about the distribution of the test statistic. It's more robust in the same way that a median is more robust than a mean.

# One common rank-based test is the Wilcoxon-Mann-Whitney test, which is like a non-parametric t-test.

# Select the weight_kilograms and late columns
weight_vs_late = late_shipments[['weight_kilograms','late']]

# Convert weight_vs_late into wide format
weight_vs_late_wide = weight_vs_late.pivot(columns='late', 
                                           values='weight_kilograms')

# Run a two-sided Wilcoxon-Mann-Whitney test on weight_kilograms vs. late
wmw_test = pingouin.mwu(x=weight_vs_late_wide['No'],
                        y=weight_vs_late_wide['Yes'],
                        alternative='two-sided')

# Print the test results
print(wmw_test)

# They tried to make me use parameters, but I said "No, no, no". The small p-value here leads us to suspect that a difference does exist in the weight of the shipment and whether or not it was late. The Wilcoxon-Mann-Whitney test is useful when you cannot satisfy the assumptions for a parametric test comparing two means, like the t-test.

##---------------------------------------------------------------------------------------------------
# Recall that the Kruskal-Wallis test is a non-parametric version of an ANOVA test, comparing the means across multiple groups.

# Run a Kruskal-Wallis test on weight_kilograms vs. shipment_mode
kw_test = pingouin.kruskal(data=late_shipments,
                           dv='weight_kilograms',
                           between='shipment_mode')

# Print the results
print(kw_test)

# Great work! The Kruskal-Wallis test yielded a very small p-value, so there is evidence that at least one of the three groups of shipment mode has a different weight distribution than the others. Th Kruskal-Wallis test is comparable to an ANOVA, which tests for a difference in means across multiple groups.