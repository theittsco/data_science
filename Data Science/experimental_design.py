'''
Data!
-Allows us to ask questions.
-Need rigorous methods to get answers.

Build hypotheses with EDA
Test hypotheses with statistical tests

Discrete vs continuous variables.
Mapping using plots or fills.
'''

# Import plotnine
import plotnine as p9

# Create the scatter plot
print(p9.ggplot(People)+ p9.aes(x='Height', y='Weight', color='Sample')+ p9.geom_point())

# Create boxplot of Weight
print(p9.ggplot(People)+ p9.aes(x='Sample', y='Weight', fill='Sample')+ p9.geom_boxplot())

# Create boxplot of Height
print(p9.ggplot(People)+ p9.aes(x='Sample', y='Height', fill='Sample')+ p9.geom_boxplot())

# Create density plot of Height
print(p9.ggplot(People)+ p9.aes(x='Height', fill='Sample')+ p9.geom_density(alpha=0.5))

# Create density plot of Weight
print(p9.ggplot(People)+ p9.aes(x='Weight', fill='Sample')+ p9.geom_density(alpha=0.5))

###############################################################################
'''
Student's t-Test

How do we go from observation to result?

Two hypotheses:
A = B
- Observed patterns are product of random chance.
A != B
- Observed patterns represent real differences.

p-Value: Likelihood of pattern under null hypothesis
alpha: Crucial threshold of p-value, usually 0.05 - reject null hypothesis

Student's t-test: Comparing two sets of continuous variables
'''
# Perform t-test and print result
t_result=stats.ttest_1samp(Sample_A, 65)
print(t_result)

# Test significance
alpha= 0.05
if (t_result[1] < alpha):
    print("mean value of Sample A differs from given value")
else:
	print("No significant difference found")

###############################################################################
# Create the density plot
(p9.ggplot(eudata)+ p9.aes('Sex_ratio')+ p9.geom_density(alpha=0.5))

# Perform the one-sample t-test
t_result= stats.ttest_1samp(eudata.Sex_ratio, 100)
print(t_result)

# Test significance
alpha = 0.05
if t_result[1] < alpha:
    print("Sex ratios are significantly biased")
else:
    print("No significant bias found")

###############################################################################
# Create the density plot
print(p9.ggplot(euasdata)+ p9.aes('Sex_ratio', fill="Continent")+ p9.geom_density(alpha=0.5))

# Create two arrays
Europe_Sex_ratio = euasdata[euasdata.Continent == "Europe"].Sex_ratio
Asia_Sex_ratio = euasdata[euasdata.Continent == "Asia"].Sex_ratio

# Perform the two-sample t-test
t_result= stats.ttest_ind(Europe_Sex_ratio, Asia_Sex_ratio)
print(t_result)

# Test significance
alpha= 0.05
if (t_result[1] < alpha):
    print("Europe and Asia have different mean sex ratios")
else: print("No significant difference found")

###############################################################################
'''
Proportion and Correlation

t-Test: Compares means of continuous variables

Chi-squared Test: Examine proportions of discrete categories
    - Observed outcomes (fit/don't fit) distribution

Fisher exact Test: Examine proportions of discrete categories
    - Two sample version of Chi-squared Test
    - Two samples have the (same/different) distribution (Null / Alternative)

Pearson Test: Examine if continuous variables are correlated
'''
# Extract sex ratio
sexratio = athletes['Sex'].value_counts()

# Perform Chi-square test
chi= stats.chisquare(sexratio)
print(chi)

# Test significance
alpha= 0.05
if chi[1] < alpha:
    print("Difference between sexes is statistically significant")
else:
    print("No significant difference between sexes found")

# Create a table of cross-tabulations
table = pd.crosstab(athletes.MedalTF, athletes.Sport)
print(table)

# Perform the Fisher exact test
fisher = stats.fisher_exact(table, alternative='two-sided')
print(fisher)

# Is the result significant?
alpha = 0.05
if fisher[1] < alpha:
    print("Proportions of medal winners differ significantly")
else:
    print("No significant difference in proportions of medal winners found")

# Create the scatterplot
print(p9.ggplot(athletes)+ p9.aes('Year', 'Weight', color='Event')+  p9.geom_point())

# Run the correlation test
pearson = stats.pearsonr(athletes.Weight, athletes.Year)
print(pearson)

# Test if p-value is bigger or smaller than alpha
alpha = 0.05
if pearson[1] < alpha:
    print("Weights and year are significantly correlated")
else:
    print("No significant correlation found")

###############################################################################
###############################################################################
'''
Confounding Variables
-Additional variable not accounted for in original study design
-Alters independent and dependent variables
'''
# Create boxplot of Team versus Weight
plotTeamVWeight = p9.ggplot(athletes)+ p9.aes('Team','Weight')+ p9.geom_boxplot()

# Create boxplot of Sport versus Weight
plotSportVWeight = p9.ggplot(athletes)+ p9.aes('Sport','Weight')+ p9.geom_boxplot()

# Print plots
print(plotTeamVWeight, plotSportVWeight)

# Create crosstabulation & print table
table = pd.crosstab(athletes.Team,athletes.Sport)
print(table)

# Life expectancy density plot
plotLifeVCont = p9.ggplot(euasdata) + p9.aes(x='Life_exp', fill='Continent') +p9.geom_density(alpha=0.5)

# GDP vs life expectancy scatter plot
plotGDPVLife = p9.ggplot(euasdata) + p9.aes(x='GDP_per_cap', y='Life_exp') +p9.geom_point()

# GDP density plot
plotGDPVCont = p9.ggplot(euasdata) + p9.aes(x='GDP_per_cap', fill='Continent') +p9.geom_density(alpha=0.5)

# Print plots
print(plotLifeVCont,plotGDPVLife,plotGDPVCont)

###############################################################################
'''
Blocking and Randomization

Making comparisons
-Only variable of interest should differ between groups
Remove sources of variation
-Variation of interest

Random Sampling
-Confounding variables can complicate random sampling
-Solve using a technique known as blocking
-Special case is paired samples, control for individual variation
'''
# Define random seed
seed = 0000

# Create two subsets, one for the athletics competitors and one for the swimmers
subsetathl = athletes[athletes.Sport == 'Athletics'].sample(n=30, random_state= seed)
subsetswim = athletes[athletes.Sport == 'Swimming'].sample(n=30,
random_state= seed)

# Perform the two-sample t-test
t_result = stats.ttest_ind(subsetathl.Weight, subsetswim.Weight)
print(t_result)

# Define random seed
seed = 2397

# Create subsets
subsetathl = athletes[athletes.Sport == "Athletics"].sample(n=30, random_state= seed)
subsetswim = athletes[athletes.Sport == "Swimming"].sample(n=30, random_state= seed)

# Perform the two-sample t-test
t_result = stats.ttest_ind(subsetathl.Weight, subsetswim.Weight)
print(t_result)

seed = 9000

# Create subset blocks
subsetathlm = athletes[(athletes.Sport == "Athletics") & (athletes.Sex == 'M')].sample(n=15, random_state= seed)
subsetathlf = athletes[(athletes.Sport == "Athletics") & (athletes.Sex == 'F')].sample(n=15, random_state= seed)
subsetswimm = athletes[(athletes.Sport == "Swimming") & (athletes.Sex == 'M')].sample(n=15, random_state= seed)
subsetswimf = athletes[(athletes.Sport == "Swimming") & (athletes.Sex == 'F')].sample(n=15, random_state= seed)

# Combine blocks
subsetathl = pd.concat([subsetathlm, subsetathlf])
subsetswim = pd.concat([subsetswimm, subsetswimf])

# Perform the two-sample t-test
print(stats.ttest_ind(subsetathl.Weight,subsetswim.Weight) )

# Perform independent t-test
ttestind = stats.ttest_ind(podataframe.Yield2018,podataframe.Yield2019)
print(ttestind)

# Perform paired t-test
ttestpair = stats.ttest_rel(podataframe.Yield2018,podataframe.Yield2019)
print(ttestpair)
'''
Good work. Recall, the paired test is more sensitive than the independent test
and can pick up a difference that the independent test can't detect. This is
because the difference within the samples each year (individual field effect)
is quite large in comparison to the difference between the two years (effect of
treatment). Paired tests are useful when a large variability exists.
'''

###############################################################################
'''
ANOVA

Independent Variables: Manipulate experimentally
Dependent Variables: Try to understand their patterns

t-Test: One discrete independent variable with two levels, one dependent variable

ANOVA = Analysis of Variance
-Generalize t-Test to broader set of cases
-Examine multiple sets of factors
-Partition variation into separate components
-Multiple simultaneous tests

One-Way ANOVA: Single factor with 3+ levels
Two-Way ANOVA: Two factors with 3+ levels
'''
# Create arrays
France_athletes = athletes[athletes.Team == 'France'].Weight
US_athletes = athletes[athletes.Team == 'United States'].Weight
China_athletes = athletes[athletes.Team == 'China'].Weight

# Perform one-way ANOVA
anova = stats.f_oneway(France_athletes,US_athletes,China_athletes)
print(anova)

# Create model
formula = 'Weight ~ Sex + Team'
model = sm.api.formula.ols(formula, data=athletes).fit()

# Perform ANOVA and print table
aov_table = sm.api.stats.anova_lm(model, typ=2)
print(aov_table)

###############################################################################
'''
Interactive Effects

Additive Model: Effect of each factor on the model is additive or subtractive
'''
# Run the ANOVA
model = sm.api.formula.ols('Weight ~ Sex + Event + Sex:Event', data = athletes).fit()

# Extract our table
aov_table = sm.api.stats.anova_lm(model, typ=2)

# Print the table
print(aov_table)

# Create model
formula = 'Height ~ Sex + Team + Sex:Team'
model = sm.api.formula.ols(formula, data=athletes).fit()

# Perform ANOVA and print table
aov_table = sm.api.stats.anova_lm(model, typ=2)
print(aov_table)

###############################################################################
###############################################################################
'''
Type I Errors

TP, TN, FP, FN

FP: Type 1 error
FN: Type 2 error

Basis of tests:
- Statistical tests are probabilstic
- Quantify likelihood of results under null hypothesis

Consider:
- Significant results are improbable but not impossible under null hypothesis

Account for multiple tests
- Avoid p-value fishing, running many tests without justification for each
test
- Correct p-values for presence of multiple tests

Correction methods: Bonferroni and Siddak corrections
- Choose method based on independence of tests, not the test method itself

Bonferroni:
- Use when not independent of each other
- Conservative and simple

Siddak:
- Less conservative
- Use when tests are independent of each other
'''
# Perform three two-sample t-tests
t_result_1924v2016= stats.ttest_ind(athletes[athletes.Year == '1924'].Height, athletes[athletes.Year == '2016'].Height)
t_result_1952v2016= stats.ttest_ind(athletes[athletes.Year == '1952'].Height, athletes[athletes.Year == '2016'].Height)
t_result_1924v1952= stats.ttest_ind(athletes[athletes.Year == '1924'].Height, athletes[athletes.Year == '1952'].Height)

# Create an array of p-value results
pvals_array = [t_result_1924v2016[1], t_result_1952v2016[1], t_result_1924v1952[1]]
print(pvals_array)

# Perform Bonferroni correction
adjustedvalues = sm.stats.multitest.multipletests(pvals_array, alpha=0.05, method='b')
print(adjustedvalues)

# Perform Pearson correlations
pearson100 = stats.pearsonr(athletes[athletes.Event == '100 meters'].Height, athletes[athletes.Event == '100 meters'].Year)
pearsonHigh = stats.pearsonr(athletes[athletes.Event == 'High Jump'].Height, athletes[athletes.Event == 'High Jump'].Year)
pearsonMara = stats.pearsonr(athletes[athletes.Event == 'Marathon'].Height, athletes[athletes.Event == 'Marathon'].Year)

# Create array of p-values
pvals_array = [pearson100[1], pearsonHigh[1], pearsonMara[1]]
print(pvals_array)

# Perform Šídák correction
adjustedvalues=  sm.stats.multitest.multipletests(pvals_array, method= 's')
print(adjustedvalues)

###############################################################################
'''
Sample Size

Type II Error:
- False negative, fail to detect an effect that exists
- Can never be sure that no effect is present

Increased sample size avoids false negatives, larger sample size == more
sensitive methods

Values that affect sample size:
- alpha: Critical value of p to reject null hypothesis
- Power: Probability we correctly reject null hypothesis if alternative
hypothesis is true
- Effect Size: How much we depart from null hypothesis

Increase Sample Size:
1. Increase statistical power
2. Decrease usable alpha
3. Smaller effect size detectable

Question: What sample size do we need with effect_size = x, power = y, and
alpha = z?
'''
# Create subset with defined random seed and perform t-test
subset = athletes.sample(n=1000, random_state= 1007)
print(stats.ttest_ind(subset[subset.Sport == "Athletics"].Weight, subset[subset.Sport == "Swimming"].Weight ))

#p-value = 4.86e-6

# Create sample with defined random seed and perform t-test
subset = athletes.sample(n=200, random_state= 1007)
print(stats.ttest_ind(subset[subset.Sport == 'Athletics'].Weight, subset[subset.Sport == 'Swimming'].Weight))

#pvalue = 0.045

# Create sample with defined random seed and perform t-test
subset = athletes.sample(n=50, random_state= 1007)
print(stats.ttest_ind(subset[subset.Sport == "Athletics"].Weight,
                      subset[subset.Sport == "Swimming"].Weight))

#p-value = 0.68

# Set parameters
effect = 0.4
power = 0.8
alpha = 0.05

# Calculate ratio
swimmercount = float(len(athletes[athletes.Sport == 'Swimming']))
athletecount = float(len(athletes[athletes.Sport == 'Athletics']))
ratio = swimmercount/athletecount

# Initialize analysis and calculate sample size
analysis = pwr.TTestIndPower()
ssresult = analysis.solve_power(effect_size=effect, power=power, alpha= alpha, nobs1=None, ratio=ratio)
print(ssresult)

###############################################################################
'''
Effect Size

Effect size versus significance:

Significance
- How sure we are that an effect exists
- X% confident that A is better than B for example

Effect
- How much of a difference that effect makes
- Yields with A are higher than yields with B for example

Measuring effect size: Cohen's d
- Normalized differences between sample means

Cohen's d = (M2 - M1)/SDPooled
'''
# Set parameters
alpha = 0.05
power = 0.8
ratio = float(len(athletes[athletes.Sport == 'Swimming'])) / len(athletes[athletes.Sport == "Athletics"])
samp_size = len(athletes[athletes.Sport == 'Athletics'])

# Initialize analysis & calculate sample size
analysis = pwr.TTestIndPower()
esresult = analysis.solve_power(effect_size = None,
                                power = power,
                                nobs1 = samp_size,
                                ratio = ratio,
                                alpha = alpha)
print(esresult)

# Set parameters
alpha = 0.05
power = 0.8
ratio = float(len(athletes[athletes.Sport == 'Swimming'].sample(n=300))/len(athletes[athletes.Sport == 'Athletics'].sample(n=300)))
samp_size = 300

# Initialize analysis & calculate sample size
analysis = pwr.TTestIndPower()
esresult = analysis.solve_power(effect_size=None, power=power, nobs1=samp_size, ratio=ratio, alpha=alpha)
print(esresult)

# Create series
athl = athletes[athletes.Sport == 'Athletics'].Weight
swim = athletes[athletes.Sport == 'Swimming'].Weight

# Calculate difference between means and pooled standard deviation
diff = swim.mean() - athl.mean()
pooledstdev = ma.sqrt((athl.std()**2 + swim.std()**2)/2 )

# Calculate Cohen's d
cohend = diff / pooledstdev
print(cohend)

###############################################################################

# Create a table of cross-tabulations
table = pd.crosstab(athletes.MedalTF,athletes.Sport)
print(table)

# Perform the Fisher exact test
chi = stats.fisher_exact(table, alternative='two-sided')

# Print p-value
print("p-value of test: " + str(round(chi[1], 5))  )

# Print odds ratio
print("Odds ratio between groups: " + str(round(chi[0], 1))  )

###############################################################################
# Perform Pearson correlation
pearsonken = stats.pearsonr(ken.Weight, ken.Height)
print(pearsonken)

# Perform Pearson correlation
pearsoneth = stats.pearsonr(eth.Weight, eth.Height)
print(pearsoneth)

###############################################################################
'''
Power: Probability of detecting an effect
Ways to Increase Power:
    - Larger effect size
    - Larger sample size

Dealing with Uncertainty
- Hypothesis Tetsing
    - Estimate likelihoods
    - Cannot give absolute certainty

- Power Analysis
    - Estimate strength of answers

- With power analysis we can estimate possibility of Type II errors
    - Negative result and high power -> True Negative
    - Negative result and low power -> Possible FN

- Find balance of power
    - Higher power can increase chance of type I errors

Requires domain knowledge to make reasonable assumptions
'''
# Set parameters
effect_size = 0.42145
alpha = 0.05
samp_size = 100
ratio = 1

# Initialize analysis & calculate power
analysis = pwr.TTestIndPower()
pwresult = analysis.solve_power(effect_size=effect_size, power=None, nobs1=samp_size, ratio=ratio, alpha=alpha)
print(pwresult)

###############################################################################
###############################################################################
'''
Testing Normality: Parametric and Non-parametric Tests

Assumptions and normal distributions

Summary stats:
    - Mean
    - Median
    - Mode
    - Standard deviation

Q-Q Plot: Quantile-Quantile
- Normal Probability plot
- Graphical method to assess normality
- Compare quantiles of data with theoretical quantiles predicted under distribution
'''
# Print density plot, mean, median, and mode of Unemployment
print(p9.ggplot(countrydata)+ p9.aes(x='Unemployment')+ p9.geom_density())
print(countrydata.Unemployment.mean())
print(countrydata.Unemployment.median())
print(countrydata.Unemployment.mode())

# Print density plot, mean, median, and mode of GDP per capita
print(p9.ggplot(countrydata)+ p9.aes(x='GDP_per_cap')+ p9.geom_density())
print(countrydata.GDP_per_cap.mean())
print(countrydata.GDP_per_cap.median())
print(countrydata.GDP_per_cap.mode())

# Calculate theoretical quantiles
tq = stats.probplot(countrydata.Unemployment, dist='norm')

# Create Dataframe
df = pd.DataFrame(data= {'Theoretical Quantiles': tq[0][0],
                         "Ordered Values": countrydata.Unemployment.sort_values() })

# Create Q-Q plot
print(p9.ggplot(df)+ p9.aes('Theoretical Quantiles', 'Ordered Values') +p9.geom_point())

###############################################################################
'''
Testing for Normailty

Normal distribution:
- Mean, median, and mode are equal
- Symmetrical
- Crucial assumption of certain tests

Shapiro-Wilk Test:
- Tests for normality
- Only appropriate for small sample sizes

1. Test normality of each sample
2. Choose test/approach
3. Perform hypothesis test

Tests based on assumption of normality:
- Student's t-Test (1 and 2 sample)
- Paired t-Test
- ANOVA

As number of samples increases, sample mean approaches population mean
- For large sample size not necessary to test for normality
- For small sample size it's important to check that normality assumption is
not violated
'''
# Perform Shapiro-Wilk test on Unemployment and print result
shapiroUnem = stats.shapiro(countrydata.Unemployment)
print(shapiroUnem)

# Perform Shapiro-Wilk test on Unemployment and print result
shapiroGDP = stats.shapiro(countrydata.GDP_per_cap)
print(shapiroGDP)

'''
While the t-test and ANOVA are designed with normal distributions in mind,
they can be used for somewhat non-normal distributions where sample sizes are
large enough.
'''

###############################################################################
'''
Non-parametric tests

When assumptions don't hold: Tests are based on assumptions about data

Violations of assumptions
- Tests are no longer valid
- Solution: Non-parametric tests, looser constraints


Parametric vs non-Parametric

Parametric: Higher sensitivity, more specific
- Make many assumptions
- Population modeled by distribution with fixed parameters

Non-parametric: Lower sensitivity, less specific
- Make few assumptions
- No fixed population parameters
- Used when data doesn't fit these distributions

Wilcoxon rank sum test
- Non-parametric version of t-Test
'''
# Print t-test result
print(t_result)

# Perform Wilcoxon rank-sum test
wilc = stats.ranksums(Europe_Sex_ratio, Asia_Sex_ratio)
print(wilc)

# Print t-test result
print(ttestpair)

# Print Shapiro-Wilk test results
print(shap2018)
print(shap2019)

# Perform Wilcoxon Signed-Rank test
wilcsr = stats.wilcoxon(podataframe.Yield2018, podataframe.Yield2019)
print(wilcsr)

###############################################################################
# Separate the heights by country
NorwayHeights = athletes[athletes['Team'] == "Norway"].Height
ChinaHeights = athletes[athletes['Team'] == "China"].Height

# Shapiro-wilks test on the heights
print(stats.shapiro(NorwayHeights)[1])
print(stats.shapiro(ChinaHeights)[1])

# Perform the Wilcoxon rank-sum test
wilc = stats.ranksums(NorwayHeights, ChinaHeights)
print(wilc)

# Use Wilson rank sum test

###############################################################################
'''
Spearman Correlation

Correlation:
- Relate continuous or ordinal variable to one another
- Will variation in one predict correlationin the other

Pearson Correlation: Based on a linear model, parametric
- Based on raw values
- Sensitive to outliers
- Assumes linear and monotonic
- Pearsons r

Spearman Correlation: Non-parametric
- Based on ranks
- Robust to outliers
- Assumes monotonic relationship
- Spearmans rho
'''
# Perform Pearson and Spearman correlations
pearcorr = stats.pearsonr(athletesF.Height, athletesF.Weight)
print(pearcorr)
spearcorr = stats.spearmanr(athletesF.Height, athletesF.Weight)
print(spearcorr)

# Perform Pearson and Spearman correlations
pearcorr = stats.pearsonr(athletesM.Height, athletesM.Weight)
print(pearcorr)
spearcorr = stats.spearmanr(athletesM.Height, athletesM.Weight)
print(spearcorr)

# Perform Spearman correlation
spearcorr = stats.spearmanr(podataframe.Production, podataframe.Fertilizer)
print(spearcorr)
