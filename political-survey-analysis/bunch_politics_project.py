# Jonathan Bunch
# March 7, 2021
# DSC530

import pandas as pd
import scipy.stats
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# I started by defining the column indexes I wanted to import and giving those columns more manageable names.
col_nums = [7, 13, 20, 23, 24]
col_names = ['how_fundamentalist', 'should_fund_education', 'resp_age', 'total_family_income', 'how_conservative']

# Now we can import the data directly into a Dataframe using pandas.
gss = pd.read_csv("gss_jan_23.dat", header=None, delim_whitespace=True, engine='python',
                  usecols=col_nums, names=col_names)

# Some variables use codes for certain responses. For example, a response of "refused to answer" might be coded as 99.
# The documentation explains these codes so we can remove incomplete responses. This step is essentially removing NAs.
df = gss.query('resp_age != [89, 98, 99] & total_family_income != [0, 13, 98, 99] & should_fund_education != [0, 8, 9]'
               ' & how_conservative != [0, 8, 9] & how_fundamentalist != 9').copy()

# Because the responses are coded as numbers, pandas will assign them integer data types. It is important to convert
# these variables to categorical data types to help avoid any misinterpretation by the computer (or ourselves).
# I also made these variables ordered because the possible responses all take the general format of a scale.
df['how_conservative'] = df['how_conservative'].astype(pd.api.types.CategoricalDtype(
    categories=[1, 2, 3, 4, 5, 6, 7], ordered=True))
df['how_fundamentalist'] = df['how_fundamentalist'].astype(pd.api.types.CategoricalDtype(
    categories=[3, 2, 1], ordered=True))
df['should_fund_education'] = df['should_fund_education'].astype(pd.api.types.CategoricalDtype(
    categories=[3, 2, 1], ordered=True))
df['total_family_income'] = df['total_family_income'].astype(pd.api.types.CategoricalDtype(
    categories=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ordered=True))

# Plot histograms for each variable.
hists = {}
for name in df.columns:
    if name == 'resp_age':
        plt.hist(df[name], bins=70)
        plt.title('Histogram of Variable: ' + name)
        plt.ylabel('Frequency')
        plt.show()
        continue
    hists[name] = np.histogram(df[name], bins=list(range((df[name].cat.categories.size + 3))))

for name in hists:
    plt.bar(x=hists[name][1][:-1], height=hists[name][0])
    plt.title('Histogram of Variable: ' + name)
    plt.ylabel('Frequency')
    plt.savefig(name)
    plt.show()

# Print summary statistics for each variable.
for name in df.columns:
    print(df[name].describe())
    print(df[name].tail())

# For my PMF comparison I chose to use the variable resp_age. I will divide the observations into two groups based on
# whether the participant identified as liberal or conservative.
lib_group = df.query('how_conservative <= 3')
lib_ages = lib_group.resp_age.copy()
cons_group = df.query('how_conservative >= 5')
cons_ages = cons_group.resp_age.copy()

# Now we can plot the PMFs using the histogram function.
plt.hist(lib_ages, density=True, bins=(lib_ages.max() - lib_ages.min()), histtype='step', label='More Liberal')
plt.hist(cons_ages, density=True, bins=(cons_ages.max() - cons_ages.min()), histtype='step', label='More Conservative')
plt.title('PMFs for Age of Respondents')
plt.ylabel('Probability')
plt.xlabel('Age')
plt.legend()
plt.show()

# Now I will plot a CDF of the variable how_conservative.
cons = df.how_conservative.copy()
cx = np.sort(cons)
cy = np.arange(len(cx))/float(len(cx))
plt.plot(cx, cy)
plt.title('CDF of variable "how_conservative"')
plt.xlabel('how conservative respondents identified as')
plt.ylabel('Cumulative Probability')
plt.show()

# Next, we will try to fit some analytic distributions to our data.
# Gamma distribution.
ag, locg, scaleg = scipy.stats.gamma.fit(df['resp_age'])
rv_gamma = scipy.stats.gamma(ag, loc=locg, scale=scaleg)
# Alpha distribution.
aa, loca, scalea = scipy.stats.alpha.fit(df['resp_age'])
rv_alpha = scipy.stats.alpha(aa, loc=loca, scale=scalea)
# Rice distribution.
br, locr, scaler = scipy.stats.rice.fit(df['resp_age'])
rv_rice = scipy.stats.rice(br, loc=locr, scale=scaler)
# Plot all three models with the sample distribution for comparison.
fig, ax = plt.subplots()
x = np.linspace(15, 90, 100)
ax.hist(df['resp_age'], density=True, bins=70, label='Sample')
ax.plot(x, rv_gamma.pdf(x), label='Gamma')
ax.plot(x, rv_alpha.pdf(x), label='Alpha')
ax.plot(x, rv_rice.pdf(x), label='Rice')
ax.legend()
ax.set_xlabel('Age')
ax.set_ylabel('Probability')
ax.set_title('Comparison of Analytic Distributions to Sample Data')
plt.show()

# Now I will begin to look at multiple variables together, starting with a scatterplot.
plt.scatter(x=df.how_conservative, y=df.resp_age, s=500, alpha=0.01)
plt.xlim(0, 8)
plt.ylim(13, 93)
plt.title('Conservativeness vs. Respondent Age')
plt.xlabel('How Conservative')
plt.ylabel('Respondent Age')
plt.show()

plt.scatter(x=df.total_family_income, y=df.resp_age, s=500, alpha=0.01)
plt.title('Family Income vs. Respondent Age')
plt.xlabel('Total Family Income')
plt.ylabel('Respondent Age')
plt.show()

# Create a correlation matrix.
c = scipy.stats.spearmanr(df)
cor_matrix = pd.DataFrame(c.correlation, index=df.columns.values, columns=df.columns.values)
p_val_matrix = pd.DataFrame(c.pvalue, index=df.columns.values, columns=df.columns.values)

# Conduct a hypothesis test. I chose to do a correlation test by permutation.
xs = df.how_conservative.copy()
ys = df.should_fund_education.copy()
orig_corr = scipy.stats.spearmanr(a=xs, b=ys)
rng = np.random.default_rng()
perm_x = rng.permutation(xs)
perm_corr = scipy.stats.spearmanr(a=perm_x, b=ys)

# Finally, I will perform a regression analysis. I will attempt to create a model that can predict how_conservative.
formulas = ['how_conservative ~ how_fundamentalist',
            'how_conservative ~ resp_age', 'how_conservative ~ total_family_income',
            'how_conservative ~ should_fund_education']
for form in formulas:
    model = smf.poisson(form, data=df.astype('int64'))
    results = model.fit()
    print(results.summary())

# All of the simple regression models had very low pseudo-R-squared values, but the p-values looked acceptable.
# So, we may as well try combining them in a multiple regression model.
form_multi = 'how_conservative ~ how_fundamentalist + resp_age + total_family_income + should_fund_education'
model = smf.poisson(form_multi, data=df.astype('int64'))
results = model.fit()
print(results.summary())
