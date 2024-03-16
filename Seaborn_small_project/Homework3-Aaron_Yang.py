#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
import warnings
warnings.filterwarnings("ignore")

#%%
# Question 1
# Load the dataset
penguins = sns.load_dataset("penguins")

# Display the last 5 Observations
penguins.tail(5)

#%%
# Display the statistic data
penguins.describe()
#%%
# Question 2
# Display the number of missing value
 penguins.isnull().sum()


#%%
# Clean the dataset
penguins = penguins.dropna()

# Show the missing values again
print(penguins.isnull().sum().sum())

#%%
# Question 3
# Set the sns plot style
sns.set_style('darkgrid')

# Histogram plot for 'flipper_length_mm'
Q3_hist = sns.histplot(data=penguins['flipper_length_mm'])
Q3_hist.set_title('Penguin Flipper Histogram')

# Display the plot
plt.tight_layout()
plt.show()

# Print the observation
print('The most length of penguin flipper is about 190-195mm.')

#%%
# Question 4
# Set the sns plot style
sns.set_style('darkgrid')

# Histogram plot for 'flipper_length_mm' with binwidth = 3
Q3_hist = sns.histplot(data=penguins['flipper_length_mm'], binwidth=3)
Q3_hist.set_title('Penguin Flipper Histogram Binwidth Changed')

# Display the plot
plt.tight_layout()
plt.show()

# Print the observation
print('The most length of penguin flipper is about 190-192mm, '
      'while the least length of penguin flipper is 175-177mm.')

#%%
# Question 5
# Set the sns plot style
sns.set_style('darkgrid')

# Histogram plot for 'flipper_length_mm' with binwidth = 3
Q3_hist = sns.histplot(data=penguins['flipper_length_mm'],binwidth=3, bins=30)
Q3_hist.set_title('Penguin Flipper Histogram Bins number changed')

# Display the plot
plt.tight_layout()
plt.show()

# Print the observation
print('The most length of penguin flipper is about 191mm, '
      'while the least length of penguin flipper is 172-173, 174, 177mm.')

#%%
# Question 6
# Set the hue and displot
Q6_displot = sns.displot(data=penguins,
                         x='flipper_length_mm',
                         binwidth=3,
                         bins=30,
                         hue='species')

# Build the title for the displot
Q6_displot.set(title='Penguin Flipper Length Distribution by Species')

# Display the plot
plt.tight_layout()
plt.show()

# Print the observation
print("The Gentoo penguin fipper length is generally longer than other two species, \n"
      "while the Adelie Penguin flipper length is shorter than other two species.")

#%%
# Question 7
# Set the element is 'step'
Q7_displot = sns.displot(data=penguins,
                         x='flipper_length_mm',
                         binwidth=3,
                         bins=30,
                         hue='species',
                         element='step')

# Build the title for the displot
Q7_displot.set(title='Penguin Flipper Length Distribution with element')

# Display the plot
plt.tight_layout()
plt.show()

# Print the observation
print("Adelie penguins tend to have the shortest flipper lengths, mostly clustering around 180-190 mm. \n "
      "Chinstrap penguins have flipper lengths that fall mainly between 190-200 mm. "
      "\nGentoo penguins have the longest flippers, with a distribution that is centered around 210-220 mm.")
#%%
# Question 8
# Set the mutiple = stack
Q8_hist = sns.histplot(data=penguins,
                       x='flipper_length_mm',
                       hue='species',
                       multiple='stack')

# Set the title for Q8_hist
Q8_hist.set_title('Penguin Flipper Length stacked Histogram')

# Disply the Q8_hist
plt.tight_layout()
plt.show()

# Print the observation
print('Adelie penguins tend to have shortest flipper lengths, mostly clustering, and have the most frequently selected\n')
print('Chinstrap penguins have medium flipper lengths, but the number of them is the smallest.')


#%%
# Question 9
Q9_hist = sns.displot(data=penguins,
                      x='flipper_length_mm',
                      hue='sex',
                      multiple='dodge')


# Set the title
plt.title('Penguin flipper length by sex with dodge')


plt.tight_layout()
plt.show()

# Print the observation
print("Female penguins tend to have longer flipper lengths than male penguins.")

#%%
# Question 10
# # Call the function FacetGrid
# Q10_fig = sns.FacetGrid(data = penguins, col='sex')
#
# # Build
# Q10_fig.map(sns.displot, 'flipper_length_mm')
# Q10_fig.set_titles("Flipper Length for {col_name} Penguins")
# plt.tight_layout()
# plt.show()

Q10_fig = sns.displot(
    data=penguins,
    x='flipper_length_mm',
    col='sex',
    kde=True,
    facet_kws={'sharey':False, 'sharex':False}
)

# Set the title and labels
Q10_fig.fig.suptitle('Flipper lengths by Sex', fontsize=16)
Q10_fig.set_axis_labels('Flipper length (mm)', 'Count')
Q10_fig.fig.subplots_adjust(top=0.8)

plt.tight_layout()
plt.show()

print("Most male penguins tend to have 190-200mm flipper lengths")
print("Most Female penguins tend to have 185-195mm flipper lengths")

#%%
# Question 11
Q11_fig = sns.histplot(data=penguins,
                       x='flipper_length_mm',
                       stat='density',
                       hue='species',
                       legend=True,
                       kde=False)

# Set the title and labels
Q11_fig.set_title('Distribution of flipper lengths by species', fontsize=16)


# Show the plot
plt.tight_layout()
plt.show()

print("Adelie penguins tend to have 190-195 mm flipper lengths")
print('Chinstrap penguins tend to have 195-200mm flipper lengths')
print('Gentoo penguins tend to have 215-220mm flipper lengths')

#%%
# Question 12
Q12_fig = sns.histplot(data=penguins,
                       x='flipper_length_mm',
                       stat='density',
                       hue='sex',
                       legend=True,
                       kde=False)

# Set the title and labels
Q12_fig.set_title('Distribution of flipper lengths by sex', fontsize=16)


# Show the plot
plt.tight_layout()
plt.show()

print("Male penguins tend to have 190-195 mm flipper lengths")
print('Female penguins tend to have 190-195mm flipper lengths')

#%%
# Question 13
Q13_fig = sns.histplot(data=penguins,
                       x='flipper_length_mm',
                       stat='probability',
                       binwidth=True,
                       hue='species',
                       legend=True,
                       kde=False)

# Set the title and labels
Q13_fig.set_title('Distribution Probability of flipper lengths by species', fontsize=16)


# Show the plot
plt.tight_layout()
plt.show()

print("Adelie Male penguins who has 191 mm flipper lengths is more probable")

#%%
# Question 14
Q14_fig = sns.displot(
    data=penguins,
    hue='species',
    kind='kde',
    x='flipper_length_mm',
    legend=True,
    fill=False
)

# Set the title and labels
Q14_fig.set(title='Kernel Density Estimation of Flipper Lengths by Species')
plt.tight_layout()
plt.show()

#%%
# Question 15
Q15_fig = sns.displot(
    data=penguins,
    hue='sex',
    kind='kde',
    x='flipper_length_mm',
    legend=True
)

# Set the title and labels
Q15_fig.set(title='Kernel Density Estimation of Flipper Lengths by Sex')

plt.tight_layout()
plt.show()

#%%
# Question 16
Q16_fig = sns.displot(
    data=penguins,
    hue='species',
    kind='kde',
    x='flipper_length_mm',
    legend=True,
    multiple='stack'
)

# Set the title and labels
Q16_fig.set(title='Kernel Density Estimation of Flipper Lengths by Species with stack')
plt.tight_layout()
plt.show()

#%%
# Question 17
Q17_fig = sns.displot(
    data=penguins,
    hue='sex',
    kind='kde',
    x='flipper_length_mm',
    legend=True,
    multiple='stack'
)

# Set the title and labels
Q17_fig.set(title='Kernel Density Estimation of Flipper Lengths by Sex with stack')

plt.tight_layout()
plt.show()

#%%
# Question 18
Q18_fig = sns.displot(
    data=penguins,
    hue='species',
    kind='kde',
    x='flipper_length_mm',
    legend=True,
    fill=True
)

# Set the title and labels
Q18_fig.set(title='Kernel Density Estimation of Flipper Lengths by Species with fill')
plt.tight_layout()
plt.show()

# Print the observation
print("The peak of Adelie penguins flipper length is 190mm, the density is 0.025")
print('The peak of Chinstrap penguins flipper length is 196mm and the density is about 0.012')
print("The peak of Gentoo penguins flipper length is 215mm, the density is about 0.018")
#%%
# Question 19
Q19_fig = sns.displot(
    data=penguins,
    hue='sex',
    kind='kde',
    x='flipper_length_mm',
    legend=True,
    fill=True
)

# Set the title and labels
Q19_fig.set(title='Kernel Density Estimation of Flipper Lengths by Sex with fill')

plt.tight_layout()
plt.show()

# Print the observation
print("The peak of female penguins flipper length is about 190mm, the density is 0.0176")
print("The peak of male penguins flipper length is about 195mm, the density is 0.0148")

#%%
# Question 20
# Create a scatter and regression plot for bill_length and bill_depth
Q20_fig = sns.regplot(
    data=penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    scatter_kws={'color':'blue'},
    line_kws={'color':'green'}
)

# Set title and xlabel, ylabel
Q20_fig.set(title='Scatter and regression plot for bill_length_mm and bill_depth_mm',
            xlabel='bill_length_mm',
            ylabel='bill_depth_mm')


# Show the plot
plt.tight_layout()
plt.show()

correlation = penguins['bill_length_mm'].corr(penguins['bill_depth_mm'])
print(f'The correlation between bill_length_mm and bill_depth_mm is {correlation:.2f}')
print('bill_length_mm and bill_depth_mm is negative correlation')

#%%
# Question 21
Q21_fig = sns.countplot(
    data=penguins,
    x='island',
    hue='species'
)

# Set the title and labels
Q21_fig.set(title='Count of Penguins by Island', xlabel='island', ylabel='count')

# Show the plot
plt.tight_layout()
plt.show()

# print the observation
print("Chinstrap penguins only live in the Dream island,\n "
      "Gentoo penguins only live in the Biscoe island,"
      "\n and Adelie penguins live in the all three islands")

#%%
# Question 22
Q22_fig = sns.countplot(
    data=penguins,
    x='sex',
    hue='species'
)

# Set the title and labels
Q22_fig.set(title='Count of Penguins by Sex', xlabel='sex', ylabel='count')

# Show the plot
plt.tight_layout()
plt.show()

# print the observation
print("Each category penguins have same proportion male and female")

#%%
# Question 23
Q23_fig = sns.kdeplot(
    data=penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    hue='sex',
    fill=True,
    grid=False
)

plt.tight_layout()
plt.show()

#%%
# Question 24
Q24_fig = sns.kdeplot(
    data=penguins,
    x='bill_length_mm',
    y='flipper_length_mm',
    hue='sex',
    fill=True,
    grid=False
)

plt.tight_layout()
plt.show()


#%%
# Question 25
Q25_fig = sns.kdeplot(
    data=penguins,
    x='flipper_length_mm',
    y='bill_depth_mm',
    hue='sex',
    fill=True,
    grid=False
)

plt.tight_layout()
plt.show()

#%%
# Question 26
fig_26, axes_26 = plt.subplots(3, 1, figsize=(8, 16))

sns.kdeplot(
    data=penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    hue='sex',
    fill=True,
    grid=False,
    ax=axes_26[0]
)

sns.kdeplot(
    data=penguins,
    x='bill_length_mm',
    y='flipper_length_mm',
    hue='sex',
    fill=True,
    grid=False,
    ax=axes_26[1]
)

sns.kdeplot(
    data=penguins,
    x='flipper_length_mm',
    y='bill_depth_mm',
    hue='sex',
    fill=True,
    grid=False,
    ax=axes_26[2]
)

plt.tight_layout()
plt.show()

# Print the observation
print("For general, male penguins tend to have longer flipper, bill and deeper bill,\n"
      "and longer bill longer flipper.\n "
      "Beside that, longer flipper tend to have lighter bill,\n and longer bill tend to have lighter bill.")

#%%
# Question 27
Q27_fig = sns.histplot(
    data=penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    hue='sex'
)


plt.tight_layout()
plt.show()

#%%
# Question 28
Q28_fig = sns.histplot(
    data=penguins,
    x='bill_length_mm',
    y='flipper_length_mm',
    hue='sex'
)


plt.tight_layout()
plt.show()

#%%
# Question 29
Q29_fig = sns.histplot(
    data=penguins,
    x='flipper_length_mm',
    y='bill_depth_mm',
    hue='sex'
)


plt.tight_layout()
plt.show()