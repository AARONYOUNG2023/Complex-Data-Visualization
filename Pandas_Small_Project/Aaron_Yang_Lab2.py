#%%
# Visualization of Complex Data Section 11
# Lab 2
# By Aaron Yang

#%%
# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

#%%


# Load yf API
yf.pdr_override()
#%%
# Question 1
# Set the stocks list, the start date and end date
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = '2000-01-01'
end_date = '2023-06-06'

# Set a blank dictionary
df_rounded = {}

for stock in stocks:
    df_rounded[stock] = web.DataReader(stock, start=start_date, end=end_date).round(2)
    print(df_rounded[stock].head(5))

#%%
# Question 2
# Create a 3*2 matrix subplots using matplotlib
fig, axs = plt.subplots(3, 2)

# Flatten the axs
axs = axs.flatten()

# Create an index for the following axs[index]
index = 0

# Create a loop to put all data into subplots
for stock in stocks:
    df_rounded[stock]['High'].plot(ax=axs[index],
                                   color='blue',
                                   title=f'{stock} High History Price',
                                   xlabel='Date',
                                   ylabel='High Price USD($)',
                                   grid=True,
                                   legend=True,
                                   figsize=(16, 8))
    # To ensure after one loop, the next stock will be located the next position
    index += 1

# Show the plot
plt.tight_layout()
plt.show()

#%%
# Question 3
# Set a feature list
features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']

# Set a nested loop to build the repeated graphs for other 5 features
# Set an outer loop for each feature
for feature in features:
    # Skip the 'High' feature
    if feature == 'High':
        continue

    # Create 3*2 matrix graph using matplotlib
    fig, axs = plt.subplots(3, 2)

    # Flatten the axs
    axs = axs.flatten()

    # Create an index for the following axs[index]
    index = 0

    # Create subplots for each stock in the inner loop
    for stock in stocks:
        # Create a y_label to set a special y_label for the volume
        y_label = 'Volume (units)' if feature == 'Volume' else f'{feature} Price USD ($)'

        # Create a title to set a special title for the volume
        title = f'{stock} History Volume' if feature == 'Volume' else f'{stock} {feature} History Price'

        # Use pandas package to create plots
        df_rounded[stock][feature].plot(ax=axs[index],
                                        color='blue',
                                        xlabel='Date',
                                        ylabel=y_label,
                                        title=title,
                                        legend=True,
                                        grid=True,
                                        figsize=(16,8)
                                        )

        # After each stock, the position has to move into the next position
        index += 1

    # Show the plot
    plt.tight_layout()
    plt.show()

#%%
# Question 4
# Create a 3*2 matrix subplots using matplotlib
fig, axs = plt.subplots(3, 2)

# Flatten the axs
axs = axs.flatten()

# Create an index for the following axs[index]
index = 0

# Create a loop to put all data into Histogram
for stock in stocks:
    df_rounded[stock]['High'].plot(ax=axs[index],
                                   kind='hist',
                                   bins=50,
                                   title=f'{stock} High History Price Histogram',
                                   xlabel='Value in USD($)',
                                   ylabel='Frequency',
                                   grid=True,
                                   legend=True,
                                   figsize=(16, 8))
    # To ensure after one loop, the next stock will be located the next position
    index += 1

# Show the plot
plt.tight_layout()
plt.show()

#%%
# Question 5
# Set a feature list
features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']

# Set a nested loop to build the repeated graphs for other 5 features
# Set an outer loop for each feature
for feature in features:
    # Skip the 'High' feature
    if feature == 'High':
        continue

    # Create 3*2 matrix graph using matplotlib
    fig, axs = plt.subplots(3, 2)

    # Flatten the axs
    axs = axs.flatten()

    # Create an index for the following axs[index]
    index = 0

    # Create subplots for each stock in the inner loop
    for stock in stocks:
        # Create a x_label to set a special xlabel for the volume
        x_label = 'Volume (units)' if feature == 'Volume' else f'Value in USD($)'

        # Create a title to set a special title for the volume
        title = f'{stock} History Volume Histogram' if feature == 'Volume' else f'{stock} {feature} History Price Histogram'

        # Use pandas package to create plots
        df_rounded[stock][feature].plot(ax=axs[index],
                                        kind='hist',
                                        bins=50,
                                        xlabel=x_label,
                                        ylabel='Frequency',
                                        title=title,
                                        legend=True,
                                        grid=True,
                                        figsize=(16,8)
                                        )

        # After each stock, the position has to move into the next position
        index += 1

    # Show the plot
    plt.tight_layout()
    plt.show()

#%%
# Question 6
# Part 1: Display covariance matrix using prettytable package for 'AAPL'
from prettytable import PrettyTable

# Define a function to calculate the covariance
def cal_covariance(x, y):
    mean_x = x.mean()
    mean_y = y.mean()
    covariance = ((x-mean_x)*(y-mean_y)).sum()/(len(x)-1)
    covariance = round(covariance, 2)
    return covariance

# Initializa an empty DataFrame for the covariance matrix
covariance_matrix = pd.DataFrame(index = features, columns = features)

# Populate the covariance matrix
for feature_x in features:
    for feature_y in features:
        covariance_matrix.loc[feature_x][feature_y] = cal_covariance(df_rounded['AAPL'][feature_x], df_rounded['AAPL'][feature_y])

# Display the PrettyTable
table = PrettyTable()
table.field_names = [''] + features

for feature in features:
    row = [feature] + list(covariance_matrix.loc[feature])
    table.add_row(row)

print('Covariance Matrix for "AAPL":\n')
print(table)


# Part 2: Correlation matrix for 'AAPL'
# If I would like to know which two features have the highest correlation, and lowest correlation, I have to
# build the correlation matrix.

def cal_correlation(covariance, features):
    correlations_matrix = pd.DataFrame(index=features, columns=features)
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            correlation = covariance.iloc[i, j] / (np.sqrt(covariance.iloc[i, i]) * np.sqrt(covariance_matrix.iloc[j, j]))
            correlations_matrix.iloc[i, j] = correlation
            correlations_matrix.iloc[j, i] = correlation
    return correlations_matrix

correlations = cal_correlation(covariance_matrix, features)

# If I would like to find the two highest correlation features and two lowest correlation features, I have to use stack
# method to find them.
# Use numpy to fill all nan values
for i in range(len(correlations.columns)):
    for j in range(i+1):
        correlations.iloc[i, j] = np.nan

# Identity the pair with the highese and loweset correlation
max_corr_pair = correlations.stack().idxmax()
min_corr_pair = correlations.stack().idxmin()

print(f"The pair with the highest correlation: {max_corr_pair}")
print(f"The pair with the lowest correlation: {min_corr_pair}")

#%%
# Question 7
# Create a new convariance matrix for other 5 stocks
covariance_matrix_Q7 = pd.DataFrame(index = features, columns = features)

# Set a loop for generating the covariance matrix for other 5 stocks
for stock in stocks:
    # Skip the stock 'AAPL'
    if stock == 'AAPL':
        continue
    else:
        #
        for feature_x in features:
            for feature_y in features:
                covariance_matrix_Q7.loc[feature_x][feature_y] = cal_covariance(df_rounded[stock][feature_x], df_rounded[stock][feature_y])
        table = PrettyTable()
        table.field_names = [''] + features
        for feature in features:
            row = [feature] + list(covariance_matrix_Q7.loc[feature])
            table.add_row(row)
        print(f'Covariance Matrix for "{stock}":\n')
        print(table)

#%%
# Set a loop to find the two highest correlation features and two lowest correlation features for each stock
for stock in stocks:
    # Skip the stock 'AAPL'
    if stock == 'AAPL':
        continue
    else:
        correlations_Q7 = cal_correlation(covariance_matrix_Q7, features)
        for i in range(len(correlations_Q7.columns)):
            for j in range(i+1):
                correlations_Q7.iloc[i, j] = np.nan
        max_corr_pair_Q7 = correlations_Q7.stack().idxmax()
        min_corr_pair_Q7 = correlations_Q7.stack().idxmin()
        print(f"The pair with the highest correlation for {stock}: {max_corr_pair_Q7}")
        print(f"The pair with the lowest correlation for {stock}: {min_corr_pair_Q7}")
#%%
# Question 8
# Create the scatter plot for the stock 'AAPL'
df_Q8 = df_rounded['AAPL']

pd.plotting.scatter_matrix(df_Q8,
                           alpha=0.5,
                           s=10,
                           hist_kwds={'bins': 50},
                           diagonal='kde',
                           figsize=(12,12))

plt.suptitle('AAPL Stock Scatter Matrix')
plt.show()

#%%
# Question 9
for stock in stocks:
    if stock == 'AAPL':
        continue
    else:
        df = df_rounded[stock]

        scatter_matrix = pd.plotting.scatter_matrix(df,
                                                    alpha=0.5,
                                                    s=10,
                                                    hist_kwds={'bins': 50},
                                                    diagonal='kde',
                                                    figsize=(12, 12))

        plt.suptitle(f'{stock} Stock Scatter Matrix')
        plt.show()