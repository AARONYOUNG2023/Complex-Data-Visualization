# Visualization of Complex Data, DATAS 6401, Section 11, Homework1, Aaron_Yang

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
import yfinance as yf
from prettytable import PrettyTable
from tabulate import tabulate
#%%
# Question 1
# Load API
yf.pdr_override()

# Set the Array stocks list and the start date and end date for loading stock data
stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
start_date = '2013-01-01'
end_date = '2023-08-28'

# Create a dictionary to hold the data
df_rounded = {}

# Fetch the stock data for each stock
for stock in stocks:
    print(f"The Data for {stock}")
    df = data.get_data_yahoo(stock, start=start_date, end=end_date)
    df_rounded[stock] = df.round(2)
    # print the first 5 rows of each stock
    print(df_rounded[stock])
#%%
# Question 2
# Calculate the mean of df
# Set a dictionary variable to store the data
mean_df = {}

# Build a loop to load data into the dict
for stock in stocks:
    mean_df[stock] = df_rounded[stock].mean().round(2)

# Convert the mean dict to a dataframe
mean_df = pd.DataFrame(mean_df).round(2)

# Calculate the max and min values
max_val = mean_df.max(axis=1).round(2)
min_val = mean_df.min(axis=1).round(2)

# Sort the company with the max or min value
max_company = mean_df.idxmax(axis=1)
min_company = mean_df.idxmin(axis=1)

# Add some new columns to the mean_df dataframe
mean_df['Maximum Value'] = max_val
mean_df['Minimum Value'] = min_val
mean_df['Maximum company name'] = max_company
mean_df['Minimum company name'] = min_company

# Transpose the mean_df to a new one
mean_df = mean_df.T

# Change the order of mean_df columns name
new_col_name = {
    'Open' : "Open ($)",
    'High' : "High ($)",
    'Low' : "Low ($)",
    'Close' : "Close ($)",
    'Volume' : "Volume",
    'Adj Close' : "Adj Close ($)"
}

# Rename the dataframe columns name
mean_df.rename(columns = new_col_name, inplace=True)

# Define a new order
new_order = ['High ($)', 'Low ($)', 'Open ($)', 'Close ($)', 'Volume', 'Adj Close ($)']

# Apply the new order to the dataframe
mean_df = mean_df[new_order]

# Create the table through tabulate package
table_Q2 = tabulate(mean_df, headers= [' ', 'High ($)', 'Low ($)', 'Open ($)', 'Close ($)', 'Volume', 'Adj Close ($)'], tablefmt='fancy_grid')
#%%
print('                                            Mean Value Comparison           \n')
print(table_Q2)

#%%
# Question 3
# Calculate the variance of the data
# Set a dictionary variable to store the data
var_df = {}

# Build a loop to load data into the dict
for stock in stocks:
    var_df[stock] = df_rounded[stock].var().round(2)

# Transfer the var_df to DataFrame
var_df = pd.DataFrame(var_df).round(2)

# Calculate the max and min value from the var_df
max_val_var = var_df.max(axis=1).round(2)
min_val_var = var_df.min(axis=1).round(2)

# Sort the company with the max or min value
max_company_var = var_df.idxmax(axis=1)
min_company_var = var_df.idxmin(axis=1)

# Add some new columns to the var_df dataframe
var_df['Maximum Value'] = max_val_var
var_df['Minimum Value'] = min_val_var
var_df['Maximum company name'] = max_company_var
var_df['Minimum company name'] = min_company_var

# Transpose the mean_df to a new one
var_df = var_df.T

# Rename the var_df
var_df.rename(columns=new_col_name, inplace=True)

# Reorder the var_df
var_df = var_df[new_order]

# Create a new table
table_Q3 = tabulate(var_df,
                    headers= [' ', 'High ($)', 'Low ($)', 'Open ($)', 'Close ($)', 'Volume', 'Adj Close ($)'],
                    tablefmt='fancy_grid'
                    )

#%%
print("                                          Variance comparison\n")
print(table_Q3)

#%%
# Question 4
# Create the table 'Standard Deviation Value comparison'
# Build a blank dict
std_df = {}

# Build a loop to load data into the dict
for stock in stocks:
    std_df[stock] = df_rounded[stock].std().round(2)

# Transfer the dataset to DataFrame
std_df = pd.DataFrame(std_df).round(2)

# Calculate the max and min value
max_val_std = std_df.max(axis=1).round(2)
min_val_std = std_df.min(axis=1).round(2)

# Sort the company with the max or min value
max_company_std = std_df.idxmax(axis=1)
min_company_std = std_df.idxmin(axis=1)

# Add some new columns to the dataframe
std_df['Maximum Value'] = max_val_std
std_df['Minimum Value'] = min_val_std
std_df['Maximum company name'] = max_company_std
std_df['Minimum company name'] = min_company_std

# Transpose the dataframe to a new one
std_df = std_df.T

# Rename the dataframe
std_df.rename(columns=new_col_name, inplace=True)

# Reorder the dataframe
std_df = std_df[new_order]

# Create a new table
table_Q4 = tabulate(std_df,
                    headers= [' ', 'High ($)', 'Low ($)', 'Open ($)', 'Close ($)', 'Volume', 'Adj Close ($)'],
                    tablefmt='fancy_grid'
                    )

#%%
print("                                          Standard Deviation Value comparison\n")
print(table_Q4)

#%%
# Question 5
# Create the table 'Median Value comparison'
# Build a blank dict
med_df = {}

# Build a loop to load data into the dict
for stock in stocks:
    med_df[stock] = df_rounded[stock].median().round(2)

# Transfer the dataset to DataFrame
med_df = pd.DataFrame(med_df).round(2)

# Calculate the max and min value
max_val_med= med_df.max(axis=1).round(2)
min_val_med = med_df.min(axis=1).round(2)

# Sort the company with the max or min value
max_company_med = med_df.idxmax(axis=1)
min_company_med = med_df.idxmin(axis=1)

# Add some new columns to the dataframe
med_df['Maximum Value'] = max_val_med
med_df['Minimum Value'] = min_val_med
med_df['Maximum company name'] = max_company_med
med_df['Minimum company name'] = min_company_med

# Transpose the dataframe to a new one
med_df = med_df.T

# Rename the dataframe
med_df.rename(columns=new_col_name, inplace=True)

# Reorder the dataframe
med_df = med_df[new_order]

# Create a new table
table_Q5 = tabulate(med_df,
                    headers= [' ', 'High ($)', 'Low ($)', 'Open ($)', 'Close ($)', 'Volume', 'Adj Close ($)'],
                    tablefmt='fancy_grid'
                    )

#%%
print("                                          Median Value comparison\n")
print(table_Q5)

#%%
# Question 6
# Calculate the correlation matrix for the APPLE company with all given features
corr_matrix_aapl = df_rounded['AAPL'].corr()

# Put the correlation matrix into a tabulate table
corr_table_aapl = tabulate(corr_matrix_aapl, headers='keys', tablefmt='fancy_grid', floatfmt=".2f")

print("                   AAPL Correlation Matrix\n")
print(corr_table_aapl)

#%%
# Question 7
# Repeat the correlation matrix for each stock
corr_matrix_orcl = df_rounded['ORCL'].corr()
corr_matrix_tsla = df_rounded['TSLA'].corr()
corr_matrix_ibm = df_rounded['IBM'].corr()
corr_matrix_yelp = df_rounded['YELP'].corr()
corr_matrix_msft = df_rounded['MSFT'].corr()


# Put the correlation matrix into a tabulate table
corr_table_orcl = tabulate(corr_matrix_orcl, headers='keys', tablefmt='fancy_grid', floatfmt=".2f")
corr_table_tsla = tabulate(corr_matrix_tsla, headers='keys', tablefmt='fancy_grid', floatfmt=".2f")
corr_table_ibm = tabulate(corr_matrix_ibm, headers='keys', tablefmt='fancy_grid', floatfmt=".2f")
corr_table_yelp = tabulate(corr_matrix_yelp, headers='keys', tablefmt='fancy_grid', floatfmt=".2f")
corr_table_msft = tabulate(corr_matrix_msft, headers='keys', tablefmt='fancy_grid', floatfmt=".2f")


print("                   ORCL Correlation Matrix\n")
print(corr_table_orcl)

print("                   TSLA Correlation Matrix\n")
print(corr_table_tsla)

print("                   IBM Correlation Matrix\n")
print(corr_table_ibm)

print("                   YELP Correlation Matrix\n")
print(corr_table_yelp)

print("                   MSFT Correlation Matrix\n")
print(corr_table_msft)


#%%
# Question 8
# =================================================
# The strategy to choose one stock to invest
# =================================================

# Sort the close and adj Close into a new variable
std_close = std_df[['Close ($)', 'Adj Close ($)']]

# Create a new tabulate table
invest_table = tabulate(std_close, headers='keys', tablefmt='fancy_grid')

print("Comparison Table based on Stardard Deviation of Close and Adj Close\n")
print(invest_table)