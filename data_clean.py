# modules we'll use
import pandas as pd
import numpy as np
# read in all our data
post_read = pd.read_csv(r"G:\uta_acad\Sem 3\IE 6304 Data mining and analytics\Project\Postures.csv")
df = post_read.iloc[:, 0:38]
# print(df)
# set seed for reproducibility
np.random.seed(0)
# Read ? as Nan
df = df.replace('?', np.nan)
# look at a few rows
# print(post_read.sample(5))

# get the number of missing data points per column

missing_values_count = df.isnull().sum()
print(missing_values_count)
# look at the # of missing points in the first ten columns
# missing_values_count


# how many total missing values do we have?
total_cells = np.product(df.shape)
# print(total_cells)
total_missing = missing_values_count.sum()
print(total_missing)

# percent of data that is missing
# print((total_missing/total_cells) * 100, "Percent")

# remove all the rows that contain a missing value
df = df.dropna()
# New_df = df.dropna()
# print(New_df)

# Data for plots
df_p = df.iloc[:, 11:14]
print(df_p)

# New_df.to_csv("Postures_pruned", sep=',', index=False)

# s = New_df.describe()
# print(s)
