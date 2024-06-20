"""_summary_
"""

# Import necessary libraries
from step import Step
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.max_colwidth = None

# Create an instance of the Step class
anal = Step()

# Specify the file path for the CSV file to be read
file = f'{anal.work_dir}/step_2.csv'
anal.read_csv(file)
df = anal.df


# Pre processing
#
df.drop(df.columns[0], axis=1, inplace=True)
df["review_comment_size"] = df["review_comment"].str.len()


# Looking for duplicates of beer and user
#
duplicates = df[df.duplicated(subset=['review_user', 'beer_name' ], keep=False)]
print(f'Beer and user duplicates: {len(duplicates)}')
duplicates.to_csv(f'{anal.work_dir}/step_2__DUPLICATES__review_user__beer_name.csv')

# print(duplicates[['review_user', 'beer_name']].head(10))
print(duplicates)

# Looking for duplicates of comments
#
# duplicates = df[df.duplicated(subset=['review_comment' ], keep=False)]
# print(f'Comment duplicates: {len(duplicates)}')
# print(duplicates[['review_user', 'beer_name', 'review_num_reviews', 'review_comment']].head(10))


# print('Removing duplicates')
# self.df.drop_duplicates(inplace=True)

exit()


# General information about the data set
#
print(df.describe())
print(f'\nNumber of users: {len(df["review_user"].unique())}')
print(f'Number of reviews: {len(df)}')
print(f'Number of beers: {len(df["beer_name"].unique())}')
print(f'Number of breweries: {len(df["beer_brewery_name"].unique())}')
print(f'Number of styles: {len(df["beer_style"].unique())}')

unique_beer_count = df.groupby('beer_brewery_name')['beer_name'].nunique().sort_values(ascending=False)
print(f'\nTop Brewery beers produced:\n{unique_beer_count.head(20).to_string()}')

unique_beer_count = df.groupby('beer_brewery_name')['beer_style'].nunique().sort_values(ascending=False)
print(f'\nTop Brewery styles produced:\n{unique_beer_count.head(20).to_string()}')

unique_active = df.groupby('beer_is_active')['beer_is_active'].count().sort_values(ascending=False)
print(f'\nActive beers:\n{unique_active.to_string()}')
unique_sazonal = df.groupby('beer_is_sazonal')['beer_is_sazonal'].count().sort_values(ascending=False)
print(f'\nSazonal beers:\n{unique_sazonal.to_string()}')


# Analisys of number of reviews per user
#
dfh = df
dfh = dfh[["review_user", "review_num_reviews"]].drop_duplicates()
dfh = dfh.sort_values(by="review_num_reviews", ascending=False)
users_per_bin = 50
num_bins = len(dfh) // users_per_bin
bin_labels = []
sum_reviews = []
# Loop through each bin and calculate the sum of reviews
for i in range(num_bins):
    start_index = i * users_per_bin
    end_index = (i + 1) * users_per_bin
    bin_dfh = dfh[start_index:end_index]
    bin_sum = bin_dfh["review_num_reviews"].sum()
    bin_labels.append(f'Bin {i+1}')
    sum_reviews.append(bin_sum)
# Create the bar graph
plt.figure(figsize=(10, 10))
plt.bar(bin_labels, sum_reviews)
plt.xlabel(f'User Bins (each containing {users_per_bin} users)')
plt.ylabel('Sum of Reviews')
plt.title('Sum of Reviews per User Bin')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
print(f'\nTop 20 users with most reviews:')
print(dfh[["review_user","review_num_reviews"]].head(20))





