"""_summary_
"""

# Import necessary libraries
from step import Step
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.max_colwidth = None
# keep = False
keep = 'first'

# Create an instance of the Step class
anal = Step()

# Specify the file path for the CSV file to be read
file = f'{anal.work_dir}/step_2.csv'
anal.read_csv(file)
df = anal.df

print(f"Initial: {len(df)} lines")


# Looking for duplicates
#
dupl = df
dupl = dupl[dupl.duplicated(keep=keep)]
print(f'Duplicates: {len(dupl)}')
dupl.to_csv(f'{anal.work_dir}/step_2__DUPLICATES.csv', index=False)
df.drop(dupl.index, inplace=True)
print(f"Still {len(df)} lines")


# Looking for duplicates ignoring review_datetime
#
dupl = df
dupl = dupl.drop(columns=['review_datetime'])  # ignore date time, because could be an database error
dupl = dupl[dupl.duplicated(keep=keep)]
print(f'Removing Duplicates ignoring review_datetime: {len(dupl)}')
dupl.to_csv(f'{anal.work_dir}/step_2__DUPLICATES__ignoring__review_datetime.csv', index=False)
col_keep = [col for col in df.columns if col not in ['review_datetime']]
df.drop(dupl.index, inplace=True)
print(f"Still {len(df)} lines")


# Looking for duplicates of comments
#
dupl = df
dupl_sub_set = ['review_comment' ]
dupl = dupl[dupl.duplicated(subset=dupl_sub_set, keep=keep)]
print(f'Comment duplicates: {len(dupl)}')
dupl.to_csv(f'{anal.work_dir}/step_2__DUPLICATES__review_comment.csv', index=False)
# df.drop(dupl.index, inplace=True)
# print(f"Still {len(df)} lines")

# Removing comments with less than 4 characters
# most of the comments are garbage and some are equas "Boa"
garbage = df[df['review_comment_size'] < 4]
print(f'Removing comments with less than 4 characters: {len(garbage)}')
garbage.to_csv(f'{anal.work_dir}/step_2__DUPLICATES__review_comment__less_than_4_characters.csv', index=False)
# show distinct review_comment of garbage variable
print(garbage["review_comment"].unique())
df.drop(garbage.index, inplace=True)
print(f"Still {len(df)} lines")


# Looking for duplicates of beer and user
#
dupl = df
dupl_sub_set = ['review_user', 'beer_name' ]
dupl = dupl[dupl.duplicated(subset=dupl_sub_set, keep=keep)]
print(f'Beer and user duplicates: {len(dupl)}')
dupl.to_csv(f'{anal.work_dir}/step_2__DUPLICATES__review_user__beer_name.csv', index=False)

df.to_csv(f'{anal.work_dir}/step_2__data_analisys__BEFORE_MOVE_TO_STEP2.csv', index=False)


# General information about the data set
#
print('\nStatistics\n==========')
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





