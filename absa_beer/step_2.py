"""
- Step 2: Data preprocessing

:author: Denis Eiras

Functions:
    - 
"""

import pandas as pd
from step import Step
import matplotlib.pyplot as plt

class Step_2(Step):

    def __init__(self) -> None:
        super().__init__()

    # Function to select rows where the "review_comment" column is not empty
    def select_rows_with_comment(self):
        """Select rows where the "review_comment" column is not empty

        Args:
            df (pandas.DataFrame): input dataframe

        Returns:
            pandas.DataFrame: output dataframe
        """
        print("Removing rows without comment")
        self.df = self.df[self.df["review_comment"].notna()]

    def sanitize_column(self, column_name, min_val, max_val):
        """Removes records outside range

        Args:
            df (pandas.DataFrame): input dataframe
            column_name (string): _description_
            min_val (float): minimum value
            max_val (float): maximum value

        Returns:
            pandas.DataFrame: output dataframe
        """
        invalid_data = self.df[
            (self.df[column_name] > max_val) | (self.df[column_name] < min_val)
        ]
        print(f"Removing {len(invalid_data)} lines of invalid {column_name}")
        self.df = self.df.drop(invalid_data.index)

    def sanitize_data(self):
        """Removes records outside range calling sanitize_column

        Args:
            df (pandas.DataFrame): input dataframe

        Returns:
            pandas.DataFrame: output dataframe
        """

        print("Sanitizing data")
        self.sanitize_column("beer_alcohol", 0, 100)
        self.sanitize_column("beer_srm", 0, 80)
        self.sanitize_column("beer_ibu", 0, 120)

    def run(self):
        """Convert types and transform evaluation values to [1-5] points. Remove invalid data
        based on range of values. Filter the non sense comments using OpenAI
        Args:
            df (pandas.DataFrame):

        Returns:
            pandas.DataFrame: the preprocessed pandas DataFrame
        """
        percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
        
        print(f'\n\nRunning Step 2\n================================')
        
        file = f"{self.work_dir}/step_1.csv"
        self.read_data(file)
        print(f"{len(self.df)} lines Total")

        # Remove duplicates
        print('Removing duplicates')
        self.df.drop_duplicates(inplace=True)
        print(f"{len(self.df)} lines Total")
        
        # select only rows with comments
        self.select_rows_with_comment()
        print(f"{len(self.df)} lines Total")

        # convert types and transform avaliation values to [1-5] points
        print("Preprocessing columns")
        self.df["review_datetime"] = pd.to_datetime(self.df["review_datetime"])
        self.df["beer_alcohol"] = (
            self.df["beer_alcohol"].str.replace("% ABV", "").astype(float)
        )
        self.df["beer_srm"] = (
            self.df["beer_srm"].str.replace(".", "").str.replace(",", ".").astype(float)
        )
        for field in [
            "review_aroma",
            "review_visual",
            "review_flavor",
            "review_sensation",
            "review_general_set",
        ]:
            self.df[field] = self.df[field].apply(eval).astype(float)
            self.df[field] = self.df[field] * 5

        # Remove invalid data based on range values
        print('Removing invalid data based on range values (ABV, SRM, IBU)')
        self.sanitize_data()
        print(f"{len(self.df)} lines Total")

        # create the col review_comment_size
        self.df["review_comment_size"] = self.df["review_comment"].str.len()
        
        
        # Looking for duplicates
        #
        dupl = self.df
        dupl = dupl[dupl.duplicated(keep='first')]
        print(f'Duplicates: {len(dupl)}')
        dupl.to_csv(f'{self.work_dir}/step_2__DUPLICATES.csv', index=False)
        self.df.drop(dupl.index, inplace=True)
        print(f"Still {len(self.df)} lines")


        # Looking for duplicates ignoring review_datetime
        #
        dupl = self.df
        dupl = dupl.drop(columns=['review_datetime'])  # ignore date time, because could be an database error
        dupl = dupl[dupl.duplicated(keep='first')]
        print(f'Removing Duplicates ignoring review_datetime: {len(dupl)}')
        dupl.to_csv(f'{self.work_dir}/step_2__DUPLICATES__ignoring__review_datetime.csv', index=False)
        col_keep = [col for col in self.df.columns if col not in ['review_datetime']]
        self.df.drop(dupl.index, inplace=True)
        print(f"Still {len(self.df)} lines")


        # Looking for duplicates of comments
        #
        dupl = self.df
        dupl_sub_set = ['review_comment' ]
        dupl = dupl[dupl.duplicated(subset=dupl_sub_set, keep='first')]
        print(f'Comment duplicates: {len(dupl)}')
        dupl.to_csv(f'{self.work_dir}/step_2__DUPLICATES__review_comment.csv', index=False)
        # self.df.drop(dupl.index, inplace=True)
        # print(f"Still {len(self.df)} lines")

        # Removing comments with less than 4 characters
        # most of the comments are garbage and some are equas "Boa"
        garbage = self.df[self.df['review_comment_size'] < 4]
        print(f'Removing comments with less than 4 characters: {len(garbage)}')
        garbage.to_csv(f'{self.work_dir}/step_2__DUPLICATES__review_comment__less_than_4_characters.csv', index=False)
        # show distinct review_comment of garbage variable
        print(garbage["review_comment"].unique())
        self.df.drop(garbage.index, inplace=True)
        print(f"Still {len(self.df)} lines")


        # Looking for duplicates of beer and user
        #
        dupl = self.df
        dupl_sub_set = ['review_user', 'beer_name' ]
        dupl = dupl[dupl.duplicated(subset=dupl_sub_set, keep=False)]
        print(f'Beer and user duplicates: {len(dupl)}')
        print(f'Some users evaluated more thar once a beer. Not a problem! Some beers have the same name but different versions. E.g. Bamberg Helles')
        # sort dupl by user_name and beer_name
        dupl = dupl.sort_values(by=['review_user', 'beer_name'])
        dupl.to_csv(f'{self.work_dir}/step_2__DUPLICATES__review_user__beer_name.csv', index=False)


        # General information about the data set
        #
        print('\nStatistics')
        print(self.df.describe(percentiles))
        print(f'\nNumber of users: {len(self.df["review_user"].unique())}')
        print(f'Number of reviews: {len(self.df)}')
        print(f'Number of beers: {len(self.df["beer_name"].unique())}')
        print(f'Number of breweries: {len(self.df["beer_brewery_name"].unique())}')
        print(f'Number of styles: {len(self.df["beer_style"].unique())}')

        unique_beer_count = self.df.groupby('beer_brewery_name')['beer_name'].nunique().sort_values(ascending=False)
        print(f'\nTop Brewery beers produced:\n{unique_beer_count.head(20).to_string()}')

        unique_beer_count = self.df.groupby('beer_brewery_name')['beer_style'].nunique().sort_values(ascending=False)
        print(f'\nTop Brewery styles produced:\n{unique_beer_count.to_string()}')

        beer_count = self.df.groupby('beer_name')['beer_name'].count().sort_values(ascending=False)
        print(f'\nTop reviwed beers:\n{beer_count.head(100).to_string()}')
        print(beer_count.describe(percentiles))

        style_count = self.df.groupby('beer_style')['beer_style'].count().sort_values(ascending=False)
        print(f'\nTop reviwed styles:\n{style_count.head(10).to_string()}')
        print(style_count.describe(percentiles))

        unique_active = self.df.groupby('beer_is_active')['beer_is_active'].count().sort_values(ascending=False)
        print(f'\nActive beers:\n{unique_active.to_string()}')
        unique_sazonal = self.df.groupby('beer_is_sazonal')['beer_is_sazonal'].count().sort_values(ascending=False)
        print(f'\nSazonal beers:\n{unique_sazonal.to_string()}')


        # Analisys of number of reviews per user
        #
        dfh = self.df
        dfh = dfh[["review_user", "review_num_reviews"]].drop_duplicates()
        dfh = dfh.sort_values(by="review_num_reviews", ascending=False)
        # Investigating Jefferson Chicone (duplicate review_num_review) and others with most reviews
        # users = ["Jefferson Chicone"]
        users_list = dfh["review_user"].head(10).to_list()
        user_str = self.df[self.df["review_user"].isin(users_list)]
        user_str = user_str.groupby(["review_user", "review_num_reviews"])['review_num_reviews'].count()
        user_str = user_str.sort_values(ascending=False)
        print(f'\nMost review users, review_num_reviews and count of lines:\n{user_str.to_string()}')
        print(f'\nUpdating the review_num_reviews of all ... now its okay!')
        self.df['review_num_reviews'] = self.df.groupby('review_user')['review_user'].transform('count')

        dfh = self.df
        dfh = dfh[["review_user", "review_num_reviews"]].drop_duplicates()
        dfh = dfh.sort_values(by="review_num_reviews", ascending=False)
        print(f'\nTop 10 users with most reviews:')
        print(dfh[["review_user","review_num_reviews"]].head(10))
        print(dfh.describe(percentiles))
        # Create histogram
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

        print(f"Step 2 final count: {len(self.df)} lines")                
        # generate the base
        self.df.to_csv(f'{self.work_dir}/step_2.csv', index=False)
