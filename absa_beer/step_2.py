"""
- Step 2: Data preprocessing

:author: Denis Eiras

Functions:
    - 
"""

import pandas as pd
from step import Step

class Step_2(Step):
    
    def __init__(self) -> None:
        super().__init__()
   
    def remove_invalid_comments(self):
        """Remove invalid comments using AI

        Args:
            df (pandas.DataFrame): input dataframe
        
        Returns:
            pandas.DataFrame: output dataframe

        """
        print('Removing invalid comments')
        # invalid_comments = df[df['review_comment'].str.len() < 10]
        # df = df.drop(invalid_comments.index)
        
        # use openai_api.get_completion for selecting invalid comments
        
 

    # Function to select rows where the "review_comment" column is not empty
    def select_rows_with_comment(self):
        """Select rows where the "review_comment" column is not empty

        Args:
            df (pandas.DataFrame): input dataframe
        
        Returns:
            pandas.DataFrame: output dataframe
        """
        print('selecting rows with comment')
        self.df = self.df[self.df['review_comment'].notna()]


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
        invalid_data = self.df[(self.df[column_name] > max_val) | (self.df[column_name] < min_val)]
        print(f'Removing {len(invalid_data)} lines of invalid {column_name}')
        self.df = self.df.drop(invalid_data.index)


    def sanitize_data(self):
        """Removes records outside range calling sanitize_column

        Args:
            df (pandas.DataFrame): input dataframe

        Returns:
            pandas.DataFrame: output dataframe
        """

        print('Sanitizing data')
        self.df = self.sanitize_column('beer_alcohol', 0, 100)    
        self.df = self.sanitize_column('beer_srm', 0, 80)    
        self.df = self.sanitize_column('beer_ibu', 0, 120)    
                

    def run(self):
        """Convert types and transform evaluation values to [1-5] points. Remove invalid data
        based on range of values. Filter the non sense comments using OpenAI
        Args:
            df (pandas.DataFrame):

        Returns:
            pandas.DataFrame: the preprocessed pandas DataFrame
        """
        
        file = f'{self.work_dir}/step_1.csv'
        self.read_data(file)
        print(f'{len(self.df)} lines Total')
        
        # select only rows with comments 
        self.select_rows_with_comment()
        print(f'{len(self.df)} lines Total')

        # convert types and transform avaliation values to [1-5] points
        print('preprocessing columns')
        self.df['review_datetime'] = pd.to_datetime(self.df['review_datetime'])
        self.df['beer_alcohol'] = self.df['beer_alcohol'].str.replace('% ABV', '').astype(float)
        self.df['beer_srm'] = self.df['beer_srm'].str.replace('.', '').str.replace(',', '.').astype(float)
        for field in ['review_aroma', 'review_visual', 'review_flavor', 'review_sensation', 'review_general_set']:
            self.df[field] = self.df[field].apply(eval).astype(float)
            self.df[field] = self.df[field] * 5
        
        # Remove invalid data based on range values
        self.sanitize_data()
        print(f'{len(self.df)} lines Total')
        
        # create a column with the size of comments
        # self.df['comment_size'] = self.df['review_comment'].str.len()

        self.remove_invalid_comments()
        
        # generate the base
        self.df.to_csv(f'{self.work_dir}/step_2.csv')