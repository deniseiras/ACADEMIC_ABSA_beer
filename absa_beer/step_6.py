"""
Step 4: Aspect-Based Sentiment Analysis of Beer Characteristics (CC)

:author: Denis Eiras

Functions:
    - 
"""

import pandas as pd
import ast
from step import Step
from Prompt_AI import Prompt_AI
import re

class Step_6(Step):


    def __init__(self) -> None:
        super().__init__()


    def run(self):
        """
        This function runs Step 6: Creating "Bases Finais" 

        Args:
                self (object): The object instance that contains the data.

        Returns:
        """
        
        print(f'\n\nRunning Step 6\n================================')
        
        self.read_csv(f'{self.work_dir}/step_4.csv')
        df_base_absa = self.df

        self.read_csv(f'{self.work_dir}/step_3.csv')
        df_base_principal = self.df
               
        print(f'- Base Principal - line count: {len(df_base_principal)}')
        print(f'- Base ABSA      - line count: {len(df_base_absa)}')

        # Create a base joining df_base_principal and df_base_absa where df_base_principal review_general_set >= 4 and df_base_absa "sentiment" is "positivo" ou "muito positivo"
        df_base_principal_interested_columns = df_base_principal[['index', 'review_comment', 'review_datetime', 'beer_style', 'review_general_rate', 
            'review_aroma', 'review_visual', 'review_flavor', 'review_sensation', 'review_general_set']]
        df_absa_join = df_base_principal_interested_columns.join(df_base_absa.set_index('index'), on='index', how='inner')
        
        df_aroma_pos, df_aroma_neg = self.create_base(df_absa_join, 'review_aroma')
        df_visual_pos, df_visual_neg = self.create_base(df_absa_join, 'review_visual')
        df_flavor_pos, df_flavor_neg = self.create_base(df_absa_join, 'review_flavor')
        df_sensation_pos, df_sensation_neg = self.create_base(df_absa_join, 'review_sensation')
        df_general_set_pos, df_general_set_neg = self.create_base(df_absa_join, 'review_general_set')
        


    def create_base(self, df_absa_join, column: str ):
        """
        This function creates a base with the desired column (aroma, visual, flavor, sensation, general_set)
        
        Args:
                df_absa_join (DataFrame): The dataframe with ABSA sentiments
        """
        print(column.capitalize())
        df_absa_join_rev_pos = df_absa_join[df_absa_join[column] >= 4.0]
        df_absa_join_rev_neg = df_absa_join[df_absa_join[column] <= 2.0]
        print(f'- Reviews         POS / NEG: {len(df_absa_join_rev_pos)} / {len(df_absa_join_rev_neg)}')
        
        df_absa_join_absa_pos = df_absa_join[df_absa_join['sentiment'].isin(['positivo', 'muito positivo'])]
        df_absa_join_absa_neg = df_absa_join[df_absa_join['sentiment'].isin(['negativo', 'muito negativo'])]
        print(f'- ABSA            POS / NEG: {len(df_absa_join_absa_pos)} / {len(df_absa_join_absa_neg)}')
        
        df_pos = df_absa_join[(df_absa_join[column] >= 4.0) & (df_absa_join['sentiment'].isin(['positivo', 'muito positivo']))]
        df_pos.to_csv(f'{self.work_dir}/step_6_{column}_POS.csv', index=False)
                
        df_neg = df_absa_join[(df_absa_join[column] <= 2.0) & (df_absa_join['sentiment'].isin(['negativo', 'muito negativo']))]
        df_neg.to_csv(f'{self.work_dir}/step_6_{column}_NEG.csv', index=False)
        
        print(f'- Reviews & ABSA  POS / NEG: {len(df_pos)} / {len(df_neg)}')
        
        return df_pos, df_neg
        
        
        

        

        
