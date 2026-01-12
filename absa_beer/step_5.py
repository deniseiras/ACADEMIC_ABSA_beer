"""
Step 5: Generating Final Bases

:author: Denis Eiras

Functions:
    - 
"""

from step import Step

import pandas as pd
import re


def remove_obvious_words(aspect: str):
    """ if the word aspect starts with "<word> de" or "<word> ", remove "<word> <de>" from the aspect, where <word> is a word in word_list below

    Args:
        aspect (string): aspect to refactor

    Returns:
        string: aspect with obvious words removed
    """
    # if the word aspect starts with "<word> de" or "<word> ", remove "<word> <de>" from the aspect, where <word> is a word in word_list below
    word_list = ['aroma', 'sabor', 'amargor', 'álcool' ]
    aspect_ret = aspect
    for word in word_list:
        if word in aspect:
            # change 2 or more spaces to one space regex
            aspect_ret = re.sub(r'\s{2,}', ' ', aspect_ret)
            split = aspect_ret.split()
            if len(split) > 1:
                if len(split) == 2:
                    if split[1] == word:
                        aspect_ret = split[0]
                if split[0] == word:
                    if split[1] == 'de':
                        aspect_ret = aspect_ret.replace(f'{word} de ', '')
                    else:
                        aspect_ret = aspect_ret.replace(f'{word} ', '')
            break
        
    return aspect_ret


class Step_5(Step):


    def __init__(self) -> None:
        super().__init__()


    def run(self):
        """
        This function runs Step 5: Creating "Bases Finais" 

        Args:
                self (object): The object instance that contains the data.

        Returns:
        """
        
        print(f'\n\nRunning Step 5\n================================')
                
        self.read_csv(f'{self.work_dir}/step_4.csv')
        df_base_absa = self.df
        categories = self.get_category_list()
        df_base_absa = df_base_absa[df_base_absa['category'].isin(categories)]

        self.read_csv(f'{self.work_dir}/step_3.csv')
        df_base_principal = self.df       

        # Create a base joining df_base_principal, df_base_absa and df_base_sa
        df_base_absa_interested_columns = df_base_principal[['index', 'review_comment', 'review_datetime', 'beer_style', 'review_general_rate', 
            'review_aroma', 'review_visual', 'review_flavor', 'review_sensation', 'review_general_set']]
        df_absa_join = df_base_absa_interested_columns.join(df_base_absa.set_index('index'), on='index', how='inner')
        df_absa_join["review_datetime"] = pd.to_datetime(df_absa_join["review_datetime"])
        df_absa_join['year'] = df_absa_join['review_datetime'].dt.year
        df_absa_join['aspect'] = df_absa_join['aspect'].apply(remove_obvious_words)
                
        print(f'- Base Principal - line count: {len(df_base_principal)}')
        print(f'- Base ABSA      - line count: {len(df_base_absa)}')
        print(f'- Base Joined    - line count: {len(df_absa_join)}')
        
        # save df_absa_sa_join and df_sa_join to csv files
        df_absa_join.to_csv(f'{self.work_dir}/step_5_join_ABSA-PRINCIPAL.csv', index=False)        

    def create_base(self, df_absa_join, category:str = None ):  # column: str ):
        """
        This function creates a base with the desired column (aroma, visual, flavor, sensation, general_set)
        
        Args:
                df_absa_join (DataFrame): The dataframe with ABSA sentiments
        """
        
        print(f'Creating base for {category}')
        
        df_absa_join_absa_pos = df_absa_join[
            (df_absa_join['sentiment'].isin(['positivo', 'muito positivo'])) ]            
        df_absa_join_absa_neg = df_absa_join[
            (df_absa_join['sentiment'].isin(['negativo', 'muito negativo'])) ]
        if category is not None:
            df_absa_join_absa_pos = df_absa_join_absa_pos[df_absa_join_absa_pos['category'] == category]
            df_absa_join_absa_neg = df_absa_join_absa_neg[df_absa_join_absa_neg['category'] == category]
        print(f'- ABSA of {category} count:  POS / NEG: {len(df_absa_join_absa_pos)} / {len(df_absa_join_absa_neg)}')
                
        df_pos = df_absa_join_absa_pos
        df_neg = df_absa_join_absa_neg
        # print(f'- Reviews & ABSA  POS / NEG: {len(df_pos)} / {len(df_neg)}')
        
        if category is None:
            categ_str = 'all_cats'
        else:
            categ_str = category
        self.save_df_pos_neg(categ_str, df_pos, df_neg)
        
        return df_pos, df_neg


    def save_df_pos_neg(self, base_name, df_pos, df_neg):
        df_pos.to_csv(f'{self.work_dir}/step_5_{base_name}_POS.csv', index=False)
        df_neg.to_csv(f'{self.work_dir}/step_5_{base_name}_NEG.csv', index=False)