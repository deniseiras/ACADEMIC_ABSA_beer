"""
Step 3: Aspect-Based Sentiment Analysis of Beer Characteristics (CC)

:author: Denis Eiras

Functions:
    - 
"""
import pandas as pd
from step import Step

class Step_3(Step):
    
    def __init__(self) -> None:
        super().__init__()
        
       
    def run(self):
        """Create Prompt Base

        Args:
            df (pandas.DataFrame): input dataframe
        
        Returns:
            pandas.DataFrame: output dataframe
        """
        
        file = f'{self.work_dir}/step_2.csv'
        self.df = Step.read_data(file)
        
        # df_distinct_styles_year = self.df.groupby(['beer_style', self.df['review_datetime'].dt.year]).nunique()
        # df_distinct_styles_year.to_csv(f'{data_folder}/disitinct_styles_year.csv')
        df_distinct_styles = self.df.groupby(['beer_style']).nunique()
        print(f'Total distinct styles = {len(df_distinct_styles)}')
        # print(f'Total beer count per distinct styles\n{df_distinct_styles[["beer_name"]]}')
        
        # From here, df_distinct_styles was mapped manually to the BJCP categories in file style_count_category.csv
        dtype_options = {
            'style': str,
            'beer_count': int,
            'bjcp_category': str
        }
        df_styles = Step.read_csv('./data/style_count_category.csv', dtype_options=dtype_options)

        # TODO Check ETAPA 3.1 
