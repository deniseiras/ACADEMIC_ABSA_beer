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

        # Create a base joining df_base_principal and df_base_absa where df_base_principal review_general_set >= 4 and df_base_absa "sentiment" is "positivo" ou "muito positivo"
        df_base_principal_interested_columns = df_base_principal[['index', 'review_comment', 'review_datetime', 'beer_style', 'review_general_rate', 
            'review_aroma', 'review_visual', 'review_flavor', 'review_sensation', 'review_general_set']]
        df_absa_join = df_base_principal_interested_columns.join(df_base_absa.set_index('index'), on='index', how='inner')
        df_absa_join.to_csv(f'{self.work_dir}/step_6_join_ABSA-PRINCIPAL.csv', index=False)
        
        print(f'- Base Principal - line count: {len(df_base_principal)}')
        print(f'- Base ABSA      - line count: {len(df_base_absa)}')
        print(f'- Base Joined    - line count: {len(df_absa_join)}')
        
        df_aroma_pos, df_aroma_neg = self.create_base(df_absa_join, 'review_aroma', 'aroma')
        df_visual_pos, df_visual_neg = self.create_base(df_absa_join, 'review_visual', 'visual')
        df_flavor_pos, df_flavor_neg = self.create_base(df_absa_join, 'review_flavor', 'sabor')
        df_sensation_pos, df_sensation_neg = self.create_base(df_absa_join, 'review_sensation', 'sensação na boca')
        df_general_set_pos, df_general_set_neg = self.create_base(df_absa_join, 'review_general_set', None)
        
        self.generate_word_cloud(df_aroma_pos, 'aroma_pos')
        self.generate_word_cloud(df_aroma_neg, 'aroma_neg')
        self.generate_word_cloud(df_visual_pos, 'visual_pos')
        self.generate_word_cloud(df_visual_neg, 'visual_neg')
        self.generate_word_cloud(df_flavor_pos, 'flavor_pos')
        self.generate_word_cloud(df_flavor_neg, 'flavor_neg')
        self.generate_word_cloud(df_sensation_pos, 'sensation_pos')
        self.generate_word_cloud(df_sensation_neg, 'sensation_neg')
        self.generate_word_cloud(df_general_set_pos, 'general_set_pos')        
        self.generate_word_cloud(df_general_set_neg, 'general_set_neg')


    def create_base(self, df_absa_join, column: str, category: str ):
        """
        This function creates a base with the desired column (aroma, visual, flavor, sensation, general_set)
        
        Args:
                df_absa_join (DataFrame): The dataframe with ABSA sentiments
        """
        pos_thres = 3.0
        neg_thres = 2.0
        print(column.capitalize())
        df_absa_join_rev_pos = df_absa_join[df_absa_join[column] >= pos_thres]
        df_absa_join_rev_neg = df_absa_join[df_absa_join[column] <= neg_thres]
        print(f'- Reviews         POS / NEG: {len(df_absa_join_rev_pos)} / {len(df_absa_join_rev_neg)}')
        
        df_absa_join_absa_pos = df_absa_join[
            (df_absa_join['sentiment'].isin(['positivo', 'muito positivo'])) ]            
        df_absa_join_absa_neg = df_absa_join[
            (df_absa_join['sentiment'].isin(['negativo', 'muito negativo'])) ]
        if category is not None:
            df_absa_join_absa_pos = df_absa_join_absa_pos[df_absa_join_absa_pos['category'] == category]
            df_absa_join_absa_neg = df_absa_join_absa_neg[df_absa_join_absa_neg['category'] == category]
        print(f'- ABSA            POS / NEG: {len(df_absa_join_absa_pos)} / {len(df_absa_join_absa_neg)}')
        
        df_pos = df_absa_join[
            (df_absa_join[column] > pos_thres) & 
            (df_absa_join['sentiment'].isin(['positivo', 'muito positivo'])) ]
                
        df_neg = df_absa_join[
            (df_absa_join[column] <= neg_thres) & 
            (df_absa_join['sentiment'].isin(['negativo', 'muito negativo'])) ]
        
        if category is not None:
            df_pos = df_pos[df_pos['category'] == category]
            df_neg = df_neg[df_neg['category'] == category]
        
        df_pos.to_csv(f'{self.work_dir}/step_6_{column}_POS.csv', index=False)
        df_neg.to_csv(f'{self.work_dir}/step_6_{column}_NEG.csv', index=False)
        
        print(f'- Reviews & ABSA  POS / NEG: {len(df_pos)} / {len(df_neg)}')
        
        return df_pos, df_neg
        
    def get_stop_words(self):
        from nltk.corpus import stopwords
        import nltk

        # Define stop words (Portuguese)
        nltk.download('stopwords')
        stop_words = set(stopwords.words('portuguese'))
        stop_words.update(['aroma', 'sabor', 'notas'])
       
        print(stop_words)
        return stop_words
        
    def generate_word_cloud(self, df: pd.DataFrame, base_name: str):
        
        import pandas as pd
        from collections import Counter
        import re
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        stop_words = self.get_stop_words()
        
        # Function to clean and extract words/entities
        def extract_entities(text):
            # Convert to lowercase
            text = text.lower()
            # Remove non-alphabetic characters
            text = re.sub(r'[^a-zà-ÿ\s,]', '', text)
            # Split by commas and spaces, and strip whitespace
            words = re.split(r'[,\s]+', text)
            # Remove stop words and empty strings
            words = [word.strip() for word in words if word.strip() and word not in stop_words]
            return words

        # Apply extraction and count words
        word_list = df['aspect'].apply(extract_entities).sum()  # Flatten the list
        word_count = Counter(word_list)

        # Print word counts
        print(len(word_count))
        print(word_count)

        # Create a WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=50).generate_from_frequencies(word_count)
        wordcloud_filename = f'{self.work_dir}/wordcloud_{base_name}.png'
        wordcloud.to_file(wordcloud_filename)
        print(f"Word cloud saved as {wordcloud_filename}")
                
                

                

                
