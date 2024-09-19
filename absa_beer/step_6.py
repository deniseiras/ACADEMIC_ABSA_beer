"""
Step 4: Aspect-Based Sentiment Analysis of Beer Characteristics (CC)

:author: Denis Eiras

Functions:
    - 
"""

import pandas as pd
from step import Step
import re
import pandas as pd
from collections import Counter
import re
from wordcloud import WordCloud
from matplotlib.colors import to_hex


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
        categories = ["visual", "aroma", "sabor", "álcool", "amargor", "sensação na boca"]
        df_base_absa = df_base_absa[df_base_absa['category'].isin(categories)]

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
        
        # df_aroma_pos, df_aroma_neg = self.create_base(df_absa_join, 'review_aroma', 'aroma')
        # df_visual_pos, df_visual_neg = self.create_base(df_absa_join, 'review_visual', 'visual')
        # df_flavor_pos, df_flavor_neg = self.create_base(df_absa_join, 'review_flavor', 'sabor')
        # df_sensation_pos, df_sensation_neg = self.create_base(df_absa_join, 'review_sensation', 'sensação na boca')
        df_general_set_pos, df_general_set_neg = self.create_base(df_absa_join, 'review_general_set', None)
        
        stop_words = self.get_stop_words()
        # self.generate_word_cloud(df_aroma_pos, 'aroma_pos', stop_words)
        # self.generate_word_cloud(df_aroma_neg, 'aroma_neg', stop_words)
        # self.generate_word_cloud(df_visual_pos, 'visual_pos', stop_words)
        # self.generate_word_cloud(df_visual_neg, 'visual_neg', stop_words)
        # self.generate_word_cloud(df_flavor_pos, 'flavor_pos', stop_words)
        # self.generate_word_cloud(df_flavor_neg, 'flavor_neg', stop_words)
        # self.generate_word_cloud(df_sensation_pos, 'sensation_pos', stop_words)
        # self.generate_word_cloud(df_sensation_neg, 'sensation_neg', stop_words)
        self.generate_word_cloud(df_general_set_pos, 'general_set_pos', stop_words, categories, max_words=100)
        self.generate_word_cloud(df_general_set_neg, 'general_set_neg', stop_words, categories, max_words=100)


    def create_base(self, df_absa_join, column: str, category: str ):
        """
        This function creates a base with the desired column (aroma, visual, flavor, sensation, general_set)
        
        Args:
                df_absa_join (DataFrame): The dataframe with ABSA sentiments
        """
        pos_thres = 0.0  # 0.0 ignore rating 
        neg_thres = 5.0  # 5.0 ignore rating
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
        stop_words.update(['aroma', 'sabor', 'notas', 'muito boa', 'excelente', 'ótima','boa', 'ruim', 'gostei', 'não gostei', 'não me agradaram muito'])
       
        # print(stop_words)
        return stop_words
    
        
    def generate_word_cloud(self, df: pd.DataFrame, base_name: str, stop_words: list, categories: list, max_words=50):
        
        # Function to clean and extract words/entities
        def extract_entities(text, split_words=False):
            text = text.lower()
            text = re.sub(r'[^a-zà-ÿ\s,]', '', text)
            if split_words:
                words = re.split(r'[,\s]+', text)
                words = [word.strip() for word in words if word.strip() and word not in stop_words]
            else:
                words = [text] if text not in stop_words else []
            return words

        # Function to map category to a color
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            word_category = word_category_mapping.get(word, 'default')
            return category_colors.get(word_category, 'black')

        # # Create a list of unique categories and assign a color to each
        # cmap = get_cmap('tab10', len(categories))  # Get a colormap with enough colors for each category
        # category_colors = {category: rgb2hex(cmap(i)) for i, category in enumerate(categories)}
        
        category_colors = {
            "visual": "Blue",
            "aroma": "Orange",
            "sabor": "Green",
            "sensação na boca": "Red",
            "álcool": "Magenta",
            "amargor": "Lime"
        }
        # Convert the color names to hex codes
        category_colors_hex = {category: to_hex(color) for category, color in category_colors.items()}

        # Create a mapping between words and their corresponding category
        word_category_mapping = {}
        for index, row in df.iterrows():
            words = extract_entities(row['aspect'])
            for word in words:
                word_category_mapping[word] = row['category']

        # Apply extraction and count words
        word_list = df['aspect'].apply(extract_entities).sum()
        word_count = Counter(word_list)
        
        category_word_count = Counter(word_category_mapping.values())
        print(f'Number of words per category for {base_name}:')
        for category, count in category_word_count.items():
            print(f"{category}: {count} words")

        # Create a WordCloud with color function
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=max_words, color_func=color_func).generate_from_frequencies(word_count)

        # Save the word cloud image
        wordcloud_filename = f'{self.work_dir}/wordcloud_{base_name}_{max_words}_words.png'
        wordcloud.to_file(wordcloud_filename)

        # # Optionally show the WordCloud
        # plt.imshow(wordcloud, interpolation='bilinear')
        # plt.axis('off')
        # plt.show()
