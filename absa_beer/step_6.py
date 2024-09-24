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
        
        self.read_csv(f'{self.work_dir}/step_5.csv')
        df_base_as = self.df 
        df_base_as = df_base_as.rename(columns={'sentiment': 'sentiment_as'})

        # Create a base joining df_base_principal and df_base_absa where df_base_principal review_general_set >= 4 and df_base_absa "sentiment" is "positivo" ou "muito positivo"
        df_base_asba_interested_columns = df_base_principal[['index', 'review_comment', 'review_datetime', 'beer_style', 'review_general_rate', 
            'review_aroma', 'review_visual', 'review_flavor', 'review_sensation', 'review_general_set']]
        df_absa_as_join = df_base_asba_interested_columns.join(df_base_absa.set_index('index'), on='index', how='inner')
        df_absa_as_join = df_absa_as_join.join(df_base_as.set_index('index'), on='index', how='inner')
        df_absa_as_join.to_csv(f'{self.work_dir}/step_6_join_ABSA-AS-PRINCIPAL.csv', index=False)
        
        print(f'- Base Principal - line count: {len(df_base_principal)}')
        print(f'- Base ABSA      - line count: {len(df_base_absa)}')
        print(f'- Base Joined    - line count: {len(df_absa_as_join)}')
        
        df_aroma_pos, df_aroma_neg = self.create_base(df_absa_as_join, 'aroma')
        df_visual_pos, df_visual_neg = self.create_base(df_absa_as_join, 'visual')
        df_flavor_pos, df_flavor_neg = self.create_base(df_absa_as_join, 'sabor')
        df_sensation_pos, df_sensation_neg = self.create_base(df_absa_as_join, 'sensação na boca')
        df_amargor_pos, df_amargor_neg = self.create_base(df_absa_as_join, 'amargor')
        df_alcool_pos, df_alcool_neg = self.create_base(df_absa_as_join, 'álcool')
        
        # include all categories, also alcool, amargor
        df_all_cats_pos, df_all_cats_neg = self.create_base(df_absa_as_join)
        
        stop_words = self.get_stop_words()
        max_words = 75
        split_words = False
        self.generate_word_cloud(df_aroma_pos, 'aroma_pos', stop_words, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_aroma_neg, 'aroma_neg', stop_words, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_visual_pos, 'visual_pos', stop_words, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_visual_neg, 'visual_neg', stop_words, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_flavor_pos, 'flavor_pos', stop_words, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_flavor_neg, 'flavor_neg', stop_words, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_sensation_pos, 'sensation_pos', stop_words, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_sensation_neg, 'sensation_neg', stop_words, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_amargor_pos, 'amargor_pos', stop_words, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_amargor_neg, 'amargor_neg', stop_words, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_alcool_pos, 'alcool_pos', stop_words, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_alcool_neg, 'alcool_neg', stop_words, categories, max_words=max_words, split_words=split_words)
        
        self.generate_word_cloud(df_all_cats_pos, 'all_cats_pos', stop_words, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_all_cats_neg, 'all_cats_neg', stop_words, categories, max_words=max_words, split_words=split_words)


    def create_base(self, df_absa_join, category:str = None ):  # column: str ):
        """
        This function creates a base with the desired column (aroma, visual, flavor, sensation, general_set)
        
        Args:
                df_absa_join (DataFrame): The dataframe with ABSA sentiments
        """
        # Not using thresholds anymore
        #
        # pos_thres = 0.0  # 0.0 ignore rating 
        # neg_thres = 5.0  # 5.0 ignore rating
        # print(column.capitalize())
        # df_absa_join_rev_pos = df_absa_join[df_absa_join[column] >= pos_thres]
        # df_absa_join_rev_neg = df_absa_join[df_absa_join[column] <= neg_thres]
        # print(f'- Reviews         POS / NEG: {len(df_absa_join_rev_pos)} / {len(df_absa_join_rev_neg)}')
        
        df_absa_join_absa_pos = df_absa_join[
            (df_absa_join['sentiment'].isin(['positivo', 'muito positivo'])) ]            
        df_absa_join_absa_neg = df_absa_join[
            (df_absa_join['sentiment'].isin(['negativo', 'muito negativo'])) ]
        if category is not None:
            df_absa_join_absa_pos = df_absa_join_absa_pos[df_absa_join_absa_pos['category'] == category]
            df_absa_join_absa_neg = df_absa_join_absa_neg[df_absa_join_absa_neg['category'] == category]
        print(f'- ABSA of {category} count:  POS / NEG: {len(df_absa_join_absa_pos)} / {len(df_absa_join_absa_neg)}')
        
        # Not using thresholds anymore
        #
        # df_pos = df_absa_join[
        #     (df_absa_join[column] > pos_thres) & 
        #     (df_absa_join['sentiment'].isin(['positivo', 'muito positivo'])) ]
                
        # df_neg = df_absa_join[
        #     (df_absa_join[column] <= neg_thres) & 
        #     (df_absa_join['sentiment'].isin(['negativo', 'muito negativo'])) ]
        
        # if category is not None:
        #     df_pos = df_pos[df_pos['category'] == category]
        #     df_neg = df_neg[df_neg['category'] == category]
        
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
        df_pos.to_csv(f'{self.work_dir}/step_6_{base_name}_POS.csv', index=False)
        df_neg.to_csv(f'{self.work_dir}/step_6_{base_name}_NEG.csv', index=False)
        
    
    def get_stop_words(self):
        from nltk.corpus import stopwords
        import nltk

        # Define stop words (Portuguese)
        nltk.download('stopwords')
        stop_words = set(stopwords.words('portuguese'))
        stop_words.update(['aroma', 'sabor', 'notas', 'muito boa', 'excelente', 'ótima','boa', 'ruim', 'gostei', 
            'eu particularmente não gostei', 'não gostei', 'não me agradaram muito', 'me decepcionou', 'não é ruim', 'é muito bom', 
            'não é muito bom', 'não é muito boa' 'é ruim', 'é muito ruim', 'não é muito ruim', 'não é bom', 'não é boa' 'é bom',
            'adorei', 'gostosa', 'agradável', 'esperava um pouco mais', 'esperava muito mais', 'custo benefício ruim', 'é meio estranha',
            'notas fortes', 'não empolga', 'cerveja bem mediana' 'cheiro apagado', 'sabor agradável', 'muito gostosa', 'saborosa',
            'cerveja bem mediana', 'cerveja muito ruim', 'desequilibrado', 'desequilibrada' 'desbalanceada', 'desbalanceado', 'pena ser pequena',
            'confuso', 'cerveja muito ruim', 'não foi bacana'])
        # print(stop_words)
       
        # print(stop_words)
        return stop_words
    
        
    def generate_word_cloud(self, df: pd.DataFrame, base_name: str, stop_words: list, categories: list, max_words=50, split_words=False):
        
        # Function to clean and extract words/entities
        def extract_entities(text, split_words=split_words):
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

        category_colors = {
            "visual": "Blue",
            "aroma": "Orange",
            "sabor": "Green",
            "sensação na boca": "Red",
            "álcool": "Magenta",
            "amargor": "Lime"
        }
        # Convert the color names to hex codes
        # category_colors_hex = {category: to_hex(color) for category, color in category_colors.items()}

        # Create a mapping between words and their corresponding category
        word_category_mapping = {}
        for index, row in df.iterrows():
            words = extract_entities(row['aspect'])
            for word in words:
                word_category_mapping[word] = row['category']

        # Apply extraction and count words
        word_list = df['aspect'].apply(extract_entities).sum()
        word_count = Counter(word_list)
        # print(f'Word count for {base_name}:')
        # print(word_count)
        
        category_word_count = Counter(word_category_mapping.values())
        print(f'Number of words per category for {base_name}:')
        for category, count in category_word_count.items():
            print(f"{category}: {count} words")

        # Create a WordCloud with color function
        wordcloud = WordCloud(width=1600, height=800, background_color='white', max_words=max_words, color_func=color_func).generate_from_frequencies(word_count)

        # Save the word cloud image
        wordcloud_filename = f'{self.work_dir}/wordcloud_{base_name}_{max_words}_words{"_split_words" if split_words else ""}.png'
        wordcloud.to_file(wordcloud_filename)

        # # Optionally show the WordCloud
        # plt.imshow(wordcloud, interpolation='bilinear')
        # plt.axis('off')
        # plt.show()
