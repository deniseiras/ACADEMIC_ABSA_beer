"""
Step 4: Aspect-Based Sentiment Analysis of Beer Characteristics (CC)

:author: Denis Eiras

Functions:
    - 
"""

#TODO - separate in step 6 and 7

from step import Step

import numpy as np
import pandas as pd
import re
import nltk
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb, to_hex, rgb_to_hsv

from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm # estimação de modelos
from statsmodels.discrete.discrete_model import MNLogit # estimação do modelo
                                                        #logístico multinomial
from scipy import stats # estatística chi2                                                        

def Qui2(modelo_multinomial):
    maximo = modelo_multinomial.llf
    minimo = modelo_multinomial.llnull
    qui2 = -2*(minimo - maximo)
    pvalue = stats.distributions.chi2.sf(qui2,4)
    df = pd.DataFrame({'Qui quadrado':[qui2],
                    'pvalue':[pvalue]})
    return df


 # Function to clean and extract words/entities
def extract_entities(text, stop_words, split_words=False):
    text = text.lower()
    text = re.sub(r'[^a-zà-ÿ\s,]', '', text)
    if split_words:
        words = re.split(r'[,\s]+', text)
        words = [word.strip() for word in words if word.strip() and word not in stop_words]
    else:
        words = [text] if text not in stop_words else []
    return words

# if the value of the column "aspect" of df_absa_as_join have exactly two words, put the words in alphabetical order to avoid duplicates
def sort_two_words(aspect: str):
    """ Sort two words in aspect

    Args:
        aspect (str): _description_

    Returns:
        str: aspect with two words sorted
    """
    words = aspect.split()
    if len(words) == 2:
        return ' '.join(sorted(words))
    return aspect


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
        
        #
        # Step 6
        #
        
        self.read_csv(f'{self.work_dir}/step_4.csv')
        df_base_absa = self.df
        categories = ["visual", "aroma", "sabor", "álcool", "amargor", "sensação na boca"]
        df_base_absa = df_base_absa[df_base_absa['category'].isin(categories)]

        self.read_csv(f'{self.work_dir}/step_3.csv')
        df_base_principal = self.df       
        
        self.read_csv(f'{self.work_dir}/step_5.csv')
        df_base_as = self.df 
        df_base_as = df_base_as.rename(columns={'sentiment': 'sentiment_as'})
        # Alucination in step 5 = value "excelent" included
        df_base_as = df_base_as[df_base_as['sentiment_as'].isin(['muito negativo', 'negativo', 'neutro', 'positivo', 'muito positivo'])]

        # Create a base joining df_base_principal, df_base_absa and df_base_as
        df_base_asba_interested_columns = df_base_principal[['index', 'review_comment', 'review_datetime', 'beer_style', 'review_general_rate', 
            'review_aroma', 'review_visual', 'review_flavor', 'review_sensation', 'review_general_set']]
        df_absa_as_join = df_base_asba_interested_columns.join(df_base_absa.set_index('index'), on='index', how='inner')
        df_absa_as_join = df_absa_as_join.join(df_base_as.set_index('index'), on='index', how='inner')
        df_absa_as_join["review_datetime"] = pd.to_datetime(df_absa_as_join["review_datetime"])
        df_absa_as_join['year'] = df_absa_as_join['review_datetime'].dt.year
        df_absa_as_join['aspect'] = df_absa_as_join['aspect'].apply(remove_obvious_words)
        
        df_base_as_interested_columns = df_base_principal[['index', 'review_general_rate', 'review_general_set']]
        df_as_join = df_base_as_interested_columns.join(df_base_as.set_index('index'), on='index', how='inner')
        
        print(f'- Base Principal - line count: {len(df_base_principal)}')
        print(f'- Base ABSA      - line count: {len(df_base_absa)}')
        print(f'- Base Joined    - line count: {len(df_absa_as_join)}')
        print(f'- Base AS        - line count: {len(df_base_as)}')
        print(f'- Base AS join   - line count: {len(df_as_join)}')
        
        # save df_absa_as_join and df_as_join to csv files
        df_absa_as_join.to_csv(f'{self.work_dir}/step_6_join_ABSA-AS-PRINCIPAL.csv', index=False)
        df_as_join.to_csv(f'{self.work_dir}/step_6_join_AS-PRINCIPAL.csv', index=False)
         
        df_aroma_pos, df_aroma_neg = self.create_base(df_absa_as_join, 'aroma')
        df_visual_pos, df_visual_neg = self.create_base(df_absa_as_join, 'visual')
        df_flavor_pos, df_flavor_neg = self.create_base(df_absa_as_join, 'sabor')
        df_sensation_pos, df_sensation_neg = self.create_base(df_absa_as_join, 'sensação na boca')
        df_amargor_pos, df_amargor_neg = self.create_base(df_absa_as_join, 'amargor')
        df_alcool_pos, df_alcool_neg = self.create_base(df_absa_as_join, 'álcool')
        # include all categories, also alcool, amargor
        df_all_cats_pos, df_all_cats_neg = self.create_base(df_absa_as_join)
        
        
        #
        # Step 7
        # 
        
        # print the most common beer_style per year using df_absa_as_join
        df_styles = df_absa_as_join[['beer_style', 'year']]
        beer_style_counts = df_styles.groupby(['year', 'beer_style']).size().reset_index(name='count')
        print(beer_style_counts)
        most_common_style_per_year = beer_style_counts.loc[beer_style_counts.groupby('year')['count'].idxmax()]
        print(most_common_style_per_year)
        

        # Box plot of review_general_set by sentiment
        sentiment_order = ['muito negativo', 'negativo', 'neutro', 'positivo', 'muito positivo' ]
        palette = {
            'muito positivo': 'green', 
            'positivo': 'lightgreen', 
            'neutro': 'gray', 
            'negativo': 'lightcoral', 
            'muito negativo': 'red'
        }
       
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='sentiment_as', y='review_general_set', data=df_as_join, order=sentiment_order, palette=palette,
                    hue='sentiment_as', legend=False)
        # sns.stripplot(x='sentiment_as', y='review_general_set', data=df_as_join, order=sentiment_order, 
        #       color='black', size=8, jitter=True, alpha=0.2)
        plt.xlabel('Polaridade')
        plt.ylabel('review_general_set')
        plt.title('Boxplot da recomendação do avaliador (review_general_set) por polaridade')
        # print into a file
        file_name = f'{self.work_dir}/boxplot_review_general_set_by_sentiment.png'
        plt.savefig(file_name, bbox_inches='tight', dpi=300)
    
        # Generate word clouds
        max_words = 50
        split_words = False
        
        stop_words_sab_aro_sens_vis = self.get_stop_words_sab_aro_sens_vis()
        self.generate_word_cloud(df_aroma_pos, 'aroma_pos', stop_words_sab_aro_sens_vis, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_aroma_neg, 'aroma_neg', stop_words_sab_aro_sens_vis, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_visual_pos, 'visual_pos', stop_words_sab_aro_sens_vis, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_visual_neg, 'visual_neg', stop_words_sab_aro_sens_vis, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_flavor_pos, 'flavor_pos', stop_words_sab_aro_sens_vis, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_flavor_neg, 'flavor_neg', stop_words_sab_aro_sens_vis, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_sensation_pos, 'sensation_pos', stop_words_sab_aro_sens_vis, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_sensation_neg, 'sensation_neg', stop_words_sab_aro_sens_vis, categories, max_words=max_words, split_words=split_words)
        
        stop_words_amar_alco = self.get_stop_words_alco_amarg()
        self.generate_word_cloud(df_amargor_pos, 'amargor_pos', stop_words_amar_alco, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_amargor_neg, 'amargor_neg', stop_words_amar_alco, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_alcool_pos, 'alcool_pos', stop_words_amar_alco, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_alcool_neg, 'alcool_neg', stop_words_amar_alco, categories, max_words=max_words, split_words=split_words)
        
        stop_words_all_cats = self.get_stop_words_all_cats()
        self.generate_word_cloud(df_all_cats_pos, 'all_cats_pos', stop_words_all_cats, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_all_cats_neg, 'all_cats_neg', stop_words_all_cats, categories, max_words=max_words, split_words=split_words)

        # Generate timeline
        self.generate_bar_chart(df_all_cats_neg, stop_words_all_cats, categories, 'negativo')
        self.generate_bar_chart(df_all_cats_pos, stop_words_all_cats, categories, 'positivo')
        
        
        # Multinomial Logistic Regression for sentiment_as
        df_as_join_copy = df_as_join.copy()
        df_as_join_copy = df_as_join_copy.drop(columns=['index', 'review_general_rate'])

        le = LabelEncoder()
        df_as_join_copy['sentiment_encoded'] = le.fit_transform(df_as_join_copy['sentiment_as'])
        print(df_as_join_copy)
        # print count per sentiment
        print(df_as_join_copy['sentiment_as'].value_counts().sort_index())
        
        # Prepare the X and y variables
        x = df_as_join_copy[['review_general_set']]  # Independent variable
        y = df_as_join_copy['sentiment_encoded']     # Dependent variable (encoded sentiment)

        test_model1 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

        X_const = sm.add_constant(x)
        model_1MNLogit = MNLogit(endog=y, exog=X_const).fit()
        print(model_1MNLogit.summary())

        #  Estatística geral do modelo
        print(f'Qui2 stats = {Qui2(model_1MNLogit)}')

        # Test model
        # df_test_m1 = pd.DataFrame({'const': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'review_general_set': test_model1})
        df_test_m1 = pd.DataFrame({'const': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'review_general_set': test_model1})
        model1_results = model_1MNLogit.predict(df_test_m1).round(4)

        print(f'\n\nModel 1 Results: \n {model1_results}')
        predicted_sentiments_m1 = model1_results.idxmax(axis=1)
        decoded_predictions_m1 = le.inverse_transform(predicted_sentiments_m1)

        # Display the predictions
        for val, pred in zip(test_model1, decoded_predictions_m1):
            print(f"Review_general_set: {val} -> Predicted Sentiment: {pred}")


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
        
        print(f'Creating base for {category}')
        
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
        
    
      
    def get_stop_words_sab_aro_sens_vis(self):

        # Define stop words (Portuguese)
        nltk.download('stopwords')
        stop_words = set(stopwords.words('portuguese'))
        stop_words.update(['aroma', 'sabor', 'notas', 'muito boa', 'excelente', 'ótima','boa', 'ruim', 'gostei', 
            'eu particularmente não gostei', 'não gostei', 'não me agradaram muito', 'me decepcionou', 'não é ruim', 'é muito bom', 
            'não é muito bom', 'não é muito boa' 'é ruim', 'é muito ruim', 'não é muito ruim', 'não é bom', 'não é boa' 'é bom',
            'adorei', 'gostosa', 'agradável', 'esperava um pouco mais', 'esperava muito mais', 'custo benefício ruim', 'é meio estranha',
            'notas fortes', 'não empolga', 'cerveja bem mediana' 'cheiro apagado', 'agradável', 'muito gostosa', 'saborosa',
            'cerveja bem mediana', 'cerveja muito ruim', 'desequilibrado', 'desequilibrada' 'desbalanceada', 'desbalanceado', 'pena ser pequena',
            'confuso', 'cerveja muito ruim', 'não foi bacana', 'amargo', 'amargor estranho', 'amargor bom', 'delicioso amargor', 
            'amargor característico', 'fraco', 'muito fraco', 'falta sabor', 'não recomendo', 'falta de personalidade', 
            'falta personalidade', 'sem graça', 'faltou personalidade', 'retrogosto ruim', 'enjoativo', 'final curto', 'fraca', 'sem gosto', 
            'esperava mais', 'curta duração', 'cerveja fraca', 'agradável', 'bem agradável', 'segue o aroma', 
            'ótima cerveja', 'deliciosa', 'recomendo', 'boa breja', 'marcante', 'equilíbrio', 'boa cerveja', 'preço alto', 
            'custobenefício ruim', 'preço não justifica', 'álcool ruim ', 'mais álcool', 'custo alto', 'preço não ajuda', 'não vale o preço', 
            'preço elevado', 'custobenefício não é dos melhores', 'péssimo custobenefício', 'preço não compensa', 'álcool bom', 'álcool ruim', 
            'bom custobenefício', 'excelente custobenefício', 'ótimo custo benefício', 'ótimo custobenefício', 'álcool', 'enjoativa', 
            'um pouco enjoativa', 'sem personalidade', 'de álcool', 'excelente cerveja', 'boa formação', 'ótimas companhias', 'longa duração',
            'falta', 'estranho', 'apagado', 'e aroma fracos', 'fraquíssimo',
            'pouco aromática', 'quase inexistente', 'nenhum'
            'pouco proeminente', 'pouco intenso', 'fraquinho', 'bastante fraco', 'decepciona', 
            'pouco pronunciado', 'pouco expressivo', 'muito leve', 'não recomendo',
            'bem leve', 'nada de lúpulo', 'pouco lúpulo', 'lúpulo fraco', 'lúpulo muito fraco', 'falta lúpulo',
            'lúpulo quase inexistente', 'lúpulo bastante fraco', 'lúpulo quase imperceptível', 'lúpulo pouco proeminente', 'lúpulo pouco intenso',
            'pouco intenso', 'lúpulo pouco pronunciado', 'lúpulo pouco expressivo', 'lúpulo muito leve', 'lúpulo leve', 'pouco lúpulo',
            'lúpulo bem leve', 'muito discreto', 'lúpulo', 'bem agradável', 'pouca personalidade',
            'equilibrada', 'desequilibrada', 'desagradável','fraquinha', 'falta complexidade',
            'complexa', 'complexidade fraca', 'pouca personalidade', 'decepcionante', 'pouco complexa', 'sabor desequilibrado',
            'muito saborosa', 'sabor acompanha o aroma', 'personalidade', 'preço salgado', 'preço', 'preço muito alto', 'preço não vale', 
            'álcool', 'sabor', 'aroma', 'preço', 'bom amargor', 'gosto amargo', 'mais amarga', 'amargor bem inserido', 'amargor na medida certa',
            'bela aparência', 'sem', 'boa amargor', 'ausente', 'decepcionante', 'e sabor fracos',
            'um pouco fraco', 'pouco perceptível', 'quase sem aroma', 'e sabor fracos', 'enjoativo',
            'esquisito','discreto', 'nulo', 'complexo','aromática', 'agradável', 'falta de complexidade',
            'retrogosto fraco', 'desbalanceada', 'retrogosto desagradável', 'muito fraca', 'sem retrogosto', 'retrorgosto desagradável',
            'suave', 'final longo', 'bem equilibrada', 'equilibrado', 'leve', 'muito equilibrada', 'final duradouro',
            'bela aparência', 'preço absurdo', 'preço abusivo', 'um pouco cara', 'agradável amargor', 'cheiro ruim',
            'quase imperceptível', 'um pouco fraco', 'terrível', 'delicioso', 'muito suave',
            'suave demais', 'muito leve', 'suave demais', 'custo benefício não compensa', 'custo benefício',
            'bom preço', 'preço acessível', 'preço razoável', 'preço baixo', 'praticamente inexistente', 'bem tímido', 
            'praticamente nulo', 'bastante tímido', 'suave demais', 'decepcionout', 'gosto estranho', 'amargor', 'muito fraco',
            'quase inexistente', 'quase nulo', 'muito leve', 'leve', 'muito suave', 'suave', 'decepcionante', 'ruim', 'bom', 'gosto', 'amargo',
            'inexistente', 'gosto', 'inexistente', 'tímido', 'estranho', 'fraco', 'bem fraco',
            'pouco', 'imperceptível', 'poderia ser mais intenso', 'sabor e aroma fracos', 'sabor segue o aroma'
            
         ])
      
        stop_words_sorted = stop_words.copy()
        for word in stop_words:
            new_word = sort_two_words(word)
            stop_words_sorted.add(new_word)
        
        return stop_words_sorted
    
    
    def get_stop_words_alco_amarg(self):
        
        # Define stop words (Portuguese)
        nltk.download('stopwords')
        stop_words = set(stopwords.words('portuguese'))
        stop_words.update(['amargo', 'estranho', 'bom', 'delicioso', 'característico', 'bem inserido', 'na medida certa',
            'sem', 'boa', 'ausente', 'agradável', 'ruim', 'mais', 'bom preço', 'preço acessível', 'preço razoável', 'preço baixo',
            'preço absurdo', 'preço abusivo',  'preço salgado', 'preço', 'preço muito alto', 'preço não vale', 
            'preço elevado', 'custobenefício não é dos melhores', 'péssimo custobenefício', 'preço não compensa',
            'custo alto', 'preço não ajuda', 'não vale o preço',  'preço alto', 'boa cerveja',
            'custobenefício ruim', 'preço não justifica', 'muito cara', 'de álcool', 'alcoólica', 'amarga', 'gostoso','ótimo',
            'característico'
        ])
      
        stop_words_sorted = stop_words.copy()
        for word in stop_words:
            new_word = sort_two_words(word)
            stop_words_sorted.add(new_word)
        
        return stop_words_sorted
    
    
    def get_stop_words_all_cats(self):
        
        stop_words = self.get_stop_words_alco_amarg()
        stop_words.update(self.get_stop_words_sab_aro_sens_vis())
        stop_words_sorted = stop_words.copy()
        
        for word in stop_words:
            new_word = sort_two_words(word)
            stop_words_sorted.add(new_word)
        
        return stop_words_sorted

        

    def generate_word_cloud(self, df: pd.DataFrame, base_name: str, stop_words: list, categories: list, max_words=50, split_words=False):
        
        # Function to map category to a color
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            word_category_mapping_str_sorted = sorted(word_category_mapping[word])
            combined_categ_str = "_".join(word_category_mapping_str_sorted)
            return category_colors.get(combined_categ_str, 'black')

        # Helper function to combine two colors (average RGB values)
        def combine_colors(color1, color2):
            rgb1 = to_rgb(color1)
            rgb2 = to_rgb(color2)
            avg_rgb = [(c1 + c2) / 2 for c1, c2 in zip(rgb1, rgb2)]
            return to_hex(avg_rgb)

        category_colors = self.get_category_colors()

        word_category_mapping = {}
        
        most_common_num = max_words
        # Apply extraction and count words 
        word_list = df['aspect'].apply(lambda x: extract_entities(x, stop_words, split_words=split_words)).sum()
        # word_list = df['aspect'].apply(extract_entities).sum()
        word_count = Counter(word_list)
        word_count = Counter(dict(word_count.most_common(most_common_num)))
        most_common_words = [word for word, count in word_count.most_common(most_common_num)]
                
        print(f'\n\n================\nWord count for {base_name}:')
        print(word_count)
        
        # Create a mapping between words and their corresponding categories
        for index, row in df.iterrows():
            words = extract_entities(row['aspect'], stop_words)
            for word in words:
                if word in most_common_words:  # Filter words using the 100 most common words
                    if word in word_category_mapping:
                        word_category_mapping[word].add(row['category'])
                        word_category_mapping_str_sorted = sorted(word_category_mapping[word])
                        # Combine colors for the new category
                        colors_size = len(word_category_mapping_str_sorted)
                        if colors_size > 1:
                            combined_categ_str = "_".join(word_category_mapping_str_sorted)
                            combined_color = category_colors[word_category_mapping_str_sorted[0]]
                            for i in range(colors_size-1):
                                combined_color = combine_colors(combined_color, category_colors[word_category_mapping_str_sorted[i+1]])
                            category_colors[combined_categ_str] = combined_color
                    else:
                        word_category_mapping[word] = {row['category']}

        # Convert sets in word_category_mapping to tuples before passing them to Counter
        category_word_count = Counter([tuple(value) for value in word_category_mapping.values()])
        print(f'Number of words per category for {base_name}:')
        for category, count in category_word_count.items():
            print(f"{category}: {count} words")

        # Create a WordCloud with color function
        wordcloud = WordCloud(width=1600, height=800, background_color='white', max_words=max_words, color_func=color_func).generate_from_frequencies(word_count)

        # Save the word cloud image
        wordcloud_filename = f'{self.work_dir}/wordcloud_{base_name}_{max_words}_words{"_split_words" if split_words else ""}.png'
        wordcloud.to_file(wordcloud_filename)

        if base_name.startswith('all_cats'):
            self.create_legend(category_colors, base_name)


    def create_legend(self, colors_dict, base_name):
        # Convert RGB to HSV and sort by the hue value to create a rainbow order
        def rgb_to_hsv_tuple(color_rgb):
            return rgb_to_hsv(to_rgb(color_rgb))

        # Sort colors by hue (first component of HSV tuple), which corresponds to the rainbow spectrum
        sorted_colors = sorted(colors_dict.items(), key=lambda x: rgb_to_hsv_tuple(x[1])[0])

        fig, ax = plt.subplots(figsize=(10, 10))

        max_colors = 1000
        for i, (attribute, color) in enumerate(sorted_colors):
            # Draw the circle
            circle = plt.Circle((0.1, 0.95 - i * 0.03), 0.01, color=color, transform=ax.transAxes)
            ax.add_patch(circle)

            # Replace underscores with comma and space
            attribute_spaces = attribute.replace('_', ', ')
            ax.text(0.15, 0.95 - i * 0.03, f'{attribute_spaces}', va='center', fontsize=12)
            if i == max_colors - 1:
                break

        # Graph settings
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')  # Remove the axes

        # Save the legend as an image
        legend_name = f'{self.work_dir}/legend_{base_name}.png'
        plt.savefig(legend_name, bbox_inches='tight', dpi=300)


    def generate_bar_chart(self, df: pd.DataFrame, stop_words: list, categories: list, sentiment: str):

        category_colors = self.get_category_colors()
        
        # Process data to find most common word per category and year
        results = []
        # df = df[df['year'] >= 2014]
        for year in df['year'].unique():
            df_year = df[df['year'] == year]
            for category in categories:
                df_category = df_year[df_year['category'] == category]
                word_list = df_category['aspect'].apply(lambda x: extract_entities(x, stop_words)).sum()  # Flatten the list of words
                # word_list = df_category['aspect'].apply(extract_entities).sum()  # Flatten the list of words
                word_count = Counter(word_list)
                if word_count:
                    most_common_word, count = word_count.most_common(1)[0]
                    results.append({'year': year, 'category': category, 'word': most_common_word, 'count': count})

        result_df = pd.DataFrame(results)
        # sort result_df by year descendent then by category ascending
        result_df = result_df.sort_values(by=['year', 'category'], ascending=[True, True])
        print(result_df)
        
        # Enlarge the figure width
        plt.figure(figsize=(15, 6))  # Increased width from 10 to 15
        sns.set(style="whitegrid")
        
        # Create the barplot
        barplot = sns.barplot(x='year', y='count', hue='category', data=result_df, palette=category_colors)
        
        block = 0
        i_block = 0
        num_blocks = len(result_df['year'].unique())
       
        for i, patch in enumerate(barplot.patches):
            if i >= num_blocks*6:
                break
            i_df = i_block * len(categories) + block 
            if i_df >= len(result_df) or (i > 1 and i % num_blocks == 0):
                i_block = 0
                block += 1
                i_df = i_block * len(categories) + block 

            i_block += 1
                
            # Get the current bar's x, y position and height (count)
            height = patch.get_height()
            width = patch.get_width()
            x = patch.get_x() + width / 2
            y = height

            # Annotate with the most common word from result_df
            barplot.text(
                x=x, 
                y=y + 0.5,  # Slightly above the bar
                s=result_df['word'].iloc[i_df],  # Fetch word for this bar
                ha='center', va='bottom', 
                fontsize=10, color='black', weight='bold',
                rotation=90
            )
        
        
        
        # plt.title('Frequência das Palavras Mais Comuns por Categoria e Ano')
        plt.xlabel('Ano')
        plt.ylabel('Frequência')
        # adjust legend position to above the plot area
        plt.legend(title='Categoria', loc='upper right', bbox_to_anchor=(1, 1), ncol=3)
        # plt.legend(title='Categoria', loc='upper right', bbox_to_anchor=(1, 1))  # Adjusted legend position
        
        plt.tight_layout()
        file_name = f'{self.work_dir}/timeline_{sentiment}.png'
        plt.savefig(file_name, bbox_inches='tight', dpi=300)
        

    def get_category_colors(self):
        category_colors = {
            "visual": "Blue",
            "aroma": "Orange",
            "sabor": "Green",
            "sensação na boca": "Red",
            "álcool": "Magenta",
            "amargor": "Lime"
        }
        
        return category_colors



