"""
Step 6: Generating Results

:author: Denis Eiras

Functions:
    - 
"""

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

# from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder


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

# if the value of the column "aspect" of df_absa_sa_join have exactly two words, put the words in alphabetical order to avoid duplicates
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
        This function runs Step 6: Analyzing Final Bases

        Args:
                self (object): The object instance that contains the data.

        Returns:
        """
        
        print(f'\n\nRunning Step 6\n================================')
        
        df_absa_join = pd.read_csv(f'{self.work_dir}/step_5_join_ABSA-PRINCIPAL.csv')
        
        # print the most common beer_style per year using df_absa_sa_join
        df_styles = df_absa_join[['beer_style', 'year']]
        beer_style_counts = df_styles.groupby(['year', 'beer_style']).size().reset_index(name='count')
        print(beer_style_counts)
        most_common_style_per_year = beer_style_counts.loc[beer_style_counts.groupby('year')['count'].idxmax()]
        print(most_common_style_per_year)
        self.plot_most_common_beer_styles_per_year(most_common_style_per_year)
    
        # Generate word clouds
        max_words = 50
        split_words = False
        
        categories = self.get_category_list()
        df_aroma_pos, df_aroma_neg = self.create_base(df_absa_join, 'aroma')
        df_visual_pos, df_visual_neg = self.create_base(df_absa_join, 'visual')
        df_flavor_pos, df_flavor_neg = self.create_base(df_absa_join, 'sabor')
        df_sensation_pos, df_sensation_neg = self.create_base(df_absa_join, 'sensação na boca')
        df_bitterness_pos, df_bitterness_neg = self.create_base(df_absa_join, 'amargor')
        df_alcool_pos, df_alcool_neg = self.create_base(df_absa_join, 'álcool')
        df_all_cats_pos, df_all_cats_neg = self.create_base(df_absa_join, None)
        
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
        self.generate_word_cloud(df_bitterness_pos, 'bitterness_pos', stop_words_amar_alco, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_bitterness_neg, 'bitterness_neg', stop_words_amar_alco, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_alcool_pos, 'alcool_pos', stop_words_amar_alco, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_alcool_neg, 'alcool_neg', stop_words_amar_alco, categories, max_words=max_words, split_words=split_words)
        
        stop_words_all_cats = self.get_stop_words_all_cats()
        self.generate_word_cloud(df_all_cats_pos, 'all_cats_pos', stop_words_all_cats, categories, max_words=max_words, split_words=split_words)
        self.generate_word_cloud(df_all_cats_neg, 'all_cats_neg', stop_words_all_cats, categories, max_words=max_words, split_words=split_words)

        # Generate timeline
        df_neg = self.get_most_common_word_by_year(df_all_cats_neg, stop_words_all_cats, 'negativo')
        df_pos = self.get_most_common_word_by_year(df_all_cats_pos, stop_words_all_cats, 'positivo')
        self.plot_mirrored_timeline(df_neg, df_pos)


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
                            try:
                                combined_color = category_colors[word_category_mapping_str_sorted[0]]
                            except KeyError:
                                print(f'KeyError: {word_category_mapping_str_sorted[0]} not found in category_colors')
                                print(f'Available keys: {list(category_colors.keys())}')
                                exit(1)
                                # combined_color = '#000000'  # default to black  
                                
                            for i in range(colors_size-1):
                                try:
                                    combined_color = combine_colors(combined_color, category_colors[word_category_mapping_str_sorted[i+1]])
                                except KeyError:
                                    print(f'KeyError: {word_category_mapping_str_sorted[i+1]} not found in category_colors')
                                    print(f'Available keys: {list(category_colors.keys())}')
                                    exit(1)
                                    # combined_color = '#000000'  # default to black
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


    def get_most_common_word_by_year(self, df: pd.DataFrame, stop_words: list, sentiment: str):
        
        categories = self.get_categories_translated(df)
        results = []
        years = np.sort(df['year'].unique())
        
        # df = df[df['year'] >= 2014]
        # create a for year in each 2 years
        for y in range(2, len(years), 2):
            df_year = df[(df['year'] == years[y-1]) | (df['year'] == years[y])]
            for category in categories:
                df_category = df_year[df_year['category'] == category]
                word_list = df_category['aspect'].apply(lambda x: extract_entities(x, stop_words)).sum()
                word_count = Counter(word_list)
                if word_count:
                    most_common_word, count = word_count.most_common(1)[0]
                    results.append({'year': years[y], 'category': category, 'word': most_common_word, 'count': count})

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(by=['year', 'category'], ascending=[True, True])
        print(result_df)
        
        
        return result_df
    

    def plot_mirrored_timeline(self, df_neg, df_pos):

        # Reverse negatives
        df_neg = df_neg.copy()
        df_neg["count"] = -df_neg["count"]

        # Categories for annotation alignment
        categories = self.get_categories_translated(df_neg)

        # ---- FIGURE WITH TWO AXES (centered Y-axis) ----
        fig = plt.figure(figsize=(10, 20)) 
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05)

        ax_neg = fig.add_subplot(gs[0, 0])
        ax_pos = fig.add_subplot(gs[0, 1], sharey=ax_neg)

        sns.set(style="whitegrid")

        # ---- NEGATIVE SIDE ----
        barplot_neg = sns.barplot(
            data=df_neg,
            y='year', x='count', hue='category',
            palette=self.get_category_colors_translated(),
            orient='h',
            ax=ax_neg,
            alpha=0.7
        )

        # Annotate NEGATIVE bars
        num_blocks = len(df_neg['year'].unique())
        block = 0
        i_block = 0

        for i, patch in enumerate(barplot_neg.patches):
            if i >= num_blocks * len(categories):
                break

            i_df = i_block * len(categories) + block
            if i_df >= len(df_neg) or (i > 1 and i % num_blocks == 0):
                i_block = 0
                block += 1
                i_df = i_block * len(categories) + block

            i_block += 1

            height = patch.get_height()
            width = patch.get_width()
            y = patch.get_y() + height / 2

            ax_neg.text(
                x=0,
                y=y,
                s=df_neg['word'].iloc[i_df],
                ha='right',
                va='center',
                fontsize=14,
                color='black'
            )


        # ---- POSITIVE SIDE ----
        barplot_pos = sns.barplot(
            data=df_pos,
            y='year', x='count', hue='category',
            palette=self.get_category_colors_translated(),
            orient='h',
            ax=ax_pos,
            alpha=0.7
        )

        # Annotate POSITIVE bars
        block = 0
        i_block = 0
        for i, patch in enumerate(barplot_pos.patches):
            if i >= num_blocks * len(categories):
                break

            i_df = i_block * len(categories) + block
            if i_df >= len(df_pos) or (i > 1 and i % num_blocks == 0):
                i_block = 0
                block += 1
                i_df = i_block * len(categories) + block

            i_block += 1

            height = patch.get_height()
            width = patch.get_width()
            y = patch.get_y() + height / 2

            ax_pos.text(
                x=0,
                y=y,
                s=df_pos['word'].iloc[i_df],
                ha='left',
                va='center',
                fontsize=14,
                color='black'
            )


        # ---- Styling / Axis Improvements ----
        ax_neg.axvline(0, color='black', linewidth=1)
        ax_pos.axvline(0, color='black', linewidth=1)

        ax_neg.set_xlabel("Frequency")
        ax_pos.set_xlabel("Frequency")
        ax_pos.set_ylabel("")
        max_count = max(df_pos['count'].max(), df_neg['count'].abs().max())
        ax_neg.set_xlim(-max_count, 0)   # negative bars on left
        ax_pos.set_xlim(0, max_count)    # positive bars on right
        # Set x-axis ticks to show positive numbers
        ticks = ax_neg.get_xticks()
        ax_neg.set_xticklabels([f"{abs(int(tick))}" for tick in ticks])
        # Hide duplicated legend from left plot
        ax_neg.legend_.remove()

        # Put one combined legend on the top right inside the plot
        handles, labels = ax_pos.get_legend_handles_labels()
        ax_pos.legend(
            handles, labels,
            title="Category",
            fontsize=12,
            loc='upper right',          # inside plot
            frameon=True,               # optional, add frame
            facecolor='white',          # legend background
            framealpha=0.7              # slightly transparent
        )

        # Remove y-axis labels from right so center feels centered
        plt.setp(ax_pos.get_yticklabels(), visible=False)
        plt.tight_layout()

        fig.savefig(f"{self.work_dir}/timeline_combined.png", dpi=300, bbox_inches='tight')
        plt.close()


    def get_categories_translated(self, df_neg):
        category_translation = {
            "visual": "visual",
            "aroma": "aroma",
            "sabor": "flavor",
            "sensação na boca": "mouthfeel",
            "álcool": "alcohol",
            "amargor": "bitterness"
        }

        df_neg['category'] = df_neg['category'].replace(category_translation)
        categories = [category_translation.get(cat, cat) for cat in self.get_category_list()]
        return categories


    def get_category_colors_translated(self):
        return {
            "visual": "Blue",
            "aroma": "Orange",
            "flavor": "Green",
            "mouthfeel": "Red",
            "alcohol": "Magenta",
            "bitterness": "Lime"
        }
        

    def get_category_colors(self):
        return {
            "visual": "Blue",
            "aroma": "Orange",
            "sabor": "Green",
            "sensação na boca": "Red",
            "álcool": "Magenta",
            "amargor": "Lime"
        }
        

    def plot_most_common_beer_styles_per_year(self, df, figsize=(10, 6), palette='tab10'):
        """
        Plots a horizontal barplot of the most common beer style per year.
        
        Parameters:
        - df: DataFrame with columns ['year', 'beer_style', 'count']
        - figsize: Tuple for figure size
        - palette: Color palette for the bars
        """
        # Sort by year
        df_sorted = df.sort_values(by='year')
        print(df_sorted)
        
        # Set seaborn style
        sns.set(style='whitegrid')

        # Create the figure
        plt.figure(figsize=figsize)
        barplot = sns.barplot(
            data=df_sorted,
            y='count',
            x='year',
            hue='beer_style',
            dodge=False,
            palette=palette
        )
        
        # # Add text labels on bars
        # for i, row in df_sorted.iterrows():
        #     plt.text(row['count'] + 0.5, i, f"{row['beer_style']} ({row['count']})", va='center')
        
        # Set titles and labels
        plt.title('Most Common Beer Style Per Year')
        plt.xlabel('Count')
        plt.ylabel('Year')
        # plt.legend(title='Beer Style', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        file_name = f'{self.work_dir}/styles_per_year.png'
        plt.savefig(file_name, bbox_inches='tight', dpi=100)

