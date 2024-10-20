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

class Step_4(Step):


    def __init__(self) -> None:
        super().__init__()


    def run(self):
        """
        This function runs Step 4 of the Aspect-Based Sentiment Analysis of Beer Characteristics.
        It reads the step_3.csv (Base Principal) containing the reviews for the previous step, and then creates the 
        "Base Prompts", used to test models and nshots by testing different prompts. Finally, runs the best prompt in
        the entire Base (Base Principal)

        Args:
                self (object): The object instance that contains the data.

        Returns:
        """
        
        print(f'\n\nRunning Step 4\n================================')
        file = f'{self.work_dir}/step_3.csv'
        self.read_csv(file)
        
        # # Create "Base Selecao de Prompts" for creating one and few shot prompts based on step_3.csv
        df_selecao_prompts = self.run_step_4_1_base_selecao_prompts()
        print(df_selecao_prompts.describe())
        
        # # create "Base Prompts", used to test models and nshots
        df_base_prompts = self.run_step_4_1_base_prompts(df_selecao_prompts)
        print(df_base_prompts.describe())
        
        # do ABSA in Base Prompts for n shots and models, to select the best combination
        self.run_step_4_2_ABSA_select_model_shots(df_base_prompts)
        
        
        df_base_principal = self.df
        print(f'- df_base_principal - line count: {len(df_base_principal)}')
       
        # do ABSA for real with the best combination of models and shots
        self.run_step_4_3_evaluate_base_principal(df_base_principal)

        
    def run_step_4_1_base_prompts(self, reviews_for_prompts_df: pd.DataFrame):
        """
        This function selects reviews for Prompt ABSA based on certain criteria.
        Creates the "Base Prompts" base, for testing prompts zero, one and few shots.
        Parameters:
            self (object): The object instance that contains the data.
        Returns:
            pandas.DataFrame: A DataFrame containing the selected reviews. The DataFrame is sorted by beer style, 
            review general rate, and review number of reviews.
        """            
       
        # Base Prompts creation
        print(f'Step 4.1 - Base Prompts creation')
        print(f'- Initial line count: {len(self.df)}')
        df = self.df

        # remove registers of df containing reviews_for_prompts_df registers
        df = df[~df.isin(reviews_for_prompts_df)]
        print(f'- Removing registers from reviews_for_prompts_df - Parcial line count (Base Prompts): {len(df)}')
        
        # review_comment size >= 75% of the greatest sizes 
        greatest_review_comment_size_threshold = df['review_comment_size'].quantile(0.75) 
        df = df[df['review_comment_size'] >= greatest_review_comment_size_threshold]
        print(f'- Select greatest review comment size - Parcial line count (Base Prompts): {len(df)}')
        
        # Best and worst reviews
        df = df[(df['review_general_rate'] >= 4.0) | (df['review_general_rate'] <= 2.0)]
        print(f'- Select best and worst reviews - Parcial line count (Base Prompts): {len(df)}')
        
        # percentil 1% reviwers with most reviews “review_num_reviews” and inexperient
        greatest_reviewers_threshold = df['review_num_reviews'].quantile(0.99) 
        df = df[ (df['review_num_reviews'] >= greatest_reviewers_threshold) | (df['review_num_reviews'] == 1)]
        print(f'- Select experients and inexperients - Parcial line count (Base Prompts): {len(df)}')

        # select only one register per review_user column on df
        df = df.groupby('review_user').head(1)
        print(f'- Select only one register per review_user - Parcial line count (Base Prompts): {len(df)}')
        
        # select only one register per beer_style column on df
        df = df.groupby('beer_style').head(1)
        print(f'- Select only one register per beer_style - Parcial line count (Base Prompts): {len(df)}')
        
        
        df = df.sort_values(by=['beer_style', 'review_general_rate', 'review_num_reviews'])
        print(f'- Final line count (Base Prompts): {len(df)}')
        
        df.to_csv(f'{self.work_dir}/step_4_1__base_prompts.csv', index=False)
        return df


    def run_step_4_1_base_selecao_prompts(self):
        """
        This function selects reviews for creating Prompts ABSA based on certain criteria.
        The 16 reviews are selected manually from this base regarding the constraints
        Parameters:
                self (object): The object instance that contains the data.

        Returns:
                pandas.DataFrame: A DataFrame containing the selected reviews. The DataFrame is sorted by beer style, 
                review general rate, and review number of reviews.
        """

        print(f'Step 4.1 - Selections of reviews for Prompts ABSA')
        styles_for_prompt = ['India Pale Ale (IPA)', 'German Weizen', 'Porter', 'Witbier']
        df = self.df[ self.df['beer_style'].isin(styles_for_prompt) & 
                                     ((self.df['review_general_rate'] >= 4) | (self.df['review_general_rate'] <= 2)) & 
                                     # TODO check 368
                                     ((self.df['review_num_reviews'] >= 368) | (self.df['review_num_reviews'] == 1))]
        df = df.sort_values(by=['beer_style', 'review_general_rate', 'review_num_reviews'])
        
        df.to_csv(f'{self.work_dir}/step_4_1__base_for_prompts_selection.csv')
        
        return df


    def run_step_4_2_ABSA_select_model_shots(self, base_prompts_df):

        reviews_per_request = 20
        is_for_each_CC = True

        for model in ['sabia-3', 'gpt-4o-mini']:
            # for is_num_shots_for_each_CC in [True, False]:
            for nshots in [1, 3]:
                self.run_ABSA('step_4_2', base_prompts_df, model, nshots, reviews_per_request, is_num_shots_for_each_CC = is_for_each_CC)
                
                
    def run_step_4_3_evaluate_base_principal(self, df_base_principal):
            
        best_model = 'sabia-3'
        best_nshots = 1
        num_reviews_to_process = 10e6
        reviews_per_request = 10
        is_num_shots_for_each_CC = False
        self.run_ABSA('step_4_3', df_base_principal, best_model, best_nshots, 
                      reviews_per_request=reviews_per_request, num_reviews_to_process=num_reviews_to_process, is_num_shots_for_each_CC = is_num_shots_for_each_CC)
        
        print("Please, copy the best file of combination of this step to step_4.csv")
       
            
    def run_ABSA(self, step_name, df_base, model, nshots, reviews_per_request = 10, num_reviews_to_process = None, is_num_shots_for_each_CC = False):

        i_initial_eval_index = 0  # 0 in from begining, otherwise index of last processed element + 1
        i_final_eval_index = min(num_reviews_to_process, len(df_base))
        
        prompt_zero = self.step_4_1_get_prompt_zero_shot()
        if nshots == 0:
            prompt_n_shot = prompt_zero
        else:
            prompt_n_shot = self.step_4_1_get_prompt_few_shots(prompt_zero, nshots, is_num_shots_for_each_CC)
        
        print(f'Running {step_name} with model {model} and {nshots} shots ...')
        review_eval_count = 1
        reviews_comments = ''
        response_columns = ['index', 'aspect', 'category', 'sentiment']
        df_response = pd.DataFrame(columns=response_columns)
        n_shot_file_name = f'{self.work_dir}/{step_name}__{nshots}shots_{model}_{"per_CC" if is_num_shots_for_each_CC else ""}_{reviews_per_request}rev_per_req_from_{i_initial_eval_index}.csv'
        
        df_response.to_csv(n_shot_file_name, index=False, header=True)
        error_count = 0
        for i_general in range(i_initial_eval_index, i_final_eval_index):
            line = df_base.iloc[i_general]
            
            comm = line[['review_comment']].values[0]
            comm = self.clean_json_string(comm)
            reviews_comments += f'\n{{"{i_general}", "{comm}"}}'
            
            if review_eval_count == reviews_per_request or i_general == i_final_eval_index-1:
                # TODO - using prompt_sys in second argument makes the output json return without "[ ]"
                prompt_ai = Prompt_AI(model, f'{prompt_n_shot} {reviews_comments} ')
                
                review_eval_count = 0
                reviews_comments = ''
                
                response, finish_reason = prompt_ai.get_completion()
                if finish_reason != 'stop':
                    print(f'Finish reason not expected: {finish_reason}')
                    error_count += 1
                    print(f'Error count: {error_count}')
                    continue
                try:
                    # replaces fixes for zero shot
                    response = response.strip()
                    
                    # fix for sabia-3 alucination with "[[[" in begining
                    # pattern = r'^[\s]*\[\s*\[\s*\['
                    # response = re.sub(pattern, '[[',response)
                    
                    # fix for sabia-3 alucination with "]],[" each review
                    pattern = r'\]\s*[\r\n]*\]\s*[\r\n]*,\s*[\r\n]*\[\s*[\r\n]*'
                    response = re.sub(pattern, '],',response)
                    
                    # fix for alucination where not exists '[[' at the beginning and ']]' at the end
                    match = re.search(r'^(\s*[\r\n]*\[\s*[\r\n]*\[)', response)
                    if not match:
                        response = f'[{response}]'
                    
                    # fix for gpt allucionations
                    response = response.replace('```json', '')
                    response = response.replace('```', '')
                    
                    data_list = ast.literal_eval(response)
                    df_new = pd.DataFrame(data_list, columns=response_columns)
                    df_response = pd.concat([df_response, df_new], ignore_index=True)
                    # saves sometimes to do not loose work 
                    df_new.to_csv(n_shot_file_name, mode='a', index=False, header=False)
                
                except Exception as e:
                    print(f'\n\nException:{e}')
                    print(f'\nError creating df: Check:\n {response}')
                    error_count += 1
                    print(f'Error count: {error_count}')
                    continue

                # WARNING if it was processed all data - due to limitations of request size
                # or some data not processed due (empty response)
                if len(df_new) < reviews_per_request and i_general != i_final_eval_index-1:
                    print(f'WARNING: Not all reviews were processed, expected {reviews_per_request}, got {len(df_new)}')
                    print(f'Last review = {i_general}')
        
            review_eval_count += 1
        
        print(f'TOTAL Error count: {error_count}')
        # finally, sort to check responses and save all the results
        df_response['index'] = df_response['index'].astype(int)
        df_response = df_response.sort_values(by=['index', 'aspect'])
        df_response.to_csv(n_shot_file_name, index=False)

    def step_4_1_get_prompt_zero_shot(self):
            
        print(f'Step 4.1 - "Prompt ABSA zero-shot" creation')
        prompt_sys = """ 
Você é um extrator de aspectos de cerveja. Do texto, extraia os ‘aspectos’ e a ‘categoria’ relacionados aos aspectos da cerveja. As categorias devem estar \
dentre os valores: ‘visual’, ‘aroma’, ‘sabor’, ‘amargor’, ‘álcool’ e ‘sensação na boca’. Extraia o ‘sentimento’ dentre os valores ‘muito negativo’, ‘negativo’, ‘neutro’, \
‘positivo’ ou ‘muito positivo’ para cada par aspecto/categoria. \
Cada avaliação a ser avaliada está compreendida entre chaves. Cada item contém "index", que registra o índice da avaliação e "review_comment", que é o texto a ser avaliado. \
Não faça comentários, apenas gere a saída dos campos extraídos no formato a seguir: ['index','aspecto','categoria','sentimento'], \
"""
        return prompt_sys

    def step_4_1_get_prompt_few_shots(self, prompt_zero_shot: str, num_shots: int, is_num_shots_for_each_CC: bool = False):
        """
        This function creates the Prompt ABSA few-shots based on the Prompt ABSA zero-shot.
        The reviews were selected manually from base step_4_1__base_for_prompts_selection.csv, considering good and bad reviews 
        for 4 main styles of beer, by experienced reviweres, and 2 reviews from newbies
        Parameters:
            self (object): The object instance that contains the data.
            prompt_zero_shot (str): The prompt ABSA zero-shot.
        """
        
        print(f'Step 4.1 - "Prompt ABSA few-shots" creation')
                    
        # beer_style review_user review_num_reviews review_general_rate review_comment
        #
        # ***** Wibier
        #
        # - experienced - low rate
        # Bruno Sicchieri	531	1.1
        style1_exp_lowrate = """
"De coloração amarelada, turva. Espuma de difícil formação, altamente efervescente e sem duração. Bom aroma \
trazendo notas cítricas de laranja e semente de coentro. Na boca, início e final amargos e efervescentes, quanto ao sabor... horrível... \
agitei para capturar um pouco do fermento sedimentando no fundo e creio que foi meu erro... é difícil descrever, exceto a sensação de estar \
estragada... sabor de giz e terra. Carbonatação baixa. Corpo médio. Uma terrível [BJCP2015] 24A: Witbier. Poupe suas papilas gustativas... \
ou experimente por sua própria conta e risco. \
['0', 'cor do líquido amarelado', 'visual', 'neutro'], \
['0', 'cor do líquido turvo', 'visual', 'neutro'], \
['0', 'formação de espuma baixa', 'visual', 'negativo'], \
['0', 'espuma efervescente', 'visual', 'negativo'], \
['0', 'espuma pouco persistente', 'visual', 'negativo'], \
['0', 'notas cítricas de laranja', 'aroma', 'positivo'], \
['0', 'notas cítricas de semente de coentro', 'aroma', 'positivo'], \
['0', 'giz', 'sabor', 'muito negativo'], \
['0', 'terra', 'sabor', 'muito negativo'], \
['0', 'efervescente', 'sensação na boca', 'neutro'], \
['0', 'carbonatação baixa', 'sensação na boca', 'neutro'], \
['0', 'corpo médio', 'sensação na boca', 'neutro'] \
"
"""
        #
        # - experienced - high rate
        # Fabio Vieira	907	4.4	 
        style1_exp_highrate = """
"Temperatura de degustação: Cinco graus Celsius. Cor: Amarelo-palha medianamente turva. Creme: Média formação \
de creme branco que mantém uma fina camada persistente, deixando marcas no tumbler. Aroma: Cítrico com notas de limão, especiarias como coentro\
e pimenta, muito bom. Sabor: Maltado com cereais, frutado de limão e especiarias dominam os sentidos. O final do gole apresenta-se levemente \
amargo, levemente ácido e picante. O sabor cítrico do limão permanece por todo o gole, se prolongando no retrogosto, apresentando excelente \
drinkability e refrescância absurda! Excelente breja!! \
['0', 'cor do líquido amarelo-palha', 'visual', 'neutro'], \
['0', 'cor do líquido turvo', 'visual', 'neutro'], \
['0', 'formação de espuma média', 'visual', 'neutro'], \
['0', 'cor da espuma branca', 'visual', 'neutro'], \
['0', 'notas cítricas de limão', 'aroma', 'muito positivo'], \
['0', 'coentro', 'aroma', 'muito positivo'], \
['0', 'especiarias', 'aroma', 'muito positivo'], \
['0', 'maltado com cereais', 'sabor', 'positivo'], \
['0', 'frutado de limão', 'sabor', 'positivo'], \
['0', 'especiarias', 'sabor', 'positivo'], \
['0', 'ácido leve', 'sabor', 'positivo'], \
['0', 'picante', 'sabor', 'positivo'], \
['0', 'cítrico do limão', 'sabor', 'muito positivo'], \
['0', 'drinkability alta', 'sensação na boca', 'muito positivo'], \
['0', 'refrescância alta', 'sensação na boca', 'muito positivo'] \
"
"""

# not used - reduce size of prompt
#         #
#         # - inexperienced - low rate
#         # Thiago Meireles	1	1.7	
#         style1_inexp_lowrate = """
# "A cerveja é bem docinha; minha opinião sobre ela, no entanto, é um pouco amarga. Produzida em meio à febre \
# de cervejas artesanais que atingiu a burguesada do Rio, o rótulo foi metido goela abaixo do consumidor pelos principais pontos de venda de \
# cervejas especiais da cidade, inclusive supermercados como o Zona Sul, em que é possível comprar boas brejas, como a Coruja, por exemplo. \
# Nas prateleiras, a Niña, uma Wit Bier, ocupa mais espaço que todas as outras, inclusive as da Ambev, por isso é impossível não notá-la. Pra \
# quem tem um paladar mais sensível, pode ser até boa posto que é doce, com gosto bem forte de limão - dizem no rótulo ser da variedade \
# siciliano. Mas uma garrafa, que tem meros 300 ml, já é suficiente para enjoar do sabor. O dulçor esconde um peso, uma sensação de estufamento \
# que vem pouco tempo após o consumo, defeito inadmissível para uma cerveja que se propõe leve acima de qualquer outra característica. O final na \
# boca e na garganta é ácido. No fim, não acho que vale o preço de 11 reais na promoção no Zona Sul - o normal é encontrá-la por 14. Se daqui a \
# pouco ela estiver sendo vendida a 8 pelo menos vamos saber que não vai durar muito. Sorte aos produtores que, certamente, têm dinheiro e boas \
# conexões no mundo do varejo e da mídia.
# ['0', 'dulçor', 'sabor', 'positivo'],
# ['0', 'limão forte', 'sabor', 'positivo'],
# ['0', 'ácido', 'sabor', 'neutro']
# "
# """

# Big text from inexperienced - removed
#         #
#         # - inexperienced - high rate
#         # Robson Grespan	1	5	
#         style1_inexp_highrate = """
# "Excelente cerveja de trigo receita tipo belga. Produzida com os ingredientes: Semente de coentro, casca de \
# laranja, alfarroba, baunilha, Tamara e anis estrelado. Extremamente aromática e refrescante. O alfarroba foi inserido para ter uma espuma densa\
# e aveludada. Uma excelente cerveja para agradar os iniciantes e cervejeiros. Refermentada na própria garrafa.  Alfarroba na Cerveja 65 anos \
# Apesar de não ter a fama do cacau, a alfarroba já era usada pelos egípcios há mais de 5 mil anos. Por ser naturalmente doce, dispensa o uso de \
# açúcar na fabricação e no consumo dos produtos. Sem falar que também não possui os estimulantes cafeína e teobromina e é rica em vitaminas e \
# minerais. Na cerveja “65 anos”, produz efeito espessante, dando mais corpo e textura aveludada. Além disso, os açúcares digeridos pelas leveduras\
# trazem aromas delicados e únicos. Baunilha de Madagascar na Cerveja 65 anos A baunilha é a vagem seca de uma orquídea. O perfil aromático depende\
# das condições de cultivo e de preparação, mas também das variedades ou espécies utilizadas. A mais tradicional é a Baunilha Bourboun, utilizada \
# nesta receita e produzida em Madagascar. A idéia de utilização dela na cerveja “65 anos” é atuar no processo de refinamento dos aromas complexos \
# provenientes da levedura e reestruturação do flavor da cerveja com características únicas da baunilha. Tâmara na Cerveja 65 anos As tâmaras são \
# digeridas completamente depois de um longo período, pois são ricas em açúcares complexos; esta característica é bem apreciada por aqueles que \
# necessitam preservar um ritmo enérgico durante atividades físicas ou mentais, normalmente em desportos que testam a resistência ou em esportes de \
# duração prolongada. No caso da cerveja, esses açúcares, por serem complexos, não serão digeridos completamente pela Levedura, gerando um sabor e \
# leve dulçor bem prazeroso na cerveja.
# [['0', 'espuma densa', 'visual', 'positivo'],
#  ['0', 'espuma aveludada', 'visual', 'positivo'],
#  ['0', 'Baunilha de Madagascar', 'aroma', 'positivo'],
#  ['0', 'Tâmara', 'sabor', 'positivo'],
#  ['0', 'encorpada', 'sensação na boca', 'neutro'],
#  ['0', 'textura aveludada', 'sensação na boca', 'neutro'],
# ]]"
# """
        #
        #
        # ***** German Weizen
        # - experienced - low rate
        #  Jota Fanchin Queiroz	563	1.2	
        style2_exp_lowrate = """
"Uma weiss significativamente inferior ao padrão do estilo. E nem falo em comparação com as bávaras mas com a \
Eisenbahn por exemplo. Aparência: coloração dourada clara turva com creme de média formação e baixa persistência. Aroma: acanhado. Sabor: \
notas de banana e nada de cravo com um final doce demais. Estranho. Corpo: aguado até para pilsen que dirá weiss. Final: estranho, seco e \
curto. Conjunto: desequilibrado pelo excesso do doce e pelo descompassado do corpo e carbonatação. Drinkability baixa e refrescância \
comprometida. \
['0', 'cor do líquido dourado claro', 'visual', 'neutro'], \
['0', 'líquido turvo', 'visual', 'neutro'], \
['0', 'formação de espuma médio', 'visual', 'neutro'], \
['0', 'espuma pouco persistente', 'visual', 'negativo'], \
['0', 'notas de banana', 'sabor', 'neutro'], \
['0', 'dulçor alto', 'sabor', 'negativo'], \
['0', 'corpo aguado', 'sensação na boca', 'negativo'], \
['0', 'final seco e curto', 'sensação na boca', 'negativo'], \
['0', 'drinkability baixa', 'sensação na boca', 'negativo'], \
['0', 'refrescância baixa', 'sensação na boca', 'negativo'] \
"
"""
        #
        # - experienced - high rate
        # Eduardo Guimarães Insta @cervascomedu	2380	4,4	
        style2_exp_highrate = """
"Apresentou coloração dourada com espuma branca de média formação e longa persistência. \
No aroma temos banana, cravo, mel, floral e pão doce. Na boca as notas permanecem, complementadas por cereais, herbal sutil e toques \
picantes. Tem corpo médio, carbonatação moderada e sensação refrescante. Excelente!
['0', 'cor do líquido dourado', 'visual', 'neutro'], \
['0', 'cor da espuma branca', 'visual', 'neutro'], \
['0', 'formação de espuma média', 'visual', 'neutro'], \
['0', 'espuma persistente', 'visual', 'positivo'], \
['0', 'banana', 'aroma', 'positivo'], \
['0', 'cravo', 'aroma', 'positivo'], \
['0', 'floral', 'aroma', 'positivo'], \
['0', 'mel', 'aroma', 'positivo'], \
['0', 'pão doce', 'aroma', 'positivo'], \
['0', 'banana', 'sabor', 'positivo'], \
['0', 'cravo', 'sabor', 'positivo'], \
['0', 'floral', 'sabor', 'positivo'], \
['0', 'mel', 'sabor', 'positivo'], \
['0', 'pão doce', 'sabor', 'positivo'], \
['0', 'cereais', 'sabor', 'positivo'], \
['0', 'herbal sutil', 'sabor', 'positivo'], \
['0', 'notas picantes', 'sabor', 'positivo'], \
['0', 'corpo médio', 'sensação na boca', 'positivo'], \
['0', 'carbonatação moderada', 'sensação na boca', 'positivo'], \
['0', 'refrescância alta', 'sensação na boca', 'positivo'] \
"
"""

# Bad text - excluded
#         #
#         # - inexperienced - low rate
#         # deivis fontes	1	1	
#         style2_inexp_lowrate = """
# "deixei na geladeira por um dia e meio a garrafa em pé, percebi que ela nao apresenta tanto corpo caracteristicos \
# das cervejas de trigo, talvez por ser uma cerveja industrial
# ['0', 'corpo baixo', 'sensação na boca', 'negativo'],
# "
# """

# removed - reduce size of the prompt 
#         #
#         # - inexperienced - high rate 
#         # Marcelo Azambuja	1	4,9	
#         style2_inexp_highrate = """
# "A cor amarelada bem turva é algo que me agrada muito em uma Weiss, e a Alenda atende este quesito como poucas. \
# Eu adquiri minhas amostras diretamente com o produtor. Já vieram resfriadas e eu as mantive assim até chegarem na minha geladeira. Desta \
# forma, posso afirmar que mantém as características perfeitamente. O gosto forte, marcante, e a cremosidade do líquido são excelentes. \
# Degustamos nossas amostras em nossa casa de praia (Capão da Canoa/RS) com um vizinho alemão que passa as férias no Brasil, e a frase do \
# alemão (funcionário - diretor - da Mercedes-Benz, lá na Matriz em Affalterbach, um cara extremamente exigente e que conhece as principais\ 
# cervejarias e países do mundo): pode parabenizar este produtor, muito boa. Depois, por duas vezes, logo após ele tomar um pouco da cerveja,\
# ele parava a conversa e dizia: nossa, muito boa. Para quem conhece europeus, eles não são de muita gentileza, muito menos de falar algo \
# que não seja totalmente sincero. Esta avaliação me ajudou muito a poder considerar a Alenda uma cerveja realmente acima da média.
# ['0', 'cor do líquido amarelo', 'visual', 'muito positivo'],
# ['0', 'turva', 'visual', 'muito positivo'],
# ['0', 'cremosidade', 'sensação na boca', 'muito positivo']
# "
# """

        #
        #
        # ***** India Pale Ale (IPA)
        # - experienced - low rate
        # Wagner Gasparetto	700	1,5	
        style3_exp_lowrate = """
"Cor amarela clara, com certa turbidez, de cara fugindo um pouco da expectativa do estilo. Aroma maltado com \
cítrico muito suave e paladar maltado, pouco lupulado e quase sem presença cítrica. Longe de uma IPA. Média carbonatação e boa drinkability,\
corpo leve. Desagradou.... \
['0', 'cor do líquido amarelo', 'visual', 'negativo'], \
['0', 'líquido turvo', 'visual', 'negativo'], \
['0', 'maltado', 'aroma', 'neutro'], \
['0', 'pouco cítrico', 'aroma', 'negativo'], \
['0', 'maltado', 'sabor', 'neutro'], \
['0', 'pouco lupulado', 'sabor', 'negativo'], \
['0', 'pouco cítrico', 'sabor', 'negativo'], \
['0', 'média carbonatação', 'sensação na boca', 'neutro'], \
['0', 'drinkability boa', 'sensação na boca', 'positivo'], \
['0', 'corpo baixo', 'sensação na boca', 'neutro'] \
"
"""
        #
        # - experienced - high rate
        # Alexandre LC	571	4,7	
        style3_exp_highrate = """
"Pataqueparéu, não sei o que dizer sobre esta cerveja! Sorvida e provada logo em seguida a perigosa. Coloração âmbar\
alaranjada. Espuma levemente bege, com alta formação e boa duração. Apesar da tampinha ser o mesmo problema que a Perigosa, como foi bem \
linda no copo leva 5/5 em aparência. Aroma é fodástico, aparecendo com um buquê fenomenal. Percepção floral, cítrica, caramelada, de melaço\
e de chocolate cremoso (lembra muito o GALAK®). Com notas herbais e de laranja ao fundo. Um conjunto bem equilibrado e perfeito. \
Perfumadíssima. Aroma pra mim é 6/5! kkkk Sabor é inicialmente doce, doce de chocolate cremoso/branco, cacau, caramelo/toffe, logo mesclado\
com um amargor leve e um malte torrado bem sutil. Corpo denso e licoroso. Conjunto equilibrado e primoroso, no qual o doce inicial se acerta\
e abraça bem o amargor floral final. Final seco e levemente amargo. Retrogosto amargo e denso. SENSACIONAL. É uma IPA diferente, devido ao \
fato de o seu padrão puxar muito mais pro doce do que pro amargor lupulento, não compararei com as demais IPAs, pra mim entraria como uma \
Specialty Beer. Já está entre as minhas favoritas. Mais um preço abusivo da Bodebrown... quase R$7 por 100mL. Vacilo. \
['0', 'cor do líquido âmbar alaranjado', 'visual', 'muito positivo'], \
['0', 'cor da espuma bege leve', 'visual', 'muito positivo'], \
['0', 'formação de espuma alta', 'visual', 'muito positivo'], \
['0', 'espuma persistente', 'visual', 'muito positivo'], \
['0', 'floral', 'aroma', 'muito positivo'], \
['0', 'cítrico', 'aroma', 'muito positivo'], \
['0', 'caramelado', 'aroma', 'muito positivo'], \
['0', 'melaço', 'aroma', 'muito positivo'], \
['0', 'chocolate cremoso', 'aroma', 'muito positivo'], \
['0', 'notas herbais', 'aroma', 'muito positivo'], \
['0', 'notas de laranja', 'aroma', 'muito positivo'], \
['0', 'dulçor', 'sabor', 'positivo'], \
['0', 'doce de chocolate branco/cremoso', 'sabor', 'positivo'], \
['0', 'cacau', 'sabor', 'positivo'], \
['0', 'caramelo/toffe', 'sabor', 'positivo'], \
['0', 'malte torrado leve', 'sabor', 'positivo'], \
['0', 'amargor leve', 'amargor', 'positivo'], \
['0', 'corpo denso', 'sensação na boca', 'positivo'], \
['0', 'corpo licoroso', 'sensação na boca', 'positivo'], \
['0', 'amargor floral', 'amargor', 'positivo'], \
['0', 'final seco', 'sensação na boca', 'positivo'] \
"
"""

        #  ONE SHOT EXAMPLE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # - inexperienced - low rate
        # Thiago Coelho	1	1,5	
        style3_inexp_lowrate = """
"Rótulo agradável, em garrafa âmbar bojuda. Tampa sem rótulo, dando um aspecto desleixado à cerveja. As cervejas bastante lupuladas sempre têm uma agradável \
antecipação do aroma logo quando se abre a garrafa. Essa não tinha: mau presságio... Cor âmbar, translúcida,excelente sensação visual ao ser servida, particularmente \
pela intensa formação de espuma, que é persistente. Aroma herbáceo, suave demais,muito aquém para uma cerveja que carrega no lúpulo aromático, inclusive tendo sido  \
feito dry hopping com Cascade, um lúpulo essencialmente aromático . Se a intenção era fazer uma autêntica American IPA, vê-se aqui mais pra uma do velho continente,  \
inglesa, precisamente. No sabor,perde-se completamente: tem um amargor intenso mas adstringente, incomodativo, que parece mesmo arranhar a língua e que perdura no  \
aftertaste.Baixíssima drinkability. Vê-se muito malte, particularmente no aftertaste, quando se mantém um retrogosto de mel e pão. Não vi qualquer off-flavor no exemplar \
que degustei. Pergunto: é uma IPA ou é uma American Pale Ale lupulada em excesso (amargor excessivo, por ser incomodativo...)? \
['0', 'cor do líquido âmbar', 'visual', 'neutro'], \
['0', 'líquido translúcido', 'visual', 'neutro'], \
['0', 'formação de espuma ótima', 'visual', 'muito positivo'], \
['0', 'espuma persistente', 'visual', 'muito positivo'], \
['0', 'herbáceo suave demais', 'aroma', 'negativo'], \
['0', 'amargor excessivo', 'amargor', 'negativo'], \
['0', 'adstringente', 'amargor', 'negativo'], \
['0', 'maltado alto', 'sabor', 'neutro'], \
['0', 'retrogosto de mel', 'sabor', 'neutro'], \
['0', 'retrogosto de pão', 'sabor', 'neutro'], \
['0', 'sem off-flavor', 'aroma', 'positivo'], \
['0', 'lupulada em excesso', 'sabor', 'negativo'] \
"
"""


        #
        # - inexperienced - high rate
        # Odonio dos Anjos Filho	1	4,7	
        style3_inexp_highrate = """
"Cerveja com sabor de cerveja forte. Lúpulo e álcool presentes que dão o perfeito sabor de cerveja India \
Palle Ale. Mais fantástico ainda reconhecer uma cerveja dessa no Brasil, respeitando os processos de pureza necessários para fabricação de \
grandes cervejas. Vale tomar com comidas mais fortes e apreciar durante todo o ano. Espuma maravilhos que matém o aroma da cerveja de forma \
prolongada. Uma perfeição em termos de equilíbrio. Sensacional! \
['0', 'formação de espuma boa', 'visual', 'muito positivo'] \
"
"""
        
        #
        # ***** Porter
        # - experienced - low rate
        # Alexandre LC	571	1,7	
        style4_exp_lowrate = """
"Coloração negra opaca. Espuma bege de alta formação e pouca duração. Aroma de caramelo e açúcar mascavo. Sabor quase \
exclusivo de caramelo, com leve torrado e um dulçor muito acima da média, enjoativa demais. Praticamente uma malzbier menos doce. Totalmente \
fora do estilo. Bebi apenas um copo e deixei o resto para mulherada. \
['0', 'cor do líquido negro opaca', 'visual', 'neutro'], \
['0', 'cor da espuma bege', 'visual', 'neutro'], \
['0', 'formação de espuma alta', 'visual', 'neutro'], \
['0', 'espuma pouco persistente', 'visual', 'negativo'], \
['0', 'caramelo', 'aroma', 'neutro'], \
['0', 'ácúcar mascavo', 'aroma', 'neutro'], \
['0', 'torrado leve', 'sabor', 'positivo'], \
['0', 'caramelo', 'sabor', 'neutro'], \
['0', 'torrado leve', 'sabor', 'neutro'], \
['0', 'dulçor alto', 'sabor', 'negativo'] \
"
"""
        #
        # - experienced - high rate
        # Odimi Toge	1031	4,6	
        style4_exp_highrate = """
"Bebida desenvolvida em parceria com a Cachaçaria Nacional - maior varejista de cachaças do mundo, sediada em Belo \
Horizonte (MG).  Trata-se de um blend de Baltic Porter com a cachaça Legítima de Minas, na proporção de 10%.  Envelhecida por dois anos em \
barris de amburana, esta cachaça é produzida em Itaverava (MG) no Alambique Taverna de Minas.  A receita toda, criada pelo cervejeiro caseiro \
Fábio Ferreira, foi medalha de Ouro do XII Concurso da Acerva Mineira.  Aroma intenso de cachaça, passando por coco, canela, baunilha e mel. \
Toffee, melaço e ameixa seca surgem sinérgicos. Espetáculo! Líquido castanho avermelhado, permitindo certa passagem de luz. Servido, forma uma \
camada fina e efêmera de espuma bege clara. Na boca mostra corpo médio e reduzida carbonatação. A junção de cachaça e cerveja conversa bem, \
resultando em notas de coco queimado, canela, baunilha, ameixa seca e café - riscadas por leve dulçor maltado. Álcool inacreditavelmente bem \
inserido (sério, cadê esse álcool todo anunciado?) O final segue ligeiramente adocicado, com bastante cachaça e breve torrado.  "Drinkability" \
relativamente alta em vista de toda sua "periculosidade", por assim dizer.  Blend muito bem construído, com cerveja e cachaça na mais perfeita \
harmonia. Parabéns aos envolvidos! ???? \
['0', 'cor do líquido castanho avermelhado', 'visual', 'neutro'], \
['0', 'cor do líquido semi translúcido', 'visual', 'neutro'], \
['0', 'cor da espuma bege clara ', 'visual', 'neutro'], \
['0', 'formação de espuma baixa', 'visual', 'neutro'], \
['0', 'intenso de cachaça', 'aroma', 'muito positivo'], \
['0', 'coco', 'aroma', 'muito positivo'], \
['0', 'canela', 'aroma', 'muito positivo'], \
['0', 'baunilha', 'aroma', 'muito positivo'], \
['0', 'mel', 'aroma', 'muito positivo'], \
['0', 'toffee', 'aroma', 'muito positivo'], \
['0', 'melaço', 'aroma', 'muito positivo'], \
['0', 'ameixa seca', 'aroma', 'muito positivo'], \
['0', 'notas de coco queimado', 'sabor', 'positivo'], \
['0', 'notas de canela', 'sabor', 'positivo'], \
['0', 'notas de baunilha', 'sabor', 'positivo'], \
['0', 'notas de ameixa seca', 'sabor', 'positivo'], \
['0', 'café', 'sabor', 'positivo'], \
['0', 'dulçor maltado leve', 'sabor', 'positivo'], \
['0', 'alcool imperceptível', 'alcool', 'muito positivo'], \
['0', 'final dulçor leve', 'sabor', 'positivo'], \
['0', 'final cachaça', 'sabor', 'positivo'], \
['0', 'final leve torrado', 'sabor', 'positivo'], \
['0', 'drinkability alta', 'sensação na boca', 'positivo'] \
"
"""

# removed - reduce size of the prompt
#         #
#         # - inexperienced - low rate
#         # FABIO NASCIMENTO	1	1,5	
#         style4_inexp_lowrate = """
# "Fiz a degustação da Zehn Bier - Porter e aqui vai o que percebi. Estou iniciando no mundo cervejeiro e estou tentando \
# aprender a degustar estas ótimas cerveja. Lá vai: Aroma, achei adocicado,sabor pouco amarga, pouca espuma(Acho que fiz algo errado, pois no \
# rótulo diz que a espuma é duradoura;não cremosa ou sem nenhuma cremosidade. Cerveja leve. Senti um pouco do sabor torrado mas não o de caramelo.\
# Sabor que deixou amargo duradouro.
# ['0', 'formação de espuma baixa', 'visual', 'neutro'],
# ['0', 'espuma não cremosa', 'visual', 'neutro'],
# ['0', 'dulçor leve', 'aroma', 'neutro'],
# ['0', 'amargor leve', 'amargor', 'neutro'],
# ['0', 'espuma não cremosa', 'sensação na boca', 'neutro']
# "
# """

# removed - not necessary
#         #
#         # - inexperienced - high rate
#         # Tiago Cosmai	1	4,6	
#         style4_inexp_highrate = """
# "Cerveja deliciosa, aroma e sabor de café presentes do início ao fim, sensação de estalar no meio da língua com a baixa \
# gaseificação, cor forte típica das Porters com uma espuma pouco densa de cor caramelo escuro tão característica como o corpo da cerveja, para \
# mim a Colorado Demoiselle é a melhor nacional.
# [['0', '', 'visual', 'neutro'],
#  ['0', '', 'aroma', 'muito positivo'],
#  ['0', '', 'sabor', 'positivo'],
#  ['0', '', 'amargor', 'positivo'],
#  ['0', '', 'sensação na boca', 'muito positivo'],
# ]]"
# """

        prompt_few_shots = prompt_zero_shot + """ \
Abaixo, entre aspas, exemplos de textos de avaliações e o resultado esperado. \
Ignore o valor do campo index dos exemplos, pois são apenas para mostrar o formato de saída.
"""
   
        if not is_num_shots_for_each_CC:
            if num_shots == 1:
                prompt_few_shots += style3_inexp_lowrate
            
            elif num_shots == 3:
                prompt_few_shots += style3_inexp_lowrate
                
                prompt_few_shots += style1_exp_lowrate
                prompt_few_shots += style2_exp_highrate
            elif num_shots == 10:
                prompt_few_shots += style3_inexp_lowrate
                
                prompt_few_shots += style1_exp_lowrate
                prompt_few_shots += style1_exp_highrate
                prompt_few_shots += style2_exp_lowrate
                prompt_few_shots += style2_exp_highrate
                prompt_few_shots += style3_exp_lowrate
                prompt_few_shots += style3_exp_highrate
                prompt_few_shots += style3_inexp_highrate
                prompt_few_shots += style4_exp_lowrate
                prompt_few_shots += style4_exp_highrate

        else:
            #  ONE SHOT EXAMPLE 1 CC !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
            # - inexperienced - low rate
            # Thiago Coelho	1	1,5	
            style3_inexp_lowrate_1_CC = """
"Rótulo agradável, em garrafa âmbar bojuda. Tampa sem rótulo, dando um aspecto desleixado à cerveja. As cervejas bastante lupuladas sempre têm uma agradável \
antecipação do aroma logo quando se abre a garrafa. Essa não tinha: mau presságio... Cor âmbar, translúcida,excelente sensação visual ao ser servida, particularmente \
pela intensa formação de espuma, que é persistente. Aroma herbáceo, suave demais,muito aquém para uma cerveja que carrega no lúpulo aromático, inclusive tendo sido  \
feito dry hopping com Cascade, um lúpulo essencialmente aromático . Se a intenção era fazer uma autêntica American IPA, vê-se aqui mais pra uma do velho continente,  \
inglesa, precisamente. No sabor,perde-se completamente: tem um amargor intenso mas adstringente, incomodativo, que parece mesmo arranhar a língua e que perdura no  \
aftertaste.Baixíssima drinkability. Vê-se muito malte, particularmente no aftertaste, quando se mantém um retrogosto de mel e pão. Não vi qualquer off-flavor no exemplar \
que degustei. Pergunto: é uma IPA ou é uma American Pale Ale lupulada em excesso (amargor excessivo, por ser incomodativo...)? 
"""

            if num_shots == 1:
                prompt_few_shots += style3_inexp_lowrate_1_CC
                prompt_few_shots += """\
['0', 'cor do líquido âmbar', 'visual', 'neutro'], \
"
"""
            elif num_shots == 3:
                prompt_few_shots += style3_inexp_lowrate_1_CC
                prompt_few_shots += """\
['0', 'cor do líquido âmbar', 'visual', 'neutro'], \
['0', 'líquido translúcido', 'visual', 'neutro'], \
['0', 'formação de espuma ótima', 'visual', 'muito positivo'] \
"
"""
    
       
        # not used
        # prompt_few_shots += style1_inexp_lowrate
        # prompt_few_shots += style1_inexp_highrate
        # prompt_few_shots += style2_inexp_lowrate
        # prompt_few_shots += style2_inexp_highrate
        # prompt_few_shots += style3_inexp_lowrate
        # prompt_few_shots += style4_inexp_lowrate
        # prompt_few_shots += style4_inexp_highrate

        return prompt_few_shots

