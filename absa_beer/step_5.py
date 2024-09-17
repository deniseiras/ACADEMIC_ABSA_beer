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

class Step_5(Step):


    def __init__(self) -> None:
        super().__init__()


    def run(self):
        """
        This function runs Step 5: General Sentiment Analysis of Reviews.
        It reads the csv file step_3.csv (Base Principal), and then runs the 
        SA.

        Args:
                self (object): The object instance that contains the data.

        Returns:
        """
        
        print(f'\n\nRunning Step 5\n================================')
        file = f'{self.work_dir}/step_3.csv'
        self.read_csv(file)
        
        df_base_principal = self.df
        print(f'- df_base_principal - line count: {len(df_base_principal)}')
        
        # do SA for real with the best combination of models and shots
        self.run_step_5_evaluate_base_principal(df_base_principal)
        
                
    def run_step_5_evaluate_base_principal(self, df_base_principal):
            
        best_model = 'sabia-3'
        best_nshots = 1
        num_reviews_to_process = 1000
        reviews_per_request = 20
        self.run_SA('step_5', df_base_principal, best_model, best_nshots, 
                      reviews_per_request=reviews_per_request, num_reviews_to_process=num_reviews_to_process)
            
            
    def run_SA(self, step_name, df_base, model, nshots, reviews_per_request = 10, num_reviews_to_process = 10):

        i_initial_eval_index = 0  # 0 in from begining, otherwise index of last processed element + 1
        i_final_eval_index = min(num_reviews_to_process, len(df_base))
        
        prompt_zero = self.step_5_get_prompt_zero_shot()
        if nshots == 0:
            prompt_n_shot = prompt_zero
        else:
            prompt_n_shot = self.step_5_get_prompt_few_shots(prompt_zero, nshots)
        
        print(f'Running {step_name} with model {model} and {nshots} shots ...')
        review_eval_count = 1
        reviews_comments = ''
        response_columns = ['index', 'sentiment']
        df_response = pd.DataFrame(columns=response_columns)
        n_shot_file_name = f'{self.work_dir}/{step_name}__{nshots}shots_{model}_{reviews_per_request}rev_per_req.csv'
        
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
                response, finish_reason = prompt_ai.get_completion()
                if finish_reason != 'stop':
                    print(f'Finish reason not expected: {finish_reason}')
                    error_count += 1
                    print(f'Error count: {error_count}')
                    continue
                try:
                    # replaces fixes for zero shot
                    response = response.strip()
                    
                    # # fix for sabia-3 alucination with "]],[" each review - not needed yet ...
                    # pattern = r'\]\s*[\r\n]*\]\s*[\r\n]*,\s*[\r\n]*\[\s*[\r\n]*'
                    # response = re.sub(pattern, '],',response)
                    # fix for '[' at the beginning and end if not present already
                    match = re.search(r'^\[\s*[\r\n]*\[', response)
                    if not match:
                        response = f'[{response}]'
                    # # fix for gpt allucionations - not needed GPT yet ...
                    # response = response.replace('```json', '')
                    # response = response.replace('```', '')
                    
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

                review_eval_count = 0
                reviews_comments = ''
                # WARNING if it was processed all data - due to limitations of request size
                # or some data not processed due (empty response)
                if len(df_new) < reviews_per_request and i_general != i_final_eval_index-1:
                    print(f'WARNING: Not all reviews were processed, expected {reviews_per_request}, got {len(df_new)}')
                    print(f'Last review = {i_general}')
        
            review_eval_count += 1
        
        # finally, sort to check responses and save all the results
        df_response['index'] = df_response['index'].astype(int)
        df_response = df_response.sort_values(by=['index'])
        df_response.to_csv(n_shot_file_name, index=False)
        # just a copy
        final_file = f'{self.work_dir}/{step_name}.csv'
        df_response.to_csv(final_file, index=False)

    def step_5_get_prompt_zero_shot(self):
            
        print(f'Step 5 - "Prompt SA zero-shot" creation')
        prompt_sys = """ 
Você é um avaliador de 'sentimento' de avaliações sobre cerveja. Extraia o ‘sentimento’ dentre os valores ‘muito negativo’, ‘negativo’, ‘neutro’, ‘positivo’ ou \
‘muito positivo’. Cada avaliação a ser avaliada está compreendida entre chaves. Cada item contém "index", que registra o índice da avaliação e "review_comment", que é o texto a ser avaliado. \
Não faça comentários, apenas gere a saída dos campos extraídos no formato a seguir: ['index', 'sentimento'], \
"""
        return prompt_sys

    def step_5_get_prompt_few_shots(self, prompt_zero_shot: str, num_shots: int):
        """
        This function creates the Prompt SA few-shots based on the Prompt SA zero-shot.
        The reviews were selected manually from base step_4_1__base_for_prompts_selection.csv, considering good and bad reviews 
        for 4 main styles of beer, by experienced reviweres, and 2 reviews from newbies
        Parameters:
            self (object): The object instance that contains the data.
            prompt_zero_shot (str): The prompt SA zero-shot.
        """
        
        print(f'Step 5 - "Prompt ABSA few-shots" creation')
                    
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
['0', 'muito negativo'] \
"
"""
        #
        # - experienced - high rate
        # Eduardo Guimarães Insta @cervascomedu	2380	4,4	
        style2_exp_highrate = """
"Apresentou coloração dourada com espuma branca de média formação e longa persistência. \
No aroma temos banana, cravo, mel, floral e pão doce. Na boca as notas permanecem, complementadas por cereais, herbal sutil e toques \
picantes. Tem corpo médio, carbonatação moderada e sensação refrescante. Excelente! \
['0', 'muito positivo'], \
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
['0', 'negativo'], \
"
"""

        prompt_few_shots = prompt_zero_shot + """ \
Abaixo, entre aspas, exemplos de textos de avaliações e o resultado esperado. Ignore o valor do campo index dos exemplos, pois são apenas para mostrar o formato de saída.
"""
        
        if num_shots == 1:
            prompt_few_shots += style3_inexp_lowrate
        elif num_shots == 3:
            prompt_few_shots += style3_inexp_lowrate
            prompt_few_shots += style1_exp_lowrate
            prompt_few_shots += style2_exp_highrate
        # elif num_shots == 10:
    
        return prompt_few_shots

