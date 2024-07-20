"""
Step 3: Aspect-Based Sentiment Analysis of Beer Characteristics (CC)

:author: Denis Eiras

Functions:
    - 
"""
# TODO remove reviews not in portuguese

import pandas as pd
import json
from step import Step
from src.openai_api import get_completion

class Step_3(Step):

    def __init__(self) -> None:
        super().__init__()


    def clean_json_string(self, json_string):
        cleaned_string = json_string.replace('\t', ' ')
        translation_table = str.maketrans('', '', "[]\"{}")
        cleaned_string = cleaned_string.translate(translation_table)
        # Here we use a regex to remove non-printable characters
        # cleaned_string = re.sub(r'[^\x20-\x7E]', '', cleaned_string)
        
        return cleaned_string
    
    def run(self):
        """Create 'Prompt Base Principal and 'Base Principal' (step_3.csv)

        To create the 'Base Principal', used for the AS and ABSA executed in ASBA (Step 4) and AS (Step 5), it was selected
        the best reviews using a prompt to achieve this.

        The step_3_data_analysis sorted the step_2.csv by beer style, review_general_rate and review_num_reviews to create step_3_data_analysis.csv.

        The idea was check manually for good and bad reviews texts. In the records selection for this task was considered:
        - beer_style. Diferent beer styles shows diferent caracteristics
        - good and bad review_general_rate. This helps the IA system to check both good and bad beers
        - review_num_reviews. This tells the review was writen by an experienced user. Some reviews was selected from the good or bad reviewers.

        So, for each style, it was selected 1 good and 1 bad reviews for 1 good and 1 bad reviewers. (4 reviews by style)
        It was considered 4 different styles for this task, so 16 reviews was selected to create the prompt "Seleção de reviews":
        - American IPA
        ...

        Args:
            None

        Returns:
            None
        """
        print(f'\n\nRunning Step 3\n================================')
        file = f"{self.work_dir}/step_2.csv"
        self.read_csv(file)
        print(f"{len(self.df)} lines Total")
        

                        
        prompt_sys = """Você é um sistema de seleção de avaliações de cervejas de uma base de avaliações, que seleciona avaliações na língua \
portuguesa que citam pelo menos uma característica de cerveja. Você não faz comentários não solicitados.
"""

        # reviews_max_evaluations = len(self.df)/100
        reviews_max_evaluations = 654
        reviews_per_request = 5  # api is limiting to 10 reviews per request, even when the token limit is not reached
        review_eval_count = 1
        # df_reviews_eval = pd.DataFrame(columns=self.df.columns)
        reviews_comments = ''
        # df to validate the results 
        df_response = pd.DataFrame(columns=['index', 'selected','review_comment','reason'])
        prompt_user = """
As avaliações estão na lista compreendida entre colchetes, onde cada item é um dicionário, contendo "index", que registra o \
índice da avaliação e "review_comment" o texto a ser avaliado.
Sua resposta inicia e termina com colchetes e o formato de cada linha é { "index", "selected", "review_comment", "reason" }, onde index \
registra o indice da avaliação, "selected" indica se a avaliação foi selecionada ("YES" ou "NO"), "review_comment" o texto avaliado \
e "reason" indica o motivo pelo qual a avaliação foi ou não selecionada.
"""
        for i_general in range(0, reviews_max_evaluations):
            line = self.df.iloc[i_general]
            
            comm = line[['review_comment']].values[0]
            comm = self.clean_json_string(comm)
            reviews_comments += f'\n["{i_general}", "{comm}"]'
            
            if review_eval_count == reviews_per_request or i_general == reviews_max_evaluations-1:
                # TODO - using prompt_sys in second argument makes the output json retunr without "[ ]"
                response, finish_reason = get_completion(f'{prompt_sys} {prompt_user} {{ {reviews_comments} }}',model='gpt-4o-mini')
                if finish_reason != 'stop':
                    print(f'Finish reason not expected: {finish_reason}')
                    exit(-1)
                try:
                    response_data = json.loads(response)    
                except json.decoder.JSONDecodeError:
                    print(f'Error: decoding JSON. Check:\n {response}')
                    exit(-1)
                df_new = pd.DataFrame(response_data)
                
                # check if it was processed all data - due to limitations of request sizd
                if len(df_new) < reviews_per_request and i_general != reviews_max_evaluations-1:
                    print(f'Error: Not all reviews were processed, expected {reviews_per_request}, got {len(df_new)}')
                    print(f'Last review = {i_general}')
                    exit(-1)
                    
                df_response = pd.concat([df_response, df_new], ignore_index=True)

                review_eval_count = 0
                reviews_comments = ''
        
            review_eval_count += 1
        
        df_response.to_csv(f'{self.work_dir}/step_3__reviews_selected.csv', index=False)
        # select the values of column "index" from df_reviews_selected where column
        
        df_reviews_not_selected = df_response[df_response['selected'] == 'NO']
        self.df = self.df.drop(df_reviews_not_selected['index'].astype(int).tolist())
        self.df.reset_index(drop=True, inplace=True)
        
        self.df.to_csv(f'{self.work_dir}/step_3.csv', index=False)
            

