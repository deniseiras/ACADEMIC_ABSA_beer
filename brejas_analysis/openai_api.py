"""
Open AI API

:author: Denis Eiras

Functions:
    - get_completion: get completion using OpenAI

"""

import openai
import dotenv
import os

# Whether your API call works at all, as total tokens must be below the model’s maximum limit (4097 tokens for gpt-3.5-turbo)
# and 128k tokens for gpt-4

# Every response will include a finish_reason. The possible values for finish_reason are:
# - stop: API returned complete message, or a message terminated by one of the stop sequences provided via the stop parameter
# - length: Incomplete model output due to max_tokens parameter or token limit
# - function_call: The model decided to call a function
# - content_filter: Omitted content due to a flag from our content filters
# - null: API response still in progress or incomplete
# - Depending on input parameters, the model response may include different information.



def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        # response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    choice = response.choices[0]
    print('Response: \n', choice.message.content)
    print('\nFinish reason: ', choice.finish_reason)
    return choice.message.content, choice.finish_reason


# create main function
def main():
    
    # read openai.api_key from openai.env file

    dotenv.load_dotenv('./openai.env')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # set the prompt
    # prompt = "What is the capital of France?"
    prompt = """Translate the text between triple backticks to English:
    ```"Análise de Sentimentos Baseadas em Aspectos para identificação de características preferenciais de cervejas brasileiras" 
    Etapa 1: Coleta de dados
    Etapa 2: Pré-processamento de dados
    Etapa 3: Análise de Sentimento Baseado em Aspectos das Cacacterísticas de Cerveja (CC)
    Etapa 4: Análise de Sentimento geral dos comentários
    Etapa 5: Seleção das melhores e piores avaliações em termos da nota geral da avaliação
    Etapa 6: Identificação das CC e suas categorias, obtidas na Etapa 3, mais referenciadas na base resultante da Etapa 5
    ```
    """
    get_completion(prompt, temperature=1)

if __name__ == "__main__":
    main()
