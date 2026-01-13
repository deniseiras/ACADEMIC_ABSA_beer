"""
Step 4: Aspect-Based Sentiment Analysis (ABSA) of Beer Characteristics (BC)

:author: Denis Eiras

Functions:
    - 
"""

import pandas as pd
import ast
from step import Step
from Prompt_AI import Prompt_AI
import re
import json
import time

class Step_4(Step):


    def __init__(self) -> None:
        super().__init__()


    def run(self):
        """
        This function runs Step 4 of the Aspect-Based Sentiment Analysis of Beer Characteristics.
        It reads the step_3.csv (Main Base) containing the reviews for the previous step, creates the prompts and then
        test models and nshots by testing different prompts. Finally, runs the best prompt in the entire Base (Main Base)

        Args:
                self (object): The object instance that contains the data.

        Returns:
        """
        
        print(f'\n\nRunning Step 4\n================================')
        file = f'{self.work_dir}/step_3.csv'
        self.read_csv(file)
        
        # Creates the Base Prompts Creation: for creating one and few shot prompts based on step_3.csv
        df_selecao_prompts = self.run_step_4_1_create_base_prompts_creation()
        print(df_selecao_prompts.describe())
        
        # Creates the Base Prompts Validation: used to test models and nshots
        df_base_prompts = self.run_step_4_1_create_base_prompts_validation(df_selecao_prompts)
        print(df_base_prompts.describe())
        
        # do ABSA in Base Prompts Validation for n shots and models, to select the best combination
        self.run_step_4_2_ABSA_model_shots_evaluation(df_base_prompts)
        
        # df_main_base = self.df
        # print(f'- df_main_base - line count: {len(df_main_base)}')
       
        # # # do ABSA for real with the best combination of models and shots
        # self.run_step_4_3_evaluate_main_base(df_main_base)


    def run_step_4_1_create_base_prompts_creation(self):
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
        pre_selected_reviews = self.df[ self.df['beer_style'].isin(styles_for_prompt) & 
                                     ((self.df['review_general_rate'] >= 4) | (self.df['review_general_rate'] <= 2)) & 
                                     # TODO check 368
                                     ((self.df['review_num_reviews'] >= 368) | (self.df['review_num_reviews'] == 1))]
        pre_selected_reviews = pre_selected_reviews.sort_values(by=['beer_style', 'review_general_rate', 'review_num_reviews'])
        
        # df is the initial base for selection of reviews for prompts
        
        # Then do the manual selection of reviews from the previous pre_selected_reviews DataFrame
        selected_review_comments_starting_with_the_strings = [
            "De coloração amarelada, turva. Espuma de difícil formação, altamente efervescente e sem duração. Bom aroma",
            "Temperatura de degustação: Cinco graus Celsius. Cor: Amarelo-palha medianamente turva. Creme: Média formação",
            "Uma weiss significativamente inferior ao padrão do estilo. E nem falo em comparação com as bávaras mas com a",
            "Apresentou coloração dourada com espuma branca de média formação e longa persistência.",
            "Cor amarela clara, com certa turbidez, de cara fugindo um pouco da expectativa do estilo. Aroma maltado com",
            "Pataqueparéu, não sei o que dizer sobre esta cerveja! Sorvida e provada logo em seguida a perigosa. Coloração âmbar",
            "Rótulo agradável, em garrafa âmbar bojuda. Tampa sem rótulo, dando um aspecto desleixado à cerveja. As cervejas bastante lupuladas sempre têm uma agradável",
            "Cerveja com sabor de cerveja forte. Lúpulo e álcool presentes que dão o perfeito sabor de cerveja India",
            "Coloração negra opaca. Espuma bege de alta formação e pouca duração. Aroma de caramelo e açúcar mascavo. Sabor quase",
            "Bebida desenvolvida em parceria com a Cachaçaria Nacional - maior varejista de cachaças do mundo, sediada em Belo"
        ]
        
        manual_selected_df = pd.DataFrame()
        for starting_string in selected_review_comments_starting_with_the_strings:
            selected_row = pre_selected_reviews[pre_selected_reviews['review_comment'].str.startswith(starting_string)].head(1)
            manual_selected_df = pd.concat([manual_selected_df, selected_row], ignore_index=True)
        
        manual_selected_df.to_csv(f'{self.work_dir}/step_4_1__base_for_prompts_selection.csv')
        
        return manual_selected_df
    
        
    def run_step_4_1_create_base_prompts_validation(self, reviews_for_prompts_df: pd.DataFrame):
        """
        This function selects reviews for Prompt ABSA based on certain criteria.
        Creates the "Base Prompts Validation", for testing prompts zero, one and few shots.
        Parameters:
            self (object): The object instance that contains the data.
        Returns:
            pandas.DataFrame: A DataFrame containing the selected reviews. The DataFrame is sorted by beer style, 
            review general rate, and review number of reviews.
        """            
       
        # Base Prompts creation
        print(f'Step 4.1 - Base Prompts Validation: creating')
        print(f'- Initial line count: {len(self.df)}')
        df = self.df

        # remove registers of df containing reviews_for_prompts_df registers, to not influence on validation
        df = df[~df.isin(reviews_for_prompts_df)]
        print(f'- Removing registers from reviews_for_prompts_df - Parcial line count (Base Prompts): {len(df)}')
        
        # review_comment size >= 75% of the greatest sizes 
        greatest_review_comment_size_threshold = df['review_comment_size'].quantile(0.75) 
        df = df[df['review_comment_size'] >= greatest_review_comment_size_threshold]
        print(f'- Select greatest review comment size (quantile 75%) - Parcial line count (Base Prompts): {len(df)}')
        
        
        # group the reviews by year
        df['review_year'] = pd.to_datetime(df['review_datetime']).dt.year
        df = df.groupby('review_year')
        
        # for each group if year, filter: 
        for year in df.groups.keys():
            df_year = df.get_group(year)
            print(f'\nProcessing year: {year} - Initial line count: {len(df_year)}')
            
            # select only one register per review_user column on df;
            df_year = df_year.groupby('review_user').head(1)
            print(f'- Select maximum of one reviews per user - Parcial line count (Base Prompts): {len(df_year)}')
            
            # Best and worst reviews
            df_year = df_year[(df_year['review_general_rate'] >= 3) | (df_year['review_general_rate'] <= 2.0)]
            print(f'- Select best and worst reviews review_general_rate<=2 or >=3 - Parcial line count (Base Prompts): {len(df_year)}')
            
            # select max 5 registers per beer_style column on df
            df_year = df_year.groupby('beer_style').head(1)
            print(f'- Select maxiumum of reviews per beer_style - Parcial line count (Base Prompts): {len(df_year)}')
        
            # # percentil 2% reviwers with most reviews “review_num_reviews” and inexperient (<=2 reviews)
            # greatest_reviewers_threshold = df_year['review_num_reviews'].quantile(0.90)
            # df_year = df_year[ (df_year['review_num_reviews'] >= greatest_reviewers_threshold) | (df_year['review_num_reviews'] <= 10)]
            # print(f'- Select experients (quantile 2%) and inexperients (<=2 reviews) - Parcial line count (Base Prompts): {len(df_year)}')  
            
            if 'df_final' in locals():
                df_final = pd.concat([df_final, df_year], ignore_index=True)
            else:
                df_final = df_year.copy()
                
        df = df_final.copy()
            
      
        
        # select maximum of 6 reviews by year
        df = df.groupby('review_year').head(6)
        print(f'\nSelect maximum of 6 reviews by year - Parcial line count (Base Prompts): {len(df)}')
        
        df = df.sort_values(by=['review_datetime', 'review_general_rate', 'review_num_reviews'])
        print(f'\nFinal line count (Base Prompts Evaluation): {len(df)}')
        
        df.to_csv(f'{self.work_dir}/step_4_1__base_prompts.csv', index=False)
        return df


    def llm_batch_equivalence_judge(self, pred_df, gold_df, error_count, prompt_ai):
        """
        Itera sobre aspectos anotados (gold), chamando o LLM uma vez por gold.
        Cada chamada recebe 1 gold e todos os preds ainda não utilizados.
        Produz pareamentos um-para-um e remove preds pareados.
        """

        pred_list = [
            {
                "id": i,
                "aspect": r["aspect"],
                "category": r["category"],
                "sentiment": r["sentiment"],
            }
            for i, r in pred_df.reset_index(drop=True).iterrows()
        ]

        gold_list = [
            {
                "id": i,
                "aspect": r["aspect"],
                "category": r["category"],
                "sentiment": r["sentiment"],
            }
            for i, r in gold_df.reset_index(drop=True).iterrows()
        ]

        remaining_preds = pred_list.copy()
        matches = []

        for gold in gold_list:

            if not remaining_preds:
                matches.append(
                    {
                        "gold_id": gold["id"],
                        "pred_id": -1,
                        "aspect_ok": False,
                        "category_ok": False,
                        "sentiment_ok": False,
                    }
                )
                continue

            gold_payload = {
                "id": gold["id"],
                "aspect": gold["aspect"],
                "category": gold["category"],
            }

            pred_payload = [
                {
                    "id": p["id"],
                    "aspect": p["aspect"],
                    "category": p["category"],
                }
                for p in remaining_preds
            ]

            prompt = f"""
    Você é um avaliador de equivalência semântica para ABSA.

    TAREFA:
    Compare o ASPECTO ANOTADO com os ASPECTOS PREVISTOS e identifique
    no máximo UM pareamento semântico.

    RETORNE APENAS um JSON válido no formato:

    {{
        "gold_id": <int>,
        "pred_id": <int | -1>,
        "aspect_ok": <true|false>,
        "category_ok": <true|false>
    }}

    REGRAS:
    - Use somente os ids fornecidos.
    - Se NÃO houver aspecto previsto semanticamente equivalente:
        - aspect_ok = false
        - pred_id = -1
        - category_ok = false
    - Se houver aspecto semanticamente equivalente:
        - aspect_ok = true
        - pred_id = id correspondente
        - category_ok = true SOMENTE se as categorias forem semanticamente equivalentes.

    ASPECTO ANOTADO:
    {gold_payload}

    ASPECTOS PREVISTOS:
    {pred_payload}
    """
            prompt_ai.prompt = prompt
            response, finish_reason = prompt_ai.get_completion()

            if finish_reason != "stop":
                error_count += 1
                print(f'Error count: {error_count}')
                continue
            
            try:
                response = response.replace("```json", "").replace("```", "").strip()
                match = json.loads(response)
            except Exception:
                print(f'Error parsing response: {response}')
                error_count += 1
                print(f'Error count: {error_count}')
                continue

            # Pós-processamento mínimo de segurança
            match["sentiment_ok"] = False

            if match["aspect_ok"] is True and match["pred_id"] != -1:
                pred_id = match["pred_id"]

                pred_item = next(
                    (p for p in remaining_preds if p["id"] == pred_id), None
                )

                if pred_item is not None:
                    # Remove pred utilizado
                    remaining_preds = [
                        p for p in remaining_preds if p["id"] != pred_id
                    ]

                    # Avaliação determinística de sentimento
                    if pred_item["sentiment"] == gold["sentiment"]:
                        match["sentiment_ok"] = True

            matches.append(match)

        return matches, error_count



                    

    def run_step_4_2_ABSA_model_shots_evaluation(self, base_prompts_df):
        """
        Runs ABSA outputs against base_prompts_validation_annotated.csv
        and computes macro Precision / Recall / F1.
        """

        annotated_file = f"{self.work_dir}/base_prompts_validation_annotated.csv"
        df_gold = pd.read_csv(annotated_file, sep=",", encoding="utf-8")
        
        prompt_ai = Prompt_AI("gpt-4o-mini", None)

        reviews_per_request = 6
        num_reviews_to_process = 108

        for model in ['sabia-3','gpt-4o-mini']:
          for use_all_BC in [True, False]:
              for nshots in [1, 3]:
                    file_basename=f'{self.work_dir}/step_4_2____{nshots}shots_{model}_{"all_BC" if use_all_BC else f"{nshots}_BC"}'
                    error_count = 0
                    
                    df_pred = self.run_ABSA(
                        'step_4_2',
                        base_prompts_df,
                        model,
                        nshots,
                        reviews_per_request,
                        num_reviews_to_process=num_reviews_to_process,
                        use_all_BC=use_all_BC
                    )
                    
                    # TESTING 
                    # file_basename_read_done_exp=f'{self.work_dir}/step_4_2__{nshots}shots_{model}_{"all_BC" if use_all_BC else f"{nshots}_BC"}_6rev_per_req_from_0.csv'
                    # df_pred = pd.read_csv(file_basename_read_done_exp, sep=",", encoding="utf-8")
                    # print(f'\n\n****************************\ndf_pred - line count: {len(df_pred)} \n\n')
                    # continue
                    
                    df_scores_filename = f'{file_basename}_scores.csv'
                    try:
                        print(f'Reading {df_scores_filename}')
                        df_scores = pd.read_csv(df_scores_filename, sep=",", encoding="utf-8")
                    except:
                        print(f'{df_scores_filename} not exists')
                        df_scores = None
                    
                    per_review_scores = []

                    # test_count = 0
                    for idx in df_gold['index'].unique():
                        # if test_count > 1:
                        #     break
                        # test_count += 1

                        gold_i = df_gold[df_gold['index'] == idx]
                        pred_i = df_pred[df_pred['index'] == idx]

                        # ignore non existing reviews in the validation set or in the predicted set
                        if len(gold_i) == 0 or len(pred_i) == 0:
                            print(f'No reviews found for review {idx} !!!')
                            continue
                        
                        if df_scores is not None and idx in df_scores['index'].unique():
                            print(f'Found review {idx} in df_scores, skipping')
                            continue

                        matches, error_count = self.llm_batch_equivalence_judge(pred_i, gold_i, error_count, prompt_ai)
                        if len(matches) == 0:
                            print(f'No matches found for review {idx}')
                            continue
                            
                        print("matches", matches)

                        a_correct = sum(1 for m in matches if m["aspect_ok"])
                        b_correct = sum(1 for m in matches if m["aspect_ok"] and m["category_ok"])
                        c_correct = sum(
                            1 for m in matches
                            if m["aspect_ok"] and m["category_ok"] and m["sentiment_ok"]
                        )

                        a_total_pred = len(pred_i)
                        a_total_gold = len(gold_i)
                        a_correct = min(a_correct, a_total_pred)
                        b_correct = min(b_correct, a_total_pred)
                        c_correct = min(c_correct, a_total_pred)
                        
                        print("a_total_pred", a_total_pred)
                        print("a_total_gold", a_total_gold)
                        print("a_correct", a_correct)
                        print("b_correct", b_correct)
                        print("c_correct", c_correct)

                        per_review_scores.append({
                            "index": idx,
                            "a_correct": a_correct,
                            "b_correct": b_correct,
                            "c_correct": c_correct,
                            "a_total_pred": a_total_pred,
                            "a_total_gold": a_total_gold,
                        })
                        
                        df_scores = pd.DataFrame(per_review_scores)
                        df_scores.to_csv(df_scores_filename, index=False)

        # write final results to csv
        results = []
        
        def preci_recall(pred_correct, total_pred_or_gold):
            return pred_correct / total_pred_or_gold if total_pred_or_gold > 0 else 0
                
        def f1(prec, recall):
            return 2*prec*recall/(prec+recall) if (prec+recall) > 0 else 0
                
        for model in ['sabia-3', 'gpt-4o-mini']:
            for use_all_BC in [True, False]:
                for nshots in [1, 3]:
                    file_basename=f'{self.work_dir}/step_4_2____{nshots}shots_{model}_{"all_BC" if use_all_BC else f"{nshots}_BC"}'
                    df_scores_filename = f'{file_basename}_scores.csv'
                    # df_scores.to_csv(df_scores_filename, index=False)
                    
                    # open the df_scores_filename to df_scores
                    try:
                        df_scores = pd.read_csv(df_scores_filename, sep=",", encoding="utf-8")
                    except:
                        print(f'Error reading {df_scores_filename}')
                        continue
                    
                    # print("df_scores", df_scores)
                    a_correct_total = df_scores["a_correct"].sum()
                    b_correct_total = df_scores["b_correct"].sum()
                    c_correct_total = df_scores["c_correct"].sum()
                    a_total_gold_total = df_scores["a_total_gold"].sum()
                    a_total_pred_total = df_scores["a_total_pred"].sum()
                    
                    a_prec = preci_recall(a_correct_total, a_total_pred_total)
                    b_prec = preci_recall(b_correct_total, a_total_pred_total)
                    c_prec = preci_recall(c_correct_total, a_total_pred_total)
                    
                    a_rec = preci_recall(a_correct_total, a_total_gold_total)
                    b_rec = preci_recall(b_correct_total, a_total_gold_total)
                    c_rec = preci_recall(c_correct_total, a_total_gold_total)
                    
                    a_f1 = f1(a_prec, a_rec)
                    b_f1 = f1(b_prec, b_rec)
                    c_f1 = f1(c_prec, c_rec)
                    
                
                    metrics = {
                        "model": model,
                        "nshots": nshots,
                        "use_all_BC": use_all_BC,
                        "a_prec": a_prec,
                        "b_prec": b_prec,
                        "c_prec": c_prec,
                        "a_rec": a_rec,
                        "b_rec": b_rec,
                        "c_rec": c_rec,
                        "a_f1": a_f1,
                        "b_f1": b_f1,
                        "c_f1": c_f1,
                    }
                    results.append(metrics)

        df_results = pd.DataFrame(results)
        df_results_filename = f'{self.work_dir}/step_4_2____evaluation_metrics.csv'
        df_results.to_csv(df_results_filename, index=False)

        print("\nStep 4.2 evaluation completed")
        print(df_results)
                
                
    def run_step_4_3_evaluate_main_base(self, df_main_base):
        "The best model will run on the whole dataset"
            
        best_model = 'sabia-3'
        best_nshots = 1
        num_reviews_to_process = 10e6
        reviews_per_request = 10
        is_num_shots_for_each_CC = False
        self.run_ABSA('step_4_3', df_main_base, best_model, best_nshots, 
                      reviews_per_request=reviews_per_request, num_reviews_to_process=num_reviews_to_process, use_all_BC = is_num_shots_for_each_CC)
        
        print("Please, copy the best file of combination of this step to step_4.csv")
       
            
    def run_ABSA(self, step_name, df_base, model, nshots, reviews_per_request = 10, num_reviews_to_process = None, use_all_BC = True):

        # i_initial_eval_index = 6  # 0 in from begining, otherwise index of last processed element + 1
        # i_final_eval_index = 12
        
        i_initial_eval_index = 0  # 0 in from begining, otherwise index of last processed element + 1
        i_final_eval_index = min(num_reviews_to_process, len(df_base)) # or number of last element to be processed + 1
        
        prompt_zero = self.step_4_1_get_prompt_zero_shot()
        if nshots == 0:
            prompt_n_shot = prompt_zero
        else:
            prompt_n_shot = self.step_4_1_get_prompt_few_shots(prompt_zero, nshots, use_all_BC)
        
        print(f'Running {step_name} with model {model} and {nshots} shots ...')
        review_eval_count = 1
        reviews_comments = ''
        response_columns = ['index', 'aspect', 'category', 'sentiment']
        df_response = pd.DataFrame(columns=response_columns)
        n_shot_file_name = f'{self.work_dir}/{step_name}__{nshots}shots_{model}_{"all_BC" if use_all_BC else f"{nshots}_BC"}_{reviews_per_request}rev_per_req_from_{i_initial_eval_index}.csv'
        
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

                    # Remove leading whitespace/newlines
                    response = response.lstrip()
                    response = response.rstrip()

                    # Normalize start
                    response = re.sub(r'^\s*(?:\[\s*)+', '[[',response)

                    # Normalize end
                    response = re.sub(r'(?:\s*\])+\s*$', ']]', response)

                    # fix for sabia-3 alucination with "][" each review
                    pattern = r'\s*[\r\n]*\]\s*[\r\n]*\[\s*[\r\n]*'
                    response = re.sub(pattern, '],[',response)

                    # fix for sabia-3 alucination with "]][" each review
                    pattern = r'\s*[\r\n]*\]\s*[\r\n]*\]\s*[\r\n]*\[\s*[\r\n]*'
                    response = re.sub(pattern, '],[',response)

                    # fix for sabia-3 alucination with "][[" each review
                    pattern = r'\s*[\r\n]*\]\s*[\r\n]*\[\s*[\r\n]*\[\s*[\r\n]*'
                    response = re.sub(pattern, '],[',response)

                    # fix for sabia-3 alucination with "]][[" each review
                    pattern = r'\s*[\r\n]*\]\s*[\r\n]*\]\s*[\r\n]*\[\s*[\r\n]*\[\s*[\r\n]*'
                    response = re.sub(pattern, '],[',response)

                    # fix for sabia-3 alucination with "]],[[" each review
                    pattern = r'\s*[\r\n]*\]\s*[\r\n]*\]\s*[\r\n]*,\s*[\r\n]*\[\s*[\r\n]*\[\s*[\r\n]*'
                    response = re.sub(pattern, '],[',response)

                    # fix for sabia-3 alucination with "]],[" each review
                    pattern = r'\s*[\r\n]*\]\s*[\r\n]*\]\s*[\r\n]*,\s*[\r\n]*\[\s*[\r\n]*'
                    response = re.sub(pattern, '],[',response)

                    # fix for sabia-3 alucination with "],[[" each review
                    pattern = r'\s*[\r\n]*\]\s*[\r\n]*,\s*[\r\n]*\[\s*[\r\n]*\[\s*[\r\n]*'
                    response = re.sub(pattern, '],[',response)

                    
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
        
        return df_response


    def step_4_1_get_prompt_zero_shot(self):
            
        print(f'Step 4.1 - "Prompt ABSA zero-shot" creation')
        prompt_sys = """ 
Você é um extrator de aspectos de cerveja. Do texto, extraia os ‘aspectos’ e a ‘categoria’ relacionados aos aspectos da cerveja. As categorias devem estar \
dentre os valores: ‘visual’, ‘aroma’, ‘sabor’, ‘amargor’, ‘álcool’ e ‘sensação na boca’. Extraia o ‘sentimento’ dentre os valores ‘muito negativo’, ‘negativo’, ‘neutro’, \
‘positivo’ ou ‘muito positivo’ para cada par aspecto/categoria. \
Regras:
- Dividir o aspecto na menor unidade possível. Por exemplo: "Espuma branca de média duração" deve gerar dois aspectos: "espuma branca" e "espuma de média duração", ambas categorias: "visual" \
- O entendimento sobre o sentimento deve considerer o sentimento relacionado para cada aspecto. Não considerar o entendimento do modelo pré-treinado ou o sentimento geral do texto. Usar "neutro" para aspectos que não possuem um sentimento relacionado. \
    - Exemplo: "Aroma cítrico" deve ser sentimento "neutro". "Bom aroma cítrico." indica um sentimento "positivo" para o aspecto. Outro exemplo: "muito Sabor de café": não indica um sentimento muito positivo, mas "sabor de café muito agradável" sim. \
    - Exceções: "espuma de boa retenção" e os adjetivos "refrescante", "cremosa", "balanceado", "equilibrado" sempre indicam um sentimento positivo. \
- Se houver expressãoes parecidas com "o sabor acompanha o aroma", copiar todos os aspectos da categoria "aroma" para a categoria "sabor", bem como o sentimento relacionado. \
Cada avaliação a ser avaliada está compreendida entre chaves. Cada item contém "index", que registra o índice da avaliação e "review_comment", que é o texto a ser avaliado. \
Não faça comentários, apenas gere a saída dos campos extraídos no formato a seguir: ['index','aspecto','categoria','sentimento'],\
"""
        return prompt_sys


    def step_4_1_get_prompt_few_shots(self, prompt_zero_shot: str, num_shots: int, use_all_BC: bool = True):
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


        prompt_few_shots = prompt_zero_shot + """ \
Abaixo, entre aspas, exemplos de textos de avaliações e o resultado esperado. \
Ignore o valor do campo index dos exemplos, pois são apenas para mostrar o formato de saída.
"""
   
        if use_all_BC:
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

        return prompt_few_shots

