# import pandas as pd

# # Example DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 3],
#     'B': [4, 5, 6],
#     'C': [7, 8, 9]
# })


# # Create an empty DataFrame with the same columns
# empty_df = pd.DataFrame(columns=df.columns)

# import re
# cleaned_string = 'Olá "mundo"\'\t\t\tteste\n\n teste' 
# print(cleaned_string)

# cleaned_string = cleaned_string.replace('\t', ' ')
# translation_table = str.maketrans('', '', "[]\"\'{}")
# cleaned_string = cleaned_string.translate(translation_table)
# # cleaned_string = re.sub(r'[^\x20-\x7E]', '', cleaned_string)
# print(cleaned_string)



# =================================================================

# from src.openai_api import get_completion

# prompt = """
# what is the maximum content before context_length_exceeded exception?
# """
# response, finish_reason = get_completion(f'{prompt}')



# =================================================================
# import pandas as pd
# import ast

# # Definindo a string fornecida
# data = """[
#     ["21171", "YES"],
#     ["21172", "YES"],
#     ["21173", "YES"],
#     ["21174", "YES"],
#     ["21175", "YES"]
# ]"""

# # Convertendo a string para uma lista de listas
# data_list = ast.literal_eval(data)

# # Criando o DataFrame
# df = pd.DataFrame(data_list, columns=["index", "selected"])

# print(df)

# =================================================================
# Concatenting the dataframes

from step import Step
import pandas as pd

step = Step()

step.read_csv(step.work_dir + "/step_3__reviews_selected___GPT4omini_0_653.csv")
df1 = step.df[["index", "selected"]]
# print(df1.head(10))
step.read_csv(step.work_dir + "/step_3__reviews_selected___GPT4omini_654_7439.csv")
df2 = step.df[["index", "selected"]]
step.read_csv(step.work_dir + "/step_3__reviews_selected___GPT4omini_7440_18119.csv")
df3 = step.df[["index", "selected"]]
step.read_csv(step.work_dir + "/step_3__reviews_selected___GPT4omini_18120_20145.csv")
df4 = step.df[["index", "selected"]]
step.read_csv(step.work_dir + "/step_3__reviews_selected___GPT4omini_20145_to_end.csv")
df5 = step.df[["index", "selected"]]

df_concatenated = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
print(df_concatenated.describe())
print(df_concatenated[["selected"]].value_counts())

indexes = df_concatenated["index"].tolist()
for i in range(0, 65405):
    if i not in indexes:
        df_new = pd.DataFrame([[i, "NO"]], columns=["index", "selected"])
        df_concatenated = pd.concat([df_new, df_concatenated], ignore_index=True)
        
print(df_concatenated.describe())
print(df_concatenated[["selected"]].value_counts())

df_concatenated.sort_values(by=["index"])
df_concatenated.to_csv(step.work_dir + "/step_3__reviews_selected.csv")

df_reviews_not_selected = df_concatenated[df_concatenated['selected'] == 'NO']
step2 = Step()
step2.read_csv(step.work_dir + "/step_2.csv")
step2.df = step2.df.drop(df_reviews_not_selected['index'].astype(int).tolist())
step2.df.reset_index(drop=True, inplace=True)
step2.df.to_csv(f'{step2.work_dir}/step_3.csv', index=False)