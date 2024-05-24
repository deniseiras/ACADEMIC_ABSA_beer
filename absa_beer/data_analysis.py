from step import Step
import pandas as pd

anal = Step()

file = f'{anal.work_dir}/step_2.csv'
anal.read_csv(file)

# First, count the distict styles
df_distinct_styles = anal.df.groupby(['beer_style']).nunique()
print(f'Total distinct styles = {len(df_distinct_styles)}')
print(f'Total beer count per distinct styles\n{df_distinct_styles[["beer_name"]]}')

# We used this results to manually map the styles to the BJCP categories in file beer_count_by_bjc_category.csv
# "-1" was setted to the bjcp_categories when the style was not found or not equivalent.
beer_count_by_bjcp_category = pd.read_csv('./data/beer_count_by_bjcp_category.csv', sep=',', encoding='utf-8')
print(beer_count_by_bjcp_category.head())        

""" 'Seleção de reviews' prompt creation

Here it was sorted the step_2.csv by beer style, review_general_rate and review_num_reviews to step_2_sorted.csv. 
The idea was check manually for good and bad reviews texts. The lines checked select for this task was considered:
- beer_style. Diferent beer styles shows diferent caracteristics
- good and bad review_general_rate. This helps the IA system to check both good and bad beers
- review_num_reviews. This tells the review was writen by an experienced user. Some reviews was selected from the good or bad reviewers.

So, for each style, it was selected 1 good and 1 bad reviews for 1 good and 1 bad reviewers. (4 reviews by style)
It was considered 4 different styles for this task, so 16 reviews was selected to create the prompt "Seleção de reviews":
- American IPA 
...
"""

anal.df.sort_values(by=['beer_style', 'review_general_rate', 'review_num_reviews'], inplace=True)
anal.df.to_csv(f'{anal.work_dir}/step_2_sorted.csv')



# print(anal.df.info())
# anal.generate_descriptive_statistics(f'{anal.work_dir}/step_2_analysis.csv')
