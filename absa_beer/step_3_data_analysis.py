"""_summary_
"""

# Import necessary libraries
from step import Step
import pandas as pd

# Create an instance of the Step class
anal = Step()

# Specify the file path for the CSV file to be read
file = f"{anal.work_dir}/step_2.csv"
anal.read_csv(file)

# Count the distinct beer styles using groupby and nunique, and display the total count
df_distinct_styles = anal.df.groupby(["beer_style"]).nunique()
print(f"Total distinct styles = {len(df_distinct_styles)}")
print(f'\n\nTotal beer count per distinct styles\n{df_distinct_styles[["beer_name"]]}')

# Manually map beer styles to BJCP categories from the file 'beer_count_by_bjc_category.csv'
# Use "-1" for categories where the style was not found or not equivalent
# Some styles from Step 2 were used in Step 3 for 'Prompt Base Principal' creation
beer_count_by_bjcp_category = pd.read_csv(
    "./data/step_3_beer_count_by_bjcp_category.csv", sep=",", encoding="utf-8"
)
print(f"\n\nStyles in BJCP categories:")
print(beer_count_by_bjcp_category)

# Sort the dataframe by 'beer_style', 'review_general_rate', and 'review_num_reviews' columns
anal.df.sort_values(
    by=["beer_style", "review_general_rate", "review_num_reviews"], inplace=True
)
# Save the sorted data to a new CSV file for further analysis
anal.df.to_csv(f"{anal.work_dir}/step_3_data_analysis.csv")
