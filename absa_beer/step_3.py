"""
Step 3: Aspect-Based Sentiment Analysis of Beer Characteristics (CC)

:author: Denis Eiras

Functions:
    - 
"""

import pandas as pd
from step import Step


class Step_3(Step):

    def __init__(self) -> None:
        super().__init__()

    def remove_invalid_comments(self):
        """Remove invalid comments using AI

        Args:
            df (pandas.DataFrame): input dataframe

        Returns:
            pandas.DataFrame: output dataframe

        """
        print("Removing invalid comments")
        # invalid_comments = df[df['review_comment'].str.len() < 10]
        # df = df.drop(invalid_comments.index)

        # use openai_api.get_completion for selecting invalid comments

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
            df (pandas.DataFrame): input dataframe

        Returns:
            pandas.DataFrame: output dataframe
        """

        file = f"{self.work_dir}/step_2.csv"
        self.read_csv(file)
        print(f"{len(self.df)} lines Total")


