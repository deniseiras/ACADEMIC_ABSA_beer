import os
import dotenv
import pandas as pd
import re
import unicodedata

class Step:

    def read_csv(self, filename, dtype_options=None):
        self.df = pd.read_csv(filename, sep=",", encoding="utf-8", dtype=dtype_options)

    def read_data(self, filename):
        """Reads the raw data into a DataFrame

        Args:
            file (string): file name to read from

        Returns:
            pandas.DataFrame: output DataFrame
        """

        dtype_options = {
            "beer_name": str,
            "beer_brewery_name": str,
            "beer_brewery_url": str,
            "beer_style": str,
            "beer_alcohol": str,
            "beer_is_active": str,
            "beer_is_sazonal": str,
            "beer_srm": str,  # uses coma separated
            "beer_ibu": float,  # uses decimal point
            "beer_ingredients": str,
            "review_user": str,
            "review_num_reviews": int,
            "review_datetime": str,
            "review_general_rate": float,  # uses decimal point
            "review_aroma": str,
            "review_visual": str,
            "review_flavor": str,
            "review_sensation": str,
            "review_general_set": str,
            "review_comment": str,
        }

        self.read_csv(filename, dtype_options=dtype_options)

    def generate_descriptive_statistics(self, file_to_save=None):
        """Generate descriptive statistics for non-empty columns

        Args:
            df (pandas.DataFrame): input dataframe
            file_to_save (string): file name to save to

        Returns:
            pandas.DataFrame: output dataframe
        """

        print("generating descriptive statistics")
        df = self.df[["beer_alcohol", "beer_srm", "beer_ibu", "review_num_reviews", "review_general_rate", 
                      "review_aroma", "review_visual", "review_flavor", "review_sensation", "review_general_set", "review_comment_size",]]
        statistics = df.describe(include="all")
        if file_to_save is None:
            print(statistics)
        else:
            file_path = os.path.join(self.work_dir, file_to_save)
            statistics.to_csv(file_path)

    def __init__(self) -> None:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)

        dotenv.load_dotenv("./.env")

        self.work_dir = os.getenv("WORK_DIR")
        self.df = None

    def run(self):
        pass


    def clean_json_string(self, json_string):
        cleaned_string = json_string.replace('\t', ' ')
        translation_table = str.maketrans('', '', "[]\"{}\'\`")
        cleaned_string = cleaned_string.translate(translation_table)
        # Here we use a regex to remove non-printable characters
        # cleaned_string = re.sub(r'[^\x20-\x7E]', '', cleaned_string)
        
        return cleaned_string
