import pandas as pd
from datetime import datetime
from etapa_2 import run as run_etapa_2
from etapa_3 import run as run_etapa_3


def read_data(file):
    """Reads the raw data into a DataFrame

    Args:
        file (string): file name to read from

    Returns:
        pandas.DataFrame: output DataFrame
    """
    
    dtype_options = {
    'beer_name': str,
    'beer_brewery_name': str,
    'beer_brewery_url': str,
    'beer_style': str,
    'beer_alcohol': str,
    'beer_is_active': str,
    'beer_is_sazonal': str,
    'beer_srm': str,  # uses coma separated
    'beer_ibu': float,  # uses decimal point
    'beer_ingredients': str,
    'review_user': str,
    'review_num_reviews': int,
    'review_datetime': str,
    'review_general_rate': float,  # uses decimal point
    'review_aroma': str,
    'review_visual': str,
    'review_flavor': str,
    'review_sensation': str,
    'review_general_set': str,
    'review_comment': str
    }

    df = pd.read_csv(file, sep=',', encoding='utf-8', dtype=dtype_options)
    return df


def select_complete_rows(df):
    """Function to select rows where all columns contain data

    Args:
        df (pandas.DataFrame): input dataframe
    
    Returns:
        pandas.DataFrame: output dataframe
    """
    complete_rows = df.dropna()
    return complete_rows


def generate_descriptive_statistics(df, file_to_save):
    """Generate descriptive statistics for non-empty columns

    Args:
        df (pandas.DataFrame): input dataframe
        file_to_save (string): file name to save to
    
    Returns:
        pandas.DataFrame: output dataframe
    """    
    
    print('generating descriptive statistics')
    statistics = df.describe(include='all')
    statistics.to_csv(file_to_save)

                       
# Main function
def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    data_folder = '/media/denis/dados/_COM_BACKUP/MBA'

    file = f'{data_folder}/joined_data.csv'
    
    df = read_data(file)
    print(f'{len(df)} lines Total')
    
    df = run_etapa_2(df, data_folder) 
    generate_descriptive_statistics(df, f'{data_folder}/etapa2_stats.csv')
    
    df = run_etapa_3(df, data_folder)
    # generate_descriptive_statistics(df, f'{data_folder}/etapa3_stats.csv')
    

  
    

if __name__ == "__main__":
    main()
