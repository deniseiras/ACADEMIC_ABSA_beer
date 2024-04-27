import pandas as pd

# Function to read data directly into a DataFrame
def read_data(file):
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

# Function to select rows where all columns contain data
def select_complete_rows(df):
    complete_rows = df.dropna()
    return complete_rows

# Function to select rows where the "review_comment" column is not empty
def select_rows_with_comment(df):
    rows_with_comment = df[df['review_comment'].notna()]
    return rows_with_comment

# Function to generate descriptive statistics for non-empty columns
def generate_descriptive_statistics(df, file_to_save):
    statistics = df.describe(include='all')
    statistics.to_csv(file_to_save)
    return statistics

# Main function
def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    data_folder = '/media/denis/dados/_COM_BACKUP/MBA'
    file = f'{data_folder}/joined_data.csv'
    
    df = read_data(file)

    df_tmp = select_complete_rows(df)
    file_complete = f'{data_folder}/complete.csv'
    df_tmp.to_csv(file_complete)
    
    file_complete_stats = f'{data_folder}/complete_stats.csv'
    _ = generate_descriptive_statistics(df_tmp, file_complete_stats)
    
    df_tmp = select_rows_with_comment(df)
    file_wcomments = f'{data_folder}/wcommnets.csv'
    df_tmp.to_csv(file_wcomments)
    
    file_wcomments_stats = f'{data_folder}/wcommnets_stats.csv'
    _ = generate_descriptive_statistics(df_tmp, file_wcomments_stats)
    

if __name__ == "__main__":
    main()
