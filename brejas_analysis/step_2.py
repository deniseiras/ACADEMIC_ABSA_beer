"""
- Step 2: Data preprocessing

:author: Denis Eiras

Functions:
    - 
"""

import pandas as pd

def remove_invalid_comments(df):
    """Remove invalid comments using AI

    Args:
        df (pandas.DataFrame): input dataframe
    
    Returns:
        pandas.DataFrame: output dataframe

    """
    print('Removing invalid comments')
    # invalid_comments = df[df['review_comment'].str.len() < 10]
    # df = df.drop(invalid_comments.index)



    return df


# Function to select rows where the "review_comment" column is not empty
def select_rows_with_comment(df):
    """Select rows where the "review_comment" column is not empty

    Args:
        df (pandas.DataFrame): input dataframe
    
    Returns:
        pandas.DataFrame: output dataframe
    """
    print('selecting rows with comment')
    rows_with_comment = df[df['review_comment'].notna()]
    return rows_with_comment


def sanitize_column(df, column_name, min_val, max_val):
    """Removes records outside range

    Args:
        df (pandas.DataFrame): input dataframe
        column_name (string): _description_
        min_val (float): minimum value
        max_val (float): maximum value

    Returns:
        pandas.DataFrame: output dataframe
    """
    invalid_data = df[(df[column_name] > max_val) | (df[column_name] < min_val)]
    print(f'Removing {len(invalid_data)} lines of invalid {column_name}')
    df = df.drop(invalid_data.index)
    return df


def sanitize_data(df):
    """Removes records outside range calling sanitize_column

    Args:
        df (pandas.DataFrame): input dataframe

    Returns:
        pandas.DataFrame: output dataframe
    """

    print('Sanitizing data')
    df = sanitize_column(df, 'beer_alcohol', 0, 100)    
    df = sanitize_column(df, 'beer_srm', 0, 80)    
    df = sanitize_column(df, 'beer_ibu', 0, 120)    
            
    return df

def run(df, data_folder):
    """Convert types and transform evaluation values to [1-5] points. Remove invalid data
    based on range of values. Filter the non sense comments using OpenAI
    Args:
        df (pandas.DataFrame):

    Returns:
        pandas.DataFrame: the preprocessed pandas DataFrame
    """
    # convert types and transform avaliation values to [1-5] points
    print('preprocessing columns')
    df['review_datetime'] = pd.to_datetime(df['review_datetime'])
    df['beer_alcohol'] = df['beer_alcohol'].str.replace('% ABV', '').astype(float)
    df['beer_srm'] = df['beer_srm'].str.replace('.', '').str.replace(',', '.').astype(float)
    for field in ['review_aroma', 'review_visual', 'review_flavor', 'review_sensation', 'review_general_set']:
        df[field] = df[field].apply(eval).astype(float)
        df[field] = df[field] * 5
       
    # filter rows with comments 
    df = select_rows_with_comment(df)
    print(f'{len(df)} lines Total')

    # Remove invalid data based on range values
    df = sanitize_data(df)
    print(f'{len(df)} lines Total')
    
    # create a column with the size of comments
    df['comment_size'] = df['review_comment'].str.len()
    
    df = remove_invalid_comments(df)
    
    # generate the base
    df.to_csv(f'{data_folder}/etapa2.csv')
        
    return df