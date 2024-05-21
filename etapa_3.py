def run(df):
    """Create Prompt base

    Args:
        df (pandas.DataFrame): input dataframe
    
    Returns:
        pandas.DataFrame: output dataframe
    """
    
    # df_distinct_styles_year = df.groupby(['beer_style', df['review_datetime'].dt.year]).nunique()
    # df_distinct_styles_year.to_csv(f'{data_folder}/disitinct_styles_year.csv')
    df_distinct_styles = df.groupby(['beer_style']).nunique()
    print(f'Total distinct styles = {len(df_distinct_styles)}')
    # print(f'Total beer count per distinct styles\n{df_distinct_styles[["beer_name"]]}')
    
    # From here, df_distinct_styles was mapped manually to the BJCP categories in file style_count_category.csv
    dtype_options = {
        'style': str,
        'beer_count': int,
        'bjcp_category': str
    }
    
    df_styles = pd.read_csv('style_count_category.csv', sep=',', encoding='utf-8', dtype=dtype_options)
    print(df_styles)
    
    # TODO Check ETAPA 3.1 
