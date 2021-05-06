import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Returns a data frame containing the messages and their corresponding labelled categories in raw form
    Inputs
        messages_filepath: path to csv file containing the messages
        categories_filepath: path to csv file containing the category labels for each message
    Output
        df: data frame that merges both raw data sources on the message id
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='inner', on='id')
    return df

def clean_data(df):
    """
    Returns a cleaned data frame showing which categories each message was labelled under
    Input
        df: data frame containing raw message and category data
    Output
        df: cleaned data frame with a column for each potential category and with duplicates removed
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe. use this row to extract a list of new column names for categories.
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str)

        # convert column from string to numeric
        categories[column] = categories[column].str[-1].astype(int)
    # replace 2's in column 'related' with 1's
    categories.loc[categories.related==2, 'related']=1 
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Saves the cleaned data in an SQLite database
    Input
        df: cleaned data frame with messages and corresponding categories
        database_filename: filepath where database is to be saved, e.g. disaster_response.db
    """
    prefix = 'sqlite:///'
    engine = create_engine(prefix + database_filename)
    df.to_sql('disaster_messages', engine, if_exists='replace', index=False) 

def main():
    """
    Runs above functions to read in raw data, clean the data and save the data to an SQLite database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()