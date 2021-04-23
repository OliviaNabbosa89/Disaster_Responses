import sys
import pandas as pd

from sqlalchemy import create_engine


def load_data(disaster_messages, disaster_categories):
    """
    Function loads the raw data

    Parameters
    ----------
    disaster_messages: csv file contains messages
    disaster_categories: csv file contains categories

    Returns
    -------
    pd.DataFrame
        DataFrame containing merged messages and disaster data
    """
    # load messages datasets
    messages = pd.read_csv('data/disaster_messages.csv')
    categories = pd.read_csv('data/disaster_categories.csv')

    # merge the messages and categories dataframe
    df = messages.merge(categories, how='left', on='id')
    return df


def clean_data(df) -> pd.DataFrame:
    """
    Function cleans and preprocesses the data

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame in which dummy variables have been created and duplicate values removed.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # Define the targert columns
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    #  Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(float)


    # Drop the duplicates
    df = df.drop('categories', axis=1)

    # Replace categories column in df with new category columns
    df = pd.concat([df, categories], axis=1)

    #  Remove all rows where related = 0
    df = df.drop(df[df['related'] == 0].index)

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, Disaster_Responses):
    """
    Function creates the database connection and the clean data is saved in the data database
    Parameters
    ----------
    df: Pd.DataFrame
    Disaster_Responses: DataBase name

    Returns
    -------

    """
    engine = create_engine('sqlite:///data/Disaster_Responses.db')
    df.to_sql('Disaster_Responses', engine, index=False)


def main():
    """
    Fuction that wraps up the data processing
    Returns
    -------

    """
    if len(sys.argv) == 4:

        disaster_messages, disaster_categories, engine = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format('data/disaster_messages.csv', 'data/disaster_categories.csv'))
        df = load_data('data/disaster_messages.csv', 'data/disaster_categories.csv')

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format('sqlite:///data/Disaster_Responses.db'))
        save_data(df, 'sqlite:///data/Disaster_Responses.db')

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
