import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from plotly.graph_objs import Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    """
    Function preprocesses the text data

    Parameters
    ----------
    text: str

    Returns
    -------
    clean_tokens: str - clean text file
    """
    # Split text into words
    tokens = word_tokenize(text)
    # Convert words into dictionary
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    def cat_counts(df, col):
        """
        Function performs value counts on categorical data

        Parameters
        ----------
        df: pd.DataFrame
        col: str - categorical data

        Returns
        -------
        dff: pd.DataFrame - counts and percentage shares.
        """
        df[col].value_counts()
        df_dict = df[col].value_counts().to_dict()
        df_list = []
        for key, value in df_dict.items():
            temp = [key, value]
            df_list.append(temp)
        dff = pd.DataFrame(df_list, columns=[col, 'Count'])
        dff['Perc'] = dff['Count'] / np.sum(dff['Count'])

        return dff

    genre_count = cat_counts(df, "genre")
    labels = genre_count['genre']
    values = genre_count['Count']

    def column_list_dict(x):
        """
        Function loops through the dummy variables and counts its frequency.
        Parameters
        ----------
        x: pd.DataFrame - dummy variables

        Returns
        -------
        pd.DataFrame
        """
        column_list_df = []
        for col_name in x.columns:
            count = df[df[col_name] == 1][col_name].value_counts().astype('str').str.extract(
                "([-+]?\d*\.\d+|\d+)").astype(float)
            y = col_name, count
            column_list_df.append(y)
        return pd.DataFrame(column_list_df, columns=['class', 'count'])

    # Work through the dataframe to simplify the plot
    categories = df.drop(['original', 'id', 'message', 'genre', 'related'], axis=1)
    counts = column_list_dict(categories)
    counts['total_count'] = counts['count'].astype(str).str[-7:]
    counts = counts.drop('count', axis=1)
    counts = counts[((counts['class'] != 'child_alone'))]
    counts['counts'] = counts['total_count'].astype(float)
    counts = counts.drop('total_count', axis=1).sort_values('counts')

    # Input to plot 2 - counts of the target responses.
    target_classes = counts['class']
    target_class_counts = counts['counts']

    # Plot Top 7 classes grouped by genre (weather,food,water,aid_related,medical_help,death and shelter)
    weather_class = df[df['weather_related'] == 1]['genre'].value_counts()
    food_class = df[df['food'] == 1]['genre'].value_counts()
    water_class = df[df['water'] == 1]['genre'].value_counts()
    aid_class = df[df['aid_related'] == 1]['genre'].value_counts()
    medical_class = df[df['medical_help'] == 1]['genre'].value_counts()
    death_class = df[df['death'] == 1]['genre'].value_counts()
    shelter_class = df[df['shelter'] == 1]['genre'].value_counts()
    class_names = ['direct', 'news', 'social']

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=target_classes,
                    y=target_class_counts
                )

            ],

            'layout': {
                'title': 'Disaster counts',
                'yaxis': {
                    'title': "Count"
                }

            }
        },

        {
            'data': [
                Bar(
                    y=weather_class,
                    x=class_names,
                    name='Weather'
                ),
                Bar(
                    y=food_class,
                    x=class_names,
                    name='Food'
                ),
                Bar(
                    y=water_class,
                    x=class_names,
                    name='Water'
                ),

                Bar(
                    y=aid_class,
                    x=class_names,
                    name='Aid'
                ),

                Bar(
                    y=medical_class,
                    x=class_names,
                    name='Medical'
                ),

                Bar(
                    y=death_class,
                    x=class_names,
                    name='Death'
                ),
                Bar(
                    y=shelter_class,
                    x=class_names,
                    name='Shelter'
                )
            ],

            'layout': {
                'title': 'Disaster per genre',
                'yaxis': {
                    'title': "Count"
                }

            }
        },

        {
            'data': [
                Pie(
                    labels=labels,
                    values=values,
                    pull=[0.1, 0, 0]
                )
            ],

            'layout': {
                'title': 'Percentage of messages per genre',

            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """

    Returns
    -------

    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()