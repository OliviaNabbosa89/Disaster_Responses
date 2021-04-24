import sys
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
import joblib

import warnings

warnings.filterwarnings('ignore')

engine = create_engine('sqlite:///data/DisasterResponse.db')


def load_data(DisasterResponse):
    """
    Function loads the clean data from the database.
    In this function the target and explanatory variables are defined

    Parameters
    ----------
    Disaster_Responses: database name

    Returns
    -------
    X: pd.DataFrame - explanatory variables
    Y: pd.DataFrame - Target variables
    category_names: str - Labels of the target variables
    """
    # Load data from the database
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)

    # Define the X and Y variables
    X = df.message.values
    Y = df.drop(['original', 'id', 'message', 'genre'], axis=1).values
    category_names = [col for col in df.columns if col not in ['original', 'id', 'message', 'genre']]
    return X, Y, category_names


def tokenize(text) -> str:
    """
    Function preprocesses the text data.

    Parameters
    ----------
    text: str

    Returns
    -------
    str: clean text data
    """
    # Split text into words
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]

    # Identify the different parts of speech
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Function build the model's pipeline
    Returns
    -------
    model wrap
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'clf__estimator__n_estimators': range(5, 20 + 1, 5),
                  'clf__estimator__max_depth': range(2, 4)}

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Functions makes predictons using the saved model and evaluates its performance.

    Parameters
    ----------
    model: transformers, gridsearch and classifier
    X_test: pd.DataFrame - Explanatory variables
    Y_test: pd.DataFrame - Target variables
    category_names

    Returns
    -------
    model
    """
    # Predict the target variables
    Y_pred = model.predict(X_test)

    n = Y_pred.shape[1]

    # Create classification_report
    for i, category_name in enumerate(category_names):
        y_truth = Y_test[:, i]
        y_preds = Y_pred[:, i]
        class_report = classification_report(y_truth, y_preds)

        print(category_names[i])
        print(class_report)
    return model


def save_model(model, classifier):
    """
    Function saves the trained model.

    Parameters
    ----------
    model: trained model
    classifier: model name

    Returns
    -------

    """
    pickle.dump(model, open(classifier, 'wb'))


def main():
    """

    Returns
    -------

    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format('data/DisasterResponse.db'))
        X, Y, category_names = load_data('data/DisasterResponse.db')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format('models/classifier.pkl'))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()