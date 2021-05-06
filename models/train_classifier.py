import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    """
    Returns the messages, message labels and category names to be used for model training
    Input
        database_filepath: path to the SQLlte database where the cleaned data is stored
    Output
        X: Series containing the disaster messages
        Y: DataFrame containig the message labels
        category_names: category names that messages could be classified under
    """    
    prefix = 'sqlite:///'
    engine = create_engine(prefix + database_filepath)
    df = pd.read_sql_table('disaster_messages', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:41]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Custom tokenization function that returns cleaned text tokens from a message
    Input
        text: string containing the message
    Output
        clean_tokens: list containing the text tokens parsed from the message
    """
    tokens = word_tokenize(text)
#     words = [w for w in words if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        tok_clean = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(tok_clean)
    return clean_tokens


def build_model():
    """
    Builds a machine learning pipeline that fits models classifying messages into multiple categories. Returns a GridSearchCV object with the fitted models
    Output
        cv: GridSearchCV object that contains models for classifying messages into multiple categories
    """
    pipeline = Pipeline([
        ('tokenize', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'tokenize__stop_words': ('english', None),
        'tfidf__use_idf': (True, False),
        'moc__estimator__n_estimators': (50, 100)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints the classification report for each category, based on output from the fitted model
    Inputs
        model: GridSearchCV or Pipeline object that can generate predictions based on test data
        X_test: validation set containing independent variables
        Y_test: validation set containing dependent variables
        category_names: category names that messages can be classified under
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    """
    Saves the best fitted model to the destination location
    Input
        model: GridSearchCV object containing fitted models
        model_filepath: filepath to where model is to be saved
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    """
    Loads clean data, trains model and saves best model to be used in app for prediction
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()