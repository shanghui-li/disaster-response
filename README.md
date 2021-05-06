# Disaster Response Pipeline Project

### Project summary
This repo contains scripts that builds a machine learning pipeline to categorise real messages sent during disasters (e.g. aid-related, medical help, water, food etc.). It also includes a web app that takes in any message as an input and predicts whether the message should fall under each of the disaster event categories.

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Description of files
- `app/templates/master.html`: html template for the web app's index page
- `app/templates/go.html`: html template displaying the output resulting from the classification of a message
- `app/run.py`: Python script containing the Flask framework for the web app
- `data/DisasterResponse.db`: example database that was generated from `process_data.py`
- `data/disaster_categories.csv`: raw data containing category labels
- `data/disaster_messages.csv`: raw data containing disaster messages
- `data/process_data.py`: Python script that reads in raw data and saves cleaned data to a database
- `models/train_classifer.py`: Python script that reads cleaned data from a database and uses the data to train a model classifying disaster messages into categories
