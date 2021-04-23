# Disaster_Responses
This respository carries the code for an application that classifies disasters based on the messages.
## **Table of contents**
1. [Installations](#installation)
2. [Project motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#user-guide)   
5. [Results](#results)
6. [Licensing, Authors, Acknowledgements](#licensing)

## Installation <a name="installation"></a>
The code will run smoothly with python 3.
This application is built on mainly 2 libraries i.e. NLTK that supports Natural language processing and sklearn
a machine learning library. The application's web has been built with flask
 ```
 pip install nltk
 pip install sklearn
 pip install flask
 ```
Other dependancies for this project can be found in the requirements section [click here](https://github.com/OliviaNabbosa89/Disaster_Responses/blob/main/requirements.txt)

## Project motivation <a name="motivation"></a>
In this project we use data provided by Figure Inc to answer the following questions:
1. Can we classify disaster based on historical messages?
2. What is the most common communication channel for reporting disaster?
3. Which disaster class is the most reported?

## File Descriptions <a name="files"></a>
To answer these questions, I have  combined of natural language processing and machine learning.
This application is built on mainly 2 libraries i.e. NLTK which supports Natural language processing and sklearn which 
supports machine learning.

The repository contains 3 main directories:
1. app [click here](https://github.com/OliviaNabbosa89/Disaster_Responses/tree/main/app): This contains the main script that runs the application and html templates that have been used to develop the web page.
2. data [click here](https://github.com/OliviaNabbosa89/Disaster_Responses/tree/main/data): This contains the messages and disaster data that has been used to train the model and hosts the database (DisasterResponse.db)
3. models [click here](https://github.com/OliviaNabbosa89/Disaster_Responses/tree/main/models): This contains the pipeline for training classification model

## Instructions <a name="user-guide"></a>
The following steps will guide you on how to run scripts:

1. Run the following commands in the project's root directory to set up your database and model.
* 'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
* python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
2. Run the following command in the app's directory to run your web app.
* python run.py

## Results <a name="results"></a>
The main results of this project can viewed in  a web browser. Follow the steps below:
1. Run ih the project's root directory to view the application
* env | grep WORK
2. Get the variables printed in the console
* SPACEDOMAIN =udacity-student-workspaces.com
* SPACEID= view6914b2f4 (this will vary from user to user)
3. Pass the SPACEDOMAIN and SPACEID in the link http://SPACEID-3001.SPACEDOMAIN
4. Open the link http://SPACEID-3001.SPACEDOMAIN in the browser.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
I must give credit to Figure Eight Inc for providing the data that has been used in this project. [here](https://www.figure-eight.com)
Special thanks to Udacity that has provided the resource material and training throughout the project [here](https://www.udacity.com/)

