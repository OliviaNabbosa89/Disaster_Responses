# Disaster_Responses
This respository stores the code for an application that classifies disasters based on the messages.
When disasters strike, the disaster response organisations receive millions of alerts/messages at a time when their resources are constrained.
The disaster teams would like to attend to the most crusial crises first. This application helps the disaster
response teams to find the most crusial disasters while also helping those in need to get quick relief or help.
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
<<<<<<< HEAD
1. app [click here](https://github.com/OliviaNabbosa89/Disaster_Responses/tree/main/app)
   * template
     - master.html: main page of web app
     - go.html: classification result page of web app
    * run.py: Flask file that runs app.
2. data [click here](https://github.com/OliviaNabbosa89/Disaster_Responses/tree/main/data)
   * disaster_categories.csv: data to process
   * disaster_messages.csv: data to process
   * process_data.py: script that process the data.
   * DisasterResponse.db: database to save clean data. 
3. models [click here](https://github.com/OliviaNabbosa89/Disaster_Responses/tree/main/models)
   * train_classier.py: script for training and evaluating the model
   * classifier.pkl: saved model.
    
4. README.md
5. Requirements.txt

## Instructions <a name="user-guide"></a>
The following steps will guide you on how to run scripts:

1. Run the following commands in the project's root directory to set up your database and model.
* 'python data/process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db'
* python models/train_classifier.py DisasterResponse.db models/classifier.pkl
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

### Outlay of the web app

![image](https://user-images.githubusercontent.com/80167199/115960260-a585c800-a510-11eb-9f5e-640e676ec620.png)

![image](https://user-images.githubusercontent.com/80167199/115960282-c3532d00-a510-11eb-8f9a-41563116598e.png)



## Licensing, Authors, Acknowledgements<a name="licensing"></a>
I must give credit to Figure Eight Inc for providing the data that has been used in this project. [here](https://www.figure-eight.com)
Special thanks to Udacity that has provided the resource material and training throughout the project [here](https://www.udacity.com/)

