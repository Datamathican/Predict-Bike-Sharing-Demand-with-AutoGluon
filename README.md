# Project: Predict Bike Sharing Demand with AutoGluon

## Project Overview
Bike-sharing demand is highly relevant to related problems companies encounter, such as Uber, Lyft, and DoorDash. Predicting demand not only helps businesses prepare for spikes in their services but also improves customer experience by limiting delays. 

In this project, we use the **AutoGluon** library to train several machine learning models for the [Bike Sharing Demand competition in Kaggle](https://www.kaggle.com/c/bike-sharing-demand). We use AutoGluon's Tabular Prediction framework to fit data from CSV files provided by the competition. 

This project demonstrates the ability to use AutoGluon to train several iterations of models, perform Exploratory Data Analysis (EDA), execute Feature Engineering, and conduct Hyperparameter Tuning (HPO) to continuously optimize the model's performance.

## Project Structure
* **`project.ipynb`**: The main Google Colab Notebook containing all the code for exploratory data analysis, feature engineering, model training, and Kaggle submissions.
* **`project.html`**: An HTML export of the completed notebook showing all executed cells and outputs.
* **`report.md`**: A detailed markdown report documenting the model iterations, hyperparameter configurations, and Kaggle score improvements.
* **`img/`**: A folder containing line plots generated during the project:
  * `model_train_score.png`: Shows the training/validation scores over three iterations.
  * `model_test_score.png`: Shows the actual Kaggle evaluation scores over three iterations.

## Environment and Setup

While this project is part of the AWS Machine Learning Engineer Nanodegree, this specific implementation was adapted to run in **Google Colab**. 

### Project Instructions

1. Create an account with Kaggle.
2. Download the Kaggle dataset using the kaggle python library.
3. Train a model using AutoGluon’s Tabular Prediction and submit predictions to Kaggle for ranking.
4. Use Pandas to do some exploratory analysis and create a new feature, saving new versions of the train and test dataset.
5. Rerun the model and submit the new predictions for ranking.
6. Tune at least 3 different hyperparameters from AutoGluon and resubmit predictions to rank higher on Kaggle.
7. Write up a report on how improvements were made by either creating additional features or tuning hyperparameters, and why you think one or the other is the best approach to invest more time in.
