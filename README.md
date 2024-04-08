# Stroke Detection Machine Learning

Implementing Multiple Classification models to predict Stroke patients.

## Table of Contents

- [Introduction](#introduction)
- [Data Source](#data-source)
- [Features](#features)
- [Learnings](#learnings)

## Introduction

The hospital has patient information and needs an estimation of exactly which variables affect whether people will be stroked or not.
- The goal is:
    - Which variables are significant in predicting whether patient will have strokes or not.
    - How well do those variables describe whether patient will have strokes or not
 
## Data Source
This data set is taken from Kaggle, the link to which is: 
  - https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset

## Features

This is a very detailed and explanatory project for beginners who are looking to get their hands dirty with Exploratory Data Science, Hypothesis Testing, and different kinds of Classification models.

- Exploratory Data Analysis of Categorical and Numerical Variables.
- Using different statuses, such as having heart disease or not or having smoked before or not
- Collecting their living area, which is very important to decide whether people are stroked or not.
- BMI, or glucose in the blood, is also one of the important things that contribute to people having stroke or not.

    ### Feature selection
    - In this dataset, we are removing the gender feature because it does not contribute much to the target. Not only make model run faster but also prevent wasting memory.

## Learnings

The first real project I worked on, taught me several things:
- Getting comfortable with working with a relatively large and noisy dataset.
- Using and showcasing several Python techniques and libraries:
  1. Numpy
  2. Pandas
  3. Matplotlib
  4. Seaborn
  5. Scikit Learn
  6. Imblearn
- Using techniques like:
  1. Outlier Detection
  2. Identifying Input Features and Target Variables.
  3. Visualizing different graphs for different variables.
  4. Visualizing relationships amongst different variables.
  5. Run all the most common models in order to choose the best one.
  6. Deal with imbalanced data
  7. Find the best param using GridSearchCV in order to optimize models.
- Hypothesis Testing:
  1. Deciding which test to conduct.
  2. The hospital field needs to focus on recall metrics to prevent the spreading of patient illness.
- Splitting the dataset into training and testing splits.
- Display the predictions and actual values via table.
- Using Classification Performance Metrics to check the accuracy of our model:
    1. Accuracy
    2. Precision
    3. Recall
    4. F1 Score
    5. Confusion matrix
- Balance data in the target feature using SMOTENC in order to make the model less likely to be biased towards one class.
- Using GridSeachCV which include k-fold cross validation to optimized our dataset and find the best hyperparameter to implement in model.
- Feature Engineering:
  1. Introducing new features from the existing ones to prevent overfitting and save computational resources.
  2. Since introducing new features may increase the chances of introducing Multi-Collinearity in the dataset.
  3. Checking for Multi-Collinearity through Heatmap.
  4. Choosing relevant features for the model.
     
Below you can find a few snippets of the project

