# Wine Quality Prediction Project

This project demonstrates an end-to-end machine learning pipeline for predicting the quality of wines. It includes data preprocessing, model training, model deployment, and a user interface for predictions.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Feature Notebook](#feature-notebook)
4. [Synthetic Wine Generator](#synthetic-wine-generator)
5. [Training Notebook](#training-notebook)
6. [User Interface](#user-interface)
7. [Batch Inference Pipeline](#batch-inference-pipeline)

## Introduction

This project aims to predict the quality of wines using machine learning. It covers various aspects, including data preparation, model training, model deployment, and user interaction.

## Dataset

The [Wine Quality Dataset](https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/wine.csv) is used for this project. The dataset contains information about various attributes of wines, including quality.

## Feature Notebook

The feature notebook is responsible for registering the wine quality dataset as a Feature Group with Hopsworks. It ensures that the data is prepared and stored appropriately for training.

## Synthetic Wine Generator

A synthetic wine generator pipeline is implemented, adding a new synthetic wine to the dataset daily.

## Training Notebook

The training notebook reads training data from a Feature View in Hopsworks, trains a [Histogram Gradient Boosting Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor) to predict wine quality, and registers the model with Hopsworks.

## User Interface

A Gradio application is provided to allow users to input or select feature values and predict the quality of a wine using the trained model.

## Batch Inference Pipeline

A batch inference pipeline predicts the quality of the new wines added, and a monitor application displays the most recent wine quality predictions along with a confusion matrix showing historical prediction performance.
