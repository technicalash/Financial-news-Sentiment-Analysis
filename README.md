# Financial News Sentiment Analysis using FinBERT

## Overview

This project is a Financial News Sentiment Analysis application that classifies financial news into **Positive**, **Neutral**, or **Negative** sentiment using a fine-tuned **FinBERT** transformer model. A simple and interactive **Streamlit** web application is provided for real-time sentiment prediction.

## Features

* Fine-tuned **FinBERT** model for financial sentiment analysis.
* Predicts **Positive**, **Neutral**, and **Negative** sentiment.
* Interactive web interface built with **Streamlit**.
* End-to-end NLP pipeline including preprocessing, tokenization, model training, evaluation, and deployment.

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers (FinBERT)
* Streamlit
* Scikit-learn
* Pandas
* Google Colab

## Dataset

The model was trained on a financial news sentiment dataset containing labeled financial sentences.
The dataset is available on Kaggle

## Model Performance

* **Model:** FinBERT (Transformer)
* **Accuracy:** **79%**

> **Note:** The dataset is imbalanced, with significantly fewer negative samples than neutral and positive samples. This affected the model's ability to classify the minority class, resulting in an overall accuracy of 79%.

## Development

* Model training and fine-tuning were performed in **Google Colab**.
* The deployment interface (`app.py`) was developed in **Visual Studio Code** using **Streamlit**.

## Future Improvements

* Improve performance on the minority class using class-weighted loss or oversampling techniques.
* Train on a larger and more balanced financial news dataset.
* Display confidence scores and sentiment probabilities in the web interface.
* Deploy the application on Streamlit Community Cloud or another hosting platform.
