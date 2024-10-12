# dvd_data
Project Process
Customer Churn Prediction:

Convert DVR data into a DataFrame using PostgreSQL.
Data cleaning and preparation (handle missing values, normalization).
Apply K-Means Clustering for unsupervised segmentation (two clusters: likely to churn vs. not likely).
Build an Artificial Neural Network (ANN) for churn prediction.
Evaluate with metrics like accuracy, precision, recall, and AUC.
Movie Genre Preference Prediction:

Extract and preprocess rental history from PostgreSQL.
Map rentals to movie genres.
Build a multi-class classification model (using neural networks) to predict customer genre preferences.
Train model using features like rental frequency and past preferences.
Evaluate using accuracy, confusion matrix, and F1-score.
Rental Demand Forecasting:

Process rental data into a time-series format.
Use Recurrent Neural Networks (RNN) and LSTM to predict future rental demand.
Train the model to forecast monthly rental demand, considering trends and seasonality.
Evaluate using Mean Squared Error (MSE) and Root Mean Square Error (RMSE).
