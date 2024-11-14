# CS5228-car-price-prediction

[xieyuan_preprocessing.ipynb link](https://colab.research.google.com/drive/19HDmme6qKjhSRaHkAHXv74ToSN7CD-67#scrollTo=LMDOda9PDFxo)
1. This notebook corrects the make/model mapping error.
2. It fills "power" missing values.
3. It creates "car_age", "coe_month_left" features.
4. It output processed train/test.csv files

training.ipynb
1. Explore different data mining methods based on the previously processed data results, train_processed.csv and test_processed.csv（from xieyuan_preprocessing.ipynb）, and analyze the outcomes.
2. Implement a method for calculating prices based on variable relationships and analyze the results.
3. Read in the results from a well-performing version of the tree model, submission.csv, and combine it with the above calculation method for prediction.

Tree_model.py
1. Data cleaning and handle the missing value.
2. Random forest and Gradient Boosting Tree.
3. Experiment on hypremater.
