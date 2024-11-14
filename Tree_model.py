import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from category_encoders import TargetEncoder
import copy

import matplotlib.pyplot as plt

DATA_PATH = "./data/"
USE_FEATURE = ['make', 'model', 'manufactured', 'reg_date', 'type_of_vehicle', 
                'category', 'transmission', 'curb_weight', 'power', 'road_tax',
                'engine_cap', 'no_of_owners', 'depreciation', 'coe', 
                'dereg_value', 'mileage', 'price']

def count_multi_category(data, col=None):
    """
    Count the number of categories in a specified column of a pandas DataFrame.
    
    Parameters:
    data (pd.DataFrame): The DataFrame to be analyzed.
    col (str): The column to be analyzed (assumed to contain comma-separated categories).

    Returns:
    dict: A dictionary containing the count of each category.
    """
    category_counts = {}
    for _, row in data[col].items():
        categories = row.split(', ')
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
    return category_counts

def category_one_hot(data, col='category'):
    """
    Convert a categorical column to one-hot encoding.

    Parameters:
    data (pd.DataFrame): The DataFrame to be processed.
    col (str): The column containing categories to be one-hot encoded.

    Returns:
    pd.DataFrame: DataFrame with one-hot encoded categories, original column dropped.
    """
    data.reset_index(drop=True, inplace=True)
    category_counts = count_multi_category(data, col)
    for category in category_counts.keys():
        one_hot = np.zeros(len(data))
        for i, row in data.iterrows():
            if category in row[col].split(', '):
                one_hot[i] = 1
        data[f'category_{category}'] = one_hot.tolist()
    data.drop(columns=[col], inplace=True)
    return data

def fill_road_tax(df):
    """
    Fill missing road tax values based on the median of the same make and model.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.

    Returns:
    pd.DataFrame: DataFrame with missing road tax values filled.
    """
    df_road_tax = df[~df['road_tax'].isnull()][['make', 'model', 'road_tax']]
    df_road_tax = df_road_tax.groupby(['make', 'model']).median().reset_index()
    for idx in df[df['road_tax'].isnull()].index:
        make = df.loc[idx, 'make']
        model = df.loc[idx, 'model']
        road_tax = df_road_tax.loc[(df_road_tax['make'] == make) & (df_road_tax['model'] == model), 'road_tax']
        if not road_tax.empty:
            df.loc[idx, 'road_tax'] = road_tax.values[0]
    return df

def data_cleaning(df, split="train", features=USE_FEATURE):
    """
    Clean the dataset and prepare it for modeling.

    Parameters:
    df (pd.DataFrame): The DataFrame to be cleaned.
    split (str): Indicates if the data is for training or testing.

    Returns:
    pd.DataFrame: Cleaned DataFrame ready for modeling.
    """
    
    if split == "test":
        features = copy.deepcopy(features)
        features.remove('price')

    df_train = df[features].copy()
    
    # Fill missing values
    df_train = fill_road_tax(df_train)
    df_train.drop(['mileage'], axis=1, inplace=True)
    if split == "train":
        df_train.dropna(inplace=True)

    # One-hot encoding for categories
    df_train = category_one_hot(df_train)
    df_train = pd.get_dummies(df_train, columns=['transmission', 'type_of_vehicle'])

    # Date encoding
    df_train['reg_date_trans'] = pd.to_datetime(df_train['reg_date'], errors='coerce')
    df_train['year'] = df_train['reg_date_trans'].dt.year
    df_train['month'] = df_train['reg_date_trans'].dt.month
    min_year = df_train['year'].min()
    df_train['month_order'] = df_train['month'] + 12 * (df_train['year'] - min_year)
    df_train.drop(['reg_date_trans', 'year', 'month', 'reg_date'], axis=1, inplace=True)
    
    return df_train

def fill_by_model_make(df1, df2, col):
    """
    Fill missing values in a specified column based on the median of the same make and model.

    Parameters:
    df1 (pd.DataFrame): DataFrame with missing values.
    df2 (pd.DataFrame): Reference DataFrame for filling missing values.
    col (str): Column name to fill.

    Returns:
    pd.DataFrame: DataFrame with missing values filled.
    """
    df_fill = df2[~df2[col].isnull()][['make', 'model', col]]
    df_fill = df_fill.groupby(['make', 'model']).median().reset_index()
    for idx in df1[df1[col].isnull()].index:
        make = df1.loc[idx, 'make']
        model = df1.loc[idx, 'model']
        road_tax = df_fill.loc[(df_fill['make'] == make) & (df_fill['model'] == model), col]
        if not road_tax.empty:
            df1.loc[idx, col] = road_tax.values[0]
    return df1

def fillna_test(df_train, df_test):
    """
    Fill missing values in the test set using data from the training set.

    Parameters:
    df_train (pd.DataFrame): Training DataFrame.
    df_test (pd.DataFrame): Testing DataFrame with missing values.

    Returns:
    pd.DataFrame: Test DataFrame with missing values filled.
    """
    na_columns = df_test.columns[df_test.isna().any()].tolist()
    df_combined = pd.concat([df_train])
    
    for col in na_columns:
        df_test = fill_by_model_make(df_test, df_combined, col)

    # KNN imputation for remaining missing values
    df_combined_mean = df_combined.mean()
    df_combined_std = df_combined.std()
    df_combined = (df_combined - df_combined_mean) / df_combined_std
    df_test = (df_test - df_combined_mean) / df_combined_std
    
    imputer = KNNImputer(n_neighbors=23)
    imputer.fit(df_combined[~df_combined.isnull()])
    test_imputed = imputer.transform(df_test)
    df_test = pd.DataFrame(test_imputed, columns=df_test.columns)
    df_test = df_test * df_combined_std + df_combined_mean

    return df_test

def random_nan(df, p=0.1):
    """
    Randomly assign NaN values to p of the rows in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.

    Returns:
    pd.DataFrame: DataFrame with random NaN values added.
    """
    random_idx = np.random.choice(df.index, int(len(df) * p), replace=False)
    na_columns = list(set(df.columns) - set(["model", "make", "price"]))
    for idx in random_idx:
        col = np.random.choice(na_columns)
        df.loc[idx, col] = np.nan
    return df

def main_train(df_train, df_test, n_estimators=100, features= USE_FEATURE, missing_value=0):
    """
    Main function for training and evaluating a regression model.

    Parameters:
    df_train (pd.DataFrame): Training DataFrame.
    df_test (pd.DataFrame): Testing DataFrame.
    n_estimators (int): Number of trees in the model.
    features (list): List of features to use for training.

    Returns:
    tuple: Best model, its RMSE.
    """
    # 1. train data
    df_train = data_cleaning(df_train, "train", features)

    # 2. test data
    print(f"test columns: {df_test.columns} in main_train")
    df_test = data_cleaning(df_test, "test", features)

    # 3. encode make and model
    normial_cols = ['make', 'model']

    # use targer encoding
    encoder = TargetEncoder(cols=normial_cols)
    df_train[normial_cols] = encoder.fit_transform(df_train[normial_cols], df_train["price"])
    df_test[normial_cols] = encoder.transform(df_test[normial_cols])

    # 通过重新索引对齐列
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

    df_test = random_nan(df_test, missing_value)
    print(f"There are {df_test.isnull().sum().sum()} missing values in the test set.")
    df_test = fillna_test(df_train, df_test)

    # df_test = df_test.drop(['price'], axis=1)

    # 4. traintest split, K-fold cross validation
    X_train, X_test = df_train.drop(['price'], axis=1), df_test.drop(['price'], axis=1)
    y_train, y_test = df_train['price'], df_test['price']

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # train model random forest
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=5228)
    # model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=5228)
    # model = AdaBoostRegressor(n_estimators=1000, random_state=5228)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse}")
    return rmse, model

def main(features=USE_FEATURE, n_estimators=1000, missing_value= 0):
    df_train = pd.read_csv(DATA_PATH + 'train_processed.csv')
    
    RMSE_list = []
    best_model = None
    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=5228)
    for i, (train_index, test_index) in enumerate(kf.split(df_train)):
        print(f"Fold {i+1}/{K}")
        rmse, model = main_train(copy.deepcopy(df_train.iloc[train_index]), copy.deepcopy(df_train.iloc[test_index]), n_estimators=n_estimators, features=features, missing_value=missing_value)
        RMSE_list.append(rmse)
        if rmse == np.min(RMSE_list):
            best_model = model
    
    print(f"RMSE Means: {np.mean(RMSE_list)}, RMSE Std: {np.std(RMSE_list)}")
    print(f"Best RMSE: {np.min(RMSE_list)}")
    return best_model, np.min(RMSE_list), np.mean(RMSE_list)

def predict(best_model):
    # 1. train data
    df_train = pd.read_csv(DATA_PATH + 'train_processed.csv')
    df_train = data_cleaning(df_train, "train")

    # 2. test data
    df_test = pd.read_csv(DATA_PATH + 'test_processed.csv')
    df_test = data_cleaning(df_test, "test")

    # 3. encode make and model
    normial_cols = ['make', 'model']

    # use targer encoding
    encoder = TargetEncoder(cols=normial_cols)
    df_train[normial_cols] = encoder.fit_transform(df_train[normial_cols], df_train["price"])
    df_test[normial_cols] = encoder.transform(df_test[normial_cols])

    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

    df_test = fillna_test(df_train, df_test)

    df_test = df_test.drop(['price'], axis=1)
    print(df_train.columns)
    print(df_test.columns)
    print(f"Train data shape: {df_train.shape}, Test data shape: {df_test.shape}")

    # 5. predict test data
    pred = best_model.predict(df_test)
    submission = pd.DataFrame({'Id': range(len(pred)), 'Predicted': pred})
    submission.to_csv(DATA_PATH + 'submission.csv', index=False)

def draw_bar_chart(X, y, title):
    """
    Draw a bar chart of feature frequency.

    Parameters:
    X (list): List of feature names.
    y (list): List of feature frequencies.
    title (str): Chart title.
    """

    fig = plt.figure(figsize=(10, 5), dpi=100)
    plt.bar(X, y)
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    for i, v in enumerate(y):
        plt.text(i, v + 0.05, str(v), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f'./{title}.png')

def feature_experment():
    """
    Experiment with feature selection.
    """
    result = []
    for i in USE_FEATURE:
        feature = [x for x in USE_FEATURE if x != i]
        try:
            _, best_rmse, mean_rmse = main(feature)
            result.append({'delete_feature': i, 'best_rmse': best_rmse,'mean_rmse': mean_rmse})
            print(f"Delete feature {i}, best_rmse: {best_rmse}, mean_rmse: {mean_rmse}")
        except Exception as e:
            print(f"Error with {i}")
            print(e)
        
        print(f"Result:{result}")
        print("="*50)
    _, best_rmse, mean_rmse = main()
    result.append({'delete_feature': None, 'best_rmse': best_rmse,'mean_rmse': mean_rmse})
    print(result)
    pd.DataFrame(result).to_csv('./feature_selection.csv', index=False)

def n_estimators_experment():
    """
    Experiment with n_estimators parameter.
    """
    result = []
    for i in range(100, 1001, 100):
        try:
            _, best_rmse, mean_rmse = main(n_estimators=i)
            result.append({'n_estimators': i, 'best_rmse': best_rmse,'mean_rmse': mean_rmse})
            print(f"n_estimators: {i}, best_rmse: {best_rmse}, mean_rmse: {mean_rmse}")
        except Exception as e:
            print(f"Error with {i}")
            print(e)
        
        print(f"Result:{result}")
        print("="*50)
    print(result)
    pd.DataFrame(result).to_csv('./n_estimators_selection.csv', index=False)

def missing_value_experment():
    """
    Experiment with missing value imputation.
    """
    df_train = pd.read_csv(DATA_PATH + 'train_processed.csv')
    # count the frequency of missing values
    missing_values = df_train.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    result = missing_values.sort_values(ascending=False)
    print(missing_values.sum())
    print(result)
    draw_bar_chart(result.index, result.values, 'Missing Value Frequency')

def missing_value_fill_experment():
    """
    Experiment with missing value imputation.
    """
    result = []
    for i in range(0, 55, 5):
        try:
            _, best_rmse, mean_rmse = main(missing_value=1.0 * i / 100)
            result.append({'missing_value_percent': i, 'best_rmse': best_rmse,'mean_rmse': mean_rmse})
            print(f"missing_value_percent: {i}, best_rmse: {best_rmse}, mean_rmse: {mean_rmse}")
        except Exception as e:
            print(f"Error with {i}")
            print(e)
        
        print(f"Result:{result}")
        print("="*50)
    print(result)
    pd.DataFrame(result).to_csv('./missing_value_percent.csv', index=False)

if __name__ == '__main__':

    _, best_rmse, mean_rmse = main()
    print(f"Best RMSE: {best_rmse}, Mean RMSE: {mean_rmse}")