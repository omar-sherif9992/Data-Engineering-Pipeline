

# Standard Library
import os
import random
import string
import time
import math
import requests

# Web Requests
import requests

# Data Manipulation & Preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler,RobustScaler
from tabulate import tabulate

# Parquet
import pyarrow as pa
import pyarrow.parquet as pq




# Machine Learning
from sklearn.linear_model import LinearRegression


# Date and Time
import datetime

# Warnings
import warnings

# Regular Expressions
import re

# Other
import shutil
from typing import List, Tuple, Dict, Any, Union


# Set Random Seed
RANDOM_SEED = 42
RANDOM_STATE = 42


def set_seed(seed=42):
    global RANDOM_SEED, RANDOM_STATE
    RANDOM_SEED = seed
    RANDOM_STATE = seed
    random.seed(seed)
    np.random.seed(seed)



# Set your Google Maps API key here
api_key = "AIzaSyBXV_Q4_CWvV7btH9drTwc3BYRoj2GwozQ"

warnings.filterwarnings('ignore')


# ## Constants <a id='constants'></a>
# 
# <p align="right"><a href='#table-of-content'>Go To Top</a></p>
# 

# In[5]:


MYINFO = 'DEW23 Omar Sherif Ali 49-3324 MET'  # Information about a person.

MILESTONE = 1  # The current milestone.

ROOT_DIR = os.path.abspath('.')

# This where all the Packages are cached instead of reinstalling them every new runtime
PACKAGES_DIR = f'{ROOT_DIR}/Packages'  # Directory for caching packages.

DATASET_DIR = f'{ROOT_DIR}/Dataset'  # Directory for storing datasets.
DATASET_NAME = 'green_tripdata_2016-10.csv'  # Name of the dataset file.

# This is where the figures are saved
FIGURES_DIR = f'{ROOT_DIR}/Figures'  # Directory for saving figures.

FIGURE_COUNTS = 1  # Number of figures.
SAVED_FIGURES = {}  # A dictionary for saving figures.

FEATURED_ENGINEER_COLS = []  # A list of columns that are featured engineered.
FIGURES = {}  # A dictionary for storing figures.

RENAME_NAMES = [()]  # A list containing tuples for renaming names.

EXPECTED_DATA_TYPES = {}  # A dictionary for expected data types.

HISTORY_TRACK = []  # A list for tracking historical data.

ENCODED_PREFIX = 'encoded_'  # Prefix for encoded data.
CLEAN_PREFIX = 'cleaned_'  # Prefix for cleaned data.
# Prefix for cleaned and imputed data.
CLEAN_IMPUTED_PREFIX = CLEAN_PREFIX + 'imputed_'

DAY_NAME_MAPPING = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}  # A dictionary mapping numerical day values to their names.

# So that I can flag whether
STEPS = {
    'BEFORE': 0,
    'AFTER': 1,
}  # A dictionary representing various data processing steps and their corresponding numeric values.

# Initialize the current step to the EDA (Exploratory Data Analysis) step.
CURRENT_STEP = STEPS['BEFORE']

CHECKPOINT_COUNTER = 0  # A counter for the number of checkpoints.
CHECKPOINT_DIR = f'{ROOT_DIR}/Checkpoints'  # Directory for saving checkpoints.

IMPUTE_TABLES_DIR = f'{ROOT_DIR}/impute'  # Directory for saving impute tables.
ENCODE_TABLES_DIR = f'{ROOT_DIR}/encode'  # Directory for saving encode tables.
LOOKUP_TABLES_DIR = f'{ROOT_DIR}/data'  # Directory for saving lookup tables.


FIXED_COLUMNS_NAMES = []
ORIGINAL_COLUMNS_NAMES = []


# #### Lookup tables dictionaries <a id="lookup"></a>
# 
# <p align="right"><a href='#table-of-content'>Go To Top</a></p>
# 

# In[6]:


LOOKUP_TABLE = {}    # An empty list used as a general lookup table.




def load_dataframe(file_path: str, file_format='csv', **kwargs) -> pd.DataFrame:
    """
    Load a DataFrame from a file in CSV or Parquet format.
    """
    # Check if the specified format is valid
    if file_format not in ['csv', 'parquet']:
        raise ValueError("Invalid file format. Use 'csv' or 'parquet'.")

    # Load the DataFrame in the specified format
    if file_format == 'csv':
        df = pd.read_csv(file_path, **kwargs)
    elif file_format == 'parquet':

        table = pq.read_table(file_path)
        df = table.to_pandas()

    print(f"DataFrame loaded from {file_format} file: {file_path}")
    return df


def save_dataframe(df: pd.DataFrame, file_path: str, file_format='csv'):
    """
    Save a DataFrame to a file in CSV or Parquet format, creating directories if they don't exist.
    """
    # Check if the specified format is valid
    if file_format not in ['csv', 'parquet']:
        raise ValueError("Invalid file format. Use 'csv' or 'parquet'.")

    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df is not a DataFrame.")
    # Create the directory if it doesn't exist

    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the DataFrame in the specified format
    if file_format == 'csv':
        if not file_path.endswith('.csv'):
            file_path += '.csv'
        df.to_csv(file_path, index=False)
    elif file_format == 'parquet':
        if not file_path.endswith('.parquet'):
            file_path += '.parquet.gzip'
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path, compression='gzip')

    print(f"DataFrame saved to {file_format} file: {file_path}")
    return file_path


# In[8]:


def insert_to_lookup_table(column_name: str, original_value: Any, imputed: Any, encoded: Any, message: str) -> None:
    """
    Inserts a value to the lookup table.
    """
    if column_name not in LOOKUP_TABLE:
        LOOKUP_TABLE[column_name] = []
    isExist = False
    for item in LOOKUP_TABLE[column_name]:
        if "original" in item and "imputed" in item and "encoded" in item:
            if item["original"] == original_value and (item["imputed"] == imputed or item["encoded"] == encoded):
                isExist = True
                break
    if not isExist:
        value = {
            "original": original_value,
            "imputed": imputed,
            "encoded": encoded,
            "message": message
        }
        LOOKUP_TABLE[column_name].append(value)


def save_lookup_table(file_name: str):
    """
    Saves all the lookup table to a single CSV file.
    """
    # Create an empty list to store all dataframes
    all_dfs = []

    for key in LOOKUP_TABLE:
        table = LOOKUP_TABLE[key]
        df = pd.DataFrame(table)

        if 'imputed' in df.columns and 'encoded' in df.columns:
            if len(table) > 0:
                df.drop_duplicates(inplace=True)
                df['column_name'] = key
                all_dfs.append(df)

    # Concatenate all dataframes together
    concatenated_df = pd.concat(all_dfs, ignore_index=True)
    print("============== Lookup Table =================")
    
    # Sort the DataFrame by the 'column_name' column
    concatenated_df.sort_values(by='column_name', inplace=True)
    
    # Reset the index based on the sorted order
    concatenated_df = concatenated_df.reset_index(drop=True)
    
    # Save the concatenated dataframe to a single CSV file
    save_dataframe(concatenated_df, os.path.join(
        LOOKUP_TABLES_DIR, file_name), 'csv')


def save_encode_table():
    """
    Saves each encode table to a CSV file.
    """
    for key in LOOKUP_TABLE:
        df = pd.DataFrame(LOOKUP_TABLE[key])

        # Check if 'imputed' and 'encoded' columns exist
        if 'imputed' in df.columns and 'encoded' in df.columns:
            filtered_df = df[(df['imputed'].isna()) & (~df['encoded'].isna())]
            
            if not filtered_df.empty:
                filtered_df.drop_duplicates(inplace=True)

                save_dataframe(filtered_df, os.path.join(
                    ENCODE_TABLES_DIR, f'{key}.csv'))
                print("=" * 60)


def save_impute_table():
    """
    Saves each impute table to a CSV file.
    """
    for key in LOOKUP_TABLE:
        df = pd.DataFrame(LOOKUP_TABLE[key])
        # Check if 'imputed' and 'encoded' columns exist
        if 'imputed' in df.columns and 'encoded' in df.columns:
            filtered_df = df[(~df['imputed'].isna()) & (df['encoded'].isna())]
            if not filtered_df.empty:
                filtered_df.drop_duplicates(inplace=True)
                save_dataframe(filtered_df, os.path.join(
                    IMPUTE_TABLES_DIR, f'{key}.csv'))
                print("="*60)


# In[9]:


def fix_types(df: pd.DataFrame):
    """
    Fixes the data types of the given dataframe.
    """
    global EXPECTED_DATA_TYPES
    for col in df.columns:
        try:
            if col.endswith('datetime'):
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
            elif col in EXPECTED_DATA_TYPES:
                df[col] = df[col].astype(EXPECTED_DATA_TYPES[col])
        except Exception as e:

            print(f'Failed to convert {col} to {EXPECTED_DATA_TYPES[col]}')
            print(e)
    return df




def feature_engineer(name: str, message: str, type: str):
    """
    Append the feature engineer name and message to the list.
    and update the EXPECTED_DATA_TYPES dictionary.
    """
    if name not in FEATURED_ENGINEER_COLS:
        EXPECTED_DATA_TYPES[name] = type
        # Append the feature engineer name and message to the list if it is not already in the list.
        FEATURED_ENGINEER_COLS.append(
            {'name': name, 'message': message, 'type': type})


# In[11]:


def format_large_number(large_number: float) -> str:
    """
    Format a large number with comma separators.
    """
    return "{:,.2f}".format(large_number)  # Comma separator for thousands, 2 decimal places

 # to avoid the scientific notation(to be more readable)
pd.set_option('display.float_format', format_large_number)


# In[12]:


def format_string(string: str) -> str:
    """
    Format a string by removing leading and trailing whitespaces, converting it to lowercase,
    and replacing any non-alphanumeric characters except comma with an empty space.
    """
    string = string.strip().lower()
    string = re.sub(r'[^a-z0-9,]', ' ', string).strip()
    return string


# In[13]:


def empty_folders():
    """
    Empties a folder by deleting all files and subdirectories.
    """
    for folder_path in [FIGURES_DIR, CHECKPOINT_DIR, ENCODE_TABLES_DIR, IMPUTE_TABLES_DIR]:
        if os.path.exists(folder_path):
            if os.path.isdir(folder_path):
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting file: {file_path} - {e}")

                    for directory in dirs:
                        dir_path = os.path.join(root, directory)
                        try:
                            shutil.rmtree(dir_path)
                            print(f"Deleted directory: {dir_path}")
                        except Exception as e:
                            print(
                                f"Error deleting directory: {dir_path} - {e}")
            else:
                print(f"{folder_path} is not a directory.")
        else:
            print(f"The folder {folder_path} does not exist.")


empty_folders()






def drop_column(df: pd.DataFrame, column_name: str):
    """
    Drops a column from a Pandas DataFrame.
    """
    df.drop(column_name, axis=1, inplace=True)
    if (column_name in FIXED_COLUMNS_NAMES):
        FIXED_COLUMNS_NAMES.remove(column_name)
    if (column_name in ORIGINAL_COLUMNS_NAMES):
        ORIGINAL_COLUMNS_NAMES.remove(column_name)


# In[30]:


def check_column_data_types(df: pd.DataFrame, expected_data_types=EXPECTED_DATA_TYPES):
    """
    Check if columns in a DataFrame have the correct data types.
    """
    for column_name in df.columns:
        actual_type = df[column_name].dtype
        expected_type = expected_data_types.get(column_name)
        if expected_type is None:
            print(
                f'No expected data type specified for column "{column_name}"')
        elif actual_type == expected_type:
            print(
                f'Column "{column_name}" has the correct type: {actual_type}')
        else:
            print(
                f'Column "{column_name}" has an incorrect type: {actual_type}, expected type: {expected_type}')


# In[31]:


def remove_unwanted_values(df, feature, unwanted_value):
    """
    Removes rows with unwanted values from the given feature column.
    """
    return df[df[feature] != unwanted_value]




def impute_missing_values_linear_regression(df: pd.DataFrame, target_columns: List[str], predictors=None):
    """
    Impute missing values in the specified target columns using linear regression.
    predictors can be same as target columns or different
    """
    if predictors == None:
        predictors = target_columns
    # Step 0: Split the dataset into two parts: data with missing values and data without missing values
    data_without_missing = df.dropna(subset=target_columns)
    for target_column in target_columns:

        # Step 1: Data Preparation

        new_predictors = [col for col in predictors if col != target_column]
        data_with_missing = df[df[target_column].isnull()]

        data_with_missing = data_with_missing.dropna(subset=new_predictors)

        # Step 2: Linear Regression Model
        model = LinearRegression()

        # Check if there are data points for imputation after filtering
        if data_with_missing.shape[0] == 0:
            print(
                f"No data points to impute for {target_column} after filtering")
            continue

        model.fit(data_without_missing[new_predictors],
                  data_without_missing[target_column])

        # Step 3: Imputation
        imputed_values = model.predict(data_with_missing[new_predictors])
        data_with_missing[target_column] = imputed_values
        print(
            f"number of imputed values for {target_column} is {len(imputed_values)}")

        # Step 4: Update the Original Dataset
        df.update(data_with_missing)


# In[33]:


def detect_and_fix_outliers_by_group(df: pd.DataFrame, outlier_column: str, group_columns: List[str], skew_threshold=1) -> pd.DataFrame:
    """
    Detect and fix outliers in a DataFrame by grouping the data and getting its mean.
    """
    # Step 1: Detect outliers in trip distance using a boxplot
    Q1 = df[outlier_column].quantile(0.25)
    Q3 = df[outlier_column].quantile(0.75)
    IQR = Q3 - Q1
        
        
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify and flag outliers
    is_outlier = (df[outlier_column] < lower_bound) | (
        df[outlier_column] > upper_bound)

    skewness = df[outlier_column].skew()

    # Calculate the median of the target feature for each category in the categorical feature

    if skewness >= skew_threshold:
        medians = df.groupby(group_columns)[outlier_column].transform('median')
        # Impute using the median if the column is highly skewed (right-skewed)
        df[outlier_column].fillna(medians, inplace=True)
    else:
        # Impute using the mean if the column is not highly skewed
        mean = df[outlier_column].mean()
        df[outlier_column].fillna(mean, inplace=True)

    # Step 2: Fix distance outliers by replacing them with the mean distance for the same hour of the day
    mean_group = df.groupby(group_columns)[
        outlier_column].transform('mean').round(1)

    mean = df[outlier_column].mean()
    mean_group.fillna(mean)

    df[outlier_column] = df[outlier_column].where(~is_outlier, mean_group)

    return df


# In[34]:


# Quantile based flooring and capping
def floor_cap_quantile(df: pd.DataFrame, col_name: str, floor_percentile: float, cap_percentile: float) -> tuple:
    """
    Calculate specified percentiles of a specified column in a DataFrame.
    """
    floor = df[col_name].quantile(floor_percentile)
    cap = df[col_name].quantile(cap_percentile)
    print(f"Floor: {floor} , Cap: {cap}")
    return floor, cap


def replace_outliers(df: pd.DataFrame, col_name: str, floor: float, cap: float):
    """
    Replace outliers in a specific column by capping values below the floor and above the cap.
    """
    df[col_name] = np.where(df[col_name] < floor, floor, df[col_name])
    df[col_name] = np.where(df[col_name] > cap, cap, df[col_name])


# In[35]:


# impute na with constant value
def impute_na_with_constant(df: pd.DataFrame, column_name: str, value: Any):
    """
    Impute null values in a column with a constant value.
    """
    df[column_name].fillna(value, inplace=True)
    insert_to_lookup_table(column_name, None, value, None,
                           f'Imputed null values in column "{column_name}" with constant value: {value}')


# In[36]:


def impute_multi(df, col_name_impute, col_categorical, method: str, missing_pattern="^unknown$|^missing$|^na$|^uknown"):
    """   
    Impute missing or unknown values in a DataFrame based on the mode, mean, or median of a column
    grouped by a categorical column.
    """
    if (df[col_name_impute].dtype == "object" or df[col_name_impute].dtype.name == "category") and method != "mode":
        print(f"{col_name_impute} is categorical so method must be mode.")
        return

    # Validate the imputation method
    if method not in ["mode", "mean", "median", "-1"]:
        raise ValueError(
            "Invalid imputation method. Use 'mode', 'mean', 'median' or '-1.")

    # Get the mode of the targeted feature in each value of the categorical feature
    if method == "mode":
        values = df.groupby(col_categorical)[col_name_impute].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    elif method == "mean":
        values = df.groupby(col_categorical)[
            col_name_impute].apply(lambda x: x.mean())

    elif method == "median":
        values = df.groupby(col_categorical)[
            col_name_impute].apply(lambda x: x.median())
    elif method == "-1":
        values = df.groupby(col_categorical)[
            col_name_impute].apply(lambda x: -1)

    # Store as key-value pairs (dictionary)
    values_dict = values.to_dict()

    # Replace missing values based on the value of col_categorical and the missing pattern

    def replace_missing(row):
        if pd.isna(row[col_name_impute]) or re.match(missing_pattern, str(row[col_name_impute])):
            original_value = row[col_name_impute]
            imputed_value = values_dict.get(
                row[col_categorical], row[col_name_impute])
            # here we insert the imputed value to the lookup table
            insert_to_lookup_table(col_name_impute,  original_value,  imputed_value, None,
                                   f"Imputed {col_name_impute} from {original_value} to {imputed_value}")
            return imputed_value
        return row[col_name_impute]

    df[col_name_impute] = df.apply(replace_missing, axis=1)


# In[37]:


def impute_outliers_with_mean(df: pd.DataFrame, col_name: str, total_column=None, threshold=0.0):
    """
    Impute outliers in a feature using the mean of values below a given threshold.
    """

    # Handle the case when the total_column is not provided
    if total_column and total_column not in df.columns:
        raise ValueError(
            f"'{total_column}' not found in the DataFrame columns.")

    # Calculate the mean excluding outliers, negative and zero values
    valid_data = df[(df[col_name] < threshold) & (df[col_name] > 0)][col_name]
    if valid_data.empty:
        # Handle the case when there are no valid data points
        raise ValueError("No valid data points found for imputation.")

    mean_val = valid_data.mean()

    # Create a mask for rows that are outliers
    outliers_mask = df[(df[col_name] > threshold) | (df[col_name] < 0)]

    if total_column:
        # Handle the case when the total column contains missing values
        if df[total_column].isnull().any():
            raise ValueError(
                f"'{total_column}' contains missing values. Imputation cannot be performed.")

        # Adjust the total column based on the difference between the outlier and mean value
        df.loc[outliers_mask, total_column] = df.loc[outliers_mask,
                                                     total_column] - df.loc[outliers_mask, col_name] + mean_val

    # Impute the outliers with the mean value
    # df.loc[outliers_mask, col_name] = mean_val
    # Impute the outliers with the mean value
    for index, row in outliers_mask.iterrows():
        original_value = row[col_name]
        imputed_value = mean_val
        insert_to_lookup_table(col_name, original_value, imputed_value, None,
                               f"Imputed {col_name} from {original_value} to {imputed_value}")
        df.at[index, col_name] = imputed_value

    return df


# In[38]:


def contains_missing_values(df: pd.DataFrame, column_name: str, missing_pattern=r"(?i)^unknown$|^missing$|^unknown,nv$|Unknown,Unknown|unknown|Unknownr|^unknown$|^missing$|^na$|^uknown$|^Unknown,Unknown$|^Uknown$|^unknown,unknown$|^Unknown,unknown$|^unknown,Unknown$|unknown,NV") -> bool:
    """
    Check if a column contains missing values indicated by a regular expression pattern.
    it also replace the missing values or pattern that indicates missing with "unknown" and return the new dataframe
    if it contains missing values return True else return False
    """

    # Create a regular expression pattern object
    pattern = re.compile(missing_pattern, flags=re.IGNORECASE)

    # Check if any value in the column matches the pattern
    flag = False
    missing_values = None

    if df[column_name].dtype == 'object':
        missing_values = df[column_name].str.strip()
        missing_values = df[column_name].str.contains(pattern)
        if missing_values.any():
            num_missing_values = missing_values.sum()  # Count the number of True values
            print(
                f'Column "{column_name}" contains {num_missing_values} missing values as a string which is {(num_missing_values/len(df))*100}%.')
            flag = True
            # Replace matched values with 'unknown' for object columns
            df[column_name] = df[column_name].str.replace(pattern, 'unknown')

        else:
            print(
                f'Column "{column_name}" does not contain missing values as a string.')

    missing_values_nan = df[column_name].isna().sum()

    if missing_values_nan > 0:
        print(f'Column "{column_name}"  {missing_values_nan} contains missing values as NaN which is {((missing_values_nan)/len(df))*100}%.')
        flag = True
    else:
        print(
            f'Column "{column_name}" does not contain missing values as NaN.')

    return flag, missing_values, missing_values_nan


# In[39]:


def identify_columns_needing_imputation(df: pd.DataFrame, columns=None):
    """
    Identify columns that contain missing values and display relevant information.
    """
    columns_needing_imputation = []
    if columns is None:
        columns = df.columns

    for column in columns:
        flag, missing_values, missing_values_nan = contains_missing_values(
            df, column)

        if flag:
            if missing_values is not None and missing_values.any():
                total_missing_values = missing_values.sum()
            else:
                total_missing_values = 0

            columns_needing_imputation.append({'column': column, 'missing_values_unknown': total_missing_values, 'missing_values_unknown_%': (
                total_missing_values/len(df))*100, 'missing_values_nan': missing_values_nan, 'missing_values_nan_%': (missing_values_nan/len(df))*100})

    if columns_needing_imputation:
        result_df = pd.DataFrame(columns_needing_imputation).sort_values(
            by=['missing_values_unknown', 'missing_values_nan'], ascending=False).reset_index(drop=True)
    else:
        print("No columns need imputation.")


# ##### Discretization <a id="discretization"> </a>
# 
# <p align="right"><a href='#table-of-content'>Go To Top</a></p>
# 

# In[40]:


def create_bins(df: pd.DataFrame, target_column: str, bin_column_name: str, labels: List[str]):
    """
    Create bins for a numeric column and add a new column with the bin labels.
    """
    # Calculate the minimum and maximum values of the target column
    min_ = df[target_column].min()
    max_ = df[target_column].max()
    num_bins = len(labels)

    # Calculate the width of each bin
    width = (max_ - min_) / num_bins

    # Create a list of bin boundaries
    bins = [min_ + i * width for i in range(num_bins)] + [max_]

    print(f'Bin boundaries: {bins}')
    for i in range(num_bins):
        bin_boundary = f'{bins[i]} - {bins[i + 1] if i < num_bins - 1 else max_}'
        insert_to_lookup_table(bin_column_name, bin_boundary,
                               labels[i], None, f"Created bins for Bin {i}: {bin_boundary} with label: {labels[i]}")
    # Create a new column in the DataFrame with the specified bin name
    df[bin_column_name] = pd.cut(
        df[target_column], bins, labels=labels, include_lowest=True)
    feature_engineer(
        bin_column_name, f'bins for {target_column}', df[bin_column_name].dtype.name)




def label_encode_column(df: pd.DataFrame, column_name: str, prefix: str = ENCODED_PREFIX, map=None):
    """
    Perform label encoding on a column in a DataFrame and update the lookup table with the mapping.
    """
    global LOOKUP_TABLE
    containsMissing = df[column_name].isnull().sum() > 0
    if containsMissing:
        print(f'You must impute first before encoding {column_name}')
        
        
    if map is not None:
        df[prefix + column_name] = df[column_name].map(map)
        for original_value, mapping in map.items():
            insert_to_lookup_table(column_name, original_value, None, mapping,
                                   f"Encoded {column_name} from {original_value} to {mapping} by Label Encoding")
        encoded_df = pd.DataFrame(LOOKUP_TABLE[column_name])
        return 

    label_encoder = LabelEncoder() if label_encoder is None else label_encoder

    if LOOKUP_TABLE is not None and ((column_name) in LOOKUP_TABLE):
        # If the column is already in the lookup table, retrieve the original and encoded values and initialize the label encoder
        original_values = [item['original'] for item in LOOKUP_TABLE[(
            column_name)] if 'encoded' in item and 'original' in item and item['encoded'] is not None and item['original'] is not None]
        label_encoder.fit(original_values)
        print(
            f'{prefix + column_name} is not empty; taking care of new values, old values are preserved')

    # Fit the label encoder on the column data
    encoded_column = label_encoder.fit_transform(df[column_name])

    # Create a mapping of labels to encoded values
    label_mapping = dict(zip(df[column_name], encoded_column))

    # Update the DataFrame with the encoded column
    df[prefix + column_name] = encoded_column

    # Update the lookup table with the label mapping
    for original_value, mapping in label_mapping.items():
        insert_to_lookup_table(column_name, original_value, None, mapping,
                               f"Encoded {column_name} from {original_value} to {mapping} by Label Encoding")

    encoded_df = pd.DataFrame(LOOKUP_TABLE[column_name])


def one_hot_encode_column(df: pd.DataFrame, column_name: str, prefix: str = ENCODED_PREFIX) -> pd.DataFrame:
    """
    Perform one-hot encoding on a column in a DataFrame and return the mapping as a dictionary.
    """
    containsMissing = df[column_name].isnull().sum() > 0
    if containsMissing:
        print(f'You must impute first before encoding {column_name}')

    # Perform one-hot encoding using Pandas get_dummies
    encoded_df = pd.get_dummies(df[column_name], prefix=prefix+column_name)

    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    df = pd.concat([df, encoded_df], axis=1)
    for encoded_column_name in encoded_df.columns:
        insert_to_lookup_table(column_name, encoded_column_name, None, encoded_column_name,
                               f"One-Hot Encoded {column_name} from column {column_name} to {encoded_column_name} by One Hot Encoding")
    encoded_df = pd.DataFrame(LOOKUP_TABLE[column_name])

    return df


def binarize_column(df: pd.DataFrame,column_name:str,map: Dict[str,int],prefix: str = ENCODED_PREFIX) -> pd.DataFrame:
    """
    Binarize the column in a DataFrame and return the mapping as a dictionary.
    """
    df[prefix + column_name] = df[column_name].map(map)
    
    for original_value, mapping in map.items():
        insert_to_lookup_table(column_name, original_value, None, mapping,
                               f"Encoded {column_name} from {original_value} to {mapping} by Binarization into column {prefix + column_name}")
    encoded_df = pd.DataFrame(LOOKUP_TABLE[column_name])
    return df
    
    


# #### Decoding functions for lookup tables <a id='decode'></a>
# 
# <p align="right"><a href='#table-of-content'>Go To Top</a></p>
# 

# In[42]:



def label_decode_column(encoded_df: pd.DataFrame, column_name: str, lookup_table: List[dict] = LOOKUP_TABLE, prefix: str = ENCODED_PREFIX) -> pd.DataFrame:
    """
    Perform label decoding on an encoded DataFrame and return the original values.
    """
    if encoded_df.empty:
        print("Input DataFrame is empty. Nothing to decode.")
        return encoded_df

    if prefix + column_name not in lookup_table:
        print(
            f'{prefix + column_name} lookup table is empty. Encoding is required before decoding.')
        return encoded_df

    # Create a mapping of encoded values to original labels
    label_mapping = {entry['encoded_value']: entry['original_value']
                     for entry in lookup_table[prefix + column_name]}

    # Reverse the encoding using the mapping
    encoded_column_name = prefix + column_name
    encoded_df[column_name] = encoded_df[encoded_column_name].map(
        label_mapping)

    return encoded_df


def one_hot_decode_column(encoded_df: pd.DataFrame, column_name: str, prefix: str = ENCODED_PREFIX) -> pd.DataFrame:
    """
    Perform one-hot decoding on an encoded DataFrame and return the original values.
    """
    # Create a mapping of one-hot encoded columns to original values
    label_mapping = {col: col.replace(prefix + column_name + '_', '')
                     for col in encoded_df.columns if col.startswith(prefix + column_name)}

    # Create a new DataFrame for the decoded values
    decoded_df = pd.DataFrame()

    # Decode the one-hot encoded values
    decoded_df[column_name] = encoded_df.apply(
        lambda row: label_mapping.get(row.idxmax(), None), axis=1)

    # Add the one-hot encoded columns to the decoded DataFrame
    for col in label_mapping:
        decoded_df[col] = encoded_df[col]

    return decoded_df



# loading the dataset
df = pd.read_csv(os.path.join(DATASET_DIR, DATASET_NAME))





def format_column_names(df: pd.DataFrame):
    """
    Format column names in a pandas DataFrame to follow best practices.
    """
    def clean_column_name(column_name):
        # Convert to lowercase
        formatted_name = column_name.lower()
        # Remove trailing spaces
        formatted_name = formatted_name.strip()
        # Replace spaces with underscores
        formatted_name = formatted_name.replace(' ', '_')
        # Remove special characters (replace with empty string)
        formatted_name = ''.join(
            e for e in formatted_name if e.isalnum() or e == '_')
        return formatted_name

    # Apply the clean_column_name function to each column name
    df.columns = [clean_column_name(col) for col in df.columns]

    return df


df = format_column_names(df)




def sort_columns_by_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
     Reorder the columns of the DataFrame based on the number of missing values in ascending order monotonically.
     """
    # Calculate the number of missing values (NaNs) in each column
    missing_values_count = df.isnull().sum()

    # Sort the columns based on the count of missing values in ascending order
    sorted_columns = missing_values_count.sort_values(ascending=True)

    # Reorder the DataFrame based on the sorted columns
    sorted_df = df[sorted_columns.index]
    return sorted_df


df = sort_columns_by_missing_values(df)




identify_columns_needing_imputation(df)




EXPECTED_DATA_TYPES = {
    'vendor': 'object',
    'pickup_datetime': 'datetime64[ns]',
    'dropoff_datetime': 'datetime64[ns]',
    'passenger_count': 'int64',
    'trip_distance': 'float64',
    'rate_type': 'object',
    'store_and_fwd_flag': 'object',
    'pu_location': 'object',
    'do_location': 'object',
    'payment_type': 'object',
    'fare_amount': 'float64',
    'extra': 'float64',
    'mta_tax': 'float64',
    'tip_amount': 'float64',
    'tolls_amount': 'float64',
    'improvement_surcharge': 'float64',
    'total_amount': 'float64',
    'congestion_surcharge': 'float64',
}
ORIGINAL_COLUMNS_NAMES = list(df.columns)




def process_new_introduced_numeric_column(df: pd.DataFrame, new_column_name: str, date_column_name: str):
    """
    Process the 'new_column_name' column: drop if no data in or after 2019, replace missing values with 0.
    """
    # Assuming the date column is 'pickup_date', make sure it's in datetime format
    df[date_column_name] = pd.to_datetime(df[date_column_name])

    # Check if the 'ehail_fee' column exists in the DataFrame
    if new_column_name in df.columns:
        has_data_after_2019 = (df[date_column_name] >= '2019-01-01').any()

        if not has_data_after_2019:
            # Drop 'ehail_fee' column if no data after 2019
            print(f'Dropping "{new_column_name}" column')
            drop_column(df, new_column_name)
        else:
            print(
                f'Replacing missing values in "{new_column_name}" column with 0')
            # Replace missing values in 'new_column_name' with 0
            df[new_column_name].fillna(0, inplace=True)
            # Add a lookup table entry for the imputed values
            insert_to_lookup_table(new_column_name, np.nan, 0, None,
                                   f"Imputed {new_column_name} from {np.nan} to {0} means it was empty")

    else:
        print(f'"{new_column_name}" column does not exist in the DataFrame.')


process_new_introduced_numeric_column(df, 'ehail_fee', 'lpep_pickup_datetime')


# In[57]:


process_new_introduced_numeric_column(
    df, 'congestion_surcharge', 'lpep_pickup_datetime')





dups_rows = df[df.duplicated(keep=False)][['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'do_location', 'pu_location',
                                           'total_amount', 'vendor', 'store_and_fwd_flag']].sort_values(by=['lpep_pickup_datetime', 'lpep_dropoff_datetime'])




# Drop rows where all attributes are duplicated
df = df.drop_duplicates(keep=False)





# dropping the rows that have all of the values missing because they are useless for us
df.dropna(how='all', inplace=True)





identify_columns_needing_imputation(df, ['vendor'])





df = one_hot_encode_column(df, 'vendor')




identify_columns_needing_imputation(df, ['store_and_fwd_flag'])



df = binarize_column(df, 'store_and_fwd_flag',{
    'N':0,
    'Y':1
})





def convert_to_datetime_col(df: pd.DataFrame):
    """
    Convert lpep_pickup_datetime  and  lpep_dropoff_datetime of the DataFrame from a string to datetime type.
    """
    def timeconvert(str1):
        """
        used in case if it is in 12 hour format
        """
        date = str1.split(' ')[0].replace('/', '-')
        time = ' '.join(str1.split(' ')[1:])
        if time[-2:] == "AM" and time[:2] == "12":
            return str(date + ' ' + "00" + time[2:-2])
        elif time[-2:] == "AM":
            return str(date + ' ' + time[:-2])
        elif time[-2:] == "PM" and time[:2] == "12":
            return str(date + ' ' + time[:-2])
        else:
            return str(date + ' '+str(int(time[:2]) + 12)) + time[2:8]

    # Convert to datetime
    df['lpep_pickup_datetime'] = pd.to_datetime(
        df['lpep_pickup_datetime'], infer_datetime_format=True)
    df['lpep_dropoff_datetime'] = pd.to_datetime(
        df['lpep_dropoff_datetime'], infer_datetime_format=True)


convert_to_datetime_col(df)




def fix_invalid_timestamps(df: pd.DataFrame):
    """
    Check for invalid rows where 'lpep_dropoff_datetime' is before 'lpep_pickup_datetime' or either is missing.
    If it exists, swap them together.
    """
    # Check for invalid rows where 'lpep_dropoff_datetime' is before 'lpep_pickup_datetime'
    # or either 'lpep_dropoff_datetime' or 'lpep_pickup_datetime' is missing (null or NaN)
    invalid_rows = df[
        (df['lpep_dropoff_datetime'] < df['lpep_pickup_datetime']) |
        df['lpep_dropoff_datetime'].isnull() |
        df['lpep_pickup_datetime'].isnull()
    ]

    # Count the number of invalid rows
    num_invalid_rows = len(invalid_rows)

    if num_invalid_rows > 0:
        print(
            f'There are {num_invalid_rows} rows with "lpep_dropoff_datetime" before "lpep_pickup_datetime" or missing timestamps.')
        print('Here are the details of the invalid rows:')
        print(invalid_rows)

        # Swap the values for 'lpep_pickup_datetime' and 'lpep_dropoff_datetime' in the invalid rows
        df.loc[invalid_rows.index, ['lpep_pickup_datetime', 'lpep_dropoff_datetime']
               ] = df.loc[invalid_rows.index, ['lpep_dropoff_datetime', 'lpep_pickup_datetime']]

    else:
        print('All lpep_pickup_datetime are <= lpep_dropoff_datetime are valid and no missing timestamps.')


fix_invalid_timestamps(df)


# In[76]:


DATETIME_COLS = ['lpep_pickup_datetime', 'lpep_dropoff_datetime', ]
TIME_COLS = ['total_trip_time_sec',
             'total_trip_time_hr', 'total_trip_deltatime']
identify_columns_needing_imputation(df, DATETIME_COLS)




def sort_values(df:pd.DataFrame,cols: List[str]):
    """
    sort the dataframe by the given columns
    """
    df.sort_values(by=cols, inplace=True)
    df = df.reset_index(drop=True)
    return df  


sort_values(df, DATETIME_COLS)




# In[78]:


def calculate_total_time(df: pd.DataFrame, start_date_column: str, end_date_column: str):
    """
    Calculate the total time duration between two date columns and add a new column 'total time'.
    in this method also we feature engineer other attributes
    """
    # Calculate the time duration and create a new column 'total time' in seconds
    df['total_trip_deltatime'] = df[end_date_column] - df[start_date_column]
    feature_engineer('total_trip_deltatime', 'the total trip time in seconds',
                     df['total_trip_deltatime'].dtype.name)
    df['total_trip_time_sec'] = df['total_trip_deltatime'].dt.total_seconds()
    feature_engineer('total_trip_time_sec', 'the total trip time in seconds',
                     df['total_trip_time_sec'].dtype.name)
    df['total_trip_time_hr'] = df['total_trip_time_sec'] / 3600
    feature_engineer('total_trip_time_hr', 'the total trip time in hours',
                     df['total_trip_time_hr'].dtype.name)
    # Create the 'Week number' column
    df['week_number_yearly'] = df[start_date_column].dt.week
    feature_engineer('week_number_yearly', 'the week number of the year',
                     df['week_number_yearly'].dtype.name)

    df['week_number_monthly'] = df[start_date_column].dt.week % 4
    feature_engineer('week_number_monthly', 'the week number of the month',
                     df['week_number_monthly'].dtype.name)

    # Create the 'Date range' column with the start and end dates of each week
    df['date_range'] = df[start_date_column].dt.to_period('W').dt.strftime(
        '%Y-%m-%d') + ' to ' + (df[end_date_column] + pd.DateOffset(6)).dt.strftime('%Y-%m-%d')

    feature_engineer('date_range', 'the date range of the week',
                     df['date_range'].dtype.name)



# Call the function to calculate total time
calculate_total_time(df, 'lpep_pickup_datetime', 'lpep_dropoff_datetime')


LOCATION_COLUMNS = ['pu_location', 'do_location']


def process_location_columns(df: pd.DataFrame, location_columns: List[str]):
    """
    Process the specified location columns: replace missing values with 'Unknown'.
    """
    for col_name in location_columns:
        # Check if the location column exists in the DataFrame
        if (contains_missing_values(df, col_name)):
            continue
        elif col_name in df.columns and df[col_name].dtype == 'object' and df[col_name].isnull().any():
            print(
                f'Replacing missing values in "{col_name}" column with "unknown"')
            # Replace missing values in location column with 'Unknown' temporarily
            impute_na_with_constant(df, col_name, 'unknown')
        elif col_name not in df.columns:
            print(f'"{col_name}" column does not exist in the DataFrame.')
        else:
            print(f'"{col_name}" column does not contain missing values.')
            
    # check if both are unknown or one of them is unknown
    different_locations_dups = df[df.duplicated(subset=LOCATION_COLUMNS, keep=False) & (~df[LOCATION_COLUMNS].duplicated()) & ((df[LOCATION_COLUMNS[0]] == 'unknown') | (
        (df[LOCATION_COLUMNS[1]] == 'unknown')))][LOCATION_COLUMNS+['trip_distance', 'total_trip_time_sec', 'total_amount', 'store_and_fwd_flag']].sort_values(by=LOCATION_COLUMNS)
    same_location_dups = df[df.duplicated(subset=LOCATION_COLUMNS, keep=False) & (df[LOCATION_COLUMNS].duplicated()) & ((df[LOCATION_COLUMNS[0]] == 'unknown') & (
        (df[LOCATION_COLUMNS[1]] == 'unknown')))][LOCATION_COLUMNS+['trip_distance', 'total_trip_time_sec', 'total_amount', 'store_and_fwd_flag']].sort_values(by=LOCATION_COLUMNS)

    if len(different_locations_dups) > 0:
        print(
            f'There are {len(different_locations_dups)} rows with different unknown locations.')
        print('Here are the details of the rows:')
        print('='*60)
    else:
        print(f'There are no rows with different unknown locations.')
        print('='*60)
    if len(same_location_dups) > 0:
        print(f'There are {len(same_location_dups)} rows with same locations.')
        print('Here are the details of the rows:')

        print('='*60)
    else:
        print(f'There are no rows with same locations.')
        print('='*60)
        
       


process_location_columns(df, LOCATION_COLUMNS)





def get_gps_coordinates(address: str):
    """
    Get the GPS coordinates (latitude, longitude) of a given address using the Google Maps Geocoding API.
    """
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "OK":
                location = data["results"][0]["geometry"]["location"]
                latitude = location["lat"]
                longitude = location["lng"]
                return latitude, longitude
    except:
        return None, None

    return None, None




from geopy.geocoders import Nominatim

ALTERNATIVE_DATASET = "nyc-zip-code-latitude-and-longitude.csv"
LOCATION_DATASET = 'locations.csv'


def geocode_and_save_combined_locations(df, location_col_names, output_csv=LOCATION_DATASET):
    """
    Geocode unique locations from specified columns and save the results to a CSV file.
    If coordinates are missing, an alternative dataset can be used.
    If coordinates are still missing, google maps is being used.
    """
    # Check if the dataset file already exists
    dataset_path = os.path.join(DATASET_DIR, output_csv)
    if os.path.exists(dataset_path):
        print(f"Dataset '{output_csv}' already exists. Not recomputing.")
        return

    # Initialize the geocoder
    geolocator = Nominatim(user_agent="geocoder")

    # Create a combined list of unique locations from both columns
    unique_locations = set()
    for location_col_name in location_col_names:
        unique_locations.update(df[location_col_name].dropna().unique())

    # Create a DataFrame to store the results
    location_data = pd.DataFrame(columns=['location', 'latitude', 'longitude'])
    # Alternative Dataset for missing coordinates
    alternative_df = pd.read_csv(
        os.path.join(DATASET_DIR, ALTERNATIVE_DATASET))

    for location in unique_locations:
        location_info = geolocator.geocode(location)
        if location_info is not None:
            location_data = location_data.append({'location': location, 'latitude': float(
                location_info.latitude), 'longitude': float(location_info.longitude)}, ignore_index=True)
        else:
            # If coordinates are missing, try to get them from the alternative dataset
            alternative_location_info = alternative_df[alternative_df['location'] == location]
            if not alternative_location_info.empty:
                location_data = location_data.append({'location': location, 'latitude': float(
                    alternative_location_info.iloc[0]['latitude']), 'longitude': float(alternative_location_info.iloc[0]['longitude'])}, ignore_index=True)
            else:
                # If coordinates are still missing, use the Google Maps API
                latitude, longitude = get_gps_coordinates(location)
                location_data = location_data.append({'location': location, 'latitude': float(
                    latitude), 'longitude': float(longitude)}, ignore_index=True)

    # Save the location data to a CSV file
    save_dataframe(location_data, dataset_path)


geocode_and_save_combined_locations(df, LOCATION_COLUMNS)



COORDINATE_COLUMNS = []


def integrate_gps_coordinates(df: pd.DataFrame, location_col_names: List[str]):
    """
    Integrate the city names into GPS coordinates using geocoding.
    """
    lat_col_name = 'latitude'
    lon_col_name = 'longitude'
    # Load the geocoded location data from the CSV file
    location_data = pd.read_csv(os.path.join(DATASET_DIR, LOCATION_DATASET))

    loc_names = location_col_names.copy()

    # Merge the location data back into the original DataFrame for each specified location column
    for location_column in location_col_names:
        df = df.merge(location_data, left_on=location_column,
                      right_on='location', how='left')
        COORDINATE_COLUMNS.append(location_column)
        # Remove the 'Location' column from the merged DataFrame
        df = df.drop(columns=['location'])
        # Rename the latitude and longitude columns
        df = df.rename(columns={lat_col_name: f'{location_column}_{lat_col_name}',
                       lon_col_name: f'{location_column}_{lon_col_name}'})
        if '{location_column}_{lat_col_name}' not in loc_names:
            loc_names.append(f'{location_column}_{lat_col_name}')
        if '{location_column}_{lon_col_name}' not in loc_names:
            loc_names.append(f'{location_column}_{lon_col_name}')
        contains_missing_values(df, f'{location_column}_{lat_col_name}')
        contains_missing_values(df, f'{location_column}_{lon_col_name}')
        feature_engineer(f'{location_column}_{lat_col_name}', 'the latitude of {location_column}',
                         df[f'{location_column}_{lat_col_name}'].dtype.name)
        feature_engineer(f'{location_column}_{lon_col_name}', 'the longitude of {location_column}',
                         df[f'{location_column}_{lon_col_name}'].dtype.name)
    return df


df = integrate_gps_coordinates(df, LOCATION_COLUMNS)





def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float, trip_distance: float):
    """
    Calculate the Haversine distance between two GPS coordinates.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return trip_distance

    # Radius of the Earth in miles
    R = 3959.0

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) *         math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance


haversine_distance(40.6892, -74.0445, 40.7128, -74.0060, 0.0)


# In[87]:


def calculate_distance_from_google_api(lat1: float, lon1: float, lat2: float, lon2: float):
    """
    Calculate the distance between two GPS coordinates using the Google Maps API.
    """

    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins={lat1},{lon1}&destinations={lat2},{lon2}&key={api_key}"
    r = requests.get(url)
    data = r.json()
    try:
        distance_in_miles = float(
            data['rows'][0]['elements'][0]['distance']['text'].replace(' mi', ''))
        return distance_in_miles
    except:
        return None


# Example usage:
calculate_distance_from_google_api(40.6892, -74.0445, 40.7128, -74.0060)


# In[88]:


def calculate_distance(df: pd.DataFrame):
    # Calculate trip distance for each row
    if 'trip_distance_haversine' in df.columns:
        print('trip_distance_haversine already exists')
    else:
        df['trip_distance_haversine'] = df.apply(lambda row: haversine_distance(
            row['pu_location_latitude'], row['pu_location_longitude'],
            row['do_location_latitude'], row['do_location_longitude'],
            None
        ), axis=1)
        feature_engineer('trip_distance_haversine', 'the distance of the trip through haversine formula',
                         df['trip_distance_haversine'].dtype.name)

    doesItWork = None

    if (doesItWork != None):
        # Calculate trip distance using Google Maps API
        if 'trip_distance_google_map' in df.columns:
            print('trip_distance_google_map already exists')
        else:
            feature_engineer('trip_distance_google_map', 'the distance of the trip through google maps api',
                             df['trip_distance_google_map'].dtype.name)
        df['trip_distance_google_map'] = df.apply(lambda row: calculate_distance_from_google_api(
            row['pu_location_latitude'], row['pu_location_longitude'],
            row['do_location_latitude'], row['do_location_longitude']
        ), axis=1)

    else:
        print('Google API is not working')


calculate_distance(df)



def same_or_diff_neighberhood():
    different_locations_dups = df[df.duplicated(subset=LOCATION_COLUMNS, keep=False) & (~df[LOCATION_COLUMNS].duplicated(
    ))][LOCATION_COLUMNS+['trip_distance', 'trip_distance_haversine', 'total_trip_time_sec', 'total_amount', 'tolls_amount']].sort_values(by=LOCATION_COLUMNS)
    same_location_dups = df[df.duplicated(subset=LOCATION_COLUMNS, keep=False) & (df[LOCATION_COLUMNS].duplicated())][LOCATION_COLUMNS+[
        'trip_distance', 'trip_distance_haversine', 'total_trip_time_sec', 'total_amount', 'tolls_amount']].sort_values(by=LOCATION_COLUMNS)
    # Calculate the total count for each category
    total_same_location_count = same_location_dups.shape[0]
    total_different_locations_count = different_locations_dups.shape[0]

    print(
        f'There are {(same_location_dups.shape[0]/len(df))*100}% duplicated trips from-to (are same) in the dataset.')
    print(
        f'There are {(different_locations_dups.shape[0]/len(df))*100}% duplicated trips from-to are different in the dataset.')



same_or_diff_neighberhood()



identify_columns_needing_imputation(
    df, ['trip_distance', 'total_trip_time_hr'])





def calculate_mph(df: pd.DataFrame, distance_column: str, time_column: str):
    """
    Calculate the speed in miles per hour (mph) and add a new column 'mph'.
    """
    # Calculate the speed in miles per hour, but handle the case where time is zero
    df['mph'] = df.apply(lambda row: (row[distance_column] / (row[time_column] / 3600))
                         if (not pd.isna(row[time_column]) and row[time_column] != 0) else 10, axis=1)

    print(f'mph type: {df["mph"].dtypes}')


calculate_mph(df, 'trip_distance', 'total_trip_time_sec')







replace_outliers(df, 'trip_distance', 0,70)


# In[99]:


df = detect_and_fix_outliers_by_group(
    df, 'trip_distance', ['pu_location', 'do_location'])


# In[100]:


def fix_time_mph_outliers(df, speed_upperbound=25, speed_lowerbound=10):

    # Step 2: Cap mph outliers at 25 mph
    df['mph'] = df['mph'].apply(lambda x: max(
        min(x, speed_upperbound), speed_lowerbound))

    # Step 3: Recompute total_trip_time_sec using the new distance and capped mph values
    df['total_trip_time_sec'] = (df['trip_distance'] / df['mph'] * 3600) * 1.10
    df['total_trip_time_hr'] =  df['total_trip_time_sec'] / 3600
    df['total_trip_deltatime'] = pd.to_timedelta(
        df['total_trip_time_sec'], unit='s')


fix_time_mph_outliers(df)


# In[101]:


def same_time_location_fix(df: pd.DataFrame) -> pd.DataFrame:
    """
     Reset distance, mph, time, and total_trip_deltatime to 0 when pickup and dropoff times and locations are the same
     returns the masked df
    """
    same_time_location_mask = ((df['lpep_pickup_datetime'] == df['lpep_dropoff_datetime']) & (
        df['pu_location'].str.strip().str.lower() == df['do_location'].str.strip().str.lower())) | (df['trip_distance'] == 0)

    df.loc[same_time_location_mask, ['trip_distance', 'mph',
                                     'total_trip_time_hr', 'total_trip_time_sec', 'total_trip_deltatime']] = 0
    print(f'There are {same_time_location_mask.sum()} rows with the same pickup and dropoff times and locations that are kept the same.')
    return same_time_location_mask


same_time_location_mask = same_time_location_fix(df)





# we reached to 0.999992 by trial and error
floor, cap = floor_cap_quantile(df, 'trip_distance', 0, 0.999992)


# In[105]:


replace_outliers(df, 'trip_distance', floor, cap)


# In[106]:


# we need to fix the trip distane dependent attributes again
fix_time_mph_outliers(df)



# Step 4: Compute the new dropoff_datetime by adding pickup_datetime and total_trip_deltatime
def update_dropoff_datetime(df: pd.DataFrame):
    """
    Compute the new dropoff_datetime by adding pickup_datetime and total_trip_deltatime
    """
    df['lpep_dropoff_datetime'] = df['lpep_pickup_datetime'] +         pd.to_timedelta(df['total_trip_deltatime'], unit='s')


update_dropoff_datetime(df)




# since the outliers are removed, we can calculate the total time again and recalculate the feature engineered columns that depends on it
calculate_total_time(df, 'lpep_pickup_datetime', 'lpep_dropoff_datetime')


# In[110]:


identify_columns_needing_imputation(
    df, ['trip_distance', 'total_trip_time_hr', 'mph'])


# - as we can see here we dont have missing values for the trip distance and total_trip_time_hr but we will make impute function in case there is future missing values in the trip distance and total trip time in future data using linear regression
# 

# In[111]:


impute_missing_values_linear_regression(
    df, ['trip_distance', 'total_trip_time_hr', 'mph'])





same_time_location_df = df[same_time_location_mask]




def bin_trip_distance(df: pd.DataFrame):
    """
    Bin the trip distance into 5 categories: Very Short, Short, Medium, Long, Very Long
    """
    create_bins(df, 'trip_distance', 'trip_distance_bins', [
                'Very Short', 'Short', 'Medium', 'Long', 'Very Long'])


bin_trip_distance(df)




label_encode_column(df, 'trip_distance_bins',map={
    'Very Short':0,
    'Short':1,
    'Medium':2,
    'Long':3,
    'Very Long':4
})





def bin_rush_hour(df: pd.DataFrame):
    """
    Bin rush hour and non-rush hour trips based on the number of trips in each hour.
    """
    count_df = df.groupby(df['lpep_pickup_datetime'].dt.hour).size().reset_index(
        name='count')['count'].to_dict()
    df['count'] = df['lpep_pickup_datetime'].dt.hour.map(count_df)
    # bin whether that row in rush hour or not based on counts relative to this hour
    create_bins(df, 'count', 'is_rush_hour', [0, 1])
    df.drop(columns=['count'], inplace=True)
    df['hour'] = df['lpep_pickup_datetime'].dt.hour
    df.drop(columns=['hour'], inplace=True)


bin_rush_hour(df)




LOCATION_LOOKUP_TABLE = {}


def encode_location(df, prefix: str = ENCODED_PREFIX):
    """
    Encode the location columns using a lookup table.
    """
    # unique locations
    unique_locations = set()
    
    # reformat strings if theres duplicates and make it more standardized by regex
    for location_col_name in LOCATION_COLUMNS:
        df[location_col_name] = df[location_col_name].apply(format_string)
        
    # get unique locations
    for location_col_name in LOCATION_COLUMNS:
        unique_locations.update(df[location_col_name].dropna().unique())
        
    LOCATION_LOOKUP_TABLE['unknown'] = -1
    unique_locations.remove('unknown')
    # Create a lookup table
    for i, location in enumerate(unique_locations):
        
        LOCATION_LOOKUP_TABLE[location] = i
        insert_to_lookup_table(prefix + 'location', location, None,
                               i, f"Encoded {location_col_name} from {location} to {i}")
    # Encode the location columns
    for location_col_name in LOCATION_COLUMNS:
        df[prefix +
            location_col_name] = df[location_col_name].map(LOCATION_LOOKUP_TABLE)


encode_location(df)



df[df['pu_location'] == df['do_location']][['do_location', 'pu_location',
                                            'encoded_do_location', 'encoded_pu_location']].sample(5)




identify_columns_needing_imputation(df, ['passenger_count'])




def fix_passenger_count(df: pd.DataFrame, col_name: str, upper_bound=6, lower_bound=1):
    """
    Fix the passenger_count column:
      - Extract the first digit.
      - Replace values greater than the upper bound with the upper bound.
      - Handle NaN, None, and zero values by setting them to the lower_bound.
    """
    def calc_digit(x):
        return max(min(int(str(int(x))[0]) if not pd.isna(x) and str(int(x)).isdigit() else lower_bound, upper_bound), lower_bound)

    df[col_name] = df[col_name].apply(calc_digit)

    # Convert the passenger_count column to an integer data type
    df[col_name] = df[col_name].astype('int64')




fix_passenger_count(df, 'passenger_count')




# re-check for missing values
identify_columns_needing_imputation(df, ['passenger_count'])





# In[136]:


identify_columns_needing_imputation(df, ['rate_type'])




def replace_with(df, col_name: str, method: str, original='unknown', replacement_value=None):
    if method == 'mode':
        value = df[col_name].mode().iloc[0]  # Get the mode value
        if pd.isna(value):
            print(f"No mode for {col_name}")
            return
        replacement_value = value
    df[col_name] = df[col_name].replace(original, replacement_value)


replace_with(df, 'rate_type', 'mode')




df = one_hot_encode_column(df, 'rate_type')



def categorize_trips(row:pd.Series):
    """
    here we categorize the trips into 3 categories based on the rate type
    """
    if row['rate_type'] in ['JFK', 'Newark']:
        return 'airport_trips'
    elif row['rate_type'] in ['Negotiated fare', 'Nassau or Westchester']:
        return 'outside_NYC_trips'
    elif row['rate_type'] in ['Group ride', 'Standard rate']:
        return 'inside_NYC_trips'
    else:
        return 'other'


df['trip_category'] = df.apply(categorize_trips, axis=1)
feature_engineer(
    'trip_category', 'trip category is categorized into 3 columns', df['trip_category'].dtype)





df = one_hot_encode_column(df, 'trip_category')




replace_with(df, 'trip_type', 'mode')




df = one_hot_encode_column(df, 'trip_type')




# Create a correlation matrix
TOTAL_AMOUNT = ['improvement_surcharge', 'tolls_amount',
                'tip_amount', 'mta_tax', 'fare_amount', 'extra']





def absolute_negatives(df: pd.DataFrame, numerical_cols: List[str]):
    """
    Convert negative values to absolute values for numerical columns
    """
    for col in numerical_cols:
        if (df[col].dtype == 'float64'):
            df[col] = df[col].abs()


absolute_negatives(df, [col for col in TOTAL_AMOUNT if col not in [
                   'fare_amount', 'tolls_amount']])




def fix_toll_and_fare_amount(df: pd.DataFrame) -> pd.DataFrame:
    # Condition to check if the payment type is not 'Dispute' and not 'No charge'
    not_no_charge_or_dispute = ~((df['payment_type'] == 'Dispute') | (df['payment_type'] == 'No charge'))
    
    # Make 'Toll Amount' and 'Fare Amount' positive for rows where the condition is met
    df.loc[not_no_charge_or_dispute, 'tolls_amount'] = df['tolls_amount'].abs()
    df.loc[not_no_charge_or_dispute, 'fare_amount'] = df['fare_amount'].abs()
    
    return df

df = fix_toll_and_fare_amount(df)




def fix_fare_amount(row: pd.Series):
    """
    we are accounting for the initial charge of 3 dollars if the trip started if it wasn't accounted for
    """
    fare = row['fare_amount']
    trip_time_sec = row['total_trip_time_sec']

    if 0 < fare < 3 and trip_time_sec != 0:
        # If fare is between 0 and 3 and trip time is not zero, increase fare by 3
        return fare + 3
    elif 0 > fare > -3 and trip_time_sec != 0:
        # If fare is between -3 and 0 and trip time is not zero, decrease fare by 3
        return fare - 3
    else:
        # Otherwise, keep the fare unchanged
        return fare

# Apply the enhanced function to the 'fare_amount' column
df['fare_amount'] = df.apply(fix_fare_amount, axis=1)




def detect_and_replace_fare_outliers(df:pd.DataFrame):
    """
    Detect and replace fare outliers using a linear regression model based on
    trip_distance, total_trip_time_sec, and if dispute or no charge to account for negatives because they are highly correlated with fare_amount in logical sense
    """
    
    not_no_charge_or_dispute = ~((df['payment_type'] == 'Dispute') | (df['payment_type'] == 'No charge'))
    not_no_charge_or_dispute = not_no_charge_or_dispute.astype(int)
    df['not_no_charge_or_dispute'] = not_no_charge_or_dispute
    
    # Step 1: Detect outliers in the 'fare_amount' column
    Q1 = df['fare_amount'].quantile(0.25)
    Q3 = df['fare_amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1 * IQR

    
    # Identify and flag outliers
    is_outlier = (df['fare_amount'] < lower_bound) | (
        df['fare_amount'] > upper_bound)

    
    # Step 2: Split the data into non-outlier and outlier subsets
    non_outlier_data = df[~is_outlier]
    outlier_data = df[is_outlier]
    #encoded_payment_type_columns = [col for col in df.columns if col.startswith('encoded_payment_type')]

    x = ['trip_distance', 'total_trip_time_sec','not_no_charge_or_dispute']

    # Step 3: Train a linear regression model on the non-outlier data
    model = LinearRegression()
    model.fit(non_outlier_data[x], non_outlier_data['fare_amount'])

    # Step 4: Predict fare_amount for the outlier data
    predicted_fares = model.predict(outlier_data[x])

    # Step 5: Replace outlier values with predicted values
    df.loc[is_outlier, 'fare_amount'] = predicted_fares
    
    df.drop(columns=['not_no_charge_or_dispute'], inplace=True)

    return df


# Call the function to detect and replace fare outliers
df = detect_and_replace_fare_outliers(df)


def handle_extra_fees(x: pd.Series):

    if (not pd.isna(x['extra']) and x['extra'] in [0, 0.5, 1]):
        return x['extra']
    hour = x['hour']
    # we have to check if the hour is rush hour or not by using the is_rush_hour column that we feature engineered before
    isRushHour = 'is_rush_hour' in x and x['is_rush_hour'] == 1
    # rush hour
    if (isRushHour):
        return 1
    # begining and end of day
    elif ((hour >= 20 and hour <= 23) or (hour >= 0 and hour <= 5)):
        return 0.5
    return 0





def fix_extra_fees(df: pd.DataFrame):
    """
    Fix extra fees by replacing them with the correct extra fee based on if it is rush hour or not or overnight
    """
    df['hour'] = df['lpep_pickup_datetime'].dt.hour
    df['extra'] = df.apply(handle_extra_fees, axis=1)
    df.drop(columns=['hour'], inplace=True)


fix_extra_fees(df)


# - as we can see the extra column values are fixed according to the hour of the day and the values seems more reasonable according the data description

# ### mta_tax
# 
# - 50 cents MTA State Surcharge. This surcharge is required for all trips that end in New York City or Nassau, Suffolk, Westchester, Rockland, Dutchess, Orange or Putnam Counties. This surcharge is not required for trips that end in Newark, or in any other New Jersey municipality.
# 

# In[170]:


identify_columns_needing_imputation(df, ['mta_tax'])



def fix_mta_tax(row):
    """
    Fix mta tax by replacing them with the correct mta tax
    """
    if pd.isna(row['mta_tax']) or row['mta_tax'] not in [0, 0.5]:
        return 0.5
    return row['mta_tax']


df['mta_tax'] = df.apply(fix_mta_tax, axis=1)



replace_outliers(df, 'tolls_amount', -151, 151)




# here we are fixing the tolls amount by replacing the outliers with the mean of the tolls amount according to the grouping
df = detect_and_fix_outliers_by_group(
    df, 'tolls_amount', ['pu_location', 'do_location'])


def fix_improvement_surcharge(df):
    if df['trip_type'] == 'Street-hail':
        return 0.3
    elif df['trip_type'] == 'Dispatch':
        return 0
    else:
        return df['improvement_surcharge']




def fix_improvement_surcharge(row: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the improvement surcharge for trips based on the conditions provided.
    """
    # Filter trips that began in or after 2015
    has_surcharge = (df['lpep_pickup_datetime'].dt.year >=
                     2015) & (df['trip_type'] == 'Street-hail')

    # Set the improvement surcharge to 0.30 dollars for trips with the surcharge
    df.loc[has_surcharge, 'improvement_surcharge'] = 0.3

    dont_has_surcharge = (df['lpep_pickup_datetime'].dt.year < 2015) | (
        df['trip_type'] == 'Dispatch')

    # Set the improvement surcharge to 0.3 for 'Street-hail' trips and 0 for 'Dispatch' trips
    df.loc[dont_has_surcharge, 'improvement_surcharge'] = 0

    return df


df = fix_improvement_surcharge(df)



def clip_tip_amount(row: pd.Series):
    """
    Clip tip_amount 
    """
    if pd.isna(row['tip_amount']) or row['tip_amount'] < 0:
        return 0
    if (row['fare_amount'] > 0):
        return min(row['tip_amount'], row['fare_amount'])
    return row['tip_amount']


# In[187]:


df['tip_amount'] = df.apply(clip_tip_amount, axis=1)




# from nice place to nice place
df = detect_and_fix_outliers_by_group(df, 'tip_amount', ['pu_location', 'do_location'])





def sum_money_attr(df: pd.DataFrame):
    """
       Sum all money attributes into one column
    """
    df['total_amount'] = (df['fare_amount'] + df['extra'] + df['mta_tax'] +
                          df['tip_amount'] + df['tolls_amount'] + df['improvement_surcharge'])


# In[195]:


sum_money_attr(df)





# we will bin the total amount into 5 categories
create_bins(df, 'total_amount', 'total_amount_bins', [
            'Very Low', 'Low', 'Medium', 'High', 'Very High'])





label_encode_column(df, 'total_amount_bins',map={
    'Very Low': 0,
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
})





identify_columns_needing_imputation(df, ['payment_type'])



def impute_payment_type_neg_fare_amount(df):
    # Step 1: impute payment_type with -ve fare amounts because most probably they are no charge trips
    neg_mode = df[df['fare_amount'] <
                  0]['payment_type'].value_counts().index[0]
    df.loc[df['fare_amount'] < 0, 'payment_type'] = df[df['fare_amount']
                                                       < 0]['payment_type'].fillna(neg_mode)


impute_payment_type_neg_fare_amount(df)


# In[213]:


identify_columns_needing_imputation(df, ['payment_type'])


impute_multi(df, 'payment_type', 'vendor', 'mode')




identify_columns_needing_imputation(df, ['payment_type'])



df = one_hot_encode_column(df, 'payment_type')






def scale_column(df:pd.DataFrame, column_name:str, scaling_technique='min-max'):
    """
    Scale a specific column in a DataFrame using the specified scaling technique.
    """
    if scaling_technique not in ['min-max', 'standard', 'robust']:
        raise ValueError("Invalid scaling technique. Use 'min-max', 'standard', or 'robust'.")

    if scaling_technique == 'min-max':
        scaler = MinMaxScaler()
    elif scaling_technique == 'standard':
        scaler = StandardScaler()
    elif scaling_technique == 'robust':
        scaler = RobustScaler()

    scaled_data = scaler.fit_transform(df[[column_name]])
    scaled_column = pd.DataFrame(scaled_data, columns=[column_name])

    return scaled_column




import pandas as pd
from sklearn.utils import resample

def balance_classes(df, class_column, method='oversample', random_state=None):
    """
    Balance classes in a DataFrame using different methods.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        class_column (str): The name of the class column.
        method (str): The balancing method. Options: 'oversample', 'undersample'.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame with balanced classes.
    """
    # Check if the specified method is valid
    if method not in ['oversample', 'undersample']:
        raise ValueError("Invalid balancing method. Use 'oversample' or 'undersample'.")

    # Count the number of samples in each class
    class_counts = df[class_column].value_counts()

    # Find the majority and minority class
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    if method == 'oversample':
        # Oversample the minority class to match the majority class
        majority_count = class_counts[majority_class]
        minority_count = class_counts[minority_class]

        minority_df = df[df[class_column] == minority_class]
        oversampled_minority = resample(minority_df, replace=True, n_samples=majority_count, random_state=random_state)

        balanced_df = pd.concat([df[df[class_column] == majority_class], oversampled_minority])
        
    elif method == 'undersample':
        # Undersample the majority class to match the minority class
        majority_count = class_counts[majority_class]
        minority_count = class_counts[minority_class]

        majority_df = df[df[class_column] == majority_class]
        undersampled_majority = resample(majority_df, replace=False, n_samples=minority_count, random_state=random_state)

        balanced_df = pd.concat([df[df[class_column] == minority_class], undersampled_majority])

    return balanced_df



identify_columns_needing_imputation(df, df.columns)


def dropping_empty_rows(df: pd.DataFrame):
    """
    Drop rows that have all the columns missing
    """
    # drop row if the all the columns are missing None or NaN or unknown
    df[['do_location','pu_location']] = df[['do_location','pu_location']].replace('unknown', np.nan)
    df.dropna(how='all', inplace=True)
    df[['do_location','pu_location']] = df[['do_location','pu_location']].fillna('unknown')

dropping_empty_rows(df)    


def impute_arbiterary(df: pd.DataFrame, column_name: str, value: str):
    """
    Impute the specified column with the specified value.
    """
        # Check if NaN values existed before
    nan_count = df[column_name].isna().sum()
    if nan_count == 0:
        print(f"No NaN values in {column_name}")
        return
    
    df[column_name].fillna(value, inplace=True)
    insert_to_lookup_table(column_name, 'nan', value,None, f'imputed nan with {value}')


# In[231]:


impute_arbiterary(df,'trip_distance_haversine',-1)
# range of values -90 to 90 for latitude and -180 to 180 for longitude according to google maps
impute_arbiterary(df,'do_location_latitude',-255)
impute_arbiterary(df,'do_location_longitude',-255)
impute_arbiterary(df,'pu_location_latitude',-255)
impute_arbiterary(df,'pu_location_longitude',-255)


save_lookup_table('lookup_table_green_taxis.csv')
save_encode_table()
save_impute_table()






def drop_categorical_columns(df):
    all_dfs = []

    for key in LOOKUP_TABLE:
        table = LOOKUP_TABLE[key]
        curr_df = pd.DataFrame(table)

        if 'imputed' in curr_df.columns and 'encoded' in curr_df.columns:
            if len(table) > 0:
                curr_df.drop_duplicates(inplace=True)
                curr_df['column name'] = key  # Updated to 'column name'
                all_dfs.append(curr_df)

    # Concatenate all dataframes together
    concatenated_df = pd.concat(all_dfs, ignore_index=True)

    # Filter the rows where 'encoded' column is not empty
    non_empty_encoded = concatenated_df[concatenated_df['encoded'].notna()]

    # Get the unique column names with non-empty 'encoded' values
    column_names_with_encoded = non_empty_encoded['column name'].unique()
    
    column_names = [col for col in column_names_with_encoded if col in df.columns] + LOCATION_COLUMNS

    # Drop the columns with non-empty 'encoded' values
    df = df.drop(columns=column_names)
    print(df.columns)

# we will remove the columns that we have encoded to reduce space
drop_categorical_columns(df)





year = 2016
month = 10
cleaned_dataset_file_name = f'green_trip_data_{year}-{month}clean'

save_dataframe(df, os.path.join(
    DATASET_DIR, cleaned_dataset_file_name), file_format='csv')
file_path = save_dataframe(df, os.path.join(
    DATASET_DIR, cleaned_dataset_file_name), file_format='parquet')





