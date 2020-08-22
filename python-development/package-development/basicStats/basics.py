import numpy as np
import pandas as pd
#from scipy.stats import *


def get_unique_values(column):
    return np.unique(column[~np.isnan(column)])


def unique_values_count(column):
    """calculate number of non-null unique values"""
    return len(get_unique_values(column))


def binary_column_recognition(column: 'np.array') -> 'boolean value':
    """Recognise columns with only two unique values for later ETL"""
    return unique_values_count(column) == 2


def categorical_column_recognition(column):
    """Recognise existing or potential categorical variables"""
    condition1 = 2 < unique_values_count(column) <= 30
    condition2 = isinstance(get_unique_values(column)[0], (np.str, np.integer))
    return condition1 and condition2


def continuous_variable_recognition(column):
    column = get_unique_values(column) # drop nulls
    first_value = column[0]
    return unique_values_count(column) > 2 and (isinstance(first_value, (np.integer, np.float)))


def column_with_missing_value(column: 'np.array') -> 'boolean value':
    """Recognise columns with missing values
    only covers apparent cases such as
    whitespaces, nan, null, na, none, question-mark, empty parenthesis"""
    common_missing_value_format = [np.nan, ' ', '""', "''", '()', '[]', '{}', '?', '*', '.']
    for value in column:
        if value in common_missing_value_format:
            return True
        else:
            return False


class raw_data:

    def __init__(self, raw_table):
        self.raw_data = raw_table

    def get_row_number(self):
        if self.raw_data.shape[0] > 0:
            return self.raw_data.shape[0]
        else:
            print('Hey, looks like you have not read the file properly.\nTry again!')

    def get_column_number(self):
        if self.raw_data.shape[1] > 0:
            return self.raw_data.shape[1]
        else:
            print('Hey, looks like you have not read the file properly.\nTry again!')

    def sample_n_rows(self, n):
        if n < 1:
            print('Error: please enter integer >= 1')
        else:
            return self.raw_data.sample(round(n))

    def get_values(self):
        if self.get_row_number() > 0:
            return self.raw_data
        else:
            print('Hey, looks like you have not read the file properly.\nTry again!')

    def get_columns(self):
        if len(list(self.raw_data.columns.values)) > 0:
            return list(self.raw_data.columns.values)
        else:
            print('Hey, looks like you have not read the file properly.\nTry again!')

    def get_binary_column_list(self):
        binary_columns = self.raw_data.columns[self.raw_data.apply(binary_column_recognition, axis=0)]
        if len(binary_columns) > 0:
            return list(binary_columns.values)
        else:
            print('Looks like there is no binary column')

    def get_non_binary_column_list(self):
        non_binary_columns = [x for x in self.get_columns() if x not in self.get_binary_column_list()]
        if len(non_binary_columns) > 0:
            return non_binary_columns
        else:
            print('Looks like there is no non-binary column')

    def get_numeric_column_list(self):
        numeric_columns = self.raw_data.columns[self.raw_data.apply(continuous_variable_recognition, axis=0)]
        if len(numeric_columns) > 0:
            return numeric_columns
        else:
            print('Looks like there is no numerical column')

    def get_categorical_column_list(self):
        cate_columns = self.raw_data.columns[self.raw_data.apply(categorical_column_recognition, axis=0)]
        if len(cate_columns) > 0:
            return cate_columns
        else:
            print('Looks like there is no categorical column')

    def get_columns_with_apparent_missing_values(self):
        missing_value_columns = self.raw_data.columns[self.raw_data.apply(column_with_missing_value, axis=0)]
        if len(missing_value_columns) > 0:
            return list(missing_value_columns.values)
        else:
            print('Congratulations! There is no columns with obvious missing values.')