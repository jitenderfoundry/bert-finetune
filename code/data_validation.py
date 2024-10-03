import pandas as pd
import ast

def validate_classification_data(df):
    # Check if the required columns are present
    required_columns = ['query', 'categories']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"DataFrame must contain the columns: {required_columns}")
    
    # Check if the columns are of string type
    if not all(df[column].apply(lambda x: isinstance(x, str)).all() for column in required_columns):
        raise ValueError(f"Both 'query' and 'categories' columns must contain string data.")
    
    # Check if the 'categories' column contains only tuples
    if not df['categories'].apply(is_string_of_tuple).all():
        raise ValueError("The 'categories' column must contain only strings representing tuples.")
    
    print("DataFrame is valid.")

def is_string_of_tuple(s):
        try:
            # Attempt to convert the string to a tuple
            result = ast.literal_eval(s)
            # Check if the result is a tuple
            return isinstance(result, tuple)
        except (ValueError, SyntaxError):
            return False
