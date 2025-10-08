import pandas as pd
import pprint

data = pd.read_csv('data/raw/users.csv')

number_of_rows = data.shape[0]
number_of_columns = data.shape[1]
data_types = data.dtypes
missing_values = data.isnull().sum()
descriptive_stats = data.describe()
basic_structure = {
    'number_of_rows': number_of_rows,
    'number_of_columns': number_of_columns,
    'data_types': data_types.to_dict(),
    'missing_values': missing_values.to_dict(),
    'descriptive_stats': descriptive_stats.to_dict()
}
pprint.pprint(basic_structure)