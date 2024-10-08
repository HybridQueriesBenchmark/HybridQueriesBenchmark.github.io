import pandas as pd
import numpy as np
from tqdm import tqdm

LOWER_BOUND = 100
TOTAL = 121057

df = pd.read_csv('./metadata/fungi_train_metadata.csv')


def generate_random_value(column_info):
    column_type = column_info[0]
    generate_way = column_info[2]
    if column_type == 'int':
        if generate_way == 'distribution':
            return np.random.randint(*column_info[3])
        elif generate_way == 'choice':
            return np.random.choice(column_info[3])
    elif column_type == 'float':
        if generate_way == 'distribution':
            return np.random.uniform(*column_info[3])
        elif generate_way == 'choice':
            return np.random.choice(column_info[3])
    elif column_type == 'bool':
        assert generate_way == 'choice'
        return np.random.choice(column_info[3])
    elif column_type == 'str':
        assert generate_way == 'choice'
        return f"'{np.random.choice(column_info[3])}'"


def generate_single_column_bool_expression(column, column_info):
    operator = np.random.choice(column_info[1])
    value = generate_random_value(column_info)
    return f"{column} {operator} {value}"


def generate_bool_expression(columns, column2infos, num_expressions=5):
    expressions = []

    for _ in range(num_expressions):
        column = np.random.choice(columns)
        column_info = column2infos[column]
        expression = generate_single_column_bool_expression(
            column, column_info)
        expressions.append(expression)


    for i in range(num_expressions - 1):
        operator = np.random.choice(
            ['and', 'and', 'and', 'or'])
        expressions.insert(i * 2 + 1, operator)

    bool_expression = ' '.join(expressions)
    return bool_expression


def get_column_generate_params(df, column, generate_way):
    if generate_way == 'distribution':
        min_value = df[column].min()
        max_value = df[column].max()
        return [min_value, max_value]
    elif generate_way == 'choice':
        lst = df[column].unique().tolist()
        return lst



columns = ['id', 'year', 'month', 'day', 'countryCode', 'scientificName', 'Substrate',
           'Latitude', 'Longitude', 'Habitat', 'poisonous']

choice = 'choice'
distribution = 'distribution'
column2infos = {
    'id': ['int', ['>', '>=',  '<', '<=', '!='], distribution, get_column_generate_params(df, 'id', distribution)],
    'year': ['int', ['>', '>=',  '<', '<=', '!=', '=='], distribution, get_column_generate_params(df, 'year', distribution)],
    'month': ['int', ['>', '>=',  '<', '<=', '!=', '=='], distribution, get_column_generate_params(df, 'month', distribution)],
    'day': ['int', ['>', '>=',  '<', '<=', '!=', '=='], distribution, get_column_generate_params(df, 'day', distribution)],
    'countryCode': ['str', ['==', '!='], choice, get_column_generate_params(df, 'countryCode', choice)],
    'scientificName': ['str', ['==', '!='], choice, get_column_generate_params(df, 'scientificName', choice)],
    'Substrate': ['str', ['==', '!='], choice, get_column_generate_params(df, 'Substrate', choice)],
    'Latitude': ['float', ['>', '>=',  '<', '<='], distribution, get_column_generate_params(df, 'Latitude', distribution)],
    'Longitude': ['float', ['>', '>=',  '<', '<='], distribution, get_column_generate_params(df, 'Longitude', distribution)],
    'Habitat': ['str', ['==', '!='], choice, get_column_generate_params(df, 'Habitat', choice)],
    'poisonous': ['bool', ['=='], choice, [0, 1]]
}

answer = []

for i in tqdm(range(TOTAL)):
    count_satisfy = 0
    expression_num = 0
    while count_satisfy < LOWER_BOUND:
        expression_num = np.random.randint(1, 6)
        bool_expression = generate_bool_expression(
            columns, column2infos, expression_num)
        count_satisfy = df.query(bool_expression).shape[0]
    answer.append([i, count_satisfy, expression_num, bool_expression])

answer = pd.DataFrame(
    answer, columns=['id', 'count_satisfy', 'expression_num', 'bool_expression'])
answer.to_csv('./metadata/fungi_bool_filter.csv', index=False)
