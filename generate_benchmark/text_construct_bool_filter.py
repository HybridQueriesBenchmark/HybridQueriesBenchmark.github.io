import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import os
import torch
import mysql.connector
from tqdm import tqdm
import threading
import time


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)

LOWER_BOUND = 100
TOTAL = 20000

tables = [
    'movies',
    'genres',
    'movies_genres',
    'speakers',
    'conversations',
    'utterances'
]

table2columns = {
    tables[0]: ['movie_id', 'movie_name', 'release_year', 'rating', 'votes'],
    tables[1]: ['genre_id', 'genre'],
    tables[2]: ['movie_id', 'genre_id'],
    tables[3]: ['speaker_id', 'character_name', 'movie_id', 'gender', 'credit_pos'],
    tables[4]: ['conversation_id', 'movie_id'],
    tables[5]: ['utterance_id', 'speaker_id', 'conversation_id', 'reply_to']
}


allow_join = {
    tables[5]: [tables[4], tables[3]],
    tables[4]: [tables[0]],
    tables[3]: [tables[0]],
    tables[2]: [tables[0], tables[1]],
    tables[1]: [tables[2]],
    tables[0]: [tables[2], tables[3], tables[4]]
}


join_conditions = {
    (tables[5], tables[4]): 'utterances.conversation_id = conversations.conversation_id',
    (tables[5], tables[3]): 'utterances.speaker_id = speakers.speaker_id',
    (tables[4], tables[0]): 'conversations.movie_id = movies.movie_id',
    (tables[3], tables[0]): 'speakers.movie_id = movies.movie_id',
    (tables[2], tables[0]): 'movies_genres.movie_id = movies.movie_id',
    (tables[2], tables[1]): 'movies_genres.genre_id = genres.genre_id'
}

choice = 'choice'
distribution = 'distribution'



def get_table_column_values(table, column, generate_way):
    df = pd.read_csv(f'./metadata/movie/{table}.csv')
    column_values = df[column].values
    if generate_way == distribution:
        return [column_values.min(), column_values.max()]
    elif generate_way == choice:
        return list(set(column_values))
    else:
        raise ValueError(f'Unknown generate_way: {generate_way}')


tablecolumn2operators = {
    tables[0]: {
        table2columns[tables[0]][0]: [['=', '>', '<', '>=', '<=', '!='], distribution, get_table_column_values(tables[0], table2columns[tables[0]][0], distribution)],
        table2columns[tables[0]][1]: [['=', '!='], choice, get_table_column_values(tables[0], table2columns[tables[0]][1], choice)],
        table2columns[tables[0]][2]: [['=', '>', '<', '>=', '<=', '!='], choice, get_table_column_values(tables[0], table2columns[tables[0]][2], choice)],
        table2columns[tables[0]][3]: [['>', '<', '>=', '<='], distribution, get_table_column_values(tables[0], table2columns[tables[0]][3], distribution)],
        table2columns[tables[0]][4]: [['>', '<', '>=', '<='], distribution, get_table_column_values(
            tables[0], table2columns[tables[0]][4], distribution)]
    },
    tables[1]: {
        table2columns[tables[1]][0]: [['=', '>', '<', '>=', '<=', '!='], distribution, get_table_column_values(tables[1], table2columns[tables[1]][0], distribution)],
        table2columns[tables[1]][1]: [['=', '!='], choice, get_table_column_values(
            tables[1], table2columns[tables[1]][1], choice)]
    },
    tables[2]: {
        table2columns[tables[2]][0]: [['=', '!='], choice, get_table_column_values(tables[2], table2columns[tables[2]][0], choice)],
        table2columns[tables[2]][1]: [['=', '!='], choice, get_table_column_values(
            tables[2], table2columns[tables[2]][1], choice)]
    },
    tables[3]: {
        table2columns[tables[3]][0]: [['>', '<', '>=', '<=', '!='], choice, get_table_column_values(tables[3], table2columns[tables[3]][0], choice)],
        table2columns[tables[3]][1]: [['=', '!='], choice, get_table_column_values(tables[3], table2columns[tables[3]][1], choice)],
        table2columns[tables[3]][2]: [['=', '!='], choice, get_table_column_values(tables[3], table2columns[tables[3]][2], choice)],
        table2columns[tables[3]][3]: [['=', '!='], choice, get_table_column_values(tables[3], table2columns[tables[3]][3], choice)],
        table2columns[tables[3]][4]: [['=', '>', '<', '>=', '<=', '!='], distribution,
                                      get_table_column_values(tables[3], table2columns[tables[3]][4], distribution)]
    },
    tables[4]: {
        table2columns[tables[4]][0]: [['>', '<', '>=', '<=', '!='], distribution, get_table_column_values(tables[4], table2columns[tables[4]][0], distribution)],
        table2columns[tables[4]][1]: [['=', '!='], choice, get_table_column_values(
            tables[4], table2columns[tables[4]][1], choice)]
    },
    tables[5]: {
        table2columns[tables[5]][0]: [['>', '<', '>=', '<=', '!='], choice, get_table_column_values(tables[5], table2columns[tables[5]][0], choice)],
        table2columns[tables[5]][1]: [['>', '<', '>=', '<=', '!='], choice, get_table_column_values(tables[3], table2columns[tables[3]][0], choice)],
        table2columns[tables[5]][2]: [['=', '!='], choice, get_table_column_values(tables[5], table2columns[tables[5]][2], choice)],
        table2columns[tables[5]][3]: [['=', '!='], choice, [-1]]
    }
}


def generate_single_column_bool_expression(table, column, column_info):
    operators, generate_way, values = column_info
    operator = random.choice(operators)
    if generate_way == distribution:
        value = random.randint(int(values[0]), int(values[1]))
    elif generate_way == choice:
        value = random.choice(values)
    if isinstance(value, str):
        value = f"'{value}'"
    condition = f"{table}.{column} {operator} {value}"
    return condition


'''
    SELECT COUNT(utterances.utterance_id)
    FROM utterances
    [ JOIN [table_name] [ ON [join_condition] ] ]
    ...
    WHERE [bool_expression];
'''


def generate_random_sql():
    query_table = 'utterances'

    sql_query = f"SELECT DISTINCT utterances.utterance_id FROM utterances "


    join_clause = ""
    join_tables = set([tables[5]])
    while random.random() < 0.7:
        join_table = random.choice(list(allow_join[query_table]))
        selected_table = join_table
        if join_table in join_tables:
            continue
        join_tables.add(join_table)
        join_condition = join_conditions.get((query_table, join_table), '')
        if not join_condition:
            join_condition = join_conditions.get((join_table, query_table))
            if not join_condition:
                continue

            query_table, join_table = join_table, query_table
        join_condition = join_conditions[(query_table, join_table)]
        join_clause += f" JOIN {selected_table} ON {join_condition}"
        query_table = selected_table


    where_clause = "WHERE "
    conditions = []
    bool_expression_num = random.randint(1, 6)
    join_tables = list(join_tables)
    for _ in range(bool_expression_num):
        table = random.choice(join_tables)
        column = random.choice(table2columns[table])
        column_info = tablecolumn2operators[table][column]
        condition = generate_single_column_bool_expression(
            table, column, column_info)
        conditions.append(condition)


    for i in range(bool_expression_num - 1):
        and_or = random.choice([' AND ', ' OR '])
        conditions.insert(i * 2 + 1, and_or)
    where_clause += ' '.join(conditions)


    select_clause = sql_query
    sql_query += join_clause + " " + where_clause + ";"

    return sql_query, select_clause, join_clause, where_clause



def execute_sql_query(sql_query, result_list):
    try:
        cursor.execute(sql_query)
        result = cursor.fetchall()
        result_list.append(result)
    except Exception as e:
        print(f"Error occurred while executing SQL query: {e}")


db_config = {
    'host': 'localhost',
    'user': 'arcsin2',
    'password': 'arcsin2',
    'database': 'movie_corpus'
}


conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()


with open('movie_bool_filter.csv', 'w') as f:
    f.write("id,count_satisfy,select_clause,join_clause,where_clause\n")
    for i in tqdm(range(TOTAL)):
        while True:
            random_sql, select_clause, join_clause, where_clause = generate_random_sql()
            result_list = []
            thread = threading.Thread(
                target=execute_sql_query, args=(random_sql, result_list))
            thread.start()

            start_time = time.time()
            while thread.is_alive():
                if time.time() - start_time > 15:
                    print("Timeout occurred. Terminating current SQL query...")
                    thread.join()
                    break
                time.sleep(1.0)

            if result_list:
                utterance_id_lst = result_list[0]
                number_of_rows = len(utterance_id_lst)
                if number_of_rows >= LOWER_BOUND:
                    id_lst = np.array(utterance_id_lst).flatten()
                    np.save(f'./metadata/movie/return_utterance_id/{i}.npy', id_lst)
                    f.write(
                        f"{i},{number_of_rows},{select_clause},{join_clause},{where_clause}\n")
                    break


cursor.close()
conn.close()
