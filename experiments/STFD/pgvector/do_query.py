import numpy as np
import json
from sqlalchemy import create_engine
from tqdm import tqdm
import pandas as pd
import os
import threading
import time
import psycopg2
import json


data_root = r"/home/ubuntu/code/hybrid_query_experiments/datasets/single_table_fungi_dataset"


def getSTFDTest():
    """
    Get the STFD dataset
    """
    bool_filter_path = os.path.join(data_root, "bool_filter", "test_filter.csv")
    vectors_path = os.path.join(data_root, "vectors", "test_vectors.npy")
    answer_path = os.path.join(data_root, "answers", "test_knn.csv")
    df = pd.read_csv(bool_filter_path)
    vectors = np.load(vectors_path)
    answer_df = pd.read_csv(answer_path)
    return df, vectors, answer_df


df, vectors, answer_df = getSTFDTest()

def generate_sql(df, vectors, index):
    sql = "SET hnsw.ef_search = 100; SELECT id FROM fungis WHERE "
    bool_filter = format_where_clause(df.iloc[index]['bool_expression'])
    sql += bool_filter
    vector_json = json.dumps(vectors[index].tolist())
    sql += " ORDER BY embedding <-> '" + vector_json + "' LIMIT 100;"
    print(sql)
    return sql


columns = pd.read_csv(os.path.join(data_root, "metadata", "base_metadata.csv")).columns


def format_where_clause(condition_string):
    lst = condition_string.split()
    for i in range(len(lst)):
        if lst[i] == 'poisonous':
            lst[i + 2] = 'True' if lst[i + 2] == '1' else 'False'   # Change '1' to 'True' and '0' to 'False'
        if lst[i] in columns:
            lst[i] = '"' + lst[i] + '"'  # Add quotes to column names
        elif lst[i] == '==':
            lst[i] = '='        # Change '==' to '='
    res = ' '.join(lst)
    return res


class ConcurrentCounter:
    def __init__(self, initial_value=0):
        self._lock = threading.Lock()
        self._value = initial_value

    def increment_and_get_old(self):
        with self._lock:
            old_value = self._value
            self._value += 1
            if old_value % 1000 == 0:
                print(old_value)
            return old_value

    def get_value(self):
        with self._lock:
            return self._value

counter = ConcurrentCounter()

class QueryThread(threading.Thread):
    def __init__(self, connection, thread_idx, total_queries):
        threading.Thread.__init__(self)
        self.connection = connection
        self.thread_idx = thread_idx
        self.total_queries = total_queries
        self.run_time = 0
        self.query_result_lst = []

    def run(self):
        cursor = self.connection.cursor()

        while True:
            idx = counter.increment_and_get_old()
            if idx >= self.total_queries:
                break
            query = generate_sql(df, vectors, idx)
            try:
                start_time = time.perf_counter()
                cursor.execute(query)
                self.run_time += (time.perf_counter() - start_time) * 1000
                query_result = cursor.fetchall()
                self.query_result_lst.append((idx, query_result))
            except Exception as e:
                print(query)
                print("Error running query:", e)

        cursor.close()
        self.connection.close()


def connect_to_postgresql():
    try:
        connection = psycopg2.connect(
            dbname="stfd",
            user="postgres",
            password="",
            host="localhost",
            port="5432"
        )
        return connection
    except Exception as e:
        print("Error connecting to PostgreSQL:", e)
        return None


def split_queries(total_queries, num_threads):
    indices = np.arange(total_queries) % num_threads
    result = [np.where(indices == i)[0] for i in range(num_threads)]
    return result


def get_query_result(threads : QueryThread):
    query_result_lst = []
    for thread in threads:
        query_result_lst += thread.query_result_lst
    query_result_lst.sort(key=lambda x: x[0])
    return query_result_lst


def query_recall(query_result_lst, answer):
    query_result_set = set(query_result_lst)
    answer_set = set(answer)
    return len(query_result_set & answer_set) / len(answer_set)


def calculate_recall(query_result_lst, df):
    recall = 0.0
    print(len(query_result_lst), len(df))
    for idx, query_result in query_result_lst:
        answer = df.iloc[idx]['L2_nearest_ids'] # str
        answer = json.loads(answer)     # list
        query_result = [x[0] for x in query_result]  # list
        recall += query_recall(query_result, answer)
    return recall / len(query_result_lst)


if __name__ == '__main__':
    total_queries = len(df)
    num_threads = 1

    print(f'Total queries: {total_queries}, num_threads: {num_threads}')

    # query_idx_lst_split = split_queries(total_queries, num_threads)

    connection = connect_to_postgresql()
    if connection:
        try:
            threads = []

            for thread_idx in range(num_threads):
                thread = QueryThread(connect_to_postgresql(), thread_idx, total_queries)
                threads.append(thread)

            total_start_time = time.perf_counter()
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            run_time_lst = [thread.run_time for thread in threads]
            print(run_time_lst)
            total_execution_time = max(run_time_lst)

            if total_execution_time == 0:
                qps = 0
            else:
                print("Total execution time (ms):", total_execution_time)
                qps = total_queries / total_execution_time * 1000
            print(f"QPS: {qps:.2f}")

            query_result_lst = get_query_result(threads)
            recall = calculate_recall(query_result_lst, answer_df)
            print(f"Recall: {recall:.5f}")

        finally:
            connection.close()

