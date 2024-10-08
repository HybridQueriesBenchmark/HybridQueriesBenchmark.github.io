import numpy as np
import json
import requests
import pandas as pd
import os
import threading
import time

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
dsl_df = pd.read_csv("output_with_dsl.csv")

def generate_es_query(df, vectors, index):
    query_vector = vectors[index].tolist()  # Convert to list for JSON serialization
    # bool_filter = format_where_clause(df.iloc[index]['bool_expression'])
    # print(bool_filter)

    try:
        resp = eval(dsl_df.iloc[index]['dsl_query'])
        bool_dsl = resp['query']
    except Exception as e:
        bool_dsl = {"bool": {}}
        # some sql failed to translated to ES dsl through ES _sql API
        # print(index, resp)

    # print(bool_dsl)

    query = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": 100,  # Set the number of nearest neighbors you want to retrieve
            "num_candidates": 100,
            "filter":  bool_dsl # Apply the boolean filter
        },
        "fields": ["id"],
        "_source": False,
        "size": 100
    }
    
    return query

columns = pd.read_csv(os.path.join(data_root, "metadata", "base_metadata.csv")).columns

def format_where_clause(condition_string):
    lst = condition_string.split()
    for i in range(len(lst)):
        if lst[i] == 'poisonous':
            lst[i + 2] = 'true' if lst[i + 2] == 'True' else 'false'
        if lst[i] in columns:
            lst[i] = f'"{lst[i]}"'
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
    def __init__(self, total_queries):
        threading.Thread.__init__(self)
        self.total_queries = total_queries
        self.run_time = 0
        self.query_result_lst = []

    def run(self):
        while True:
            idx = counter.increment_and_get_old()
            if idx >= self.total_queries:
                break
            query = generate_es_query(df, vectors, idx)
            try:
                start_time = time.perf_counter()
                response = requests.post("http://localhost:9200/fungis/_search", json=query)
                response.raise_for_status()  # Raise an error for bad responses
                self.run_time += (time.perf_counter() - start_time) * 1000
                # print(len(response.json()['hits']['hits']))
                self.query_result_lst.append((idx, response.json()['hits']['hits']))
            except Exception as e:
                print("query idx: ", idx)
                print("Error running query:", e)

def calculate_recall(query_result_lst, df):
    recall = 0.0
    for idx, query_result in query_result_lst:
        answer = df.iloc[idx]['L2_nearest_ids']  # str
        answer = json.loads(answer)  # list
        query_result = [int(hit['_id']) for hit in query_result]  # list of document IDs
        # print("after:", len(query_result))
        query_result_set = set(query_result)
        answer_set = set(answer)
        recall += len(query_result_set & answer_set) / len(answer_set)
    return recall / len(query_result_lst)

if __name__ == '__main__':
    total_queries = len(df)
    num_threads = 1

    print(f'Total queries: {total_queries}, num_threads: {num_threads}')

    try:
        threads = []

        for thread_idx in range(num_threads):
            thread = QueryThread(total_queries)
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

        query_result_lst = []
        for thread in threads:
            query_result_lst += thread.query_result_lst
        
        recall = calculate_recall(query_result_lst, answer_df)
        print(f"Recall: {recall:.5f}")

    except Exception as e:
        print("Error:", e)
