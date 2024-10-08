from pymilvus import MilvusClient
import time
import pandas as pd
import numpy as np
import json
import os
import threading


data_root = "/home/ubuntu/code/hybrid_query_experiments/datasets/single_table_fungi_dataset"


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


def format_boolean_filter(boolean_filter):
    lst = boolean_filter.split()
    for i in range(len(lst)):
        if lst[i] == 'poisonous':
            lst[i + 2] = 'True' if lst[i + 2] == '1' else 'False'   # Change '1' to 'True' and '0' to 'False'
    res = ' '.join(lst)
    # print('Formatted boolean filter:', res)
    return res


df, vectors, answer_df = getSTFDTest()

vectors = [vector.tolist() for vector in vectors]
df['boolean_filter'] = df['bool_expression'].map(lambda x: format_boolean_filter(x))


class ConcurrentCounter:
    def __init__(self, initial_value=0):
        self._lock = threading.Lock()
        self._value = initial_value

    def increment_and_get_old(self):
        with self._lock:
            old_value = self._value
            self._value += 1
            return old_value

    def get_value(self):
        with self._lock:
            return self._value

counter = ConcurrentCounter()


class QueryThread(threading.Thread):
    def __init__(self, thread_idx, total_queries):
        threading.Thread.__init__(self)
        self.client = MilvusClient(uri="http://localhost:19530")
        self.client.load_collection("fungis")
        self.thread_idx = thread_idx
        self.total_queries = total_queries
        self.run_time = 0
        self.query_result_lst = []

    def run(self):
        while True:
            idx = counter.increment_and_get_old()
            if idx >= self.total_queries:
                break
            try:
                query_vector = [vectors[idx]]
                boolean_filter = df.iloc[idx]['boolean_filter']
                start_time = time.perf_counter()
                query_result = self.client.search(
                    collection_name="fungis",
                    data=query_vector,
                    search_params={"metric_type": "L2", "params": {"ef": 100}},
                    limit=100,
                    output_fields=["id"],
                    filter=boolean_filter
                )
                self.run_time += (time.perf_counter() - start_time) * 1000
                self.query_result_lst.append((idx, query_result))
            except Exception as e:
                print(idx)
                print("Error running query:", e)

        self.client.close()


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
        query_result = query_result[0]
        query_result = [x['id'] for x in query_result]  # list
        recall += query_recall(query_result, answer)
    return recall / len(query_result_lst)


if __name__ == '__main__':
    total_queries = len(df)
    num_threads = 1

    print(f'Total queries: {total_queries}, num_threads: {num_threads}')

    try:
        threads = []

        for thread_idx in range(num_threads):
            thread = QueryThread(thread_idx, total_queries)
            threads.append(thread)
        
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

    except Exception as e:
        print("Error:", e)
