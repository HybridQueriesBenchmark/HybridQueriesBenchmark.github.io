import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import random
import heapq
import json


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


LEARN_SIZE = 10000
TEST_SIZE = 10000


def compute_l2_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def simple_knn(query_vector, dataframe, embeddings_matrix, expr, k=100, statisfy=100):
    subset_indices = dataframe.query(expr).index
    assert statisfy == len(subset_indices)
    distances = []

    for idx in subset_indices:
        current_vec = embeddings_matrix[idx]
        sub_vec = query_vector - current_vec
        dis = np.sqrt(np.sum(sub_vec ** 2))
        if abs(compute_l2_distance(query_vector, current_vec) - dis) >= 1e-5:
            print(compute_l2_distance(query_vector, current_vec), dis)
        distances.append((idx, dis))

    distances = sorted(distances, key=lambda x: x[1])
    return [x[0] for x in distances[:k]], [x[1] for x in distances[:k]]



def knn(query_vector, subset_indices, embeddings_matrix, idx2dbid, k=100):
    nearest_heap = []


    for index in subset_indices:
        current_vector = embeddings_matrix[index]


        distance = compute_l2_distance(query_vector, current_vector)


        if len(nearest_heap) < k:
            heapq.heappush(nearest_heap, (-distance, index))
        else:
            max_distance = -nearest_heap[0][0]
            if distance < max_distance:
                heapq.heappop(nearest_heap)
                heapq.heappush(nearest_heap, (-distance, index))


    distance = [(-distance, idx) for distance, idx in nearest_heap]
    distance = sorted(distance, key=lambda x: x[0])
    nearest_indices = [idx for distance, idx in distance]
    nearest_ids = [idx2dbid[idx] for idx in nearest_indices]
    nearest_distances = [distance for distance, idx in distance]
    return nearest_ids, nearest_distances


def database_ididx(utterances_csv_path):
    dbid2idx = {}
    idx2dbid = {}

    df = pd.read_csv(utterances_csv_path)
    n = len(df)
    for i in range(n):
        dbid = df['utterance_id'][i]
        dbid2idx[dbid] = i
        idx2dbid[i] = dbid

    return dbid2idx, idx2dbid


if __name__ == '__main__':
    seed_everything(42)

    bool_df = pd.read_csv('./metadata/movie/movie_bool_filter.csv')

    assert len(bool_df) == LEARN_SIZE + TEST_SIZE

    features = np.load('./features/bert_movie-corpus_text_features_train.npy')

    learn_querys = np.load(
        './features/bert_movie-corpus_text_features_learn.npy')
    test_querys = np.load(
        './features/bert_movie-corpus_text_features_test.npy')

    print(features.shape, learn_querys.shape, test_querys.shape)

    selected_idx_root = r"E:\code\build_dataset\metadata\movie\return_utterance_id"

    dbid2idx, idx2dbid = database_ididx('./metadata/movie/utterances.csv')

    # learn querys KNN
    learn_knn = []
    for i in tqdm(range(LEARN_SIZE)):
        query_vector = learn_querys[i]
        selected_id_lst = np.load(os.path.join(selected_idx_root, f"{i}.npy"))
        selected_idx = [dbid2idx[x] for x in selected_id_lst]
        nearest_indices, nearest_distances = knn(
            query_vector, selected_idx, features, idx2dbid)
        learn_knn.append([i, nearest_distances[-1], nearest_indices])

    learn_knn_df = pd.DataFrame(
        learn_knn, columns=['id', 'max_distance', 'L2_nearest_ids'])
    learn_knn_df.to_csv('./metadata/movie_learn_knn.csv', index=False)

    # test querys KNN
    test_knn = []
    for i in tqdm(range(TEST_SIZE)):
        query_vector = test_querys[i]
        selected_id_lst = np.load(os.path.join(
            selected_idx_root, f"{LEARN_SIZE + i}.npy"))
        selected_idx = [dbid2idx[x] for x in selected_id_lst]
        nearest_indices, nearest_distances = knn(
            query_vector, selected_idx, features, idx2dbid)
        test_knn.append([i, nearest_distances[-1], nearest_indices])

    test_knn_df = pd.DataFrame(
        test_knn, columns=['id', 'max_distance', 'L2_nearest_ids'])
    test_knn_df.to_csv('./metadata/movie_test_knn.csv', index=False)
