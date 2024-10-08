import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import random
import heapq
import math


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


LEARN_SIZE = 60832
TEST_SIZE = 60225



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



def knn(query_vector, dataframe, embeddings_matrix, expr, k=100):
    subset_indices = dataframe.query(expr).index

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
    nearest_distances = [distance for distance, idx in distance]
    return nearest_indices, nearest_distances



if __name__ == '__main__':
    seed_everything(42)

    df = pd.read_csv('./metadata/fungi_train_metadata.csv')
    bool_df = pd.read_csv('./metadata/fungi_bool_filter.csv')

    assert len(bool_df) == LEARN_SIZE + TEST_SIZE

    features = np.load('./features/vit_small_patch16_224_train_features.npy')

    learn_querys = np.load('./features/vit_small_patch16_224_val_features.npy')
    test_querys = np.load('./features/vit_small_patch16_224_test_features.npy')

    print(features.shape, learn_querys.shape, test_querys.shape)

    # learn querys KNN
    # learn_knn = []
    # for i in tqdm(range(LEARN_SIZE)):
    #     query_vector = learn_querys[i]
    #     expr = bool_df['bool_expression'][i]
    #     nearest_indices, nearest_distances = knn(
    #         query_vector, df, features, expr)
    #     learn_knn.append([i, nearest_distances[-1], nearest_indices])

    #     # simple_indices, simple_distances = simple_knn(
    #     #     query_vector, df, features, expr, k=100, statisfy=bool_df['count_satisfy'][i])
    #     # learn_knn.append([i, simple_distances[-1], simple_indices])
    # learn_knn_df = pd.DataFrame(
    #     learn_knn, columns=['id', 'max_distance', 'L2_nearest_indices'])
    # learn_knn_df.to_csv('./metadata/fungi_learn_knn.csv', index=False)

    # test querys KNN
    test_knn = []
    for i in tqdm(range(TEST_SIZE)):
        query_vector = test_querys[i]
        expr = bool_df['bool_expression'][LEARN_SIZE + i]
        nearest_indices, nearest_distances = knn(
            query_vector, df, features, expr)
        test_knn.append([i, nearest_distances[-1], nearest_indices])
        
    test_knn_df = pd.DataFrame(
        test_knn, columns=['id', 'max_distance', 'L2_nearest_indices'])
    test_knn_df.to_csv('./metadata/fungi_test_knn.csv', index=False)
