import os

import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine
from tqdm import tqdm

engine = create_engine(
    'postgresql://postgres:xxx@localhost:5432/mtmd')

data_root = r"/home/ubuntu/code/hybrid_query_experiments/datasets/multi_table_movie_dataset"


def insert_structured_data(table_nane):
    df_csv = pd.read_csv(os.path.join(
        data_root, "metadata", table_nane + ".csv"))

    df_csv.to_sql(table_nane, engine, if_exists='append', index=False)


def insert_vector_table():
    df_csv = pd.read_csv(os.path.join(data_root, "metadata", "utterances.csv"))

    vectors_path = os.path.join(data_root, "vectors", "base_vectors.npy")
    vectors = np.load(vectors_path)
    print(type(vectors), vectors.shape)
    print(df_csv.shape)

    df_csv['embedding'] = [vector.tolist() for vector in vectors]
    print("to list over")

    tqdm.pandas(desc="Converting to JSON")
    df_csv['embedding'] = df_csv['embedding'].progress_apply(json.dumps)

    print("do insert data...")

    df_csv.to_sql('utterances', engine, if_exists='append', index=False)


if __name__ == '__main__':
    struct_table_names = ["movies", "genres",
                          "movies_genres", "speakers", "conversations"]

    for table_name in struct_table_names:
        insert_structured_data(table_name)

    insert_vector_table()
    engine.dispose()
