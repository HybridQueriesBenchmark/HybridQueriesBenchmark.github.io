import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine
from tqdm import tqdm


engine = create_engine('postgresql://postgres:xxx@localhost:5432/stfd')

meta_csv_path = r"/home/ubuntu/code/hybrid_query_experiments/datasets/single_table_fungi_dataset/metadata/base_metadata.csv"

df_csv = pd.read_csv(meta_csv_path)


vectors_path = r"/home/ubuntu/code/hybrid_query_experiments/datasets/single_table_fungi_dataset/vectors/base_vectors.npy"

vectors = np.load(vectors_path)
print(type(vectors), vectors.shape)
print(df_csv.shape)


df_csv['poisonous'] = df_csv['poisonous'].astype(bool)


df_csv['embedding'] = [vector.tolist() for vector in vectors]
print("to list over")


tqdm.pandas(desc="Converting to JSON")
df_csv['embedding'] = df_csv['embedding'].progress_apply(json.dumps)

print("do insert data...")


df_csv.to_sql('fungis', engine, if_exists='append', index=False)

engine.dispose()
