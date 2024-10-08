import pandas as pd
import numpy as np
from pymilvus import Collection, MilvusClient
from tqdm import tqdm


meta_csv_path = "/home/ubuntu/code/hybrid_query_experiments/datasets/single_table_fungi_dataset/metadata/base_metadata.csv"

df_csv = pd.read_csv(meta_csv_path)


vectors_path = "/home/ubuntu/code/hybrid_query_experiments/datasets/single_table_fungi_dataset/vectors/base_vectors.npy"

vectors = np.load(vectors_path)
print(type(vectors), vectors.shape)
print(df_csv.shape)

df_csv['poisonous'] = df_csv['poisonous'].astype(bool)
df_csv['year'] = df_csv['year'].fillna(-1)
df_csv['year'] = df_csv['year'].astype(np.int32)
df_csv['month'] = df_csv['month'].fillna(-1)
df_csv['month'] = df_csv['month'].astype(np.int32)
df_csv['day'] = df_csv['day'].fillna(-1)
df_csv['day'] = df_csv['day'].astype(np.int32)

def fill_empty_strings(df):
    columns = df.columns

    for column in columns:
        if df[column].dtype == 'object':
            print(column, df[column].dtype)
            df[column].fillna("", inplace=True)
    
    return df

df_csv = fill_empty_strings(df_csv)

df_csv.to_csv('temp.csv', index=False)

df_csv['embedding'] = [vector.tolist() for vector in vectors]

client = MilvusClient(
    uri="http://localhost:19530"
)

for idx in tqdm(range(len(df_csv))):
    res = client.insert(
        collection_name="fungis",
        data=df_csv.iloc[idx].to_dict()
    )

client.close()
