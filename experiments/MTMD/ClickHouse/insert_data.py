import pandas as pd
import numpy as np
import json
import clickhouse_connect
from tqdm import tqdm


clickhouse_host = 'localhost'
clickhouse_port = 8123
clickhouse_user = 'default'
clickhouse_password = ''
clickhouse_database = 'mtmd'

client = clickhouse_connect.get_client(
    host=clickhouse_host,
    port=clickhouse_port,
    username=clickhouse_user,
    password=clickhouse_password
)


client.command(f'USE {clickhouse_database}')


meta_csv_path = r"/home/ubuntu/code/hybrid_query_experiments/datasets/multi_table_movie_dataset/metadata/utterances.csv"
vectors_path = r"/home/ubuntu/code/hybrid_query_experiments/datasets/multi_table_movie_dataset/vectors/base_vectors.npy"


df_csv = pd.read_csv(meta_csv_path)
vectors = np.load(vectors_path).astype(np.float32)


print(f"CSV Shape: {df_csv.shape}")
print(f"Vectors Shape: {vectors.shape}")


assert vectors.shape[1] == 768, "Embedding vector length does not match the table constraint!"


insert_query = """
INSERT INTO utterances (utterance_id, conversation_id, speaker_id, reply_to, embedding) 
VALUES
"""

batch_size = 1000
batch_values = []


def format_value(val):
    if pd.isna(val):
        return 'NULL'
    elif isinstance(val, str):
        return f"'{val}'"
    else:
        return str(val)


for i, row in tqdm(df_csv.iterrows(), total=df_csv.shape[0], desc="Inserting data"):
    vector = vectors[i]
    
    embedding_json = json.dumps(vector.tolist())
    
    value = f"({format_value(row['utterance_id'])}, {format_value(row['conversation_id'])}, {format_value(row['speaker_id'])}, {format_value(row['reply_to'])}, {embedding_json})"
    
    batch_values.append(value)
    
    if len(batch_values) == batch_size:
        query = insert_query + ', '.join(batch_values)
        client.command(query)
        batch_values.clear()

if batch_values:
    query = insert_query + ', '.join(batch_values)
    client.command(query)

print("Data inserted successfully!")

client.close()
