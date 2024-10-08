import pandas as pd
import numpy as np
import json
import clickhouse_connect
from tqdm import tqdm


clickhouse_host = 'localhost'
clickhouse_port = 8123
clickhouse_user = 'default'
clickhouse_password = ''
clickhouse_database = 'stfd'


client = clickhouse_connect.get_client(
    host=clickhouse_host,
    port=clickhouse_port,
    username=clickhouse_user,
    password=clickhouse_password
)


client.command(f'USE {clickhouse_database}')


meta_csv_path = r"/home/ubuntu/code/hybrid_query_experiments/datasets/single_table_fungi_dataset/metadata/base_metadata.csv"
df_csv = pd.read_csv(meta_csv_path)


vectors_path = r"/home/ubuntu/code/hybrid_query_experiments/datasets/single_table_fungi_dataset/vectors/base_vectors.npy"
vectors = np.load(vectors_path).astype(np.float32)
print(type(vectors), vectors.shape)
print(df_csv.shape)


df_csv['poisonous'] = df_csv['poisonous'].astype(int)


insert_query = """
INSERT INTO fungis (id, year, month, day, countryCode, scientificName, Substrate, Latitude, Longitude, Habitat, poisonous, embedding) 
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
    
    value = f"({format_value(row['id'])}, {format_value(row['year'])}, {format_value(row['month'])}, {format_value(row['day'])}, {format_value(row['countryCode'])}, {format_value(row['scientificName'])}, {format_value(row['Substrate'])}, {format_value(row['Latitude'])}, {format_value(row['Longitude'])}, {format_value(row['Habitat'])}, {format_value(row['poisonous'])}, {embedding_json})"
    
    batch_values.append(value)
    
    if len(batch_values) == batch_size:
        query = insert_query + ', '.join(batch_values)
        # print(query)
        client.command(query)
        batch_values.clear()


if batch_values:
    query = insert_query + ', '.join(batch_values)
    client.command(query)

print("Data inserted successfully!")


client.close()
