import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

elasticsearch_host = 'localhost'
elasticsearch_port = 9200
elasticsearch_index = 'fungis'

es = Elasticsearch([{
    'host': elasticsearch_host,
    'port': elasticsearch_port,
    'scheme': 'http'
}])


meta_csv_path = r"/home/ubuntu/code/hybrid_query_experiments/datasets/single_table_fungi_dataset/metadata/base_metadata.csv"
df_csv = pd.read_csv(meta_csv_path)


vectors_path = r"/home/ubuntu/code/hybrid_query_experiments/datasets/single_table_fungi_dataset/vectors/base_vectors.npy"
vectors = np.load(vectors_path).astype(np.float32)


df_csv['poisonous'] = df_csv['poisonous'].astype(bool)
df_csv['Substrate'] = df_csv['Substrate'].fillna('')


batch_size = 1000
actions = []

for i in tqdm(range(len(df_csv)), desc="Preparing data for insertion"):
    vector = vectors[i].tolist()
    
    year = df_csv.iloc[i]['year'] if pd.notna(df_csv.iloc[i]['year']) else None
    month = df_csv.iloc[i]['month'] if pd.notna(df_csv.iloc[i]['month']) else None
    day = df_csv.iloc[i]['day'] if pd.notna(df_csv.iloc[i]['day']) else None
    
    habitat = df_csv.iloc[i]['Habitat'] if pd.notna(df_csv.iloc[i]['Habitat']) else None
    
    document = {
        "year": year,
        "month": month,
        "day": day,
        "countryCode": df_csv.iloc[i]['countryCode'],
        "scientificName": df_csv.iloc[i]['scientificName'],
        "Substrate": df_csv.iloc[i]['Substrate'],
        "Latitude": df_csv.iloc[i]['Latitude'] if pd.notna(df_csv.iloc[i]['Latitude']) else None,
        "Longitude": df_csv.iloc[i]['Longitude'] if pd.notna(df_csv.iloc[i]['Longitude']) else None,
        "Habitat": habitat,
        "poisonous": df_csv.iloc[i]['poisonous'],
        "embedding": vector
    }
    
    actions.append({
        "_index": elasticsearch_index,
        "_id": str(df_csv.iloc[i]['id']),
        "_source": document
    })


    if len(actions) >= batch_size:
        try:
            response = helpers.bulk(es, actions)
        except Exception as e:
            for action in actions:
                try:
                    es.index(index=elasticsearch_index, id=action["_id"], body=action["_source"]
                except Exception as index_error:
                    print(f"id: {action['_id']} , error: {index_error}")
        finally:
            actions.clear()


if actions:
    try:
        response = helpers.bulk(es, actions)
    except Exception as e:
        for action in actions:
            try:
                es.index(index=elasticsearch_index, id=action["_id"], body=action["_source"])
            except Exception as index_error:
                print(f"id: {action['_id']} , error: {index_error}")

es.close()
