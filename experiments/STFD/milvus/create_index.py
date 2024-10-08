from pymilvus import MilvusClient
import time


client = MilvusClient(
    uri="http://localhost:19530"
)


column_lst = ['id', 'year', 'month', 'day', 'countryCode', 'scientificName', 'Substrate', 'Latitude', 'Longitude', 'Habitat', 'poisonous']

for column in column_lst:
    index_params = MilvusClient.prepare_index_params()
    index_name = column + "_index"
    client.drop_index(
        collection_name="fungis",
        index_name=index_name
    )
    # create scalar index
    if column == 'poisonous':
        index_params.add_index(
            field_name=column,
            index_type="INVERTED",
            index_name=index_name
        )
        client.create_index(
            collection_name="fungis",
            index_params=index_params
        )
    else:
        index_params.add_index(
            field_name=column,
            index_name=index_name
        )
        client.create_index(
            collection_name="fungis",
            index_params=index_params
        )
    print("Create index on column: ", column)


# create vector index
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="embedding",
    metric_type="L2",
    index_type="HNSW",
    M=16,
    efConstruction=64,
    index_name="embedding_index"
)
client.drop_index(
    collection_name="fungis",
    index_name="embedding_index"
)

start = time.perf_counter()
client.create_index(
    collection_name="fungis",
    index_params=index_params
)

build_idx_time = time.perf_counter() - start
print(f"Time: {build_idx_time:.4f}")    # 48.2946 seconds

client.close()
