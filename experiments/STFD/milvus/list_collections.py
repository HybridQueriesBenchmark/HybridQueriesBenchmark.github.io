from pymilvus import MilvusClient
import pymilvus

client = MilvusClient("http://localhost:19530")
# print(client.list_collections())

res = client.describe_collection(
    collection_name="fungis"
)
print(res)

res = client.get_collection_stats(collection_name="fungis")
print(res)

res = client.list_indexes(
    collection_name="fungis"
)

print(res)

client.close()
