from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema, Collection
import pymilvus
import random
import time


client = MilvusClient(
    uri="http://localhost:19530"
)


id_field = FieldSchema(name="id", dtype=DataType.INT64,
                       is_primary=True, description="primary id")
year_field = FieldSchema(name="year", dtype=DataType.INT32, description="year")
month_field = FieldSchema(
    name="month", dtype=DataType.INT32, description="month")
day_field = FieldSchema(name="day", dtype=DataType.INT32, description="day")
countryCode_field = FieldSchema(
    name="countryCode", dtype=DataType.VARCHAR, max_length=2)
scientificName_field = FieldSchema(
    name="scientificName", dtype=DataType.VARCHAR, max_length=110)
Substrate_field = FieldSchema(
    name="Substrate", dtype=DataType.VARCHAR, max_length=50)
Latitude_field = FieldSchema(name="Latitude", dtype=DataType.FLOAT)
Longitude_field = FieldSchema(name="Longitude", dtype=DataType.FLOAT)
Habitat_field = FieldSchema(
    name="Habitat", dtype=DataType.VARCHAR, max_length=60)
poisonous_field = FieldSchema(name="poisonous", dtype=DataType.BOOL)
embedding_field = FieldSchema(
    name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768, description="vector")

schema = CollectionSchema(fields=[id_field, year_field, month_field, day_field, countryCode_field, scientificName_field, Substrate_field, Latitude_field,
                          Longitude_field, Habitat_field, poisonous_field, embedding_field], auto_id=False, enable_dynamic_field=True, description="desc of a collection")

collection_name = "fungis"
client.drop_collection(collection_name)
print("Before Create: ", client.list_collections())
connection = pymilvus.connections.connect(host='localhost', port='19530', alias='default')
collection = Collection(name=collection_name, schema=schema, using='default', metrictype='L2')

print("After: ", client.list_collections())
client.close()
