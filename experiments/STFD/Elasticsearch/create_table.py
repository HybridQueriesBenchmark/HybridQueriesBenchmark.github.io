from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")  # Replace with your Elasticsearch address and port

# Drop the index if it exists
index_name = "fungis"
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# Define the index mapping with HNSW settings
mapping = {
    "mappings": {
        "properties": {
            "id": {"type": "integer"},  # ID of the document
            "year": {"type": "integer"},  # Year (can be missing)
            "month": {"type": "integer"},  # Month (can be missing)
            "day": {"type": "integer"},    # Day (can be missing)
            "countryCode": {"type": "keyword"},  # Country code, exact match
            "scientificName": {"type": "keyword"},  # Scientific name, full-text search
            "Substrate": {"type": "keyword"},        # Substrate type, full-text search
            "Latitude": {"type": "float"},        # Latitude
            "Longitude": {"type": "float"},       # Longitude
            "Habitat": {"type": "keyword"},          # Habitat type, full-text search
            "poisonous": {"type": "boolean"},     # Indicates if the fungus is poisonous
            "embedding": {  # Vector field for embeddings with HNSW parameters
                "type": "dense_vector",
                "dims": 768,  # Set to the number of dimensions of your vector
                "element_type": "float",  # Data type for the vector elements
                "index": True,  # Enable indexing
                "similarity": "l2_norm",  # Set the similarity metric
                "index_options": {
                    "type": "hnsw",  # Specify HNSW index type
                    "m": 16,  # Number of neighbors each node will be connected to
                    "ef_construction": 64  # Number of candidates to track while assembling the list of nearest neighbors
                }
            }
        }
    }
}

# Create the index with the defined mapping
es.indices.create(index=index_name, body=mapping)

print(f"Index '{index_name}' created successfully.")
