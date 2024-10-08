import pandas as pd
from elasticsearch import Elasticsearch
from tqdm import tqdm


es = Elasticsearch(["http://localhost:9200"])


df = pd.read_csv('/home/ubuntu/code/hybrid_query_experiments/datasets/single_table_fungi_dataset/bool_filter/test_filter.csv')
print(len(df))


dsl_queries = []


for index, row in tqdm(df.iterrows()):
    bool_expression = row['bool_expression'].replace('==', '=')
    
    bool_expression = bool_expression.replace("poisonous = 0", "poisonous = true")
    bool_expression = bool_expression.replace("poisonous = 1", "poisonous = false")
    sql_query = f"SELECT id FROM fungis WHERE {bool_expression}"


    # print(sql_query)
    
    response = es.sql.translate(body={"query": sql_query})
    
    dsl_query = response
    
    dsl_queries.append(dsl_query)

df['dsl_query'] = dsl_queries
df.to_csv('output_with_dsl.csv', index=False)

print("DSL queries have been saved to 'output_with_dsl.csv'")
