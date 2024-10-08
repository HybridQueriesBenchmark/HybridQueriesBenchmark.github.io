import pandas as pd
import clickhouse_connect

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


csv_path = '/home/ubuntu/code/hybrid_query_experiments/datasets/multi_table_movie_dataset/metadata/conversations.csv'
df = pd.read_csv(csv_path)


data_tuples = list(df.itertuples(index=False, name=None))


client.insert('conversations', data_tuples)

print("Data inserted successfully into conversations table!")


client.close()
