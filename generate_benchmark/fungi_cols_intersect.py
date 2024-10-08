import pandas as pd

df = pd.read_csv('./metadata/FungiCLEF2023_train_metadata_PRODUCTION.csv')


df.describe(include='all').to_csv('./fungi_train_summary.csv')

count = len(df[df['countryCode'] != 'DK'])
print(count)


columns_to_drop = ['observationID', 'locality', 'taxonID', 'kingdom', 'phylum',
                   'class', 'order', 'family', 'genus', 'specificEpithet', 'taxonRank',
                   'species', 'level0Gid', 'level0Name', 'level1Gid', 'level1Name', 'level2Gid', 'level2Name', 'ImageUniqueID',
                   'rightsHolder', 'CoorUncert', 'image_path', 'class_id', 'MetaSubstrate']


df.drop(columns=columns_to_drop, inplace=True)


df.to_csv("./metadata/fungi_train_metadata.csv", index=True, index_label='id')
