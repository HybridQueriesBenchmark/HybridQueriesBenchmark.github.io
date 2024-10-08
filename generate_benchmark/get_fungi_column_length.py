import pandas as pd


def get_fungi_column_length(column_name, df):
    """
    Get the length of the column values in the dataframe
    :param column_name: str, the name of the column
    :param df: pd.DataFrame, the dataframe
    :return: list, the length of the column values
    """
    return max(list(df[column_name].apply(lambda x: len(str(x)))))


if __name__ == '__main__':
    df = pd.read_csv('./metadata/fungi_train_metadata.csv')
    res = get_fungi_column_length('Habitat', df)
    print(res)
