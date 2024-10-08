import pandas as pd

LEARN_SIZE = 60832
TEST_SIZE = 60225


def split_csv(input_file, output_file_head, output_file_tail, column_names, n):
    df = pd.read_csv(input_file)

    selected_columns = df[column_names]

    selected_columns.head(n).to_csv(output_file_head, index_label='id')

    tail_df = selected_columns.tail(len(df) - n).reset_index(drop=True)
    tail_df.to_csv(output_file_tail, index_label='id')


if __name__ == '__main__':
    split_csv('./metadata/fungi_bool_filter.csv', './metadata/fungi_learn_filter.csv', './metadata/fungi_test_filter.csv',
              ['expression_num', 'bool_expression'], LEARN_SIZE)
    print('Done')
