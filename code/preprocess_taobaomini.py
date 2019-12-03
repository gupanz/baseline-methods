import pandas as pd
import numpy as np

np.random.seed(5)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


def sample_data():
    def select_n(df, n):
        all_user = df.SessionId.unique()
        selected_user = np.random.choice(all_user, n)
        return df[df.SessionId.isin(selected_user)]

    train_fi = pd.read_csv('../data/Taobao/train.tsv', sep='\t', dtype={0: str, 1: str, 2: str, 3: np.int64, 4: str, 5: str, 6: str})
    test_fi = pd.read_csv('../data/Taobao/test.tsv', sep='\t', dtype={0: str, 1: str, 2: str, 3: np.int64, 4: str, 5: str, 6: str})
    valid_fi = pd.read_csv('../data/Taobao/valid.tsv', sep='\t', dtype={0: str, 1: str, 2: str, 3: np.int64, 4: str, 5: str, 6: str})

    train_fi = select_n(train_fi, 1000)
    to_predict = train_fi.ItemId.unique()

    valid_fi = valid_fi[valid_fi['ItemId'].isin(to_predict)]
    valid_fi = valid_fi[valid_fi['SessionId'].groupby(valid_fi['SessionId']).transform('size') > 1]
    valid_fi = select_n(valid_fi, 1000)

    test_fi = test_fi[test_fi['ItemId'].isin(to_predict)]
    test_fi = test_fi[test_fi['SessionId'].groupby(test_fi['SessionId']).transform('size') > 1]
    test_fi = select_n(test_fi, 1000)

    def reset_id(data, id_map, column_name='UserId'):
        mapped_id = data[column_name].map(id_map)
        data[column_name] = mapped_id
        if column_name == 'UserId':
            session_id = [str(uid) + '_' + str(tid) for uid, tid in zip(data['UserId'], data['TimeId'])]
            data['SessionId'] = session_id
        return data

    total_df = pd.concat([train_fi, valid_fi, test_fi])
    user_map = dict(zip(total_df.UserId.unique(), range(total_df.UserId.nunique())))
    item_map = dict(zip(total_df.ItemId.unique(), range(1, 1 + total_df.ItemId.nunique())))
    with open('user_id_map.tsv', 'w') as fout:
        for k, v in user_map.items():
            fout.write(str(k) + '\t' + str(v) + '\n')
    with open('item_id_map.tsv', 'w') as fout:
        for k, v in item_map.items():
            fout.write(str(k) + '\t' + str(v) + '\n')
    num_users = len(user_map)
    num_items = len(item_map)
    reset_id(total_df, user_map)
    reset_id(train_fi, user_map)
    reset_id(valid_fi, user_map)
    reset_id(test_fi, user_map)
    reset_id(total_df, item_map, 'ItemId')
    reset_id(train_fi, item_map, 'ItemId')
    reset_id(valid_fi, item_map, 'ItemId')
    reset_id(test_fi, item_map, 'ItemId')
    

    def output(train_tr, df_name):
        print('{} set\n\tEvents: {}\n\tSessions: {}\n\tUsers: {}\n\tItems: {}\n\tAvg length: {}'.format(df_name, len(train_tr), train_tr.SessionId.nunique(), train_tr.UserId.nunique(),
                                                                                                    train_tr.ItemId.nunique(),
                                                                                                    train_tr.groupby('SessionId').size().mean()))
    output(train_fi, "train")
    output(test_fi, "test")
    output(valid_fi, "valid")

    train_fi.to_csv('../data/TaobaoMini/train.tsv', sep='\t', index=False)
    valid_fi.to_csv('../data/TaobaoMini/valid.tsv', sep='\t', index=False)
    test_fi.to_csv('../data/TaobaoMini/test.tsv', sep='\t', index=False)


if __name__ == '__main__':
    sample_data()