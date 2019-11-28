import pandas as pd
import numpy as np
import math
from collections import Counter

PATH_TO_DATA = './Douban/'

SOCIAL_NETWORK_FILE = PATH_TO_DATA + 'socialnet/socialnet.tsv'
RATING_FILE = PATH_TO_DATA + 'movie/douban_movie.tsv'
max_length = 30


def process_rating(day=7):  # segment session in every $day days.
    df = pd.read_csv(RATING_FILE, sep='\t', dtype={0: str, 1: str, 2: np.int32, 3: np.float32})
    df = df[df['Rating'].between(1, 6, inclusive=True)]
    span_left = 1.2e9
    span_right = 1.485e9
    df = df[df['Timestamp'].between(span_left, span_right, inclusive=True)]
    min_timestamp = df['Timestamp'].min()
    time_id = [int(math.floor((t - min_timestamp) / (86400 * day))) for t in df['Timestamp']]
    df['TimeId'] = time_id
    session_id = [str(uid) + '_' + str(tid) for uid, tid in zip(df['UserId'], df['TimeId'])]
    df['SessionId'] = session_id
    print('Statistics of user ratings:')
    print('\tNumber of total ratings: {}'.format(len(df)))
    print('\tNumber of users: {}'.format(df.UserId.nunique()))
    print('\tNumber of items: {}'.format(df.ItemId.nunique()))
    print('\tAverage ratings per user:{}'.format(df.groupby('UserId').size().mean()))
    return df


def process_social():  # read in social network.
    net = pd.read_csv(SOCIAL_NETWORK_FILE, sep='\t', dtype={0: str, 1: str})
    net.drop_duplicates(subset=['Follower', 'Followee'], inplace=True)
    friend_size = net.groupby('Follower').size()
    # net = net[np.in1d(net.Follower, friend_size[friend_size>=5].index)]
    print('Statistics of social network:')
    print('\tTotal user in social network:{}.\n\tTotal edges(links) in social network:{}.'.format( \
        net.Follower.nunique(), len(net)))
    print('\tAverage number of friends for users: {}'.format(net.groupby('Follower').size().mean()))
    return net


def reset_id(data, id_map, column_name='UserId'):
    mapped_id = data[column_name].map(id_map)
    data[column_name] = mapped_id
    if column_name == 'UserId':
        session_id = [str(uid) + '_' + str(tid) for uid, tid in zip(data['UserId'], data['TimeId'])]
        data['SessionId'] = session_id
    return data


def split_data(day):  # split data for training/validation/testing.
    df_data = process_rating(day)
    df_net = process_social()
    df_net = df_net.loc[df_net['Follower'].isin(df_data['UserId'].unique())]
    df_net = df_net.loc[df_net['Followee'].isin(df_data['UserId'].unique())]
    df_data = df_data.loc[df_data['UserId'].isin(df_net.Follower.unique())]

    print("columns of orignal input data  : ", df_data.shape[0])

    # # will be deleted
    # df_data = df_data.sample(n=500000).reset_index(drop=True)

    # restrict session length in [2, max_length]. We set a max_length because too long sequence may come from a fake user.
    df_data = df_data[df_data['SessionId'].groupby(df_data['SessionId']).transform('size') > 1]
    df_data = df_data[df_data['SessionId'].groupby(df_data['SessionId']).transform('size') <= max_length]
    # length_supports = df_data.groupby('SessionId').size()
    # df_data = df_data[np.in1d(df_data.SessionId, length_supports[length_supports<=max_length].index)]

    # split train, test, valid.
    tmax = df_data.TimeId.max()
    session_max_times = df_data.groupby('SessionId').TimeId.max()
    session_train = session_max_times[session_max_times < tmax - 26].index
    session_holdout = session_max_times[session_max_times >= tmax - 26].index
    train_tr = df_data[df_data['SessionId'].isin(session_train)]
    holdout_data = df_data[df_data['SessionId'].isin(session_holdout)]

    print('Number of train/test: {}/{}'.format(len(train_tr), len(holdout_data)))

    train_tr = train_tr[train_tr['ItemId'].groupby(train_tr['ItemId']).transform('size') >= 20]
    train_tr = train_tr[train_tr['SessionId'].groupby(train_tr['SessionId']).transform('size') > 1]

    print('Item size in train data: {}'.format(train_tr['ItemId'].nunique()))
    train_item_counter = Counter(train_tr.ItemId)
    to_predict = Counter(el for el in train_item_counter.elements() if train_item_counter[el] >= 50).keys()
    print('Size of to predict: {}'.format(len(to_predict)))

    # split holdout to valid and test.
    holdout_cn = holdout_data.SessionId.nunique()
    holdout_ids = holdout_data.SessionId.unique()
    np.random.shuffle(holdout_ids)
    valid_cn = int(holdout_cn * 0.5)
    session_valid = holdout_ids[0: valid_cn]
    session_test = holdout_ids[valid_cn:]
    valid = holdout_data[holdout_data['SessionId'].isin(session_valid)]
    test = holdout_data[holdout_data['SessionId'].isin(session_test)]

    valid = valid[valid['ItemId'].isin(to_predict)]
    valid = valid[valid['SessionId'].groupby(valid['SessionId']).transform('size') > 1]

    test = test[test['ItemId'].isin(to_predict)]
    test = test[test['SessionId'].groupby(test['SessionId']).transform('size') > 1]

    total_df = pd.concat([train_tr, valid, test])
    df_net = df_net.loc[df_net['Follower'].isin(total_df['UserId'].unique())]
    df_net = df_net.loc[df_net['Followee'].isin(total_df['UserId'].unique())]
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
    reset_id(train_tr, user_map)
    reset_id(valid, user_map)
    reset_id(test, user_map)
    reset_id(df_net, user_map, 'Follower')
    reset_id(df_net, user_map, 'Followee')
    reset_id(total_df, item_map, 'ItemId')
    reset_id(train_tr, item_map, 'ItemId')
    reset_id(valid, item_map, 'ItemId')
    reset_id(test, item_map, 'ItemId')

    # add item-item-adj
    item_adj = []
    data = train_tr.sort_values(by=['Timestamp']).groupby('SessionId')['ItemId'].apply(list).to_dict()
    for k, v in data.items():
        if len(v) < 2:
            continue
        item_adj.extend([[v[i + 1], v[i]] for i in range(len(v) - 1)])
    item_adj_df = pd.DataFrame(item_adj, columns=["Follower", "Followee"])
    item_adj_df = item_adj_df.groupby(["Follower", "Followee"]).size().reset_index(name='Weight')
    item_adj_df['Rate'] = item_adj_df['Weight'] / item_adj_df.groupby('Follower')['Weight'].transform('sum')

    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tAvg length: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique(),
                                                                                          train_tr.groupby('SessionId').size().mean()))
    print('Valid set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tAvg length: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique(), valid.groupby('SessionId').size().mean()))
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tAvg length: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), test.groupby('SessionId').size().mean()))

    user2sessions = total_df.groupby('UserId')['SessionId'].apply(set).to_dict()
    user_latest_session = []
    for idx in range(num_users):
        sessions = user2sessions[idx]
        latest = []
        for t in range(tmax + 1):
            if t == 0:
                latest.append('NULL')
            else:
                sess_id_tmp = str(idx) + '_' + str(t - 1)
                if sess_id_tmp in sessions:
                    latest.append(sess_id_tmp)
                else:
                    latest.append(latest[t - 1])
        user_latest_session.append(latest)

    train_tr.to_csv('train.tsv', sep='\t', index=False)
    valid.to_csv('valid.tsv', sep='\t', index=False)
    test.to_csv('test.tsv', sep='\t', index=False)
    df_net.to_csv('adj.tsv', sep='\t', index=False)
    item_adj_df.to_csv("item_adj.tsv", sep='\t', index=False)
    with open('latest_sessions.txt', 'w') as fout:
        for idx in range(num_users):
            fout.write(','.join(user_latest_session[idx]) + '\n')


if __name__ == '__main__':
    day = 7
    split_data(day)
