import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import random
import time
from collections import Counter
from datetime import datetime

np.random.seed(5)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


def load_data(fn, user_cnt):
    #
    # user_cnt = 100000
    train = pd.read_csv(fn, sep=',', names=["UserId", "ItemId", "CategoryId", "BehaviorType", "Timestamp"])
    # 按照行为
    all_user = train.UserId.unique()
    selected_user = np.random.choice(all_user, user_cnt)
    train = train[train.UserId.isin(selected_user)]
    return train


def filter_time(train, leftValue, rightValue):
    #
    # 2017.11.25 to 2017.12.03
    def getTimestamp(value):
        d = datetime.strptime(value, "%Y.%m.%d %H:%M:%S")
        t = d.timetuple()
        span = int(time.mktime(t))
        return span

    span_left = getTimestamp(leftValue)
    span_right = getTimestamp(rightValue)

    train = train[train['Timestamp'].between(span_left, span_right, inclusive=True)]
    return train


def add_session(train, day):
    min_timestamp = train.Timestamp.min()
    time_id = [int(math.floor((t - min_timestamp) / (86400 * day))) for t in train['Timestamp']]
    train['TimeId'] = time_id

    session_id = [str(uid) + '_' + str(tid) for uid, tid in zip(train['UserId'], train['TimeId'])]
    train['SessionId'] = session_id
    train = train[["SessionId", "UserId", "ItemId", "Timestamp", "BehaviorType", 'TimeId', "CategoryId"]]
    return train


def filter_data(train, ITEM_MIN, USER_MIN, MAX_LENGTH):
    train = train[train['ItemId'].groupby(train['ItemId']).transform('size') >= ITEM_MIN]
    train = train[train['UserId'].groupby(train['UserId']).transform('size') >= USER_MIN]
    train = train[train['SessionId'].groupby(train['SessionId']).transform('size') > 1]
    train = train[train['SessionId'].groupby(train['SessionId']).transform('size') < MAX_LENGTH]
    return train


def split_data(train, ITEM_MIN):
    # Split data into train and test
    tmax = train.TimeId.max()
    session_max_times = train.groupby('SessionId').TimeId.max()

    session_train = session_max_times[session_max_times < tmax].index
    session_holdout = session_max_times[session_max_times >= tmax].index
    train_tr = train[train['SessionId'].isin(session_train)]
    holdout_data = train[train['SessionId'].isin(session_holdout)]

    print('Number of train/test: {}/{}'.format(len(train_tr), len(holdout_data)))

    train_tr = train_tr[train_tr['ItemId'].groupby(train_tr['ItemId']).transform('size') >= ITEM_MIN]
    train_tr = train_tr[train_tr['SessionId'].groupby(train_tr['SessionId']).transform('size') > 1]

    print('Item size in train data: {}'.format(train_tr['ItemId'].nunique()))

    train_item_counter = Counter(train_tr.ItemId)
    to_predict = Counter(el for el in train_item_counter.elements() if train_item_counter[el] >= ITEM_MIN).keys()
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

    def reset_id(data, id_map, column_name='UserId'):
        mapped_id = data[column_name].map(id_map)
        data[column_name] = mapped_id
        if column_name == 'UserId':
            session_id = [str(uid) + '_' + str(tid) for uid, tid in zip(data['UserId'], data['TimeId'])]
            data['SessionId'] = session_id
        return data

    total_df = pd.concat([train_tr, valid, test])
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
    reset_id(total_df, item_map, 'ItemId')
    reset_id(train_tr, item_map, 'ItemId')
    reset_id(valid, item_map, 'ItemId')
    reset_id(test, item_map, 'ItemId')

    print(
        'Train set\n\tEvents: {}\n\tSessions: {}\n\tUsers: {}\n\tItems: {}\n\tAvg length: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.UserId.nunique(), train_tr.ItemId.nunique(),
                                                                                                     train_tr.groupby('SessionId').size().mean()))
    print('Valid set\n\tEvents: {}\n\tSessions: {}\n\tUsers: {}\n\tItems: {}\n\tAvg length: {}'.format(len(valid), valid.SessionId.nunique(), valid.UserId.nunique(), valid.ItemId.nunique(),
                                                                                                       valid.groupby('SessionId').size().mean()))
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tUsers: {}\n\tItems: {}\n\tAvg length: {}'.format(len(test), test.SessionId.nunique(), test.UserId.nunique(), test.ItemId.nunique(),
                                                                                                      test.groupby('SessionId').size().mean()))

    train_tr = train_tr.sort_values(by=["SessionId", "Timestamp"], ascending=[True, True])
    valid = valid.sort_values(by=["SessionId", "Timestamp"], ascending=[True, True])
    test = test.sort_values(by=["SessionId", "Timestamp"], ascending=[True, True])

    train_tr.to_csv('../data/Taobao/train.tsv', sep='\t', index=False)
    valid.to_csv('../data/Taobao/valid.tsv', sep='\t', index=False)
    test.to_csv('../data/Taobao/test.tsv', sep='\t', index=False)


if __name__ == '__main__':
    day = 1
    leftValue = '2017.11.25 00:00:00'
    rightValue = '2017.12.04 00:00:00'
    train = load_data(fn='../data/Taobao/UserBehavior.csv', user_cnt=100000)
    train = filter_time(train, leftValue, rightValue)
    train = add_session(train, day)
    ITEM_MIN = 44
    USER_MIN = 25
    MAX_LENGTH = 50
    train = filter_data(train, ITEM_MIN, USER_MIN, MAX_LENGTH)
    split_data(train, ITEM_MIN)
