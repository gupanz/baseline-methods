#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-06
# @Author  : Yuqiang Han(hanyuqiang1991@gmail.com)
# @Version : 1.0
import os
import numpy as np
import pandas as pd
from datetime import datetime


class ItemKNN:
    '''
    ItemKNN(n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time')

    Item-to-item predictor that computes the the similarity to all items to the given item.

    Similarity of two items is given by:

    .. math::
        s_{i,j}=\sum_{s}I\{(s,i)\in D & (s,j)\in D\} / (supp_i+\\lambda)^{\\alpha}(supp_j+\\lambda)^{1-\\alpha}

    Parameters
    --------
    n_sims : int
        Only give back non-zero scores to the N most similar items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    lmbd : float
        Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)
    alpha : float
        Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')

    '''

    def __init__(self, n_sims=10, lmbd=20, alpha=0.5, session_key='session_id', item_key='item_id', time_key='ts'):
        self.n_sims = n_sims
        self.lmbd = lmbd
        self.alpha = alpha
        self.item_key = item_key
        self.session_key = session_key
        self.time_key = time_key

    def fit(self, data):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs,
            one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set
            during the initialization of the network (session_key, item_key, time_key properties).
        '''
        data.set_index(np.arange(len(data)), inplace=True)
        itemids = data[self.item_key].unique()
        n_items = len(itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': np.arange(len(itemids))}),
                        on=self.item_key, how='inner')
        sessionids = data[self.session_key].unique()
        data = pd.merge(data, pd.DataFrame({self.session_key: sessionids, 'SessionIdx': np.arange(len(sessionids))}),
                        on=self.session_key, how='inner')

        supp = data.groupby('SessionIdx').size()
        session_offsets = np.zeros(len(supp) + 1, dtype=np.int32)
        session_offsets[1:] = supp.cumsum()
        index_by_sessions = data.sort_values(['SessionIdx', self.time_key]).index.values
        supp = data.groupby('ItemIdx').size()
        item_offsets = np.zeros(n_items + 1, dtype=np.int32)
        item_offsets[1:] = supp.cumsum()
        index_by_items = data.sort_values(['ItemIdx', self.time_key]).index.values
        self.sims = dict()
        for i in range(n_items):
            iarray = np.zeros(n_items)
            start = item_offsets[i]
            end = item_offsets[i + 1]
            for e in index_by_items[start:end]:
                uidx = data.SessionIdx.values[e]
                ustart = session_offsets[uidx]
                uend = session_offsets[uidx + 1]
                user_events = index_by_sessions[ustart:uend]
                iarray[data.ItemIdx.values[user_events]] += 1
            iarray[i] = 0
            norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
            norm[norm == 0] = 1
            iarray = iarray / norm
            indices = np.argsort(iarray)[-1:-1 - self.n_sims:-1]
            self.sims[itemids[i]] = pd.Series(data=iarray[indices], index=itemids[indices])

    def predict_next(self, test_ids, labels):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs
            of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session.
            Indexed by the item IDs.
        '''
        rec_5, rec_10 = 0, 0
        mrr_5, mrr_10 = 0, 0
        for i in range(len(test_ids)):
            test_id = test_ids[i]
            predict_items = pd.Series()
            # calculate the nearest neighbour of all items in the sequence
            for item in test_id:
                pred_candidates = self.sims[item]
                predict_items = predict_items.append(pred_candidates)

            item_sim_dic = {"ItemIdx": predict_items.index, "val": predict_items.values}
            df_item_sim = pd.DataFrame(item_sim_dic)
            uitem_sim_dic = dict(df_item_sim.groupby("ItemIdx")["val"].sum())
            uitem_sim_list = sorted(uitem_sim_dic.items(), key=lambda d: d[1], reverse=True)
            preds = [i for i, j in uitem_sim_list[:10]]

            label = labels[i]
            if label in preds:
                rank = np.where(preds == label)[0][0] + 1
                if rank <= 5:
                    rec_5 += 1
                    mrr_5 += 1 / rank
                rec_10 += 1
                mrr_10 += 1 / rank

        test_num = len(test_ids)
        rec5 = rec_5 / test_num
        rec10 = rec_10 / test_num
        mrr5 = mrr_5 / test_num
        mrr10 = mrr_10 / test_num
        print('Rec@5 is: %.4f, Rec@10 is: %.4f' % (rec5, rec10))
        print('MRR@5 is: %.4f, MRR@10 is: %.4f' % (mrr5, mrr10))

        with open('../results/itemKNN_results.txt', 'w') as f:
            f.write(str(rec5)[:6] + ' ' + str(rec10)[:6] + ' ' + str(mrr5)[:6] + ' ' + str(mrr10)[:6])

    def process_seqs(self, iseqs):
        out_seqs = []
        labs = []
        for seq in iseqs:
            for i in range(1, len(seq)):
                tar = seq[-i]
                labs += [tar]
                out_seqs += [seq[:-i]]
        return out_seqs, labs


if __name__ == '__main__':

    # dataset = 'Retailrocket'
    dataset = 'Taobao'

    if dataset == 'Taobao':
        session_key = 'SessionId'
        item_key = 'ItemId'
        time_key = 'Timestamp'
    elif dataset == 'Retailrocket':
        session_key = 'SessionId'
        item_key = 'ItemId'
        time_key = 'Timestamp'
    else:
        raise FileNotFoundError

    data_root = '../data/' + dataset
    interactions = pd.read_csv(os.path.join(data_root, 'train_mini.tsv'), sep='\t')
    test_data = pd.read_csv(os.path.join(data_root, 'test_mini.tsv'), sep='\t')

    # print(interactions.head())

    # interactions = interactions[:1000]
    itemknn = ItemKNN(session_key=session_key, item_key=item_key, time_key=time_key)
    session_ids = []
    item_ids = []
    test_ids = []
    labels = []
    ts = []
    test_ts = []

    users = interactions[session_key].unique()
    test_users = test_data[session_key].unique()

    # users = users[:2]
    for user in users:
        clicks = interactions[interactions[session_key] == user]
        session_ids.extend(clicks[session_key].values)
        item_ids.extend(clicks[item_key].values)
        ts.extend(clicks[time_key].values)
    train_data = pd.DataFrame({session_key: session_ids,
                               item_key: item_ids,
                               time_key: ts})
    start = datetime.now()
    itemknn.fit(train_data)
    end = datetime.now()
    print('training time: %.4f minutes' % ((end - start).seconds / 60))

    for user in test_users:
        clicks = test_data[test_data[session_key] == user]
        test_ids.append(clicks[item_key].values)
        # test_ts.append(clicks[time_key].values)
    out_seqs, labs = itemknn.process_seqs(test_ids)
    print("length of  prediction sequence:", len(out_seqs))

    itemknn.predict_next(out_seqs, labs)
