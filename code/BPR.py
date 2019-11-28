#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-10
# @Author  : Yuqiang Han(hanyuqiang1991@gmail.com)
# @Version : 1.0
import os
import pandas as pd
import numpy as np
from datetime import datetime


class BPR:
    '''
    BPR(n_factors = 100, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_item = 0.0, sigma = 0.05,
    init_normal = False, session_key = 'SessionId', item_key = 'ItemId')

    Bayesian Personalized Ranking Matrix Factorization (BPR-MF). During prediction time, the current state of
    the session is modelled as the average of the feature vectors of the items that have occurred in it so far.

    Parameters
    --------
    n_factor : int
        The number of features in a feature vector. (Default value: 100)
    n_iterations : int
        The number of epoch for training. (Default value: 10)
    learning_rate : float
        Learning rate. (Default value: 0.01)
    lambda_session : float
        Regularization for session features. (Default value: 0.0)
    lambda_item : float
        Regularization for item features. (Default value: 0.0)
    sigma : float
        The width of the initialization. (Default value: 0.05)
    init_normal : boolean
        Whether to use uniform or normal distribution based initialization.
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    '''

    def __init__(self, n_factors=100, n_iterations=10, n_sims=10, learning_rate=0.01, lambda_session=0.0, lambda_item=0.0,
                 sigma=0.05, init_normal=False, session_key='SessionId', item_key='ItemId'):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.lambda_session = lambda_session
        self.lambda_item = lambda_item
        self.sigma = sigma
        self.init_normal = init_normal
        self.session_key = session_key
        self.item_key = item_key
        self.n_sims = n_sims
        # self.current_session = None

    def init(self):
        self.U = np.random.rand(self.n_sessions, self.n_factors) * 2 * self.sigma - self.sigma \
            if not self.init_normal else np.random.randn(self.n_sessions, self.n_factors) * self.sigma
        self.I = np.random.rand(self.n_items, self.n_factors) * 2 * self.sigma - self.sigma \
            if not self.init_normal else np.random.randn(self.n_items, self.n_factors) * self.sigma
        self.bU = np.zeros(self.n_sessions)
        self.bI = np.zeros(self.n_items)

    def update(self, uidx, p, n):
        uF = np.copy(self.U[uidx, :])
        iF1 = np.copy(self.I[p, :])
        iF2 = np.copy(self.I[n, :])
        sigm = self.sigmoid(iF1.T.dot(uF) - iF2.T.dot(uF) + self.bI[p] - self.bI[n])
        c = 1.0 - sigm
        self.U[uidx, :] += self.learning_rate * (c * (iF1 - iF2) - self.lambda_session * uF)
        self.I[p, :] += self.learning_rate * (c * uF - self.lambda_item * iF1)
        self.I[n, :] += self.learning_rate * (-c * uF - self.lambda_item * iF2)
        return np.log(sigm)

    def fit(self, data):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item
            IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the
            initialization of the network (session_key, item_key, time_key properties).

        '''
        itemids = data[self.item_key].unique()
        self.itemids = itemids
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': np.arange(self.n_items)}),
                        on=self.item_key, how='inner')
        data = pd.merge(data, pd.DataFrame({self.session_key: sessionids, 'SessionIdx': np.arange(self.n_sessions)}),
                        on=self.session_key, how='inner')
        self.init()
        for it in range(self.n_iterations):
            c = []
            for e in np.random.permutation(len(data)):
                uidx = data.SessionIdx.values[e]
                iidx = data.ItemIdx.values[e]
                iidx2 = data.ItemIdx.values[np.random.randint(self.n_items)]
                err = self.update(uidx, iidx, iidx2)
                c.append(err)
            print(it, np.mean(c))

    def predict_next(self, input_item_ids, predict_for_item_ids):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_ids : int or string
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
        # iidx = self.itemidmap[input_item_id]
        # if self.current_session is None or self.current_session != session_id:
        #     self.current_session = session_id
        #     self.session = [iidx]
        # else:
        #     self.session.append(iidx)
        # uF = self.I[self.session].mean(axis=0)
        # self.session = [self.itemidmap[k] for k in input_item_ids]
        self.session = self.itemidmap[input_item_ids]
        uF = self.I[self.session].mean(axis=0)
        iIdxs = self.itemidmap[predict_for_item_ids]
        iarray = self.I[iIdxs].dot(uF) + self.bI[iIdxs]
        indices = np.argsort(iarray)[-1:-1 - self.n_sims:-1]
        return self.itemids[indices]
        # return pd.Series(data=iarray[indices], index=self.itemids[indices])
        # return pd.Series(data=self.I[iIdxs].dot(uF) + self.bI[iIdxs], index=predict_for_item_ids)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def process_seqs(self, iseqs):
        out_seqs = []
        out_dates = []
        labs = []
        for seq in iseqs:
            for i in range(1, len(seq)):
                tar = seq[-i]
                labs += [tar]
                out_seqs += [seq[:-i]]
        return out_seqs, labs


if __name__ == '__main__':
    # taobao
    # dataset = 'Retailrocket'
    dataset = 'Taobao'


    if dataset == 'Taobao':
        session_key = 'UserId'
        item_key = 'ItemId'

    elif dataset == 'Retailrocket':
        session_key = 'UserId'
        item_key = 'ItemId'
    else:
        raise FileNotFoundError

    data_root = '../data/' + dataset

    interactions = pd.read_csv(os.path.join(data_root, 'train_mini.tsv'), sep='\t')
    test_data = pd.read_csv(os.path.join(data_root, 'test_mini.tsv'), sep='\t')

    # interactions = interactions[:1000]
    session_ids = []
    item_ids = []
    labels = []
    users = interactions[session_key].unique()
    # users = users[:2]

    for user in users:
        clicks = interactions[interactions[session_key] == user]
        session_ids.extend(clicks[session_key].values)
        item_ids.extend(clicks[item_key].values)
        # labels.append(clicks[item_key].values[-1])
    train_data = pd.DataFrame({session_key: session_ids,
                               item_key: item_ids})
    start = datetime.now()
    bpr = BPR(session_key=session_key, item_key=item_key)
    bpr.fit(train_data)
    end = datetime.now()
    print('training time: %.4f minutes' % ((end - start).seconds / 60))

    # test
    rec_5, rec_10 = 0, 0
    mrr_5, mrr_10 = 0, 0
    test_users = test_data[session_key].unique()
    candidates = train_data[item_key].unique()

    test_ids = []
    test_ts = []
    for user in test_users:
        clicks = test_data[test_data[session_key] == user]
        test_ids.append(clicks[item_key].values)
    out_seqs, labs = bpr.process_seqs(test_ids)

    for input_item_ids, label in zip(out_seqs, labs):
        preds = bpr.predict_next(input_item_ids, candidates)
        if label in preds:
            rank = np.where(preds == label)[0][0] + 1
            if rank <= 5:
                rec_5 += 1
                mrr_5 += 1 / rank
            rec_10 += 1
            mrr_10 += 1 / rank

    test_num = len(out_seqs)
    rec5 = rec_5 / test_num
    rec10 = rec_10 / test_num
    mrr5 = mrr_5 / test_num
    mrr10 = mrr_10 / test_num
    print('Rec@5 is: %.4f, Rec@10 is: %.4f' % (rec5, rec10))
    print('MRR@5 is: %.4f, MRR@10 is: %.4f' % (mrr5, mrr10))
    # with open(os.path.join(data_root, 'results.txt'), 'w') as f:
    with open('../results/bpr_results.txt', 'w') as f:
        f.write(str(rec5)[:6] + ' ' + str(rec10)[:6] + ' ' + str(mrr5)[:6] + ' ' + str(mrr10)[:6])
