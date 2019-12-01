#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-06
# @Author  : Yuqiang Han(hanyuqiang1991@gmail.com)
# @Version : 1.0
import pandas as pd
import os
# import numpy as np
# import random
from collections import Counter


class PPOP:
    def __init__(self, data, user_id='user_id', item_id='item_id'):
        self.data = data
        self.user_id = user_id
        self.item_id = item_id

    def predict(self, test_ids, labels):
        rec_5, rec_10 = 0, 0
        mrr_5, mrr_10 = 0, 0

        for i in range(len(test_ids)):
            inputs = test_ids[i]
            label = labels[i]
            pop_counter = Counter(inputs).most_common(10)
            pop_list = [pop_counter[i][0] for i in range(len(pop_counter))]
            if label in pop_list:
                rank = pop_list.index(label) + 1
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

    elif dataset == 'Retailrocket':
        session_key = 'SessionId'
        item_key = 'ItemId'
    else:
        raise FileNotFoundError

    data_root = '../data/' + dataset
    test_data = pd.read_csv(os.path.join(data_root, 'test_mini.tsv'), sep='\t')
    test_users = test_data[session_key].unique()
    test_ids = []

    for user in test_users:
        clicks = test_data[test_data[session_key] == user]
        test_ids.append(clicks[item_key].values)
        # test_ts.append(clicks[time_key].values)

    pop = PPOP(test_data, user_id=session_key, item_id=item_key)
    out_seqs, labs = pop.process_seqs(test_ids)
    print("length of  prediction sequence:", len(out_seqs))


    pop.predict(out_seqs, labs)

#     要保证每个session中最后一个item在所有的inputs中出现过 ？
