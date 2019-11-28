#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-06
# @Author  : Yuqiang Han(hanyuqiang1991@gmail.com)
# @Version : 1.0
import pandas as pd
# import numpy as np
# import random
from collections import Counter


class PPOP:
    def __init__(self, data, user_id='user_id', item_id='item_id'):
        self.data = data
        self.user_id = user_id
        self.item_id = item_id

    def predict(self):
        users = self.data[self.user_id].unique()
        rec_5, rec_10 = 0, 0
        mrr_5, mrr_10 = 0, 0
        for user in users:
            clicks = self.data[self.data[self.user_id] == user][self.item_id].values
            label = clicks[-1]
            inputs = clicks[:-1]
            pop_counter = Counter(inputs).most_common(10)
            # pop_set = set([ppop[i][0] for i in range(len(ppop))])
            pop_list = [pop_counter[i][0] for i in range(len(pop_counter))]
            if label in pop_list:
                rank = pop_list.index(label) + 1
                if rank <= 5:
                    rec_5 += 1
                    mrr_5 += 1 / rank
                rec_10 += 1
                mrr_10 += 1 / rank

        test_num = len(users)
        rec5 = rec_5 / test_num
        rec10 = rec_10 / test_num
        mrr5 = mrr_5 / test_num
        mrr10 = mrr_10 / test_num
        print('Rec@5 is: %.4f, Rec@10 is: %.4f' % (rec5, rec10))
        print('MRR@5 is: %.4f, MRR@10 is: %.4f' % (mrr5, mrr10))


if __name__ == '__main__':
    # Taobao
    # interactions = pd.read_csv('/Users/hyq/Documents/Dataset/Taobao/interactions.csv')
    # pop = PPOP(interactions)
    # pop.predict()

    # Retailrocket
    interactions = pd.read_csv('/Users/hyq/Documents/Dataset/Retailrocket/interactions.csv')
    pop = PPOP(interactions, user_id='visitorid', item_id='itemid')
    pop.predict()


#     要保证每个session中最后一个item在所有的inputs中出现过 ？









