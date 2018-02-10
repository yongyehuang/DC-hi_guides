# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division

import features.action as action
import features.comment as comment
import features.history as history
import features.profile as profile
from my_utils import check_path

import numpy as np
import pandas as pd
import os
import time
import pickle

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

"""Model assessment tools.
"""

df_profile_train = pd.read_csv('data/trainingset/userProfile_train.csv')
df_action_train = pd.read_csv('data/trainingset/action_train.csv')
df_history_train = pd.read_csv('data/trainingset/orderHistory_train.csv')
df_comment_train = pd.read_csv('data/trainingset/userComment_train.csv')
df_future_train = pd.read_csv('data/trainingset/orderFuture_train.csv')

df_profile_test = pd.read_csv('data/test/userProfile_test.csv')
df_action_test = pd.read_csv('data/test/action_test.csv')
df_history_test = pd.read_csv('data/test/orderHistory_test.csv')
df_comment_test = pd.read_csv('data/test/userComment_test.csv')
df_future_test = pd.read_csv('data/test/orderFuture_test.csv')
df_future_train.rename(columns={'orderType': 'label'}, inplace=True)


def get_action_rate_before_jp(order_type=1):
    """统计订单下单前的单个动作比例。"""
    save_path = 'dict_last_action_rate_{}.pkl'.format(order_type)
    if os.path.exists(save_path):
        dict_last_action_rate = pickle.load(open(save_path, 'rb'))
        return dict_last_action_rate
    df_history = pd.concat([df_history_train, df_history_test])
    df_history = df_history[df_history.orderType == order_type].copy()
    df_history.rename(columns={'orderTime': 'orderTime_{}'.format(order_type)}, inplace=True)
    df_actions = pd.concat([df_action_train, df_action_test])
    df_actions = pd.merge(df_actions, df_history[['userid', 'orderTime_{}'.format(order_type)]], on='userid',
                          how='left')
    df_actions['time_after_order'] = df_actions['actionTime'] - df_actions['orderTime_{}'.format(order_type)]
    df_actions = df_actions[df_actions['time_after_order'] <= 0]
    df_user_last_action_type = df_actions[['userid', 'actionType']].groupby('userid', as_index=False).tail(1)
    sr_action_count = df_user_last_action_type.actionType.value_counts()
    action_types = sr_action_count.index
    action_counts = sr_action_count.values
    n_total_actions = len(df_user_last_action_type)
    dict_last_action_rate = dict()
    for i in range(len(action_types)):
        dict_last_action_rate[action_types[i]] = action_counts[i] / n_total_actions
    pickle.dump(dict_last_action_rate, open(save_path, 'wb'))
    print('Save dict_last_action_rate to {}'.format(save_path))
    return dict_last_action_rate


def get_pair_rate_before_jp(order_type=1):
    """统计订单下单前的连续两个动作比例。"""
    save_path = 'dict_last_pair_rate_{}.pkl'.format(order_type)
    if os.path.exists(save_path):
        dict_last_pair_rate = pickle.load(open(save_path, 'rb'))
        return dict_last_pair_rate
    df_history = pd.concat([df_history_train, df_history_test])
    df_history = df_history[df_history.orderType == order_type].copy()
    df_history.rename(columns={'orderTime': 'orderTime_{}'.format(order_type)}, inplace=True)
    df_actions = pd.concat([df_action_train, df_action_test])
    df_actions = pd.merge(df_actions, df_history[['userid', 'orderTime_{}'.format(order_type)]], on='userid',
                          how='left')
    df_actions['last_userid'] = df_actions.userid.shift(1)
    df_actions['last_actionType'] = df_actions.actionType.shift(1)
    df_actions = df_actions.loc[df_actions.last_userid == df_actions.userid].copy()
    action_pairs = list(zip(df_actions.last_actionType.values, df_actions.actionType.values))
    action_pairs = map(lambda s: str(int(s[0])) + '_' + str(int(s[1])), action_pairs)
    df_actions['action_pair'] = action_pairs

    df_actions['time_after_order'] = df_actions['actionTime'] - df_actions['orderTime_{}'.format(order_type)]
    df_actions = df_actions[df_actions['time_after_order'] <= 0]
    df_user_last_pair_type = df_actions[['userid', 'action_pair']].groupby('userid', as_index=False).tail(1)
    sr_pair_count = df_user_last_pair_type.action_pair.value_counts()
    pair_types = sr_pair_count.index
    pair_counts = sr_pair_count.values
    n_total_pairs = len(df_user_last_pair_type)
    dict_last_pair_rate = dict()
    for i in range(len(pair_types)):
        dict_last_pair_rate[pair_types[i]] = pair_counts[i] / n_total_pairs
    pickle.dump(dict_last_pair_rate, open(save_path, 'wb'))
    print('Save dict_last_pair_rate to {}'.format(save_path))
    return dict_last_pair_rate


def get_triple_rate_before_jp(order_type=1):
    """统计订单下单前的连续3个动作比例。"""
    save_path = 'dict_last_triple_rate_{}.pkl'.format(order_type)
    if os.path.exists(save_path):
        dict_last_triple_rate = pickle.load(open(save_path, 'rb'))
        return dict_last_triple_rate
    df_history = pd.concat([df_history_train, df_history_test])
    df_history = df_history[df_history.orderType == order_type].copy()
    df_history.rename(columns={'orderTime': 'orderTime_{}'.format(order_type)}, inplace=True)
    df_actions = pd.concat([df_action_train, df_action_test])
    df_actions = pd.merge(df_actions, df_history[['userid', 'orderTime_{}'.format(order_type)]], on='userid',
                          how='left')
    df_actions['last_userid_2'] = df_actions.userid.shift(2)
    df_actions['last_actionType_2'] = df_actions.actionType.shift(2)
    df_actions['last_userid'] = df_actions.userid.shift(1)
    df_actions['last_actionType'] = df_actions.actionType.shift(1)
    df_actions = df_actions.loc[df_actions.last_userid_2 == df_actions.userid].copy()

    action_triples = list(
        zip(df_actions.last_actionType_2.values, df_actions.last_actionType.values, df_actions.actionType.values))
    action_triples = map(lambda s: str(int(s[0])) + '_' + str(int(s[1])) + '_' + str(int(s[2])), action_triples)
    df_actions['action_triple'] = action_triples

    df_actions['time_after_order'] = df_actions['actionTime'] - df_actions['orderTime_{}'.format(order_type)]
    df_actions = df_actions[df_actions['time_after_order'] <= 0]
    df_user_last_triple_type = df_actions[['userid', 'action_triple']].groupby('userid', as_index=False).tail(1)
    sr_triple_count = df_user_last_triple_type.action_triple.value_counts()
    triple_types = sr_triple_count.index
    triple_counts = sr_triple_count.values
    n_total_triples = len(df_user_last_triple_type)
    dict_last_triple_rate = dict()
    for i in range(len(triple_types)):
        dict_last_triple_rate[triple_types[i]] = triple_counts[i] / n_total_triples
    pickle.dump(dict_last_triple_rate, open(save_path, 'wb'))
    print('Save dict_last_triple_rate to {}'.format(save_path))
    return dict_last_triple_rate


def get_gender_convert_rate():
    """统计历史订单中转化率:
    dict_gender_rate: 男女转化率。
    dict_province_rate：省份转化率。
    dict_age_rate: 年龄转化率。
    """
    df_history = pd.concat([df_history_train, df_history_test])[['userid', 'orderType']].copy()
    df_profile = pd.concat([df_profile_train, df_profile_test])
    df_order_rate = pd.merge(df_history, df_profile, on='userid', how='left')
    mean_convert_rate = np.sum(df_order_rate.orderType.values) / len(df_order_rate)
    # 性别转化率
    gender_rate_path = 'dict_gender_convert_rate.pkl'
    dict_gender = {'mean': mean_convert_rate}
    genders = set(df_order_rate['gender'].values.tolist())
    print(genders)
    for gender in genders:
        if gender is np.nan:
            continue
        print(gender)
        n_total = len(df_order_rate[df_order_rate['gender'] == gender])
        n_jp = len(df_order_rate[(df_order_rate['gender'] == gender) & (df_order_rate['orderType'] == 1)])
        convert_rate = n_jp / n_total
        dict_gender[gender] = convert_rate
    print(dict_gender)
    pickle.dump(dict_gender, open(gender_rate_path, 'wb'))

    # 省份转化率
    province_rate_path = 'dict_province_convert_rate.pkl'
    dict_province = {'mean': mean_convert_rate}
    provinces = set(df_order_rate['province'].values.tolist())
    print(provinces)
    for province in provinces:
        if province is np.nan:
            continue
        print(province)
        n_total = len(df_order_rate[df_order_rate['province'] == province])
        n_jp = len(df_order_rate[(df_order_rate['province'] == province) & (df_order_rate['orderType'] == 1)])
        convert_rate = n_jp / n_total
        dict_province[province] = convert_rate
    print(dict_province)
    pickle.dump(dict_province, open(province_rate_path, 'wb'))

    # 年龄转化率
    age_rate_path = 'dict_age_convert_rate.pkl'
    dict_age = {'mean': mean_convert_rate}
    ages = set(df_order_rate['age'].values.tolist())
    print(ages)
    for age in ages:
        if age is np.nan:
            continue
        print(age)
        n_total = len(df_order_rate[df_order_rate['age'] == age])
        n_jp = len(df_order_rate[(df_order_rate['age'] == age) & (df_order_rate['orderType'] == 1)])
        convert_rate = n_jp / n_total
        dict_age[age] = convert_rate
    print(dict_age)
    pickle.dump(dict_age, open(age_rate_path, 'wb'))


def get_feat(df_future, df_history, df_actions, df_profile, df_comment):
    """Get features."""
    dict_last_action_rate_0 = get_action_rate_before_jp(0)
    dict_last_action_rate_1 = get_action_rate_before_jp(1)
    dict_last_pair_rate_0 = get_pair_rate_before_jp(0)
    dict_last_pair_rate_1 = get_pair_rate_before_jp(1)
    dict_last_triple_rate_0 = get_triple_rate_before_jp(0)
    dict_last_triple_rate_1 = get_triple_rate_before_jp(1)

    df_history_feat = history.get_history_feat(df_history)
    df_features = pd.merge(df_future, df_history_feat, on='userid', how='left')
    # 用户信息特征
    df_profile_feat = profile.get_profile_feat(df_profile)
    df_features = pd.merge(df_features, df_profile_feat, on='userid', how='left')
    # 评论特征
    df_comment_feat = comment.get_comment_feat(df_comment)
    df_features = pd.merge(df_features, df_comment_feat, on='userid', how='left')
    # 行为特征
    df_action_feat = action.get_action_feat(df_actions,
                                            dict_last_action_rate_0,
                                            dict_last_action_rate_1,
                                            dict_last_pair_rate_0,
                                            dict_last_pair_rate_1,
                                            dict_last_triple_rate_0,
                                            dict_last_triple_rate_1)
    df_features = pd.merge(df_features, df_action_feat, on='userid', how='left')
    return df_features


def load_feat(re_get=False, feature_path=None):
    """Load the features.
    Args:
        re_get: If true, run the get_feat function to get the new features. Else, load feature from feature_path.
            If you modified the function to get feature, re_get should be True.
        feature_path: The path to load(or save) features.
    Returns:
        The features extract from train data and test data.
    """
    train_data_path = '{}train_data.csv'.format(feature_path)
    test_data_path = '{}test_data.csv'.format(feature_path)
    if not re_get and os.path.exists(feature_path):
        print('Loading features from {}'.format(feature_path))
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
    else:
        tic = time.time()
        print('Data pre-processing, please wait for minutes.')
        train_data = get_feat(df_future_train, df_history_train, df_action_train, df_profile_train, df_comment_train)
        test_data = get_feat(df_future_test, df_history_test, df_action_test, df_profile_test, df_comment_test)
        print('Data prepared, cost time {}s'.format(time.time() - tic))
        if feature_path:
            check_path(feature_path)
            train_data.to_csv(train_data_path, index=False)
            test_data.to_csv(test_data_path, index=False)
            message = 'Saved featrues to {}'.format(feature_path)
            print(message)
    # **1.对特征进行特定处理
    # 对所有的 diff 取log， 结果变差，放弃
    # column_names = train_data.columns.values
    # diff_columns = list(filter(lambda col_name: 'diff' in col_name, column_names))
    # for diff_column in diff_columns:
    #     train_data[diff_column] = train_data[diff_column].apply(lambda x: np.log(x))
    #     test_data[diff_column] = test_data[diff_column].apply(lambda x: np.log(x))
    # **2.减去最小值，结果也不好
    # **3.统计转化率
    # **4.构造特征组合
    return train_data, test_data
