# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division

from my_utils import time_to_date
import pandas as pd
import numpy as np
import pickle


def merge_action(action_type):
    """将动作进行归类"""
    if action_type in [2, 3, 4]:
        return 2
    if action_type in [7, 8, 9]:
        return 7
    else:
        return action_type


def get_action_ratio(df_actions, n_seconds=None):
    """统计每个用户的各种动作比例，取用户最后的 n_seconds 的行为进行统计。
    Args:
        df_actions: 用户行为数据
        n_seconds: 取该用户最后的 n_seconds 时间段进行统计。
        如果 n_seconds==None,统计用户所有的动作。此外，主要统计 1小时（3600），12小时（3600*12）,48小时（3600*24）。
    Returns:
        action{}_ratio_{}.format(1~9, n_seconds): 动作比例
        action_count_{}.format(n_seconds): 动作总数
    """
    if n_seconds is not None:
        df_last_action_time = df_actions[['userid', 'actionTime']].groupby('userid').tail(1).copy()  # 取最后一个动作的时刻
        df_last_action_time.rename(columns={'actionTime': 'last_actionTime'}, inplace=True)
        df_actions = pd.merge(df_actions, df_last_action_time, on='userid', how='left')
        df_actions['diff_with_last_action'] = df_actions.last_actionTime - df_actions.actionTime
        # 取距离小于 n_seconds 的动作
        df_actions = df_actions[df_actions.diff_with_last_action <= n_seconds]
    else:
        n_seconds = 'all'
    # 统计动作的比例
    df_actions['action_count'] = 1
    df_count = df_actions[['userid', 'action_count']].groupby('userid', as_index=False).count()
    df_action_type = pd.get_dummies(df_actions['actionType'], prefix='actionType_{}'.format(n_seconds))
    df_action_ratio = pd.concat([df_actions['userid'], df_action_type], axis=1)
    df_action_ratio = df_action_ratio.groupby('userid', as_index=False).sum()
    for action_type in range(1, df_action_ratio.shape[1]):
        col_name = 'actionType_{}_{}'.format(n_seconds, action_type)
        df_action_ratio[col_name] = df_action_ratio[col_name] / df_count['action_count']
    df_action_ratio['n_action_{}'.format(n_seconds)] = df_count['action_count'].values
    return df_action_ratio


def get_first_last_action_time(df_actions):
    """获取最早一次行为的时间和最晚一次行为的时间，还有两次行为的时间距离。"""
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    grouped_action_time = df_actions[['userid', 'actionTime']].groupby('userid', as_index=False)
    df_mintime = grouped_action_time.min()
    df_mintime.rename(columns={'actionTime': 'first_action_time'}, inplace=True)
    # 最后一行为距离“当前时刻”
    df_maxtime = grouped_action_time.max()
    df_maxtime.rename(columns={'actionTime': 'last_action_time'}, inplace=True)
    df_actions_feat = pd.merge(df_mintime, df_maxtime, on='userid', how='left')
    df_actions_feat['diff_last_first_action'] = df_actions_feat['last_action_time'] - df_actions_feat[
        'first_action_time']
    return df_actions_feat


def get_last_action_date_time(df_actions, action_type='all'):
    """获取最后一次动作 action_type 的日期。"""
    if action_type is not 'all':
        df_actions = df_actions[df_actions.actionType == action_type].copy()
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    df_actions = df_actions[['userid', 'actionTime']].groupby('userid', as_index=False).tail(1).copy()
    df_actions['last_action_date'] = df_actions['actionTime'].apply(time_to_date)
    df_actions['last_action_date'] = pd.to_datetime(df_actions.last_action_date,
                                                    format="%Y-%m-%d %H:%M:%S")
    # df_actions['last_action_year'] = df_actions.last_action_date.apply(lambda dt: dt.year)
    # df_actions['last_action_month'] = df_actions.last_action_date.apply(lambda dt: dt.month)
    # df_actions['last_action_day'] = df_actions.last_action_date.apply(lambda dt: dt.day)
    df_actions['last_action_weekday_{}'.format(action_type)] = df_actions.last_action_date.apply(
        lambda dt: dt.weekday())
    df_actions['last_action_hour_{}'.format(action_type)] = df_actions.last_action_date.apply(lambda dt: dt.hour)
    df_actions['last_action_minute_{}'.format(action_type)] = df_actions.last_action_date.apply(lambda dt: dt.minute)
    df_last_action_date = df_actions.drop(['actionTime', 'last_action_date'], axis=1)
    return df_last_action_date


def get_each_action_last_time(df_actions, action_type):
    """获取动作 action_type 的最早一次时间和最后一次时间。"""
    df_actions = df_actions.loc[df_actions.actionType == action_type][['userid', 'actionTime']]
    grouped = df_actions.groupby('userid')
    df_first_action = grouped.head(1).copy()
    df_first_action.rename(columns={'actionTime': 'first_action_{}_time'.format(action_type)}, inplace=True)
    df_last_action = grouped.tail(1).copy()
    df_last_action.rename(columns={'actionTime': 'last_action_{}_time'.format(action_type)}, inplace=True)
    df_each_action_feat = pd.merge(df_first_action, df_last_action, on='userid', how='left')
    diff_col = 'first_last_action_{}_diff'.format(action_type)
    df_each_action_feat[diff_col] = df_each_action_feat['last_action_{}_time'.format(action_type)] - \
                                    df_each_action_feat['first_action_{}_time'.format(action_type)]
    return df_each_action_feat


def get_each_action_to_end_diff(df_actions, action_type):
    """获取最后一次动作 action_type 的距离最后的一次时间距离。
    TODO: 把 2，3，4 归为一类动作。
    TODO：获取用户每次历史订单时间距离每个动作的时间距离。
    """
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    df_last_action_time = df_actions[['userid', 'actionTime']].groupby('userid').tail(1).copy()
    df_last_action_time.rename(columns={'actionTime': 'last_action_time'}, inplace=True)
    df_actions = pd.merge(df_actions, df_last_action_time, on='userid', how='left')
    df_actions = df_actions.loc[df_actions.actionType == action_type][['userid', 'actionTime', 'last_action_time']]
    df_actions['{}_to_end_diff'.format(action_type)] = df_actions['last_action_time'] - df_actions['actionTime']
    df_action_to_end_diff = df_actions[['userid', '{}_to_end_diff'.format(action_type)]].groupby('userid').tail(
        1).copy()
    return df_action_to_end_diff


def get_last_diff_statistic(df_actions, n_last_diff='all'):
    """获取最后 n_last_diff 个动作之间时间间隔的统计特征。
    """
    df_actions['next_userid'] = df_actions.userid.shift(-1)
    df_actions['next_actionTime'] = df_actions.actionTime.shift(-1)
    df_actions = df_actions.loc[df_actions.next_userid == df_actions.userid].copy()
    df_actions['action_diff'] = df_actions['next_actionTime'] - df_actions['actionTime']
    if n_last_diff is not 'all':
        df_n_last_diff = df_actions.groupby('userid', as_index=False).tail(n_last_diff).copy()
        df_last_diff_statistic = df_n_last_diff.groupby('userid', as_index=False).action_diff.agg({
            # 'last_{}_action_diff_max'.format(n_last_diff): np.max,
            # 'last_{}_action_diff_min'.format(n_last_diff): np.min,
            'last_{}_action_diff_mean'.format(n_last_diff): np.mean,
            'last_{}_action_diff_std'.format(n_last_diff): np.std
        })
    else:
        grouped_user = df_actions.groupby('userid', as_index=False)
        df_last_diff_statistic = grouped_user.action_diff.agg({
            # 'last_{}_action_diff_max'.format(n_last_diff): np.max,
            # 'last_{}_action_diff_min'.format(n_last_diff): np.min,
            'last_{}_action_diff_mean'.format(n_last_diff): np.mean,
            'last_{}_action_diff_std'.format(n_last_diff): np.std
        })
    return df_last_diff_statistic


def get_last_diff_divide_statistic(df_1, df_2, n_last_diff1, n_last_diff2):
    """获取最后 n_last_diff 个动作之间时间间隔的统计特征。
    """
    df_last_diff_divide = pd.merge(df_1, df_2, on='userid', how='left')
    df_last_diff_divide['mean_divide_{}_{}'.format(n_last_diff1, n_last_diff2)] = df_last_diff_divide[
                                                                                      'last_{}_action_diff_mean'.format(
                                                                                          n_last_diff1)] / \
                                                                                  df_last_diff_divide[
                                                                                      'last_{}_action_diff_mean'.format(
                                                                                          n_last_diff2)]
    df_last_diff_divide['std_divide_{}_{}'.format(n_last_diff1, n_last_diff2)] = df_last_diff_divide[
                                                                                     'last_{}_action_diff_std'.format(
                                                                                         n_last_diff1)] / \
                                                                                 df_last_diff_divide[
                                                                                     'last_{}_action_diff_std'.format(
                                                                                         n_last_diff2)]
    df_last_diff_divide = df_last_diff_divide[['userid', 'mean_divide_{}_{}'.format(n_last_diff1, n_last_diff2),
                                               'std_divide_{}_{}'.format(n_last_diff1, n_last_diff2)]].copy()
    return df_last_diff_divide


def get_last_diff(df_actions, n_last_diff=10):
    """获取最后 n_last_diff 个动作之间的时间间隔.
    Args:
        df_actions: 用户行为数据
        n_last_diff: 最近的 n_last_diff 个动作
    Returns:
        final_diff：该用户最近的 n_last_diff 个动作之间的时间间隔。
    """
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    df_actions['next_userid'] = df_actions.userid.shift(-1)
    df_actions['next_actionTime'] = df_actions.actionTime.shift(-1)
    df_actions = df_actions.loc[df_actions.next_userid == df_actions.userid].copy()
    df_actions['action_diff'] = df_actions['next_actionTime'] - df_actions['actionTime']

    df_actions['rank_'] = range(df_actions.shape[0])
    df_max_rank = df_actions.groupby('userid', as_index=False)['rank_'].agg({'user_max_rank': np.max})
    df_actions = pd.merge(df_actions, df_max_rank, on='userid', how="left")
    df_actions['last_rank'] = df_actions['user_max_rank'] - df_actions['rank_']
    df_actions = df_actions[df_actions['last_rank'] < n_last_diff]  # 取最后的 n_last_diff 个间隔

    df_last_diffs = df_actions[['userid', 'last_rank', 'action_diff']].copy()
    df_last_diffs = df_last_diffs.set_index(["userid", "last_rank"]).unstack()
    df_last_diffs.reset_index(inplace=True)
    df_last_diffs.columns = ["userid"] + ["final_diff_{}".format(i) for i in range(n_last_diff)]
    return df_last_diffs


def get_last_action_type(df_actions, n_action=1, to_one_hot=True):
    """获取最后 n_action 个特征的类型。
    Args:
        df_actions: 用户行为数据
        n_action: 最后 n_action 个特征。
        to_one_hot: 是否进行 one-hot 处理。
    Returns:
        最后 的 n_action 个特征。
    """
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    grouped_user = df_actions.groupby('userid', as_index=False)
    if n_action == 1:
        df_last_action_types = grouped_user['userid', 'actionType'].tail(1).copy()
        df_last_action_types.rename(columns={'actionType': 'last_action_type'}, inplace=True)
        if to_one_hot:
            df_last_action_types = pd.get_dummies(df_last_action_types, columns=['last_action_type'])
        return df_last_action_types
    df_actions['rank_'] = range(df_actions.shape[0])
    # 每个用户最后的一个动作 rank
    user_rank_max = df_actions.groupby('userid', as_index=False)['rank_'].agg({'user_rank_max': np.max})
    df_actions = pd.merge(df_actions, user_rank_max, on='userid', how="left")
    # 每个用户的倒数第几个动作
    df_actions["user_rank_sub"] = df_actions['user_rank_max'] - df_actions['rank_']
    df_last_action_types = user_rank_max[['userid']]
    for k in range(n_action):
        df_last_action = df_actions[df_actions.user_rank_sub == k][['userid', 'actionType']]
        if to_one_hot:
            df_last_action = pd.get_dummies(df_last_action, columns=['actionType'],
                                            prefix='user_last_{}_action_type'.format(k))
        else:
            df_last_action.rename(columns={'actionType': 'last_action_type_{}'.format(k)})
        df_last_action_types = pd.merge(df_last_action_types, df_last_action, on='userid', how="left")
    return df_last_action_types


def get_last_pair(df_actions, dict_pair_rate_0, dict_pair_rate_1):
    """获取最后的一组动作，并将其转为对应的转化概率。"""
    df_actions['last_userid'] = df_actions.userid.shift(1)
    df_actions['last_actionType'] = df_actions.actionType.shift(1)
    df_actions = df_actions.loc[df_actions.last_userid == df_actions.userid].copy()
    action_pairs = list(zip(df_actions.last_actionType.values, df_actions.actionType.values))
    action_pairs = map(lambda s: str(int(s[0])) + '_' + str(int(s[1])), action_pairs)
    df_actions['action_pair'] = action_pairs
    df_user_last_pair_type = df_actions[['userid', 'action_pair']].groupby('userid', as_index=False).tail(1)
    df_user_last_pair_type['last_pair_rate_0'] = df_user_last_pair_type['action_pair'].apply(
        lambda s: dict_pair_rate_0.get(s, 0))
    df_user_last_pair_type['last_pair_rate_1'] = df_user_last_pair_type['action_pair'].apply(
        lambda s: dict_pair_rate_1.get(s, 0))
    df_user_last_pair_type['last_pair_convert_rate'] = df_user_last_pair_type['last_pair_rate_1'] / (
        df_user_last_pair_type['last_pair_rate_0'] + df_user_last_pair_type['last_pair_rate_1'])
    df_user_last_pair_type['last_pair_convert_rate'] = df_user_last_pair_type['last_pair_convert_rate'].apply(
        lambda x: 0 if x == np.inf else x)
    # df_last_pair_rate = df_user_last_pair_type[
    #     ['userid', 'last_pair_rate_0', 'last_pair_rate_1', 'last_pair_convert_rate']]
    df_last_pair_rate = df_user_last_pair_type[['userid', 'last_pair_rate_0', 'last_pair_rate_1']]
    return df_last_pair_rate


def get_last_triple(df_actions, dict_triple_rate_0, dict_triple_rate_1):
    """获取最后的一组动作，并将其转为对应的转化概率。"""
    df_actions['last_userid_2'] = df_actions.userid.shift(2)
    df_actions['last_actionType_2'] = df_actions.actionType.shift(2)
    df_actions['last_userid'] = df_actions.userid.shift(1)
    df_actions['last_actionType'] = df_actions.actionType.shift(1)
    df_actions = df_actions.loc[df_actions.last_userid_2 == df_actions.userid].copy()

    action_triples = list(
        zip(df_actions.last_actionType_2.values, df_actions.last_actionType.values, df_actions.actionType.values))
    action_triples = map(lambda s: str(int(s[0])) + '_' + str(int(s[1])) + '_' + str(int(s[2])), action_triples)
    df_actions['action_triple'] = action_triples
    df_user_last_triple_type = df_actions[['userid', 'action_triple']].groupby('userid', as_index=False).tail(1)
    df_user_last_triple_type['last_triple_rate_0'] = df_user_last_triple_type['action_triple'].apply(
        lambda s: dict_triple_rate_0.get(s, 0))
    df_user_last_triple_type['last_triple_rate_1'] = df_user_last_triple_type['action_triple'].apply(
        lambda s: dict_triple_rate_1.get(s, 0))
    df_user_last_triple_type['last_triple_convert_rate'] = df_user_last_triple_type['last_triple_rate_1'] / (
        df_user_last_triple_type['last_triple_rate_0'] + df_user_last_triple_type['last_triple_rate_1'])
    df_user_last_triple_type['last_triple_convert_rate'] = df_user_last_triple_type['last_triple_convert_rate'].apply(
        lambda x: 0 if x == np.inf else x)
    # df_last_triple_rate = df_user_last_triple_type[
    #     ['userid', 'last_triple_rate_0', 'last_triple_rate_1', 'last_triple_convert_rate']]
    df_last_triple_rate = df_user_last_triple_type[['userid', 'last_triple_rate_0', 'last_triple_rate_1']]
    return df_last_triple_rate


def get_after_action_feat(df_actions, action_type, n_diff=2):
    """获取最近一个动作（action_type，如 6） 后面的统计特征：
    Args:
        df_actions: 用户行为数据
        action_type: int,动作类型：1~9。
        n_diff: int,返回该动作后面 n_diff 个动作的时间间隔。
    Returns:
        'n_action_after_{}'.action_type:  动作个数
        'diff1_after_{}'.action_type: 该动作后第 1 个动作的时间间隔
        'diff2...': 该动作后第 2 个动作的时间间隔
        'mean_diff_after_{}'.action_type 时间间隔均值
        'std_diff_after_{}'.action_type 时间间隔标准差。
    """
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    df_after_action_feats = df_actions.groupby('userid', as_index=False).tail(1)[['userid']]
    df_actions['rank_'] = range(df_actions.shape[0])
    df_actions_op = df_actions[df_actions.actionType == action_type].copy()
    # 每个用户最后的一个动作op 的 rank
    op_max_rank = df_actions_op.groupby('userid', as_index=False)['rank_'].agg({'op_rank_max': np.max})
    df_actions = pd.merge(df_actions, op_max_rank, on='userid', how="left")
    df_actions['after_op'] = df_actions['rank_'] - df_actions['op_rank_max']
    df_actions = df_actions[df_actions['after_op'] >= 0]

    df_actions['count'] = 1
    df_count = df_actions[['userid', 'count']].groupby('userid', as_index=False).count()
    df_count.rename(columns={'count': 'n_action_after_{}'.format(action_type)}, inplace=True)
    df_count['n_action_after_{}'.format(action_type)] = df_count['n_action_after_{}'.format(action_type)].apply(
        lambda x: x - 1)

    # 统计后面动作的时间间隔
    df_actions["next_userid"] = df_actions.userid.shift(-1)
    df_actions["next_action_time"] = df_actions.actionTime.shift(-1)
    df_actions["next_action_diff"] = df_actions["next_action_time"] - df_actions["actionTime"]
    df_actions = df_actions[df_actions.userid == df_actions.next_userid].copy()
    user_op_time = df_actions.groupby("userid", as_index=False)["next_action_diff"].agg({
        "mean_diff_after_{}".format(action_type): np.mean,
        "std_diff_after_{}".format(action_type): np.std,
        "min_diff_after_{}".format(action_type): np.min,
        "max_diff_after_{}".format(action_type): np.max
    })
    df_after_action_feats = pd.merge(df_after_action_feats, df_count, on='userid', how='left')

    # 获取该动作后面的 n_diff 个动作
    for k in range(n_diff):
        df_last_diff = df_actions[df_actions.after_op == k][['userid', 'next_action_diff']]
        df_last_diff.rename(columns={'next_action_diff': 'diff_{}_after_{}'.format(k, action_type)}, inplace=True)
        df_after_action_feats = pd.merge(df_after_action_feats, df_last_diff, on='userid', how="left")
    df_after_action_feats = pd.merge(df_after_action_feats, user_op_time, on='userid', how='left')
    return df_after_action_feats


def get_action_pair_feats(df_actions, action1=5, action2=6):
    """获取两个特定动作之间的时间距离。
       必须 action1 在前，否则返回 np.nan
    Args:
        action_pair：动作对.
    Returns:
        mean_diff_action1_action2: action1 到 action_2 的平均时间距离
        last_diff_action1_action2: 最后一次 action1 到 action_2 的时间距离。
        diff_rate_action1_action2: = last_diff_action1_action2 / mean_diff_action1_action2
    """
    df_action_pair_feat = df_actions.groupby('userid', as_index=False).tail(1)[['userid']].copy()
    df_actions["rank_"] = range(df_actions.shape[0])
    df_actions = df_actions.loc[(df_actions.actionType == action1) | (df_actions.actionType == action2)].copy()
    df_actions['next_userid'] = df_actions.userid.shift(-1)
    df_actions["next_actionType"] = df_actions.actionType.shift(-1)
    df_actions["next_action_time"] = df_actions.actionTime.shift(-1)
    df_actions["next_action_rank"] = df_actions.rank_.shift(-1)
    df_actions = df_actions.loc[df_actions.next_userid == df_actions.userid].copy()
    df_actions["next_action_diff"] = df_actions["next_action_time"] - df_actions["actionTime"]
    df_actions["next_action_count"] = df_actions["next_action_rank"] - df_actions["rank_"]
    df_actions = df_actions.loc[(df_actions.actionType == action1) & (df_actions.next_actionType == action2)].copy()
    df_actions = df_actions[["userid", "next_action_count", "next_action_diff"]]
    df_action1_action2_count = df_actions.groupby("userid", as_index=False)["next_action_count"].agg({
        "{}_{}_action_count_mean".format(action1, action2): np.mean,
        "{}_{}_action_count_max".format(action1, action2): np.max,
        "{}_{}_action_count_min".format(action1, action2): np.min,
        "{}_{}_action_count_std".format(action1, action2): np.std,
    })
    df_action1_action2_time = df_actions.groupby("userid", as_index=False)["next_action_diff"].agg({
        "{}_{}_action_diff_mean".format(action1, action2): np.mean,
        "{}_{}_action_diff_max".format(action1, action2): np.max,
        "{}_{}_action_diff_min".format(action1, action2): np.min,
        "{}_{}_action_diff_std".format(action1, action2): np.std
    })
    df_last_a1_a2_action = df_actions.groupby("userid", as_index=False).tail(1).copy()
    df_last_a1_a2_action.rename(columns={
        "next_action_count": "last_{}_{}_action_count".format(action1, action2),
        "next_action_diff": "last_{}_{}_action_diff".format(action1, action2)
    }, inplace=True)

    df_action_pair_feat = pd.merge(df_action_pair_feat, df_action1_action2_count, on='userid', how='left')
    df_action_pair_feat = pd.merge(df_action_pair_feat, df_action1_action2_time, on='userid', how='left')
    df_action_pair_feat = pd.merge(df_action_pair_feat, df_last_a1_a2_action, on='userid', how='left')
    df_action_pair_feat['last_count_divide_mean_{}_{}'.format(action1, action2)] = df_action_pair_feat[
                                                                                       "last_{}_{}_action_count".format(
                                                                                           action1, action2)] / \
                                                                                   df_action_pair_feat[
                                                                                       "{}_{}_action_count_mean".format(
                                                                                           action1, action2)]

    df_action_pair_feat['last_diff_divide_mean_{}_{}'.format(action1, action2)] = df_action_pair_feat[
                                                                                      "last_{}_{}_action_diff".format(
                                                                                          action1, action2)] / \
                                                                                  df_action_pair_feat[
                                                                                      "{}_{}_action_diff_mean".format(
                                                                                          action1, action2)]
    return df_action_pair_feat


def get_action_pair_count(df_actions, action1, action2):
    """统计用户出现 (action1, action2) 连续动作的次数（TODO：和时间间隔）。
    TODO: 划分时间窗口后统计，如最近一天内的统计。
    """
    df_actions['next_userid'] = df_actions.userid.shift(-1)
    df_actions['next_actionType'] = df_actions.actionType.shift(-1)
    df_actions['next_actionTime'] = df_actions.actionTime.shift(-1)
    df_actions = df_actions.loc[df_actions.next_userid == df_actions.userid]
    df_actions = df_actions.loc[(df_actions.actionType == action1) & (df_actions.next_actionType == action2)]
    df_actions['diff'] = df_actions['next_actionTime'] - df_actions['actionTime']
    df_actions['action_{}_{}_count'.format(action1, action2)] = 1
    # 统计动作出现的次数
    grouped_user = df_actions.groupby('userid', as_index=False)
    df_action_pair_count = grouped_user['action_{}_{}_count'.format(action1, action2)].count()
    # 统计时间间隔
    df_action_pair_diff = grouped_user['diff'].agg(
        {'pair_count_action_diff_mean_{}_{}'.format(action1, action2): np.mean,
         'pair_count_action_diff_std_{}_{}'.format(action1, action2): np.std})
    df_action_pair_count_feats = pd.merge(df_action_pair_count, df_action_pair_diff, on='userid', how='left')
    return df_action_pair_count_feats


def get_pair_rate(df_actions):
    """获取不同动作组合的比例。
    统计每种动作最后一次出现距离最后的时间。
    """
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    df_end_time = df_actions[['userid', 'actionTime']].groupby('userid', as_index=False).tail(1)
    df_end_time.rename(columns={'actionTime': 'end_time'}, inplace=True)
    df_actions = pd.merge(df_actions, df_end_time, on='userid', how='left')
    df_actions.actionType = df_actions.actionType.apply(merge_action)  # 动作归类
    df_actions['last_userid'] = df_actions.userid.shift(1)
    df_actions['last_actionType'] = df_actions.actionType.shift(1)
    df_actions = df_actions.loc[df_actions.last_userid == df_actions.userid].copy()
    action_type = df_actions.actionType.values.astype(int)
    last_action_type = df_actions.last_actionType.values.astype(int)
    action_pairs = list(zip(last_action_type, action_type))
    action_pairs = list(map(lambda s: str(s[0]) + '_' + str(s[1]), action_pairs))
    df_actions['action_pair'] = action_pairs
    df_actions['diff_to_end'] = df_actions['end_time'] - df_actions['actionTime']
    # 统计每种动作最后出现的时间
    df_last_pair_feat = df_actions.drop_duplicates(['userid'])[['userid']].copy()
    action_pair_types = set(action_pairs)
    for action_pair in action_pair_types:
        df_actions_ = df_actions[df_actions.action_pair == action_pair][['userid', 'diff_to_end']].copy()
        df_last_pair_to_end = df_actions_.groupby('userid', as_index=False).tail(1)
        df_last_pair_to_end.rename(columns={'diff_to_end': 'pair_to_end_{}'.format(action_pair)}, inplace=True)
        df_last_pair_feat = pd.merge(df_last_pair_feat, df_last_pair_to_end, on='userid', how='left')
    # 统计每种动作组合出现的频次
    df_action_pair = pd.get_dummies(df_actions[['userid', 'action_pair']], columns=['action_pair'],
                                    prefix='action_pair')
    df_action_pair['total_count'] = 1
    grouped_user = df_action_pair.groupby('userid', as_index=False)
    df_pair_rate = grouped_user.sum()
    pair_columns = list(filter(lambda x: x.startswith('action_pair'), df_pair_rate.columns.values))
    for pair_column in pair_columns:
        df_pair_rate[pair_column] = df_pair_rate[pair_column] / df_pair_rate['total_count']
    df_pair_rate.drop(['total_count'], axis=1, inplace=True)
    df_pair_rate_feat = pd.merge(df_pair_rate, df_last_pair_feat, on='userid', how='left')
    return df_pair_rate_feat


def get_action_feat(df_actions,
                    dict_last_action_rate_0, dict_last_action_rate_1,
                    dict_last_pair_rate_0, dict_last_pair_rate_1,
                    dict_last_triple_rate_0, dict_last_triple_rate_1):
    """获取所有的动作相关的特征。"""
    # **1.所有动作的比例
    df_action_feat = get_action_ratio(df_actions.copy(), n_seconds=None)
    # **2.最早一次行为的时间和最晚一次行为的时间，还有两次行为的时间距离
    df_first_last_action_time = get_first_last_action_time(df_actions.copy())  # 最早和最后一次动作
    df_action_feat = pd.merge(df_action_feat, df_first_last_action_time, on='userid', how='left')
    # **3.动作 action_type 的最早一次时间和最后一次时间，还有两次时间差
    for action_type in range(1, 10):
        df_each_action_feat = get_each_action_last_time(df_actions.copy(), action_type=action_type)
        df_action_feat = pd.merge(df_action_feat, df_each_action_feat.copy(), on='userid', how='left')
    # **4.动作 action_type 到用户最后一次动作的时间距离
    for action_type in range(1, 10):
        df_action_to_end_diff = get_each_action_to_end_diff(df_actions.copy(), action_type=action_type)
        df_action_feat = pd.merge(df_action_feat, df_action_to_end_diff.copy(), on='userid', how='left')
    # **5.最后 10 个动作之间的时间间隔
    df_last_diffs = get_last_diff(df_actions.copy(), n_last_diff=10)
    df_action_feat = pd.merge(df_action_feat, df_last_diffs, on='userid', how='left')

    # n_last_diffs = [3, 10, 15, 20, None]
    # for n_last_diff in n_last_diffs:
    #     df_last_diff_statistic = get_last_diff_statistic(df_actions, n_last_diff=n_last_diff)
    #     df_action_feat = pd.merge(df_action_feat, df_last_diff_statistic, on='userid', how='left')

    # # **6.所有动作之间时间间隔的统计特征
    # df_last_diff_statistic_all = get_last_diff_statistic(df_actions, n_last_diff=None)
    # df_action_feat = pd.merge(df_action_feat, df_last_diff_statistic_all, on='userid', how='left')

    # **6.所有动作之间时间间隔的统计特征
    df_last_diff_statistic_3 = get_last_diff_statistic(df_actions.copy(), n_last_diff=3)
    df_action_feat = pd.merge(df_action_feat, df_last_diff_statistic_3, on='userid', how='left')

    # **7.最后 10 个动作之间时间间隔的统计特征
    df_last_diff_statistic_10 = get_last_diff_statistic(df_actions.copy(), n_last_diff=10)
    df_action_feat = pd.merge(df_action_feat, df_last_diff_statistic_10, on='userid', how='left')

    # 最后三个动作和是个动作间隔比值
    df_last_diff_statistic_3_10 = get_last_diff_divide_statistic(df_last_diff_statistic_3, df_last_diff_statistic_10, 3,
                                                                 10)
    df_action_feat = pd.merge(df_action_feat, df_last_diff_statistic_3_10, on='userid', how='left')

    # **8.最后n(=1)个动作的类型
    df_last_action_types = get_last_action_type(df_actions.copy(), n_action=1, to_one_hot=False)
    df_action_feat = pd.merge(df_action_feat, df_last_action_types, on='userid', how='left')

    # **最后的一个动作转为对应的概率：
    df_action_feat['last_action_type_rate_0'] = df_action_feat['last_action_type'].apply(
        lambda s: dict_last_action_rate_0.get(s, 0))
    df_action_feat['last_action_type_rate_1'] = df_action_feat['last_action_type'].apply(
        lambda s: dict_last_action_rate_1.get(s, 0))
    # df_action_feat['last_action_type_convert_rate'] = df_action_feat['last_action_type_rate_1'] / (
    #     df_action_feat['last_action_type_rate_1'] + df_action_feat['last_action_type_rate_0'])
    # df_action_feat['last_action_type_convert_rate'] = df_action_feat['last_action_type_convert_rate'].apply(
    #     lambda x: 0 if x == np.inf else x)

    # **最后的一组动作及其对应的转化概率：
    df_last_pair_rate = get_last_pair(df_actions.copy(), dict_last_pair_rate_0, dict_last_pair_rate_1)
    df_action_feat = pd.merge(df_action_feat, df_last_pair_rate, on='userid', how='left')

    # **最后的3个动作组合及其转化概率：
    df_last_triple_rate = get_last_triple(df_actions.copy(), dict_last_triple_rate_0, dict_last_triple_rate_1)
    df_action_feat = pd.merge(df_action_feat, df_last_triple_rate, on='userid', how='left')

    # **9.最近一个动作（action_type，如 6） 后面的统计特征
    for action_type in range(1, 10):
        df_after_action_feats = get_after_action_feat(df_actions.copy(), action_type, n_diff=2)
        df_action_feat = pd.merge(df_action_feat, df_after_action_feats.copy(), on='userid', how='left')
    # **10.获取两个特定动作之间的时间距离
    # action_pairs = [[5, 6], [6, 7], [7, 8], [5, 5], [6, 6], [7, 7], [5, 9]]
    action_pairs = [[5, 5], [5, 6], [1, 1], [6, 6], [6, 7], [5, 7], [7, 7]]
    for action1, action2 in action_pairs:
        df_action1_action2_feats = get_action_pair_feats(df_actions.copy(), action1, action2)
        df_action_feat = pd.merge(df_action_feat, df_action1_action2_feats.copy(), on='userid', how='left')
    # # **11.统计用户出现 (action1, action2) 连续动作的次数
    # action_pairs = [[5, 6]]
    # for action1, action2 in action_pairs:
    #     df_action_pair_count_feats = get_action_pair_count(df_actions.copy(), action1, action2)
    #     df_action_feat = pd.merge(df_action_feat, df_action_pair_count_feats.copy(), on='userid', how='left')

    # **12.将 2-4 类动作全部替换成 '2' 统计 '2' 和其他动作之间的时间距离
    df_actions_24 = df_actions.copy()
    df_actions_24.actionType = df_actions_24.actionType.apply(lambda x: x if x not in [2, 3, 4] else 2)
    action_pairs = [[2, 5], [2, 6]]
    for action1, action2 in action_pairs:
        df_action1_action2_feats = get_action_pair_feats(df_actions_24, action1, action2)
        df_action_feat = pd.merge(df_action_feat, df_action1_action2_feats.copy(), on='userid', how='left')

    # 最后一个动作的年月日，时分
    action_types = ['all', 1, 5, 6, 7, 8, 2, 3, 4, 9]
    for action_type in action_types:
        df_last_action_date = get_last_action_date_time(df_actions.copy(), action_type).copy()
        df_action_feat = pd.merge(df_action_feat, df_last_action_date, on='userid', how='left')

    # 获取各种动作组合的比例
    df_pair_rate = get_pair_rate(df_actions.copy())
    df_action_feat = pd.merge(df_action_feat, df_pair_rate, on='userid', how='left')
    return df_action_feat
