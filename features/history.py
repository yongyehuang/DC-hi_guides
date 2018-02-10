# -*- coding:utf-8 -*- 

from __future__ import print_function
from __future__ import division

from my_utils import time_to_date

import pandas as pd
import numpy as np


def map_country(country):
    """根据每个国家的精品率进行分组合并不同国家"""
    if country in [u'斐济', u'中国香港', u'新加坡']:
        return 1
    if country in [u'爱尔兰',u'澳大利亚', u'泰国', u'越南', u'马来西亚']:
        return 2
    if country in [u'法国', u'英国', u'菲律宾', u'加拿大', u'柬埔寨', u'新西兰', u'美国']:
        return 3
    if country in [u'日本', u'韩国', u'中国台湾']:
        return 4
    if country in [u'印度尼西亚', u'阿联酋', u'中国澳门', u'荷兰', u'挪威', u'希腊']:
        return 5
    if country in [u'缅甸', u'比利时', u'土耳其', u'瑞典', u'埃及', u'德国', u'芬兰', u'西班牙', u'意大利']:
        return 6
    if country in [u'波兰', u'冰岛', u'巴西', u'毛里求斯', u'瑞士', u'丹麦', u'奥地利', u'匈牙利', u'葡萄牙', u'捷克', u'俄罗斯']:
        return 7
    else: # [u'老挝', u'摩洛哥', u'尼泊尔', u'墨西哥', u'卡塔尔', u'南非', u'中国']
        return 0


def map_continent(continent):
    if continent in [u'大洋洲', u'北美洲']:
        return 0
    if continent in [u'亚洲']:
        return 1
    else: #[u'南美洲', u'非洲', u'欧洲']
        return 2


def get_first_last_order_feat(df_history):
    """获取最早一单的类型和最早一单距离当前时间。
    Args:
        df_history: 历史订单数据。
    Returns:
        first_order_type: 0 or 1，最后一单的类型。
        first_order_time: 最后一单到的时间。
    """
    df_history = df_history.sort_values(by='orderTime', ascending=True)[['userid', 'orderType', 'orderTime']]
    # 最早，最后一单的类型
    grouped_user = df_history.groupby(['userid'], as_index=False)
    df_first_order_type = grouped_user['userid', 'orderType'].head(1)
    df_first_order_type.rename(columns={'orderType': 'first_order_type'}, inplace=True)
    df_last_order_type = grouped_user['userid', 'orderType'].tail(1)
    df_last_order_type.rename(columns={'orderType': 'last_order_type'}, inplace=True)
    df_first_order_feats = pd.merge(df_first_order_type, df_last_order_type, on='userid', how='left')

    # 最早，最后一单的时间
    df_first_order_time = grouped_user['userid', 'orderTime'].head(1)
    df_first_order_time.rename(columns={'orderTime': 'first_order_time'}, inplace=True)
    df_last_order_time = grouped_user['userid', 'orderTime'].tail(1)
    df_last_order_time.rename(columns={'orderTime': 'last_order_time'}, inplace=True)

    # 获取最后一单的年月日
    df_last_order_time['last_order_date'] = df_last_order_time['last_order_time'].apply(time_to_date)
    df_last_order_time['last_order_date'] = pd.to_datetime(df_last_order_time.last_order_date,
                                                           format="%Y-%m-%d %H:%M:%S")
    df_last_order_time['last_order_year'] = df_last_order_time.last_order_date.apply(lambda dt: dt.year)
    df_last_order_time['last_order_month'] = df_last_order_time.last_order_date.apply(lambda dt: dt.month)
    df_last_order_time['last_order_day'] = df_last_order_time.last_order_date.apply(lambda dt: dt.day)
    df_last_order_time['last_order_weekday'] = df_last_order_time.last_order_date.apply(lambda dt: dt.weekday())
    df_last_order_time['last_order_hour'] = df_last_order_time.last_order_date.apply(lambda dt: dt.hour)
    df_last_order_time['last_order_minute'] = df_last_order_time.last_order_date.apply(lambda dt: dt.minute)
    df_last_order_time.drop('last_order_date', axis=1, inplace=True)

    df_first_order_feats = pd.merge(df_first_order_feats, df_first_order_time, on='userid', how='left')
    df_first_order_feats = pd.merge(df_first_order_feats, df_last_order_time, on='userid', how='left')
    return df_first_order_feats


def get_jp_ratio(df_history):
    """统计每个用户下单的数量和购买精品的比例。"""
    df_real_type = df_history.groupby(['userid', 'orderTime'])['orderType'].agg([('real_type', np.sum)]).reset_index()
    df_real_type.real_type = df_real_type.real_type.apply(lambda x: 1 if x > 0 else 0)
    grouped_user = df_real_type[['userid', 'real_type']].groupby('userid', as_index=False)
    df_total_count = grouped_user.count()
    df_total_count.rename(columns={'real_type': 'n_total_order'}, inplace=True)
    df_jp_count = grouped_user.sum()
    df_jp_count.rename(columns={'real_type': 'n_jp_order'}, inplace=True)
    df_jp_ratio_feat = pd.merge(df_total_count, df_jp_count, on='userid', how='left')
    df_jp_ratio_feat['jp_rate'] = df_jp_ratio_feat['n_jp_order'] / df_jp_ratio_feat['n_total_order']
    return df_jp_ratio_feat


def get_country_count(df_history):
    """统计每个用户到每个国家的比例。注意去重,用户同一时间的订单去重."""
    df_history = df_history.drop_duplicates(['userid', 'orderTime']).copy()
    df_history.country = df_history.country.apply(map_country)
    # 每个用户所有的国家和
    df_total_count = df_history[['userid', 'country']].groupby(["userid"], as_index=False).count()
    df_total_count.rename(columns={'country': 'n_total_trip'}, inplace=True)
    # 每个用户每个国家的次数
    df_country_count = df_history.groupby(["userid", "country"])["country"].agg(
        [("country_count", np.size)]).reset_index()
    df_country_count = pd.pivot_table(df_country_count, values="country_count", index="userid",
                                      columns="country").reset_index()
    df_country_count = df_country_count.fillna(0)
    df_country_count.columns = ['userid'] + ['country_{}'.format(i) for i in range(8)]
    df_country_count = pd.merge(df_total_count, df_country_count, on='userid', how='left')
    for i in range(8):
        df_country_count['country_{}'.format(i)] = df_country_count['country_{}'.format(i)] / df_country_count['n_total_trip']
    df_country_count_feat = df_country_count.drop(['n_total_trip'], axis=1)
    return df_country_count_feat


def get_city_count(df_history):
    """统计每个用户到每个城市的次数。"""
    df_city_count_feat = df_history.drop_duplicates('userid')[['userid']]
    df_history = df_history.drop_duplicates(['userid', 'orderTime'])
    df_city_count = df_history.groupby(["userid", "city"])["city"].agg([("city_count", np.size)]).reset_index()
    df_city_count = pd.pivot_table(df_city_count, values="city_count", index="userid", columns="city").reset_index()
    df_city_count = df_city_count.fillna(0)
    df_city_count_feat = pd.merge(df_city_count_feat, df_city_count, on='userid', how='left')
    return df_city_count_feat


def get_continent_count(df_history):
    """统计每个用户到每个国家的比例。注意去重,用户同一时间的订单去重."""
    df_history = df_history.drop_duplicates(['userid', 'orderTime']).copy()
    df_history.continent = df_history.continent.apply(map_continent)
    # 每个用户所有的国家和
    df_total_count = df_history[['userid', 'continent']].groupby(["userid"], as_index=False).count()
    df_total_count.rename(columns={'continent': 'n_total_trip'}, inplace=True)
    # 每个用户每个国家的次数
    df_continent_count = df_history.groupby(["userid", "continent"])["continent"].agg(
        [("continent_count", np.size)]).reset_index()
    df_continent_count = pd.pivot_table(df_continent_count, values="continent_count", index="userid",
                                      columns="continent").reset_index()
    df_continent_count = df_continent_count.fillna(0)
    df_continent_count.columns = ['userid'] + ['continent_{}'.format(i) for i in range(3)]
    df_continent_count = pd.merge(df_total_count, df_continent_count, on='userid', how='left')
    for i in range(3):
        df_continent_count['continent_{}'.format(i)] = df_continent_count['continent_{}'.format(i)] / df_continent_count['n_total_trip']
    df_continent_count_feat = df_continent_count.drop(['n_total_trip'], axis=1)
    return df_continent_count_feat


def get_history_feat(df_history):
    """获取历史订单特征。时间相同的订单，去重处理。
    统计每个用户的下单数量，和每个用户购买精品订单数量,统计每个用户下单时购买精品旅行的比例。
    """
    # 获取精品订单的比例
    df_jp_ratio_feat = get_jp_ratio(df_history)
    df_first_order_feats = get_first_last_order_feat(df_history)
    df_history_feat = pd.merge(df_jp_ratio_feat, df_first_order_feats, on='userid', how='left')
    # 去过的国家次数
    df_country_count_feat = get_country_count(df_history)
    df_history_feat = pd.merge(df_history_feat, df_country_count_feat, on='userid', how='left')
    # 去过的城市次数
    # df_city_count_feat = get_city_count(df_history)
    # df_history_feat = pd.merge(df_history_feat, df_city_count_feat, on='userid', how='left')
    # 去过的洲次数
    df_continent_count_feat = get_continent_count(df_history)
    df_history_feat = pd.merge(df_history_feat, df_continent_count_feat, on='userid', how='left')
    return df_history_feat
