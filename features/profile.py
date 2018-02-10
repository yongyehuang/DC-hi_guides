# -*- coding:utf-8 -*- 

from __future__ import print_function
from __future__ import division

import pandas as pd

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

"""Model assessment tools.
"""


def map_province(province):
    """合并省份。"""
    if (province == u'北京') | (province == u'上海'):
        return province
    if province in [u'广东', u'四川', u'辽宁', u'浙江', u'湖北', u'江苏', u'重庆', u'湖南']:
        return u'二线'
    else:
        return u'三线'


def get_province(df_profile):
    df_profile.province = df_profile.province.apply(map_province)
    df_province = pd.get_dummies(df_profile[['userid', 'province']], columns=['province'])
    return df_province


def get_gender_rate(df_profile, dict_gender_convert_rate):
    df_profile['gender_rate'] = df_profile['gender'].apply(
        lambda x: dict_gender_convert_rate.get(x, dict_gender_convert_rate['mean']))
    return df_profile[['userid', 'gender_rate']]


def get_province_rate(df_profile, dict_province_convert_rate):
    df_profile['province_rate'] = df_profile['province'].apply(
        lambda x: dict_province_convert_rate.get(x, dict_province_convert_rate['mean']))
    return df_profile[['userid', 'province_rate']]


def get_age_rate(df_profile, dict_age_convert_rate):
    df_profile['age_rate'] = df_profile['age'].apply(
        lambda x: dict_age_convert_rate.get(x, dict_age_convert_rate['mean']))
    return df_profile[['userid', 'age_rate']]


def get_profile_feat(df_profile):
    # def get_profile_feat(df_profile):
    """获取个人信息特征。"""
    df_profile_feat = df_profile.drop_duplicates(['userid'])[['userid']].copy()
    df_province = get_province(df_profile.copy())
    df_profile_feat = pd.merge(df_profile_feat, df_province, on='userid', how='left')
    df_profile_ = df_profile[['userid', 'gender', 'age']].copy()
    df_dummy_feat = pd.get_dummies(df_profile_)
    df_profile_feat = pd.merge(df_profile_feat, df_dummy_feat, on='userid', how='left')
    return df_profile_feat


if __name__ == '__main__':
    df_profile = pd.read_csv('../data/trainingset/userProfile_train.csv')
    df_profile_feat = get_profile_feat(df_profile)
    print(len(df_profile), len(df_profile_feat))
    print(df_profile_feat.head(10))
