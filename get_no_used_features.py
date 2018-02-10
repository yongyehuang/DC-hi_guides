# -*- coding:utf-8 -*- 

from __future__ import print_function
from __future__ import division

import pandas as pd
import lightgbm as lgb
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


def get_xgb_features(data_path='features/test_data.csv',
                     feature_scores_path='features/xgb_feature_score.csv',
                     xgb_features_path='features/xgb_features.csv'):
    """统计没有用到的特征"""
    df_feature_score = pd.read_csv(feature_scores_path)
    df_test_data = pd.read_csv(data_path)
    all_features = df_test_data.columns.values
    used_features = df_feature_score.feature.values
    print('n_all_features={}, n_feature_used={}'.format(len(all_features), len(used_features)))
    no_used_features = list(set(all_features).difference(set(used_features)))
    n_no_used = len(no_used_features)
    print('n_no_used_feature={}'.format(n_no_used))

    df_no_used = pd.DataFrame({'feature': no_used_features, 'score': [0] * n_no_used})
    df_xgb_features = pd.concat([df_feature_score, df_no_used])
    df_xgb_features.to_csv(xgb_features_path, index=False)


def get_lgb_features(lgb_model_path='model/lgb.txt', lgb_feature_path='features/lgb_features.csv'):
    """获取 lgb 的特征重要度"""
    lgb_model = lgb.Booster(model_file=lgb_model_path)
    feature_names = lgb_model.feature_name()
    feature_importances = lgb_model.feature_importance()
    df_lgb_features = pd.DataFrame({'feature': feature_names, 'scores': feature_importances})
    df_lgb_features = df_lgb_features.sort_values('scores', ascending=False)
    df_lgb_features.to_csv(lgb_feature_path, index=False)


if __name__ == '__main__':
    get_xgb_features()
    get_lgb_features(lgb_model_path='model/lgb.txt')
