# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division

from data_helper import *
import lightgbm as lgb


def lgb_predict_on_train(model, X_train, save_result_path=None):
    y_pred_prob = model.predict(X_train)
    df_result = df_future_train
    df_result['predict_type'] = y_pred_prob
    df_result.to_csv(save_result_path, index=False)
    print('Save the result to {}'.format(save_result_path))


if __name__ == '__main__':
    save_model_path = 'model/lgb.txt'
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_train_lgb_{}.csv'.format(now)
    check_path(result_path)
    # get feature
    feature_path = 'features/'
    train_data, test_data = load_feat(re_get=False, feature_path=feature_path)
    train_feats = train_data.columns
    test_feats = test_data.columns
    drop_oolumns = list(filter(lambda x: x not in test_feats, train_feats))
    X_train = train_data.drop(drop_oolumns, axis=1)
    y_train = train_data['label']
    X_test = test_data
    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)

    lgb_model = lgb.Booster(model_file=save_model_path)
    lgb_predict_on_train(lgb_model, X_train, result_path)