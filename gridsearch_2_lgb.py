# -*- coding:utf-8 -*- 

from __future__ import print_function
from __future__ import division

from data_helper import *
from my_utils import get_grid_params
from m2_lgb import lgb_fit, lgb_predict

import time

import logging.handlers

"""Grid search for the best parameters.
And save the best model, predict the result using the best parameters.
"""

LOG_FILE = 'log/lgb_grid_search.log'
check_path(LOG_FILE)
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('search')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def my_search(config, search_params, model_fit, X_train, y_train):
    """Simple search. 逐个参数搜索，并不是 grid search， 所以应该注意参数的顺序。"""
    best_auc = 0.0
    best_model = None
    for k, vs in search_params.items():
        best_v = vs[0]  # the best vs
        for v in vs:
            config.params[k] = v
            _model, _auc, _round, _ = model_fit(config, X_train, y_train)
            if _auc > best_auc:  # find the better param
                best_v = v
                best_auc = _auc
                best_model = _model
            message = 'Best_auc={}; {}={}, auc={}, round={}'.format(best_auc, k, v, _auc, _round)
            logger.info(message)
            print(message)
        config.params[k] = best_v  # set the best params
    search_message = 'Finished Search! Best auc={}. Best params are \n{}'.format(best_auc, config.params)
    logger.info(search_message)
    print(search_message)
    return best_model, config, best_auc


def my_grid_search(config, search_params, model_fit, X_train, y_train):
    """grid search."""
    best_auc = 0.0
    best_model = None
    best_params = None
    grid_params = get_grid_params(search_params)
    message = 'Begin grid searching. Params group number is {}'.format(len(grid_params))
    print(message)
    logger.info(message)
    for i in range(len(grid_params)):
        dict_params = grid_params[i]
        logger.info('Searching {}/{}'.format(i+1, len(grid_params)))
        for k, v in dict_params.items():
            config.params[k] = v
        _model, _auc, _round, _ = model_fit(config, X_train, y_train)
        if _auc >= best_auc:  # find the better param
            best_params = dict_params.copy()
            best_auc = _auc
            best_model = _model
        message = 'Best_auc={}; auc={}, round={}, params={}'.format(best_auc, _auc, _round, dict_params)
        logger.info(message)
        print(message)
    search_message = 'Finished Search! Best auc={}. Best params are \n{}'.format(best_auc, best_params)
    logger.info(search_message)
    print(search_message)
    return best_model, best_params, best_auc


class Config(object):
    def __init__(self):
        self.params = {
            'objective': 'binary',
            'metric': {'auc'},
            'learning_rate': 0.05,
            'num_leaves': 30,  # 叶子设置为 50 线下过拟合严重
            'min_sum_hessian_in_leaf': 0.1,
            'feature_fraction': 0.3,  # 相当于 colsample_bytree
            'bagging_fraction': 0.5,  # 相当于 subsample
            'lambda_l1': 0,
            'lambda_l2': 5,
        }
        self.max_round = 5000
        self.cv_folds = 5
        self.early_stop_round =30
        self.seed = 3
        self.save_model_path = None
        self.best_model_path = 'model/best_lgb_model.txt'


if __name__ == '__main__':
    # get feature
    feature_path = 'features/'
    train_data, test_data = load_feat(re_get=False, feature_path=feature_path)
    train_feats = train_data.columns
    test_feats = test_data.columns
    drop_oolumns = list(filter(lambda x: x not in test_feats, train_feats))
    X_train = train_data.drop(drop_oolumns, axis=1).iloc[:, :50]
    y_train = train_data['label']
    X_test = test_data.iloc[:, :50]
    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)

    # grid search
    tic = time.time()
    config = Config()
    search_params = {'num_leaves': [20, 30, 40, 50],
                     'learning_rate': [0.025, 0.05, 0.1, 0.15, 0.20]}
    logger.info(search_params)
    # best_model, config, best_auc = my_search(config, search_params, lgb_fit, X_train, y_train)  # my simple search
    best_model, best_params, best_auc = my_grid_search(config, search_params, lgb_fit, X_train, y_train)  # my grid search
    print('Time cost {}s'.format(time.time() - tic))
    check_path(config.best_model_path)
    best_model.save_model(config.best_model_path)

    # predict
    # best_model = joblib.load(config.best_model_path)  # load the trained model
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_lgb_search_{}.csv'.format(now)
    check_path(result_path)
    lgb_predict(best_model, X_test, result_path)
