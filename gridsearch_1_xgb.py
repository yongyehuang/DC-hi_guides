# -*- coding:utf-8 -*- 

from __future__ import print_function
from __future__ import division

from data_helper import *
from my_utils import feature_analyze, get_grid_params
from m1_xgb import xgb_fit, xgb_predict

from sklearn.externals import joblib
import time
import logging.handlers

"""Grid search for the best parameters.
And save the best model, predict the result using the best parameters.
"""

LOG_FILE = 'log/xgb_grid_search.log'
check_path(LOG_FILE)
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('search')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def my_search(config, search_params, model_fit, X_train, y_train):
    """Search for best params. 逐个参数搜索，并不是 grid search， 所以应该注意参数的顺序。"""
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
        self.params = {'learning_rate': 0.025,
                       'eval_metric': 'auc',
                       'n_estimators': 5000,
                       'max_depth': 5,
                       'min_child_weight': 7,
                       'gamma': 0,
                       'subsample': 0.8,
                       'colsample_bytree': 0.6,
                       'eta': 0.05,
                       'silent': 1,
                       'objective': 'binary:logistic',
                       'scale_pos_weight': 1}
        self.max_round = 5000
        self.cv_folds = 5
        self.early_stop_round = 30
        self.seed = 3
        self.save_model_path = None
        self.best_model_path = 'model/best_xgb_model.dat'


if __name__ == '__main__':
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
    logger.info(data_message)

    tic = time.time()
    config = Config()
    search_params = {'learning_rate': [0.01, 0.025, 0.05],
                     'max_depth': [5, 6, 7, 8],
                     'subsample': [0.6, 0.8],
                     'colsample_bytree': [0.5, 0.6, 0.8]}
    logger.info(search_params)
    # best_model, config, best_auc = my_search(config, search_params, xgb_fit, X_train, y_train)  # my simple search
    best_model, best_params, best_auc = my_grid_search(config, search_params, xgb_fit, X_train, y_train)  # my grid search
    print('Time cost {}s'.format(time.time() - tic))
    check_path(config.best_model_path)
    joblib.dump(best_model, config.best_model_path)

    # predict
    # best_model = joblib.load(config.best_model_path)  # load the trained model
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_xgb_search_{}.csv'.format(now)
    check_path(result_path)
    xgb_predict(best_model, X_test, result_path)

    # feature analyze
    feature_score_path = 'model/xgb_search_feature_score.csv'
    check_path(feature_score_path)
    feature_analyze(best_model, csv_path=feature_score_path)
