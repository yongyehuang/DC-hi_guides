# -*- coding:utf-8 -*- 

from __future__ import print_function
from __future__ import division

from data_helper import *

import catboost as cgb
from sklearn.model_selection import train_test_split
import time
import pickle
import logging.handlers

"""Train the lightGBM model."""

LOG_FILE = 'log/cgb_train.log'
check_path(LOG_FILE)
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('train')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def cgb_fit(config, X_train, y_train):
    """模型（交叉验证）训练，并返回最优迭代次数和最优的结果。
    Args:
        config: xgb 模型参数 {params, max_round, cv_folds, early_stop_round, seed, save_model_path}
        X_train：array like, shape = n_sample * n_feature
        y_train:  shape = n_sample * 1

    Returns:
        best_model: 训练好的最优模型
        best_auc: float, 在测试集上面的 AUC 值。
        best_round: int, 最优迭代次数。
    """
    params = config.params
    max_round = config.max_round
    cv_folds = config.cv_folds
    seed = config.seed
    save_model_path = config.save_model_path
    if cv_folds is not None:
        dtrain = cgb.Pool(X_train, label=y_train)
        cv_result = cgb.cv(dtrain, params, num_boost_round=max_round, nfold=cv_folds, seed=seed, logging_level='Verbose')
        # 最优模型，最优迭代次数
        auc_test_avg = cv_result['AUC_test_avg']
        best_round = np.argmax(auc_test_avg)
        best_auc = np.max(auc_test_avg)  # 最好的 auc 值
        best_model = cgb.train(dtrain, params, num_boost_round=best_round)
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=100)
        dtrain = cgb.Pool(X_train, label=y_train)
        dvalid = cgb.Pool(X_valid, label=y_valid)
        best_model = cgb.train(params, dtrain, num_boost_round=max_round, eval_set=dvalid)
        best_round = best_model.best_iteration
        best_auc = best_model.best_score
        cv_result = None
    if save_model_path:
        check_path(save_model_path)
        pickle.dump(best_model, open(save_model_path, 'wb'))
    return best_model, best_auc, best_round, cv_result


def cgb_predict(model, X_test, save_result_path=None):
    y_pred_prob = model.predict(X_test, prediction_type='Probability')
    y_pred_prob = y_pred_prob[:, 1]  # get the probability of class 1
    if save_result_path:
        df_result = df_future_test
        df_result['orderType'] = y_pred_prob
        df_result.to_csv(save_result_path, index=False)
        print('Save the result to {}'.format(save_result_path))
    return y_pred_prob


class Config(object):
    def __init__(self):
        self.params = {
            'learning_rate': 0.05,
            'eval_metric': 'AUC',
            'depth': 8,
            'logging_level': 'Info',
            'loss_function': 'Logloss',
            'train_dir': 'model/cgb_record/',
            'thread_count': 6
        }
        check_path(self.params['train_dir'])
        self.max_round = 1000
        self.cv_folds = 5
        self.seed = 3
        self.save_model_path = 'model/cgb.model'


if __name__ == '__main__':
    # get feature
    feature_path = 'features/'
    train_data, test_data = load_feat(re_get=False, feature_path=feature_path)
    train_feats = train_data.columns.values
    test_feats = test_data.columns.values
    drop_oolumns = list(filter(lambda x: x not in test_feats, train_feats))
    X_train = train_data.drop(drop_oolumns, axis=1)
    y_train = train_data['label']
    X_test = test_data
    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)

    config = Config()
    # train model
    # tic = time.time()
    # cgb_model, best_auc, best_round, cv_result = cgb_fit(config, X_train, y_train)
    # print('Time cost {}s'.format(time.time() - tic))
    # result_message = 'best_round={}, best_auc={}'.format(best_round, best_auc)
    # logger.info(result_message)
    # print(result_message)

    # predict
    cgb_model = pickle.load(open(config.save_model_path, 'rb'))
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_cgb_{}.csv'.format(now)
    check_path(result_path)
    cgb_predict(cgb_model, X_test, result_path)
