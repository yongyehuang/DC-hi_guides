# -*- coding:utf-8 -*- 

from __future__ import print_function
from __future__ import division

from data_helper import *

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import time
import logging.handlers

"""Train the lightGBM model."""

LOG_FILE = 'log/lgb_train.log'
check_path(LOG_FILE)
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('train')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def lgb_fit(config, X_train, y_train):
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
    early_stop_round = config.early_stop_round
    seed = config.seed
    # seed = np.random.randint(0, 10000)
    save_model_path = config.save_model_path
    if cv_folds is not None:
        dtrain = lgb.Dataset(X_train, label=y_train)
        cv_result = lgb.cv(params, dtrain, max_round, nfold=cv_folds, seed=seed, verbose_eval=True,
                           metrics='auc', early_stopping_rounds=early_stop_round, show_stdv=False)
        # 最优模型，最优迭代次数
        best_round = len(cv_result['auc-mean'])
        best_auc = cv_result['auc-mean'][-1]  # 最好的 auc 值
        best_model = lgb.train(params, dtrain, best_round)
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=100)
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)
        watchlist = [dtrain, dvalid]
        best_model = lgb.train(params, dtrain, max_round, valid_sets=watchlist, early_stopping_rounds=early_stop_round)
        best_round = best_model.best_iteration
        best_auc = best_model.best_score
        cv_result = None
    if save_model_path:
        check_path(save_model_path)
        best_model.save_model(save_model_path)
    return best_model, best_auc, best_round, cv_result


def lgb_predict(model, X_test, save_result_path=None):
    y_pred_prob = model.predict(X_test)
    if save_result_path:
        df_result = df_future_test
        df_result['orderType'] = y_pred_prob
        df_result.to_csv(save_result_path, index=False)
        print('Save the result to {}'.format(save_result_path))
    return y_pred_prob


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
            'num_thread': 6  # 线程数设置为真实的 CPU 数，一般12线程的机器有6个物理核
        }
        self.max_round = 3000
        self.cv_folds = 5
        self.early_stop_round = 30
        self.seed = 3
        self.save_model_path = 'model/lgb.txt'


def run_feat_search(X_train, X_test, y_train, feature_names):
    """根据特征重要度，逐个删除特征进行训练，获取最好的特征结果。
    同时，将每次迭代的结果求平均作为预测结果"""
    config = Config()
    # train model
    tic = time.time()
    y_pred_list = list()
    aucs = list()
    for i in range(1, 250, 3):
        drop_cols = feature_names[-i:]
        X_train_ = X_train.drop(drop_cols, axis=1)
        X_test_ = X_test.drop(drop_cols, axis=1)
        data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train_.shape, X_test_.shape)
        print(data_message)
        logger.info(data_message)
        lgb_model, best_auc, best_round, cv_result = lgb_fit(config, X_train_, y_train)
        print('Time cost {}s'.format(time.time() - tic))
        result_message = 'best_round={}, best_auc={}'.format(best_round, best_auc)
        logger.info(result_message)
        print(result_message)

        # predict
        # lgb_model = lgb.Booster(model_file=config.save_model_path)
        now = time.strftime("%m%d-%H%M%S")
        result_path = 'result/result_lgb_{}-{:.4f}.csv'.format(now, best_auc)
        check_path(result_path)
        y_pred = lgb_predict(lgb_model, X_test_, result_path)
        y_pred_list.append(y_pred)
        aucs.append(best_auc)
        y_preds_path = 'stack_preds/lgb_feat_search_pred_{}.npz'.format(i)
        check_path(y_preds_path)
        np.savez(y_preds_path, y_pred_list=y_pred_list, aucs=aucs)
        message = 'Saved y_preds to {}. Best auc is {}'.format(y_preds_path, np.max(aucs))
        logger.info(message)
        print(message)


def run_cv(X_train, X_test, y_train):
    config = Config()
    # train model
    tic = time.time()
    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)
    lgb_model, best_auc, best_round, cv_result = lgb_fit(config, X_train, y_train)
    print('Time cost {}s'.format(time.time() - tic))
    result_message = 'best_round={}, best_auc={}'.format(best_round, best_auc)
    logger.info(result_message)
    print(result_message)
    # predict
    # lgb_model = lgb.Booster(model_file=config.save_model_path)
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_lgb_{}-{:.4f}.csv'.format(now, best_auc)
    check_path(result_path)
    lgb_predict(lgb_model, X_test, result_path)


if __name__ == '__main__':
    # get feature
    feature_path = 'features/'
    train_data, test_data = load_feat(re_get=True, feature_path=feature_path)
    train_feats = train_data.columns.values
    test_feats = test_data.columns.values
    drop_columns = list(filter(lambda x: x not in test_feats, train_feats))
    X_train = train_data.drop(drop_columns, axis=1)
    y_train = train_data['label']
    X_test = test_data
    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)

    # 根据特征搜索中最好的结果丢弃部分特征
    # n_drop_col = 141
    # drop_cols = feature_names[-n_drop_col:]
    # X_train = X_train.drop(drop_cols, axis=1)
    # X_test = X_test.drop(drop_cols, axis=1)
    # 直接训练
    run_cv(X_train, X_test, y_train)

    # 特征搜索
    # get feature scores
    # try:
    #     df_lgb_feat_score = pd.read_csv('features/lgb_features.csv')
    #     feature_names = df_lgb_feat_score.feature.values
    # except Exception as e:
    #     print('You should run the get_no_used_features.py first.')
    # run_feat_search(X_train, X_test, y_train, feature_names)
