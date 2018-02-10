# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division

from m1_xgb import xgb_fit, xgb_predict
from m1_xgb import Config as XGB_Config
from m2_lgb import lgb_fit, lgb_predict
from m2_lgb import Config as LGB_Config
from m3_cgb import cgb_fit, cgb_predict
from m3_cgb import Config as CGB_Config
from data_helper import *

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.svm import SVR

import logging.handlers
import time

"""Model stacking.
my_stacking: 每个单模型需要具备两个函数：
- 1.model_fit, 返回 best model;
- 2.返回的 best_model 具有 predict 预测类别概率。
- sklearn_stacking: 所有 base_model 都是 sklearn 中的模型，这样具有统一的 fit, perdict 接口。
"""

LOG_FILE = 'log/stacking.log'
check_path(LOG_FILE)
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('stack')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

SEED = 3


def my_stacking(fit_funcs, predict_funcs, configs, X_train, y_train, X_test, n_fold=5):
    """Stacking for my customized models, like xgb, lgb, cgb.
    For each model, you should specify the fit, predict functions.
    Args:
        fit_funcs: return the best model
        predict_funcs: return the probability of the positive class
        configs: the config for each model.
        X_train: shape=[n_sample_train, n_feats], feature for training data
        y_train: shape=[n_sample_train, 1], labels for training data.
        X_test: shape=[n_sample_test, n_feats]feature for testing data.
        n_fold: n_fold for cv.
        save_path: the path to save stack features.
    Returns:
        X_train_stack: shape=[n_sample_train, n_model]
        y_train_stack: shape=[n_sample_test, 1]
        X_test_stack: shape=[n_sample_test, n_model]
    """
    # df_lgb_feat_score = pd.read_csv('features/lgb_features.csv')
    # feature_names = df_lgb_feat_score.feature.values
    # y_train = y_train.values
    if type(X_train) == pd.DataFrame:
        X_train = X_train.values
    if type(X_test) == pd.DataFrame:
        X_test = X_test.values
    if (type(y_train) == pd.DataFrame) | (type(y_train) == pd.Series):
        y_train = y_train.values
    n_train = len(X_train)
    n_test = len(X_test)
    n_model = len(fit_funcs)
    # shuffle the training data first
    new_idx = np.random.permutation(n_train)
    y_train = y_train[new_idx]
    X_train = X_train[new_idx]
    print('X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape))
    kf = KFold(n_splits=n_fold, shuffle=False)

    X_train_stack = None
    X_test_stack = None
    tic = time.time()
    for k in range(n_model):
        message = 'Training model {}/{}, pass {}s.'.format(k + 1, n_model, time.time() - tic)
        print(message)
        logger.info(message)
        fit_func = fit_funcs[k]
        predict_func = predict_funcs[k]
        config = configs[k]
        oof_train = np.zeros((n_train,))
        oof_test_skf = np.zeros((n_test, n_fold))
        for i, (train_idx, test_idx) in enumerate(kf.split(X_train)):
            X_tr = X_train[train_idx]
            y_tr = y_train[train_idx]
            X_te = X_train[test_idx]
            best_model, best_auc, _, _ = fit_func(config, X_tr, y_tr)
            message = 'Fished fold {}/{}, auc={}'.format(i + 1, n_fold, best_auc)
            logger.info(message)
            y_pred_prob = predict_func(best_model, X_te)
            oof_train[test_idx] = y_pred_prob
            oof_test_skf[:, i] = predict_func(best_model, X_test)
        oof_train = oof_train.reshape(-1, 1)
        oof_test = np.mean(oof_test_skf, axis=1).reshape(-1, 1)
        if X_train_stack is None:  # the first model
            X_train_stack = oof_train
            X_test_stack = oof_test
        else:
            X_train_stack = np.hstack((X_train_stack, oof_train))
            X_test_stack = np.hstack((X_test_stack, oof_test))
        stack_feats_path = 'features/stack_feats/round_{}.npz'.format(k + 1)  # 训练过程中进行保存
        check_path(stack_feats_path)
        np.savez(stack_feats_path, X_train=X_train_stack, y_train=y_train, X_test=X_test_stack)
    message = 'X_train_stack.shape={}, X_test_stack.shape={}'.format(X_train_stack.shape, X_test_stack.shape)
    print(message)
    logger.info(message)
    save_path = 'features/stack_feat_{}.npz'.format(time.strftime("%m%d-%H%M%S"))
    check_path(save_path)
    np.savez(save_path, X_train=X_train_stack, y_train=y_train, X_test=X_test_stack)
    message = 'Finished stacking, saved the features to {}'.format(save_path)
    logger.info(message)
    print(message)
    return X_train_stack, y_train, X_test_stack


def sklearn_stacking(base_models, X_train, y_train, X_test, n_fold=5, save_path='features/sklearn_stack_feat.npz'):
    """Stacking for sklearn models."""
    pass
    if type(X_train) == pd.DataFrame:
        X_train = X_train.values
    if type(X_test) == pd.DataFrame:
        X_test = X_test.values
    if (type(y_train) == pd.DataFrame) | (type(y_train) == pd.Series):
        y_train = y_train.values
    n_train = len(X_train)
    n_test = len(X_test)
    n_model = len(base_models)
    # shuffle the training data first
    new_idx = np.random.permutation(n_train)
    X_train = X_train[new_idx]
    y_train = y_train[new_idx]
    print('X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape))
    kf = KFold(n_splits=n_fold, shuffle=False)

    X_train_stack = None
    X_test_stack = None
    tic = time.time()
    for k in range(n_model):
        message = 'Training model {}/{}, pass {}s.'.format(k + 1, n_model, time.time() - tic)
        print(message)
        logger.info(message)
        oof_train = np.zeros((n_train,))
        oof_test_skf = np.zeros((n_test, n_fold))
        model = base_models[k]
        for i, (train_idx, test_idx) in enumerate(kf.split(X_train)):
            X_tr = X_train[train_idx]
            y_tr = y_train[train_idx]
            X_te = X_train[test_idx]
            model.fit(X_tr, y_tr)
            message = 'Fished fold {}/{}, pass {}s'.format(i + 1, n_fold, time.time() - tic)
            print(message)
            logger.info(message)
            y_pred_prob = model.predict_proba(X_te)[:, 1]
            oof_train[test_idx] = y_pred_prob
            oof_test_skf[:, i] = model.predict_proba(X_test)[:, 1]
        oof_train = oof_train.reshape(-1, 1)
        oof_test = np.mean(oof_test_skf, axis=1).reshape(-1, 1)
        if X_train_stack is None:  # the first model
            X_train_stack = oof_train
            X_test_stack = oof_test
        else:
            X_train_stack = np.hstack((X_train_stack, oof_train))
            X_test_stack = np.hstack((X_test_stack, oof_test))
        stack_feats_path = 'features/stack_feats/round_{}.npz'.format(k + 1)  # 训练过程中进行保存
        check_path(stack_feats_path)
        np.savez(stack_feats_path, X_train=X_train_stack, y_train=y_train, X_test=X_test_stack)
    message = 'X_train_stack.shape={}, X_test_stack.shape={}'.format(X_train_stack.shape, X_test_stack.shape)
    print(message)
    logger.info(message)
    if save_path:
        check_path(save_path)
        np.savez(save_path, X_train=X_train_stack, y_train=y_train, X_test=X_test_stack)
        message = 'Finished stacking, saved the features to {}'.format(save_path)
        logger.info(message)
        print(message)
    return X_train_stack, y_train, X_test_stack


def final_fit_predict(X_train, y_train, X_test, save_result_path=None):
    """Final train using the stacking features.
    Using the LogisticRegression as the model for this lever.
    """
    param_grids = {
        "C": list(np.linspace(0.0001, 10, 100))
    }
    print('Begin final fit with params:{}'.format(param_grids))
    grid = GridSearchCV(LogisticRegression(penalty='l2', max_iter=200), param_grid=param_grids, cv=5, scoring="roc_auc")
    grid.fit(X_train, y_train)
    try:
        message = 'Final fit: param_grids is: {};\n best_param is {};\n best cv_score is {};\n best_estimator is {}'.format(
            param_grids, grid.best_params_, grid.best_score_, grid.best_estimator_)
        logger.info(message)
        print(message)
    except Exception as e:
        print(e.message)
    y_pred_prob = grid.predict_proba(X_test)[:, 1]
    if save_result_path is not None:
        df_result = df_future_test
        df_result['orderType'] = y_pred_prob
        df_result.to_csv(save_result_path, index=False)
        print('Save the result to {}'.format(save_result_path))
    return y_pred_prob


def load_features(feature_path='features_lin/'):
    """Loading date."""
    train_data, test_data = load_feat(re_get=False, feature_path=feature_path)
    train_feats = train_data.columns.values
    test_feats = test_data.columns.values
    drop_columns = list(filter(lambda x: x not in test_feats, train_feats))
    X_train = train_data.drop(drop_columns, axis=1)
    y_train = train_data['label']
    X_test = test_data
    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)
    return X_train, y_train, X_test


def run_my_stack():
    """My stacking function test."""

    X_train, y_train, X_test = load_features()
    fit_funcs = list()
    predict_funcs = list()
    configs = list()
    MAX_ROUND = 3

    # lgb
    num_leaves = [31, 41, 51, 61, 71, 81, 91]
    feature_fractions = [0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3]
    for i in range(len(num_leaves)):
        lgb_config = LGB_Config()
        lgb_config.params['num_leaves'] = num_leaves[i]
        lgb_config.params['feature_fraction'] = feature_fractions[i]
        lgb_config.seed = np.random.randint(0, 10000)
        lgb_config.save_model_path = None
        # lgb_config.max_round = MAX_ROUND
        configs.append(lgb_config)
        fit_funcs.append(lgb_fit)
        predict_funcs.append(lgb_predict)

    max_depths = [6, 7]
    colsample_bytrees = [0.7, 0.6]
    for i in range(len(max_depths)):
        xgb_config = XGB_Config()
        xgb_config.params['max_depth'] = max_depths[i]
        xgb_config.params['colsample_bytree'] = colsample_bytrees[i]
        xgb_config.seed = np.random.randint(0, 10000)
        xgb_config.save_model_path = None
        # xgb_config.max_round = MAX_ROUND
        configs.append(xgb_config)
        fit_funcs.append(xgb_fit)
        predict_funcs.append(xgb_predict)

    # cgb
    max_depths = [8]
    for i in range(len(max_depths)):
        cgb_config = CGB_Config()
        cgb_config.params['depth'] = max_depths[i]
        cgb_config.seed = np.random.randint(0, 10000)
        cgb_config.save_model_path = None
        # cgb_config.max_round = MAX_ROUND
        configs.append(cgb_config)
        fit_funcs.append(cgb_fit)
        predict_funcs.append(cgb_predict)

    X_train_stack, y_train_stack, X_test_stack = my_stacking(fit_funcs, predict_funcs, configs, X_train, y_train,
                                                             X_test)
    result_path = 'result/my_stack_result-{}.csv'.format(time.strftime("%m%d-%H%M%S"))
    y_pred_prob = final_fit_predict(X_train_stack, y_train_stack, X_test_stack, save_result_path=result_path)
    return y_pred_prob


def run_sklearn_stack():
    """Stacking with sklearn model test."""
    X_train, y_train, X_test = load_features()
    base_models = [
        XGBClassifier(learning_rate=0.05,
                      eval_metric='auc',
                      # n_estimators=712,  # 750
                      n_estimators=7,  # 750
                      max_depth=5,
                      min_child_weight=7,
                      gamma=0,
                      subsample=0.8,
                      colsample_bytree=0.6,
                      eta=0.05,
                      silent=1,
                      seed=3,
                      objective='binary:logistic',
                      scale_pos_weight=1),
        LGBMClassifier(num_leaves=31,
                       learning_rate=0.05,
                       # n_estimators=543,  # 443
                       n_estimators=5,  # 443
                       objective='binary',
                       metric={'auc'},
                       seed=3,
                       colsample_bytree=0.8,
                       min_child_weight=7,
                       subsample=0.8,
                       silent=1),
        CatBoostClassifier(iterations=5,
                           learning_rate=0.05,
                           eval_metric='AUC',
                           depth=8
                           ),
    ]
    X_train_stack, y_train_stack, X_test_stack = sklearn_stacking(base_models, X_train, y_train, X_test, n_fold=5)
    result_path = 'result/sklearn_stack_result-{}.csv'.format(time.strftime("%m%d-%H%M%S"))
    check_path(result_path)
    y_pred_prob = final_fit_predict(X_train_stack, y_train_stack, X_test_stack, save_result_path=result_path)
    return y_pred_prob


if __name__ == '__main__':
    run_my_stack()
    # run_sklearn_stack()
