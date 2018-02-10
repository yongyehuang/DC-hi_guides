# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from itertools import chain
import time
import os
import sys

# reload(sys)
# sys.setdefaultencoding('utf-8')
#
# """Model assessment tools.
# """


def time_to_date(time_stamp):
    """把时间戳转成日期的形式。"""
    time_array = time.localtime(time_stamp)
    date_style_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    return date_style_time


def get_no_used_features(all_features, used_features, no_used_features_path='features/no_used_features.csv'):
    """统计没有用到的特征"""
    print('n_all_features={}, n_feature_used={}'.format(len(all_features), len(used_features)))
    no_used_features = list(set(all_features).difference(set(used_features)))
    n_no_used = len(no_used_features)
    print('n_no_used_feature={}'.format(n_no_used))
    df_no_used = pd.DataFrame({'no_used': no_used_features})
    df_no_used.to_csv(no_used_features_path)


def get_lgb_features(lgb_model, lgb_feature_path='features/lgb_features.csv'):
    """获取 lgb 的特征重要度"""
    feature_names = lgb_model.feature_name()
    feature_importances = lgb_model.feature_importance()
    df_lgb_features = pd.DataFrame({'feature': feature_names, 'scores': feature_importances})
    df_lgb_features = df_lgb_features.sort_values('scores', ascending=False)
    df_lgb_features.to_csv(lgb_feature_path, index=False)


def params_append(list_params_left, list_param_right):
    if type(list_params_left) is not list:
        list_params_left = list(map(lambda p: list([p]), list_params_left))
    n_left = len(list_params_left)
    n_right = len(list_param_right)
    list_params_left *= n_right
    list_param_right = list(chain([[p] * n_left for p in list_param_right]))
    for i in range(len(list_params_left)):
        list_params_left[i].append(list_param_right[i])
    return list_params_left


def get_grid_params(search_params):
    """遍历 grid search 的所有参数组合。
    Args:
        search_params: dict of params to be search.
        >>> search_params = {'learning_rate': [0.025, 0.05, 0.1, 0.15, 0.20],
                             'max_depth': [4, 5, 6, 7],
                             'colsample_bytree': [0.6, 0.7, 0.8]}
    Returns:
        grid_params: list, 每个元素为一个dict, 对应每次搜索的参数。
    """
    keys = list(search_params.keys())
    values = list(search_params.values())
    grid_params = list()
    if len(keys) == 1:
        for value in values[0]:
            dict_param = dict()
            dict_param[keys[0]] = value
            grid_params.append(dict_param.copy())
        return grid_params
    list_params_left = values[0]
    for i in range(1, len(values)):
        list_param_right = values[i]
        list_params_left = params_append(list_params_left, list_param_right)
    for params in list_params_left:
        dict_param = dict()
        for i in range(len(keys)):
            dict_param[keys[i]] = params[i]
        grid_params.append(dict_param.copy())
    return grid_params


def check_path(_path):
    """Check weather the _path exists. If not, make the dir."""
    if os.path.dirname(_path):
        if not os.path.exists(os.path.dirname(_path)):
            os.makedirs(os.path.dirname(_path))


def print_confusion_matrix(y_true, y_pred):
    """打印分类混淆矩阵。
    Args:
        y_true: 真实类别。
        y_pred: 预测类别。
    """
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    print("confusion_matrix(left labels: y_true, up labels: y_pred):")
    out = "labels\t"
    for i in range(len(labels)):
        out += (str(labels[i]) + "\t")
    print(out)
    for i in range(len(conf_mat)):
        out = (str(labels[i]) + "\t")
        for j in range(len(conf_mat[i])):
            out += (str(conf_mat[i][j]) + '\t')
        print(out)
    return conf_mat


def get_auc(y_true, y_pred_pos_prob, plot_ROC=False):
    """计算 AUC 值。
    Args:
        y_true: 真实标签，如 [0, 1, 1, 1, 0]
        y_pred_pos_prob: 预测每个样本为 positive 的概率。
        plot_ROC: 是否绘制  ROC 曲线。
    Returns:
       roc_auc: AUC 值.
       fpr, tpr, thresholds: see roc_curve.
    """
    fpr, tpr, thresholds = (y_true, y_pred_pos_prob)
    roc_auc = auc(fpr, tpr)  # auc 值
    if plot_ROC:
        plt.plot(fpr, tpr, '-*', lw=1, label='auc=%g' % roc_auc)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    return roc_auc, fpr, tpr, thresholds


def evaluate(y_true, y_pred):
    """二分类预测结果评估。
    Args:
        y_true: list, 真实标签，如 [1, 0, 0, 1]
        y_pred: list，预测结果，如 [1, 1, 0, 1]
    Returns:
        返回正类别的评价指标。
        p: 预测为正类别的准确率： p = tp / (tp + fp)
        r: 预测为正类别的召回率： r = tp / (tp + fn)
        f1: 预测为正类别的 f1 值： f1 = 2 * p * r / (p + r).
    """
    conf_mat = confusion_matrix(y_true, y_pred)
    all_p = np.sum(conf_mat[:, 1])
    if all_p == 0:
        p = 1.0
    else:
        p = conf_mat[1, 1] / all_p
    r = conf_mat[1, 1] / np.sum(conf_mat[1, :])
    f1 = f1_score(y_true, y_pred)
    return p, r, f1


def feature_analyze(model, to_print=False, to_plot=False, csv_path=None):
    """XGBOOST 模型特征重要性分析。

    Args:
        model: 训练好的 xgb 模型。
        to_print: bool, 是否输出每个特征重要性。
        to_plot: bool, 是否绘制特征重要性图表。
        csv_path: str, 保存分析结果到 csv 文件路径。
    """
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    if to_plot:
        features = list()
        scores = list()
        for (key, value) in feature_score:
            features.append(key)
            scores.append(value)
        plt.barh(range(len(scores)), scores)
        plt.yticks(range(len(scores)), features)
        for i in range(len(scores)):
            plt.text(scores[i] + 0.75, i - 0.25, scores[i])
        plt.xlabel('feature socre')
        plt.title('feature score evaluate')
        plt.grid()
        plt.show()
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))
    if to_print:
        print(''.join(fs))
    if csv_path is not None:
        with open(csv_path, 'w') as f:
            f.writelines("feature,score\n")
            f.writelines(fs)
    return feature_score
