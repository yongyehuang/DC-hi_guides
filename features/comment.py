# -*- coding:utf-8 -*- 

from __future__ import print_function
from __future__ import division

import pandas as pd


def get_comment_feat(df_comment):
    """获取评论特征。
    加上评分有提高，加上 tags 数量降低。
    """
    # df_comment = df_comment[['userid', 'orderid', 'rating', 'tags', 'commentsKeyWords']]
    df_comment_feat = df_comment[['userid', 'rating']].copy()
    df_comment_feat.rename(columns={'rating': 'comment_rating'}, inplace=True)
    return df_comment_feat
