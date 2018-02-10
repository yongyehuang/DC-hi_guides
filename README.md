# HI GUIDES 精品旅行服务成单预测  final rank: 12
比赛说明：[精品旅行服务成单预测](http://www.dcjingsai.com/common/cmpt/%E7%B2%BE%E5%93%81%E6%97%85%E8%A1%8C%E6%9C%8D%E5%8A%A1%E6%88%90%E5%8D%95%E9%A2%84%E6%B5%8B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)

- **竞赛背景**: **第二届智慧中国杯首发** 皇包车（HI GUIDES）是一个为中国出境游用户提供全球中文包车游服务的平台。

- **比赛目标**:我们提供了5万多名用户在旅游app中的浏览行为记录，其中有些用户在浏览之后完成了订单，且享受了精品旅游服务，而有些用户则没有下单。参赛者需要分析用户的个人信息和浏览行为，从而预测用户是否会在短期内购买精品旅游服务。预测用户是否会在短期内购买精品旅游服务


## 文件结构
|- hi_guide<br/>
|　　|- data　　　　　　　　# 比赛提供的原始数据<br/>
|　　|　　|- test　　　　　
|　　|　　|- trainingset　　　　　　 
|　　|- features　　　　　　# 特征提取函数<br/>
|　　|　　|- action.py　　　　  # 行为特征<br/>
|　　|　　|- comment.py　　　　 # 评论特征<br/>
|　　|　　|- history.py　　　　 # 历史订单特征<br/>
|　　|　　|- profile.py　　　　 # 用户信息特征<br/>
|　　|　　|- train_data.csv　　 # 保存提取的训练特征 　<br/>
|　　|　　|- test_data.csv　　　# 保存提取的测试集特征　<br/>
|　　|- log　　　　　　　　# 模型训练日志<br/>
|　　|- result　　　　　　 # 模型预测结果<br/>
|　　|- model　　　　　　  # 保存训练好的模型和特征重要度分析文件<br/>
|　　|- data_helper.py　　# 执行特征提取的代码<br/>
|　　|- my_utils.py　　   # 工具函数库，主要用到其中的 xgb 特征重要度分析函数<br/>
|　　|- m1_xgb.py　　     # xgb 模型<br/>
|　　|- m2_lgb.py　　     # lgb 模型<br/>
|　　|- m3_cgb.py　　     # catboost 模型<br/>
|　　|- stacking.py　　   # stacking 模型融合<br/>
|　　|- get_no_used_features.py　　     # 获取 xgb 和 lgb 中的特征重要度<br/>


## 使用方式

```shell
# run the single model
python -u m1_xgb.py
# run the stacking model
python -u stacking.py
# 注意事项
# train_data, test_data = load_feat(re_get=True, feature_path=feature_path)  # 如果没有修改特征，设置re_get=False，就会直接导入之前保存好的特征。
```




