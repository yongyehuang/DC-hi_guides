
### 20180113
```
这里使用 xgb.cv 的时候，会比单次的 xgb.train 中 eval 的结果要差些，但是应该会和测试结果更加接近。待提交测试结果验证一下才行。

历史订单：0.6844794
行为信息：0.7660604
历史订单+行为：0.8132784
历史订单+行为+性别： 0.814641
历史订单+行为+性别+年龄: 0.814963
历史订单+行为+性别+年龄+城市：0.8160524  (前面 max_depth=5, 这里 max_depth=7)
历史订单+行为+性别+年龄+城市+评分：0.8230778  (max_depth=7)
+ 最早一次订单距离“当前时刻”：0.825192 （+最后一次：0.8244876， 反而没有最早一次的结果好。）
+ 最早一次订单距离“当前时刻”：0.825476

+ 最后一次动作距离“当前时刻”：0.8477612
+ 最早一次动作距离“当前时刻”：0.8517812
+ （最后-最早动作的距离）：0.854584
+ 最后一次 7 距离“当前时刻”：0.8560382
+ 每种动作距离 “当前时刻”：0.861841
+ 最后 5 个动作的均值和均方差：0.926513
+ 若是改成 最后 10 个动作：0.9291472（把学习率设置为 0.05）

+ 加上动作 6 后面的统计特征：0.9413118
+ 动作 [2,3,5,6,7,8,9] 后面的统计特征：0.9579598
+ 动作 [5,6,7,8,9] 后面的统计特征：0.957664
+ 动作 [1~9] 后面的统计特征：0.959190
+ 加上最近某动作后面的 1 个时间间隔：0.962745
+ 加上最近某动作后面的 2 个时间间隔：0.9631138

# 20180130
+ 用户去过的国家次数 lgb=0.967240670954
+ 去过的城市次数 lgb=0.967025701617 变差了
+ 去过的洲次数 best_auc=0.967296241646

（下面改成 n_fold=10，线下会更高一些）
+ action_pair 取 [[5, 6], [6, 7], [7, 8]] 更好一些
+ 每个动作距离最后一次动作的时间距离：（lgb） 0.967911181955

+ action_pairs = [[5, 6], [6, 7], [7, 8], [5, 5], [6, 6], [7, 7], [5, 9]]
lgb: 0.97019201826  -> 线上
xgb: 线下0.969576 -> 线上 0.96909

+ action_pairs = [2, 5], [2, 6]  其中 2 表示动作 [2,3,4]
lgb: 0.9700109
xgb: 0.9697423

# 20180131
TODO:
1. 加上参数搜索代码
2. 整理 stacking 代码

action pair 统计特征中加上 min, max 要比只有 mean, std 好一个千分点。

action pair 加上 2，3，4

+ 最后 n_diff 之间统计特征加上 min, max
+ 修改  n_diff 的值

after parameter search:
xgb 线下： 0.9700818 -> 0.96939

换成lgb 参数以后，线下5折：0.97013 -> 0.96950


# 20180201
1.加上 5% 概率最大的样本添加到训练集中一块训练，结果没有提高。
原因：可能是这 5% 太大了，直接加到训练集中影响了正负样本的分布。

使用 1% 最大值（正类）和 5% 的最小值（负类），基本维持原本的比例。再次实验。这样由于新加进来的样本都是比较好预测的，那么线下的预测准确度肯定会提高不少，所以线下提高了，但是线上不一定能有提高。
结果还是没用。

2.对训练数据进行预测，其中部分负样本被预测为正类的话，剔除这些样本。剔除了 130 条左右负样本，结果变差了。虽然这些样本原本为 0，被预测成了 1，但是这些样本都是确确实实存在的“异常”。直接删除这些样本的话，对于那些有区分度的测试样本可能会预测得更加准确率，但是对于模棱两可的样本可能预测就会出现偏差。而且由于删除了这些“异常”的样本，那么线下的准确度肯定会更高一些。

3.TODO: 既然线下存在这么多"异常"的样本没有办法拟合，那么想办法增加这些样本的惩罚度。使用 adaboost 看看是否会更好一些。


# 20180202
+ 最后一个动作的时，分，周几；加上最后一次动作 1, 5, 6, 7 的时，分，周几。有提高。
+ TODO: 加上动作组合的比例
+ TODO: 3个动作连续
+ TODO: 最后一个动作的时间于最后一次订单的时间
+ 统计精品订单下单的组合动作比例。

# 20180204
+ my_stacking 线下 0.971078940133， 线上 0.97089

# 20180205
[为什么过多的特征（feature）导致过拟合（over-fitting)？](https://www.zhihu.com/question/47375421)

# 20180206
- 使用 search_feat 的方式训练了 lgb，逐个参数递减，一共减去了 150 个特征。
最后选择其中 auc > 0.971 的结果进行平均：最后线上 0.97149
- 使用 360 维特征 stacking 12 个模型，结果比上面差一点点。

- 将 diff 改成 next_action_time - action_time，提升了4个万分点，360维特征，单模型 0.971306896552。


TODO：
1.下面特征数值太大全部减去最小值：
'first_order_time', 'last_order_time', 'first_action_time', 'last_action_time',
'first_action_1_time', 'last_action_1_time',
'first_action_2_time', 'last_action_2_time',
...
'first_action_9_time', 'last_action_9_time',




2.下面特征差异几个数量级，取对数 log 进行处理：
'diff_last_first_action',
'first_last_action_1_diff',
'first_last_action_2_diff',
'first_last_action_3_diff',
  ...
 'first_last_action_9_diff',


'1_to_end_diff',
'2_to_end_diff',
...
'9_to_end_diff',


'final_diff_0',
'final_diff_1',
 ...
 'final_diff_9',


'last_3_action_diff_std',
'last_3_action_diff_mean',
'last_10_action_diff_std',
'last_10_action_diff_mean',


'pair_to_end_1_1',
'pair_to_end ...',

cols.startswith('pair_to_end')


***结果：*变差了 3 个万分点，放弃。



3.下面特征先统计转化率，计算用户对应的转化率：
'gender', 'country', 'province',


4.userid 减去 userid 的最小值。


## 20180207
- 选择提交结果中最好的 5 个模型进行平均，结果： 0.97180， B 榜 0.97280
```

[lgb params](http://lightgbm.readthedocs.io/en/latest/Parameters.html)
num_threads, default=OpenMP_default, type=int, alias=num_thread, nthread
number of threads for LightGBM
for the best speed, set this to the number of real CPU cores, not the number of threads (most CPU using hyper-threading to generate 2 threads per CPU core)
do not set it too large if your dataset is small (do not use 64 threads for a dataset with 10,000 rows for instance)
be aware a task manager or any similar CPU monitoring tool might report cores not being fully utilized. This is normal
for parallel learning, should not use full CPU cores since this will cause poor performance for the network

设置 num_threads=真实的CPU个数。一般来说，如果是12线程的机器有6个真实的物理核，所以应该设置为 6。默认他会使用 12，但是并不推荐使用满线程，这样速度会很慢。实验发现，如果设置满线程的话，那么整个机器必须只跑这个任务速度才会比较快，只要你在同时跑个很小的程序，都会使它的速度慢好几倍，比如你再跑个小程序只用 50%（0.5个）CPU，那么lgb 的速度都可能降低三四倍。


不同版本的 lgb 训练结果差距还是有比较大的差别。

