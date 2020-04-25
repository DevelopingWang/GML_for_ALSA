import pickle
from copy import copy
import random

with open('all_global_parse_fg.pkl', 'rb') as f:
    all_datas = pickle.load(f)

var_id_map = dict()   #维护一个id对应表 例如：{'B0074703CM_102_ANONYMOUS:6:0'：0}
for id,item in enumerate(all_datas[0]):
    var_id_map[item.name] = id

#处理features
features=list()
feature=dict()
weight_elem = dict()
feature_id = 0
#先处理relation型特征
for rel_type in ['asp2asp_sequence_oppo','asp2asp_intrasent_simi','asp2asp_sequence_simi']:
    feature['feature_id'] = feature_id
    feature['feature_type'] = rel_type
    feature['feature_name'] = rel_type
    for item in all_datas[2]:
        if item.rel_type == rel_type:
            key = (var_id_map[item.name1], var_id_map[item.name2])
            weight_value = [-2.0,0]  #暂时把featurevalue都设置为0
            weight_elem[key] = weight_value
    feature['weight'] = copy(weight_elem)
    features.append(copy(feature))
    feature_id += 1
    weight_elem.clear()
    feature.clear()

#再处理word型特征
for item in all_datas[3]:
    feature['feature_id'] = feature_id
    feature['feature_name'] = item.name1
    feature['feature_type'] =item.rel_type
    key = var_id_map[item.name2]
    weight_value = [0,item.name1]  #暂时把feature_value设置为word
    weight_elem[key] = weight_value
    feature['weight'] = copy(weight_elem)
    features.append(copy(feature))
    feature_id += 1
    weight_elem.clear()
    feature.clear()

#整理变量
variables=list()
variable=dict()
feature_set = dict()
for id,item in enumerate(all_datas[0]):
    variable['var_id'] = id
    variable['is_evidence'] = item.isEvidence
    if item.polarity is None:
        variable['is_easy'] = False
        variable['is_evidence'] = False
        variable['label'] = random.choice((1,0))   #如果是hard,就随机初始化label
    else:
        variable['is_easy'] = True
        variable['is_evidence'] = True
        variable['label'] = 1 if item.polarity == 'positive' else 0  #如果是easy，就设置为easy的标签
    variable['true_label'] = 1 if item.gold_polarity == 'positive' else 0
    variable['prior'] = item.prior
    for feature in features:
        for kv in feature['weight'].items():
            if type(kv[0]) == tuple and id in kv[0]:
                feature_set[kv[0]] = kv[1]
            elif id == kv[0]:
                feature_set[kv[0]] = kv[1][1]
    variable['feature_set'] = copy(feature_set)
    variables.append(copy(variable))
    variable.clear()
    feature_set.clear()

with open('variables.pkl','wb') as v:
    pickle.dump(variables,v)
with open('features.pkl','wb') as f:
    pickle.dump(features,f)