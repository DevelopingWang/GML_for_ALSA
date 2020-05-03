import math
import numpy as np


#存放一些工具类函数
def separate_variables(variables):
    '''将variables分成证据变量和隐变量
    修改实例对象这两个属性：observed_variables_id，poential_variables_id
    '''
    observed_variables_set = set()
    poential_variables_set = set()
    for variable in variables:
        if variable['is_evidence'] == True:
            observed_variables_set.add(variable['var_id'])
        else:
            poential_variables_set.add(variable['var_id'])
    return observed_variables_set,poential_variables_set

def init_evidence_interval(evidence_interval_count):
    '''初始化证据区间
    输出：一个包含evidence_interval_count个区间的list
    '''
    evidence_interval = list()
    step = float(1) / evidence_interval_count
    previousleft = 0
    previousright = previousleft + step
    for intervalindex in range(0, evidence_interval_count):
        currentleft = previousright
        currentright = currentleft + step
        if intervalindex == evidence_interval_count - 1:
            currentright = 1 + 1e-3
        previousleft = currentleft
        previousright = currentright
        evidence_interval.append([currentleft, currentright])
    return evidence_interval

def init_evidence(features,evidence_interval,observed_variables_set):
    '''
    初始化所有feature的evidence_interval属性和evidence_count属性
    :return:
    '''
    for feature in features:
        evidence_count = 0
        intervals = [set(), set(), set(), set(), set(), set(), set(), set(), set(), set()]
        weight = feature['weight']
        feature['evidence_interval'] = intervals
        for kv in weight.items():
            if kv[0] in observed_variables_set:
                for interval_index in range(0, len(evidence_interval)):
                    if kv[1][1] >= evidence_interval[interval_index][0] and kv[1][1] < \
                            evidence_interval[interval_index][1]:
                        feature['evidence_interval'][interval_index].add(kv[0])
                        evidence_count += 1
        feature['evidence_count'] = evidence_count

def write_labeled_var_to_evidence_interval(variables,features,var_id,evidence_interval):
    '''
    因为每个featurew维护了evidence_interval属性，所以每标记一个变量之后，需要更新这个属性
    :param var_id:
    :return:
    '''
    var_index = var_id
    feature_set = variables[var_index]['feature_set']
    for kv in feature_set.items():
        for interval_index in range(0, len(evidence_interval)):
            if kv[1][1] >= evidence_interval[interval_index][0] and kv[1][1] < \
                    evidence_interval[interval_index][1]:
                features[kv[0]]['evidence_interval'][interval_index].add(var_id)
                features[kv[0]]['evidence_count'] += 1


def entropy(probability):
    '''给定概率之后计算熵
    输入：
    probability ： 单个概率或者概率列表
    输出： 单个熵或者熵的列表
    '''
    if type(probability) == np.float64 or type(probability) == np.float32 or type(probability) == float or type(
            probability) == int:
        if math.isinf(probability) == True:
            return probability
        else:
            if probability <= 0 or probability >= 1:
                return 0
            else:
                return 0 - (probability * math.log(probability, 2) + (1 - probability) * math.log((1 - probability),
                                                                                                  2))
    else:
        if type(probability) == list:
            entropy_list = []
            for each_probability in probability:
                entropy_list.append(entropy(each_probability))
            return entropy_list
        else:
            return None

def open_p(weight):
    return float(1) / float(1 + math.exp(- weight))
