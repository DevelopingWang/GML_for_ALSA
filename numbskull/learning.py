"""TODO."""

from __future__ import print_function, absolute_import
import numba
from numba import jit
import numpy as np
import math
import random
from numbskull.inference import draw_sample, eval_factor


@jit(nopython=True, cache=True, nogil=True)
def learnthread(shardID, nshards, step, regularization, reg_param, truncation,
                var_copy, weight_copy, weight,
                variable, factor, fmap,
                vmap, factor_index, Z, fids, var_value, var_value_evid,
                weight_value, learn_non_evidence):
    """TODO."""
    # Identify start and end variable
    nvar = variable.shape[0]
    start = (shardID * nvar) // nshards
    end = ((shardID + 1) * nvar) // nshards
    for var_samp in range(start, end):  # 对每一个变量进行采样，然后随机梯度下降
        if variable[var_samp]["isEvidence"] == 4:
            # This variable is not owned by this machine
            continue
        sample_and_sgd(var_samp, step, regularization, reg_param, truncation,
                       var_copy, weight_copy, weight, variable,
                       factor, fmap, vmap,
                       factor_index, Z[shardID], fids[shardID], var_value,
                       var_value_evid, weight_value, learn_non_evidence)


@jit(nopython=True, cache=True, nogil=True)
def get_factor_id_range(variable, vmap, var_samp, val):
    """TODO."""
    varval_off = val
    if variable[var_samp]["dataType"] == 0:
        varval_off = 0
    vtf = vmap[variable[var_samp]["vtf_offset"] + varval_off]
    start = vtf["factor_index_offset"]
    end = start + vtf["factor_index_length"]
    return (start, end)


@jit(nopython=True, cache=True, nogil=True)
def sample_and_sgd(var_samp, step, regularization, reg_param, truncation,
                   var_copy, weight_copy, weight, variable, factor, fmap,
                   vmap, factor_index, Z, fids, var_value, var_value_evid,
                   weight_value, learn_non_evidence):
    """TODO."""
    # 随机梯度下降是用一个样本来近似所有的样本来调整参数theta,计算更快，结果会在最优附近。
    # 参数中var_samp是一个索引，truncation意思为截断
    # If learn_non_evidence sample twice.
    # The method corresponds to expectation-conjugate descent.  #期望共轭梯度
    if variable[var_samp]["isEvidence"] != 1:  # 如果不是观测变量就需要采样
        evidence = draw_sample(var_samp, var_copy, weight_copy,
                               weight, variable, factor,
                               fmap, vmap, factor_index, Z,
                               var_value_evid, weight_value)
        # If evidence then store the initial value in a tmp variable
    # then sample and compute the gradient.
    else:
        evidence = variable[var_samp]["initialValue"]  # 如果是观测变量直接取出初始值就可以用

    var_value_evid[var_copy][var_samp] = evidence  # var_value_evid中存放的应该是作为证据节点的变量
    # Sample the variable
    proposal = draw_sample(var_samp, var_copy, weight_copy, weight,
                           variable, factor, fmap, vmap,
                           factor_index, Z, var_value, weight_value)

    var_value[var_copy][var_samp] = proposal
    if not learn_non_evidence and variable[var_samp]["isEvidence"] != 1:
        return
    # Compute the gradient and update the weights
    # Iterate over corresponding factors

    range_fids = get_factor_id_range(variable, vmap, var_samp, evidence)  # get_factor_id_range返回因子的ID范围
    # TODO: is it possible to avoid copying around fids
    if evidence != proposal:
        range_prop = get_factor_id_range(variable, vmap, var_samp, proposal)
        s1 = range_fids[1] - range_fids[0]
        s2 = range_prop[1] - range_prop[0]
        s = s1 + s2
        fids[:s1] = factor_index[range_fids[0]:range_fids[1]]
        fids[s1:s] = factor_index[range_prop[0]:range_prop[1]]
        fids[:s].sort()
    else:
        s = range_fids[1] - range_fids[0]
        fids[:s] = factor_index[range_fids[0]:range_fids[1]]

    truncate = random.random() < 1.0 / truncation if regularization == 1 else False  # random()方法返回随机生成的一个实数，它在[0,1)范围内
    # go over all factor ids, ignoring dupes      #depus翻译：骗子？？？
    last_fid = -1  # numba 0.28 would complain if this were None
    for factor_id in fids[:s]:  # 遍历所有因子
        if factor_id == last_fid:
            continue
        last_fid = factor_id
        weight_id = factor[factor_id]["weightId"]
        if weight[weight_id]["isFixed"]:    #如果权重是固定的，就不需进行更新
            continue
        # Compute Gradient
        p0 = eval_factor(factor_id, var_samp,
                         evidence, var_copy,
                         variable, factor, fmap,
                         var_value_evid)
        p1 = eval_factor(factor_id, var_samp,
                         proposal, var_copy,
                         variable, factor, fmap,
                         var_value)
        # print('p0', p0)
        # print('p1', p1)
        #如果需要参数化
        if weight[factor[factor_id]['weightId']]['parameterize']:
            x = fmap[factor[factor_id]["ftv_offset"]]['x']      #根据偏移量来找x
            theta = fmap[factor[factor_id]["ftv_offset"]]['theta']
            # w = pow(a,2)*(x-pow(b,2))
            # gradient1 = (p1 - p0) * factor[factor_id]["featureValue"] * 2*weight[factor[factor_id]['weightId']]['a']*(x - pow(weight[factor[factor_id]['weightId']]['b'],2))
            # gradient2 = (p1 - p0) * factor[factor_id]["featureValue"] * (-2* weight[factor[factor_id]['weightId']]['b']*pow(weight[factor[factor_id]['weightId']]['a'],2))
            # w = a*(x-b)
            # if(p1 - p0) == 0:
            #     print('梯度为0')
            gradient1 = (p1 - p0) * theta * factor[factor_id]["featureValue"] * (x - weight[factor[factor_id]['weightId']]['b'])
            gradient2 = (p1 - p0) * theta * factor[factor_id]["featureValue"] * (-weight[factor[factor_id]['weightId']]['a'])
            weight[factor[factor_id]['weightId']]['a'] -= step * gradient1  # update a
            weight[factor[factor_id]['weightId']]['b'] -= step * gradient2  # Update b
            a = weight[factor[factor_id]['weightId']]['a']
            b = weight[factor[factor_id]['weightId']]['b']
        # w =   weight_value[weight_copy][weight_id]
            if regularization == 2:  # 是否需要正则化
                a *= (1.0 / (1.0 + reg_param * step))
                a -= step * gradient1
                b *= (1.0 / (1.0 + reg_param * step))
                b -= step * gradient2
            elif regularization == 1:
            # Truncated Gradient                        截断梯度法
            # "Sparse Online Learning via Truncated Gradient"  截断梯度的稀疏在线学习
            #  Langford et al. 2009
                a -= step * gradient1
                b -= step * gradient2
                if truncate:
                    l1delta = reg_param * step * truncation
                    a = max(0, a - l1delta) if a > 0 else min(0, a + l1delta)
                    b = max(0, b - l1delta) if b > 0 else min(0, b + l1delta)
            else:
                a -= step * gradient1
                b -= step * gradient2

            w = theta * a * (x - b)
        else:  #如果不需要参数化
            gradient = (p1 - p0) * factor[factor_id]["featureValue"]
        # Update weight
            w = weight_value[weight_copy][weight_id]
            if regularization == 2:
                w *= (1.0 / (1.0 + reg_param * step))
                w -= step * gradient
            elif regularization == 1:
            # Truncated Gradient
            # "Sparse Online Learning via Truncated Gradient"
            #  Langford et al. 2009
                w -= step * gradient
                if truncate:
                    l1delta = reg_param * step * truncation
                    w = max(0, w - l1delta) if w > 0 else min(0, w + l1delta)
            else:
                w -= step * gradient
        #print(w)
        weight_value[weight_copy][weight_id] = w
        weight[factor[factor_id]['weightId']]['initialValue'] = w
