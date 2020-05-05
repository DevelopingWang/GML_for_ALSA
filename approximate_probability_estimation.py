from copy import copy
from scipy.sparse import *
from scipy.stats import t
from sklearn.linear_model import LinearRegression
import logging
import numpy as np
import gml_utils

#近似概率计算的一些方法
class ApproximateProbabilityEstimation:
    def __init__(self,variables):
        self.variables = variables

    def approximate_probability_estimation_by_interval(self, var_id):
        '''
        依据近似权重计算近似概率
        :param var_id:
        :return:
        '''
        if type(var_id) == int:
            self.variables[var_id]['probability'] = gml_utils.open_p(self.variables[var_id]['approximate_weight'])
        elif type(var_id) == list or type(var_id) == set:
            for id in var_id:
                self.approximate_probability_estimation_by_interval(id)

    def approximate_probability_estimation_by_relation(self, var_id):
        pass

    def approximate_probability_estimation_by_custom(self, var_id):
        pass
