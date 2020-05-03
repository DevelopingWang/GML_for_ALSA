import heapq
import math
import pickle
from collections import namedtuple
from copy import copy
from scipy.sparse import *
from scipy.stats import t
from sklearn.linear_model import LinearRegression
import numbskull
# from easyinstancelabeling import EasyInstanceLabeling
# from featureextract import FeatureExtract
from numbskull.numbskulltypes import *
# import data_pre
import random
import logging
import time
import warnings
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics

from pyds import MassFunction
# class FeatureExtract:
#     pass
# class EasyInstanceLabeling:
#     pass

class General:
    '''暂时存放一些较为独立的全局函数,后续再迁移'''

    @staticmethod
    def open_p(weight):
        return float(1) / float(1 + math.exp(- weight))

    @staticmethod
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
                    entropy_list.append(General.entropy(each_probability))
                return entropy_list
            else:
                return None

    @staticmethod
    def print_results(dataname,datapath):
        all_true_label = list()
        label_file = pd.read_csv(datapath + dataname + '_pair_info_g.csv')
        for i in label_file.Label:
            all_true_label.append(i)
        # easys
        easys = EasyInstanceLabeling.load_easy_instance_from_file(datapath + dataname + '_easys.csv')
        easy_pred = []
        easy_true = []
        for easy in easys:
            easy_true.append(all_true_label[easy['var_id']])
            easy_pred.append(easy['label'])
        # hards
        with open(datapath + dataname + '_result.txt') as r:
            lines = r.readlines()
        hards_pred = []
        hards_true = []
        for line in lines:
            content = line.strip('\n').split(' ')
            var_id = int(content[0])
            label = int(content[1])
            hards_pred.append(label)
            hards_true.append(all_true_label[var_id])
        # total = easys + hards
        total_true = easy_true + hards_true
        total_pred = easy_pred + hards_pred

        r.close()
        print("--------------------------------------------")
        print("total:")
        print("--------------------------------------------")
        print("total precision_score: " + str(metrics.precision_score(total_true, total_pred)))
        print("total recall_score: " + str(metrics.recall_score(total_true, total_pred)))
        print("total f1_score: " + str(metrics.f1_score(total_true, total_pred)))
        print("--------------------------------------------")
        print("easys:")
        print("--------------------------------------------")
        print("easys precision_score:" + str(metrics.precision_score(easy_true, easy_pred)))
        print("easys recall_score:" + str(metrics.recall_score(easy_true, easy_pred)))
        print("easys f1_score: " + str(metrics.f1_score(easy_true, easy_pred)))
        print("--------------------------------------------")
        print("hards:")
        print("--------------------------------------------")
        print("hards precision_score: " + str(metrics.precision_score(hards_true, hards_pred)))
        print("hards recall_score: " + str(metrics.recall_score(hards_true, hards_pred)))
        print("hards f1_score: " + str(metrics.f1_score(hards_true, hards_pred)))


class Regression:
    '''线性回归相关类，对所有feature进行线性回归
    输入：一个feature
    输出： regression对象
    '''

    def __init__(self, each_feature_easys, n_job):
        '''
        todo:
         feature回归的更新策略:只回归证据支持有变化的feature
        '''
        self.effective_training_count = max(2, GML.effective_training_count_threshold)
        self.n_job = n_job
        if len(each_feature_easys) > 0:
            XY = np.array(each_feature_easys)
            self.X = XY[:, 0].reshape(-1, 1)
            self.Y = XY[:, 1].reshape(-1, 1)
        else:
            self.X = np.array([]).reshape(-1, 1)
            self.Y = np.array([]).reshape(-1, 1)
        self.balance_weight_y0_count = 0
        self.balance_weight_y1_count = 0
        for y in self.Y:
            if y > 0:
                self.balance_weight_y1_count += 1
            else:
                self.balance_weight_y0_count += 1
        self.perform()

    def perform(self):
        '''执行线性回归'''
        self.N = np.size(self.X)
        if self.N <= self.effective_training_count:
            self.regression = None
            self.residual = None
            self.meanX = None
            self.variance = None
            self.k = None
            self.b = None
        else:
            sample_weight_list = None
            if self.balance_weight_y1_count > 0 and self.balance_weight_y0_count > 0:
                sample_weight_list = list()
                sample_weight = float(self.balance_weight_y0_count) / self.balance_weight_y1_count
                for y in self.Y:
                    if y[0] > 0:
                        sample_weight_list.append(sample_weight)
                    else:
                        sample_weight_list.append(1)
            self.regression = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=self.n_job).fit(self.X, self.Y,
                                                                                                       sample_weight=sample_weight_list)
            self.residual = np.sum((self.regression.predict(self.X) - self.Y) ** 2) / (self.N - 2)
            self.meanX = np.mean(self.X)  # 此feature的所有证据变量的feature_value的平均值
            self.variance = np.sum((self.X - self.meanX) ** 2)
            z = self.regression.predict(np.array([0, 1]).reshape(-1, 1))
            self.k = (z[1] - z[0])[0]
            self.b = z[0][0]

    def append(self, appendx, appendy):
        '''实时添加训练数据'''
        self.X = np.append(self.X, [[appendx]], axis=0)
        self.Y = np.append(self.Y, [[appendy]], axis=0)
        if appendy > 0:
            self.balance_weight_y1_count += 1
        else:
            self.balance_weight_y0_count += 1
        self.perform()

    def disable(self, delx, dely):
        '''实时删除训练数据'''
        for index in range(0, len(self.X)):
            if self.X[index][0] == delx and self.Y[index][0] == dely:
                self.X = np.delete(self.X, index, axis=0)
                self.Y = np.delete(self.Y, index, axis=0)
                if dely > 0:
                    self.balance_weight_y1_count -= 1
                else:
                    self.balance_weight_y0_count -= 1
                break
        self.perform()


class GML:
    '''GML大类: 包括计算Evidential Support，Approximate Estimation of Inferred Probability，
    Construction of Inference Subgraph等；不包括Feature Extract和Easy Instance Labeling
    在实现过程中，注意区分实例变量和类变量
    '''
    # tau_and_regression_bound = 10
    delta = 2
    effective_training_count_threshold = 2
    NOT_NONE_VALUE = 1e-8
    n_job = 10
    evidence_interval_count = 10  # 区间个数10
    interval_evidence_count = 200  # 每个区间的变量数为200

    # zh add for ALSA
    word_evi_uncer_degree = 0.1
    relation_evi_uncer_degree = 0.3

    # update_cache = 10
    # openpweight = 100

    def __init__(self,dataname,datapath,variables, features, edges, easys, top_m=2000, top_k=10,
                 update_proportion=0.01, tau_and_regression_bound=10,balance = False):
        '''
        todo:
             1.目前暂不知为了支持binary_relation需要如何修改数据结构
        '''
        self.dataname = dataname
        self.datapath = datapath
        self.variables = variables
        self.features = features
        self.edges = edges
        self.easys = easys
        self.features_easys = dict()  # 存放所有features的所有easy的featurevalue   :feature_id:[[value1,bound],[value2,bound]...]
        self.observed_variables_id = set()  # 所有观测变量集合,随推理实时更新
        self.poential_variables_id = set()  # 所有隐变量集合，随推理实时更新
        self.labeled_variables_id = set()  # 所有新标记变量集合
        self.data_matrix = None
        self.evidence_interval = list()
        self.top_m = top_m
        self.top_k = top_k
        self.update_proportion = update_proportion
        self.evidence_interval = None
        self.tau_and_regression_bound = tau_and_regression_bound
        self.balance = balance
        # self.cache_subgraph = cache_subgraph   #是否缓存因子图
        # self.evidence_interval_count = evidence_interval_count   #证据区间的个数
        # self.interval_evidence_count = interval_evidence_count   #每个区间的证据个数
        # GML.delta = delta
        # GML.tau_and_regression_bound = tau_and_regression_bound
        # GML.effective_training_count_threshold = effective_training_count_threshold

        # zh add for ALSA
        # key:vid   value: [(n_samples, neg_prob, pos_prob),(n_samples, neg_prob, pos_prob)]
        self.dict_unlabvar_feature_evis = {}

        logging.basicConfig(
            level=logging.INFO,  # 设置输出信息等级
            format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s'  # 设置输出格式
        )
    # zh add for ALSA
    def zh_test(self):
        self.easy_instance_labeling()
        self.separate_variables()
        self.get_unlabvar_feature_evis()
        self.get_unlabvar_evi_support()

    def get_unlabvar_feature_evis(self):
        for vid in self.poential_variables_id:
            vindex = self.find_in_variables(vid)
            feature_set = self.variables[vindex]["feature_set"]
            pos = 0
            neg = 0
            dic_temp = []
            for key, value in feature_set.items():
                # 如果是词特征
                if self.features[key]["feature_type"] == "fea2asp_ngramfea_simi":
                    # 统计pos，neg的比例
                    weight = self.features[key]["weight"]
                    n_samples = len(weight)

                    for k, v in weight.items():
                        vindex_temp = self.find_in_variables(k)
                        if self.variables[vindex_temp]["label"] == 1:
                            pos += 1
                        if self.variables[vindex_temp]["label"] == 0:
                            neg += 1
                    if n_samples != 0:
                        dic_temp.append((n_samples, neg/n_samples, pos/n_samples))
            self.dict_unlabvar_feature_evis[vid] = dic_temp

    def get_unlabvar_evi_support(self):
        dict_unlabvar_propensity_masses = {}  # key: unlabeled variable
                                              # value: mass functions for different evidences
        for vid in self.poential_variables_id:
            mass_functions_tmp = []

            # 关系特征由wcy实现
            # if dict_unlabvar_relation_evis.get(unlabel_var):
            #     for (label_var_name, rel_type) in dict_unlabvar_relation_evis[unlabel_var]:
            #         # rel_uncer_degree = get_relation_uncer_degree(rel_type)
            #         # rel_acc = get_relation_acc(rel_type)
            #         rel_acc = dict_rel_acc[rel_type]
            #         mass_functions_tmp.append(
            #             construct_mass_function_for_propensity(relation_evi_uncer_degree, rel_acc, 1 - rel_acc))

            # 词特征
            if self.dict_unlabvar_feature_evis.get(vid):
                for (n_samples, neg_prob, pos_prob) in self.dict_unlabvar_feature_evis[vid]:
                    mass_functions_tmp.append(self.construct_mass_function_for_propensity(
                        GML.word_evi_uncer_degree, max(pos_prob, neg_prob), min(pos_prob, neg_prob)))

            if len(mass_functions_tmp) > 0:
                dict_unlabvar_propensity_masses[vid] = mass_functions_tmp

        for unlabel_var, mass_funcs in dict_unlabvar_propensity_masses.items():
            combined_mass = self.labeling_propensity_with_ds(mass_funcs)
            # value: combined mass function ({{'l'}:0.9574468085106382; {'u'}:0.04255319148936169; {'l', 'u'}:0.0})
            index = self.find_in_variables(unlabel_var)
            self.variables[index]["evidential_support"] = combined_mass["l"]
            # print()



    def construct_mass_function_for_propensity(self, uncertain_degree, label_prob, unlabel_prob):
        '''
        # l: support for labeling
        # u: support for unalbeling
        '''
        return MassFunction({'l': (1 - uncertain_degree) * label_prob,
                             'u': (1 - uncertain_degree) * unlabel_prob,
                             'lu': uncertain_degree})

    def labeling_propensity_with_ds(self, mass_functions):
        combined_mass = self.combine_evidences_with_ds(mass_functions, normalization=True)
        return combined_mass

    def combine_evidences_with_ds(self, mass_functions, normalization):
        # combine evidences from different sources
        if len(mass_functions) < 2:
            combined_mass = mass_functions[0]
        else:
            combined_mass = mass_functions[0].combine_conjunctive(mass_functions[1], normalization)

            if len(mass_functions) > 2:
                for mass_func in mass_functions[2: len(mass_functions)]:
                    combined_mass = combined_mass.combine_conjunctive(mass_func, normalization)
        return combined_mass

    def create_csr_matrix(self):
        # 创建稀疏矩阵存储所有variable的所有featureValue，用于后续计算Evidential Support
        data = list()
        row = list()
        col = list()
        for index, var in enumerate(self.variables):
            feature_set = self.variables[index]['feature_set']
            for feature_id in feature_set:
                data.append(feature_set[feature_id][1] + GML.NOT_NONE_VALUE)
                row.append(index)
                col.append(feature_id)
        self.data_matrix = csr_matrix((data, (row, col)), shape=(len(self.variables), len(self.features)))

    def init(self):
        '''
        处理此对象推理所需的全部准备工作
        '''
        self.easy_instance_labeling()
        self.separate_variables()
        self.init_evidence()  # 初始化每个feature的evidence_interval属性
        self.create_csr_matrix()  # 创建用于计算Evidential support的稀疏矩阵

    def find_in_variables(self, var_id):
        for i in range(len(self.variables)):
            if self.variables[i]['var_id'] == var_id:
                return i
                break

    def easy_instance_labeling(self):
        '''根据提供的easy列表修改标出变量中的Easy'''
        for var in self.variables:
            var['is_easy'] = False
            var['is_evidence'] = False
        for easy in self.easys:
            var_index = self.find_in_variables(easy['var_id'])
            self.variables[var_index]['is_easy'] = True
            self.variables[var_index]['is_evidence'] = True

    def separate_variables(self):
        '''将variables分成证据变量和隐变量
        修改实例对象这两个属性：observed_variables_id，poential_variables_id
        '''
        for variable in variables:
            if variable['is_evidence'] == True:
                self.observed_variables_id.add(variable['var_id'])
            else:
                self.poential_variables_id.add(variable['var_id'])

    def separate_feature_value(self):
        # 选出每个feature的easy feature value用于线性回归
        each_feature_easys = list()
        self.features_easys.clear()
        for feature in self.features:
            each_feature_easys.clear()
            for var_id, value in feature['weight'].items():
                # 每个feature拥有的easy变量的feature_value
                if var_id in self.observed_variables_id:
                    each_feature_easys.append([value[1], (1 if self.variables[self.find_in_variables(var_id)][
                                                                   'label'] == 1 else -1) * self.tau_and_regression_bound])
            self.features_easys[feature['feature_id']] = copy(each_feature_easys)

    def init_evidence_interval(self):
        '''初始化证据区间
        输出：一个包含evidence_interval_count个区间的list
        '''
        evidence_interval = list()
        step = float(1) / GML.evidence_interval_count
        previousleft = 0
        previousright = previousleft + step
        for intervalindex in range(0, GML.evidence_interval_count):
            currentleft = previousright
            currentright = currentleft + step
            if intervalindex == GML.evidence_interval_count - 1:
                currentright = 1 + 1e-3
            previousleft = currentleft
            previousright = currentright
            evidence_interval.append([currentleft, currentright])
        return evidence_interval

    def influence_modeling(self, update_feature_set):
        '''对已更新feature进行线性回归
        把回归得到的所有结果存回feature, 键为'regression'
        '''
        self.separate_feature_value()
        if len(update_feature_set) > 0:
            self.init_tau_and_alpha(update_feature_set)
            logging.info("init tau&alpha finished")
            for feature_id in update_feature_set:
                # 对于某些features_easys为空的feature,回归后regression为none
                self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id], n_job=GML.n_job)
            logging.info("feature regression finished")

    def evidential_support(self):
        '''计算所有隐变量的Evidential Support'''
        coo_data = self.data_matrix.tocoo()
        row, col, data = coo_data.row, coo_data.col, coo_data.data
        coefs = []
        intercept = []
        residuals = []
        Ns = []
        meanX = []
        variance = []
        delta = GML.delta
        zero_confidence = []
        for feature in self.features:
            if feature['regression'].regression is not None and feature['regression'].variance > 0:
                coefs.append(feature['regression'].regression.coef_[0][0])
                intercept.append(feature['regression'].regression.intercept_[0])
                zero_confidence.append(1)
            else:
                coefs.append(0)
                intercept.append(0)
                zero_confidence.append(0)
            Ns.append(feature['regression'].N if feature['regression'].N > feature[
                'regression'].effective_training_count else np.NaN)
            residuals.append(feature['regression'].residual if feature['regression'].residual is not None else np.NaN)
            meanX.append(feature['regression'].meanX if feature['regression'].meanX is not None else np.NaN)
            variance.append(feature['regression'].variance if feature['regression'].variance is not None else np.NaN)
        coefs = np.array(coefs)[col]
        intercept = np.array(intercept)[col]
        zero_confidence = np.array(zero_confidence)[col]
        residuals, Ns, meanX, variance = np.array(residuals)[col], np.array(Ns)[col], np.array(meanX)[col], \
                                         np.array(variance)[col]
        tvalue = float(delta) / (residuals * np.sqrt(1 + 1.0 / Ns + np.power(data - meanX, 2) / variance))
        confidence = np.ones_like(data)
        confidence[np.where(residuals > 0)] = (1 - t.sf(tvalue, (Ns - 2)) * 2)[np.where(residuals > 0)]
        confidence = confidence * zero_confidence
        evidential_support = (1 + confidence) / 2  # 正则化
        # 将计算出的evidential support写回
        csr_evidential_support = csr_matrix((evidential_support, (row, col)),
                                            shape=(len(self.variables), len(self.features)))
        for index, var in enumerate(self.variables):
            for feature_id in var['feature_set']:
                var['feature_set'][feature_id][0] = csr_evidential_support[index, feature_id]
        # 计算近似权重
        predict = data * coefs + intercept
        espredict = predict * evidential_support
        # espredict[np.where(polar_enforce == 0)] = np.minimum(espredict, 0)[np.where(polar_enforce == 0)]   #小于0就变成0
        # espredict[np.where(polar_enforce == 1)] = np.maximum(espredict, 0)[np.where(polar_enforce == 1)]   #大于0就变成1
        espredict = espredict * zero_confidence
        # 验证所有evidential support都在(0,1)之间
        assert len(np.where(evidential_support < 0)[0]) == 0
        assert len(np.where((1 - evidential_support) < 0)[0]) == 0
        loges = np.log(evidential_support)  # 取自然对数
        logunes = np.log(1 - evidential_support)
        evidential_support_logit = csr_matrix((loges, (row, col)), shape=(len(self.variables), len(self.features)))
        evidential_unsupport_logit = csr_matrix((logunes, (row, col)), shape=(len(self.variables), len(self.features)))
        p_es = np.exp(np.array(evidential_support_logit.sum(axis=1)))
        p_unes = np.exp(np.array(evidential_unsupport_logit.sum(axis=1)))
        approximate_weight = csr_matrix((espredict, (row, col)), shape=(len(self.variables), len(self.features)))
        approximate_weight = np.array(approximate_weight.sum(axis=1)).reshape(-1)
        # 将计算出来的近似权重写回variables
        for index, var in enumerate(self.variables):
            var['approximate_weight'] = approximate_weight[index]
        logging.info("approximate_weight calculate finished")
        # 综合计算每个隐变量的evidential suppport,并写回self.variables
        for var_id in self.poential_variables_id:
            index = self.find_in_variables(var_id)
            var_p_es = p_es[index]
            var_p_unes = p_unes[index]
            self.variables[index]['evidential_support'] = float(var_p_es / (var_p_es + var_p_unes))
        logging.info("evidential_support calculate finished")

    def approximate_probability_estimation(self, var_id):
        '''
        依据近似权重计算近似概率
        :param var_id:
        :return:
        '''
        if type(var_id) == int:
            var_index = self.find_in_variables(var_id)
            self.variables[var_index]['probability'] = General.open_p(self.variables[var_index]['approximate_weight'])
        elif type(var_id) == list or type(var_id) == set:
            for id in var_id:
                self.approximate_probability_estimation(id)

    def select_top_m_by_es(self, m):
        '''根据计算出的Evidential Support(从大到小)选前m个隐变量
        输入：
        1.  m----需要选出的隐变量的个数
        输出： 一个包含m个变量id的列表
        '''
        # 此处选只能在所有隐变量里面选
        poential_var_list = list()
        m_id_list = list()
        for var_id in self.poential_variables_id:
            poential_var_list.append([var_id, self.variables[self.find_in_variables(var_id)]['evidential_support']])
        topm_var = heapq.nlargest(m, poential_var_list, key=lambda s: s[1])
        for elem in topm_var:
            m_id_list.append(elem[0])
        logging.info('select m finished')
        return m_id_list

    def select_top_k_by_entropy(self, var_id_list, k):
        '''计算熵，选出top_k个熵小的隐变量
        输入:
        1.var_id_list: 选择范围
        2.k:需要选出的隐变量的个数
        输出： 一个包含k个id的列表
        '''
        m_list = list()
        k_id_list = list()
        for var_id in var_id_list:
            var_index = self.find_in_variables(var_id)
            self.variables[var_index]['entropy'] = General.entropy(self.variables[var_index]['probability'])
            m_list.append(self.variables[var_index])
        k_list = heapq.nsmallest(k, m_list, key=lambda x: x['entropy'])
        for var in k_list:
            k_id_list.append(var['var_id'])
        logging.info('select k finished')
        return k_id_list

    def init_evidence(self):
        '''
        初始化所有feature的evidence
        :return:
        '''
        self.evidence_interval = self.init_evidence_interval()
        for feature in self.features:
            evidence_count = 0
            intervals = [set(), set(), set(), set(), set(), set(), set(), set(), set(), set()]
            weight = feature['weight']
            feature['evidence_interval'] = intervals
            for kv in weight.items():
                if kv[0] in self.observed_variables_id:
                    for interval_index in range(0, len(self.evidence_interval)):
                        if kv[1][1] >= self.evidence_interval[interval_index][0] and kv[1][1] < \
                                self.evidence_interval[interval_index][1]:
                            feature['evidence_interval'][interval_index].add(kv[0])
                            evidence_count += 1
            feature['evidence_count'] = evidence_count

    def select_evidence(self, var_id):
        '''
        为指定的隐变量挑一定数量的证据变量：目前是每个feature划分evidence_interval_count个区间，每个区间挑不超过interval_evidence_count个
        同时确定边
        输入：var_id -- 隐变量id
        输出：
        evidence_set --证据变量的id集合
        partial_edges -- 边的集合
        connected_feature_set --能用得上的feature的集合
        '''
        evidence_set = set()
        partial_edges = list()
        connected_feature_set = set()  # 记录此隐变量上建因子图时实际保留了哪些feature
        feature_set = self.variables[self.find_in_variables(var_id)]['feature_set']
        for feature_id in feature_set.keys():
            if self.features[feature_id]['evidence_count'] > 0:  # 有些feature上没有连接证据变量，就不用再加进来
                connected_feature_set.add(feature_id)
                evidence_interval = self.features[feature_id]['evidence_interval']
                for interval in evidence_interval:
                    # 如果这个区间的证据变量小于200，就全加进来
                    if len(interval) <= GML.interval_evidence_count:
                        evidence_set = evidence_set.union(interval)
                        for id in interval:
                            partial_edges.append([feature_id, id])
                    else:
                        # 如果大于200,就随机采样200个
                        sample = random.sample(list(interval), GML.interval_evidence_count)
                        evidence_set = evidence_set.union(sample)
                        for id in sample:
                            partial_edges.append([feature_id, id])
        logging.info("var-" + str(var_id) + " select evidence finished")
        return evidence_set, partial_edges, connected_feature_set

    def init_tau_and_alpha(self, feature_set):
        '''对给定的feature计算tau和alpha

        输入：feature_set 是一个feature_id的集合
        输出：直接修改feature中的tau和alpha属性
        '''
        if type(feature_set) != list and type(feature_set) != set:
            print("输入参数错误，应为set或者list")
            return
        else:
            for feature_id in feature_set:
                # tau值固定为上界
                self.features[feature_id]["tau"] = self.tau_and_regression_bound
                weight = self.features[feature_id]["weight"]
                labelvalue0 = 0
                num0 = 0
                labelvalue1 = 0
                num1 = 0
                for key in weight:
                    if self.variables[key]["is_evidence"] and self.variables[key]["label"] == 0:
                        labelvalue0 += weight[key][1]
                        num0 += 1
                    elif self.variables[key]["is_evidence"] and self.variables[key]["label"] == 1:
                        labelvalue1 += weight[key][1]
                        num1 += 1
                if num0 == 0 and num1 == 0:
                    continue
                if num0 == 0:
                    # 如果没有和该feature相连的label0，就把值赋值为value的上界，目前先定为1
                    labelvalue0 = 1
                else:
                    # label为0的featurevalue的平均值
                    labelvalue0 /= num0
                if num1 == 0:
                    # 同上
                    labelvalue1 = 1
                else:
                    # label为1的featurevalue的平均值
                    labelvalue1 /= num1
                alpha = (labelvalue0 + labelvalue1) / 2
                self.features[feature_id]["alpha"] = alpha

    def construct_subgraph(self, var_id):
        '''在选出topk个隐变量之后建立子图
        输入：一个隐变量的id
        输出：按照numbskull的要求因子图,返回weight, variable, factor, fmap, domain_mask, edges
        '''
        var_index = self.find_in_variables(var_id)
        feature_set = self.variables[var_index]['feature_set']
        evidences = self.select_evidence(var_id)
        # 存储选出的证据和实时的变量和特征
        # with open("ProcessedCache/" + str(var_id) + '_evidences.pkl', 'wb') as e:
        #     pickle.dump(evidences, e)
        # with open("ProcessedCache/" + str(var_id) + '_variables.pkl', 'wb') as v:
        #     pickle.dump(self.variables, v)
        # with open("ProcessedCache/" + str(var_id) + '_features.pkl', 'wb') as f:
        #     pickle.dump(self.features, f)
        evidence_set, partial_edges, connected_feature_set = evidences
        #平衡化
        if self.balance:
            label0_var = set()
            label1_var = set()
            for var_id in evidence_set:
                if variables[var_id]['label'] == 1:
                    label1_var.add(var_id)
                elif variables[var_id]['label'] == 0:
                    label0_var.add(var_id)
            sampled_label0_var = set(random.sample(list(label0_var), len(label1_var)))
            new_evidence_set = label1_var.union(sampled_label0_var)
            new_partial_edges = list()
            new_connected_feature_set = set()
            for edge in partial_edges:
                if edge[1] in new_evidence_set:
                    new_partial_edges.append(edge)
                    new_connected_feature_set.add(edge[0])
            evidence_set = new_evidence_set
            partial_edges = new_partial_edges
            connected_feature_set = new_connected_feature_set
        var_map = dict()  # 用来记录self.variables与numbskull的variable变量的映射-(self,numbskull)
        # 初始化变量
        var_num = len(evidence_set) + 1  # 证据变量+隐变量
        variable = np.zeros(var_num, Variable)
        # 初始化隐变量,隐变量的id为0，注意区分总体的variables和小因子图中的variable
        variable[0]["isEvidence"] = False
        variable[0]["initialValue"] = self.variables[var_index]['label']
        variable[0]["dataType"] = 0  # datatype=0表示是bool变量，为1时表示非布尔变量。
        variable[0]["cardinality"] = 2
        var_map[var_index] = 0  # (self,numbskull)  隐变量ID为0
        i = 1
        # 初始化证据变量
        for evidence_id in evidence_set:
            var_index = self.find_in_variables(evidence_id)
            variable[i]["isEvidence"] = True  # self.variables[var_index]['is_evidence']
            variable[i]["initialValue"] = self.variables[var_index]['label']
            variable[i]["dataType"] = 0  # datatype=0表示是bool变量，为1时表示非布尔变量。
            variable[i]["cardinality"] = 2
            var_map[evidence_id] = i  # 一一记录
            i += 1
        # 初始化weight,多个因子可以共享同一个weight
        weight = np.zeros(len(connected_feature_set), Weight)  # weight的数目等于此隐变量使用的feature的数目
        feature_map_weight = dict()  # 需要记录feature id和weight id之间的映射 [feature_id,weight_id]
        weight_index = 0
        for feature_id in connected_feature_set:
            weight[weight_index]["isFixed"] = False
            weight[weight_index]["parameterize"] = True
            weight[weight_index]["a"] = self.features[feature_id]['tau']
            weight[weight_index]["b"] = self.features[feature_id]['alpha']
            weight[weight_index]["initialValue"] = random.uniform(-5,5)  # 此处一个weight会有很多个weight_value，此处随机初始化一个，后面应该用不上
            feature_map_weight[feature_id] = weight_index
            weight_index += 1

        # 按照numbskull要求初始化factor,fmap,edges
        edges_num = len(connected_feature_set) + len(partial_edges)  # 边的数目
        factor = np.zeros(edges_num, Factor)  # 目前是全当成单因子，所以有多少个边就有多少个因子
        fmap = np.zeros(edges_num, FactorToVar)
        domain_mask = np.zeros(var_num, np.bool)
        edges = list()
        edge = namedtuple('edge', ['index', 'factorId', 'varId'])  # 单变量因子的边
        factor_index = 0
        fmp_index = 0
        edge_index = 0
        # 先初始化此隐变量上连接的所有单因子
        for feature_id in connected_feature_set:  # dict是无序的，但是得到的keys是有序的
            factor[factor_index]["factorFunction"] = 18
            factor[factor_index]["weightId"] = feature_map_weight[feature_id]
            factor[factor_index]["featureValue"] = feature_set[feature_id][1]
            factor[factor_index]["arity"] = 1  # 单因子度为1
            factor[factor_index]["ftv_offset"] = fmp_index  # 偏移量每次加1
            # 先保存此隐变量上的边
            edges.append(edge(edge_index, factor_index, 0))
            fmap[fmp_index]["vid"] = 0  # edges[factor_index][2]
            fmap[fmp_index]["x"] = feature_set[feature_id][1]  # feature_value
            fmap[fmp_index]["theta"] = feature_set[feature_id][0]  # theta
            fmp_index += 1
            factor_index += 1
            edge_index += 1
        # 再初始化证据变量上连接的单因子
        for elem in partial_edges:  # [feature_id,var_id]
            var_index = self.find_in_variables(elem[1])
            factor[factor_index]["factorFunction"] = 18
            factor[factor_index]["weightId"] = feature_map_weight[elem[0]]
            factor[factor_index]["featureValue"] = self.variables[var_index]['feature_set'][elem[0]][1]
            factor[factor_index]["arity"] = 1  # 单因子度为1
            factor[factor_index]["ftv_offset"] = fmp_index  # 偏移量每次加1
            edges.append(edge(edge_index, factor_index, var_map[elem[1]]))
            fmap[fmp_index]["vid"] = edges[factor_index][2]
            fmap[fmp_index]["x"] = self.variables[var_index]['feature_set'][elem[0]][1]  # feature_value
            fmap[fmp_index]["theta"] = self.variables[var_index]['feature_set'][elem[0]][0]  # theta
            fmp_index += 1
            factor_index += 1
            edge_index += 1
        logging.info("var-" + str(var_id) + " construct subgraph succeed")
        subgraph = weight, variable, factor, fmap, domain_mask, edges_num, var_map, feature_map_weight
        # with open("ProcessedCache/" + str(var_id) + "_subgraph.pkl",'wb') as s:
        #     pickle.dump(subgraph, s)
        return weight, variable, factor, fmap, domain_mask, edges_num, var_map, feature_map_weight

    def construct_subgraph2(self, var_id):
        '''第二种建图方式
        输入：一个隐变量的id
        输出：按照numbskull的要求因子图,返回weight, variable, factor, fmap, domain_mask, edges
        '''
        index = self.find_in_variables(var_id)
        feature_set = self.variables[index]['feature_set']
        evidences = self.select_evidence(var_id)
        # 存储选出的证据和实时的变量和特征
        # with open("ProcessedCache/" + str(var_id) + '_evidences.pkl', 'wb') as e:
        #     pickle.dump(evidences, e)
        # with open("ProcessedCache/" + str(var_id) + 'variables.pkl', 'wb') as v:
        #     pickle.dump(self.variables, v)
        # with open("ProcessedCache/" + str(var_id) + 'features.pkl', 'wb') as f:
        #     pickle.dump(self.features, f)
        evidence_set, partial_edges, connected_feature_set = evidences
        var_map = dict()  # 用来记录self.variables与numbskull的variable变量的映射-(self,numbskull)
        feature_map_weight = dict()  # 用来记录feature id和weight id之间的映射 [feature_id,weight_id]

        var_num = len(evidence_set) + len(connected_feature_set)  # 证据变量+隐变量(隐变量连了几个feature，就复制了几份)
        variable = np.zeros(var_num, Variable)
        weight = np.zeros(len(connected_feature_set), Weight)  # weight的数目等于此隐变量使用的feature的数目
        edges_num = len(connected_feature_set) + len(partial_edges)  # 边的数目
        factor = np.zeros(edges_num, Factor)  # 目前是全当成单因子，所以有多少个边就有多少个因子
        fmap = np.zeros(edges_num, FactorToVar)
        domain_mask = np.zeros(var_num, np.bool)
        edges = list()
        edge = namedtuple('edge', ['index', 'factorId', 'varId'])  # 单变量因子的边

        # 1.初始化权重
        weight_index = 0
        for feature_id in connected_feature_set:
            weight[weight_index]["isFixed"] = False
            weight[weight_index]["parameterize"] = True
            weight[weight_index]["a"] = self.features[feature_id]['tau']
            weight[weight_index]["b"] = self.features[feature_id]['alpha']
            weight[weight_index]["initialValue"] = random.uniform(-5,5)  # 此处一个weight会有很多个weight_value，此处随机初始化一个，后面应该用不上
            feature_map_weight[feature_id] = weight_index
            weight_index += 1
        # 2.初始化隐变量及其相关的fmp，factor
        var_index = 0
        factor_index = 0
        fmp_index = 0
        edge_index = 0
        var_map[index] = var_index  # (self,numbskull)  隐变量ID为0
        for feature_id in connected_feature_set:  # dict是无序的，但是得到的keys是有序的
            variable[var_index]["isEvidence"] = False
            variable[var_index]["initialValue"] = self.variables[index]['label']
            variable[var_index]["dataType"] = 0  # datatype=0表示是bool变量，为1时表示非布尔变量。
            variable[var_index]["cardinality"] = 2

            factor[factor_index]["factorFunction"] = 18
            factor[factor_index]["weightId"] = feature_map_weight[feature_id]
            factor[factor_index]["featureValue"] = feature_set[feature_id][1]
            factor[factor_index]["arity"] = 1  # 单因子度为1
            factor[factor_index]["ftv_offset"] = fmp_index  # 偏移量每次加1
            # 先保存此隐变量上的边
            edges.append(edge(edge_index, factor_index, 0))
            fmap[fmp_index]["vid"] = 0  # edges[factor_index][2]
            fmap[fmp_index]["x"] = feature_set[feature_id][1]  # feature_value
            fmap[fmp_index]["theta"] = feature_set[feature_id][0]  # theta
            var_index += 1
            fmp_index += 1
            factor_index += 1
            edge_index += 1
        # 3.初始化证据变量
        for evidence_id in evidence_set:
            index = self.find_in_variables(evidence_id)
            variable[var_index]["isEvidence"] = True  # self.variables[var_index]['is_evidence']
            variable[var_index]["initialValue"] = self.variables[index]['label']
            variable[var_index]["dataType"] = 0  # datatype=0表示是bool变量，为1时表示非布尔变量。
            variable[var_index]["cardinality"] = 2
            var_map[index] = var_index  # 一一记录
            var_index += 1
        # 4.初始化证据变量及其相关的factor和fmap
        for elem in partial_edges:  # [feature_id,var_id]
            index = self.find_in_variables(elem[1])
            factor[factor_index]["factorFunction"] = 18
            factor[factor_index]["weightId"] = feature_map_weight[elem[0]]
            factor[factor_index]["featureValue"] = self.variables[index]['feature_set'][elem[0]][1]
            factor[factor_index]["arity"] = 1  # 单因子度为1
            factor[factor_index]["ftv_offset"] = fmp_index  # 偏移量每次加1
            edges.append(edge(edge_index, factor_index, var_map[elem[1]]))
            fmap[fmp_index]["vid"] = edges[factor_index][2]
            fmap[fmp_index]["x"] = self.variables[index]['feature_set'][elem[0]][1]  # feature_value
            fmap[fmp_index]["theta"] = self.variables[index]['feature_set'][elem[0]][0]  # theta
            fmp_index += 1
            factor_index += 1
            edge_index += 1

        logging.info("var-" + str(var_id) + " construct subgraph succeed")
        subgraph = weight, variable, factor, fmap, domain_mask, edges_num, var_map, feature_map_weight
        # with open("ProcessedCache/" + str(var_id) + "_subgraph.pkl",'wb') as s:
        #     pickle.dump(subgraph, s)
        return weight, variable, factor, fmap, domain_mask, edges_num, var_map, feature_map_weight

    def inference_subgraph(self, var_id):
        '''推理子图
        输入: 一个隐变量的id
        输出：推理出的此隐变量的概率，此概率需写回variables
        '''
        learn = 1000
        ns = numbskull.NumbSkull(n_inference_epoch=10,
                                 n_learning_epoch=learn,
                                 quiet=True,
                                 learn_non_evidence=True,
                                 stepsize=0.0001,
                                 burn_in=10,
                                 decay=0.001 ** (1.0 / learn),
                                 regularization=1,
                                 reg_param=0.01)
        # 弃用：如果允许并且有缓存过这个变量的图，就不需再建图,图不能缓存,
        # if self.cache_subgraph == True and var_id in self.subgraph_cache.keys():
        #     weight, variable, factor, fmap, domain_mask, edges_num, var_map, feature_map_weight =  self.subgraph_cache[var_id]
        #     subgraph = weight, variable, factor, fmap, domain_mask, edges_num
        #     ns.loadFactorGraph(*subgraph)
        #     logging.info("var-"+str(var_id)+"load subgraph from cache")
        # else:
        weight, variable, factor, fmap, domain_mask, edges_num, var_map, feature_map_weight = self.construct_subgraph(
            var_id)
        subgraph = weight, variable, factor, fmap, domain_mask, edges_num
        ns.loadFactorGraph(*subgraph)
        # 因子图参数学习
        ns.learning()
        logging.info("var-" + str(var_id) + " learning finished")
        # 因子图推理
        # 参数学习完成后将weight的isfixed属性置为true
        for w in ns.factorGraphs[0].weight:
            w["isFixed"] = True
        ns.learn_non_evidence = False
        # ns1 = numbskull.NumbSkull(n_inference_epoch=10,
        #                          n_learning_epoch=learn,
        #                          quiet=True,
        #                          learn_non_evidence=False,
        #                          stepsize=0.0001,
        #                          burn_in=10,
        #                          decay=0.001 ** (1.0 / learn),
        #                          regularization=1,
        #                          reg_param=0.01)
        # ns1.loadFactorGraph(*subgraph)
        ns.inference()
        logging.info("var-" + str(var_id) + " inference finished")
        # 写回概率到self.variables
        self.variables[self.find_in_variables(var_id)]['inference_probability'] = ns.factorGraphs[0].marginals[
            var_map[var_id]]
        logging.info("var-" + str(var_id) + " probability recored")

    def write_labeled_var_to_evidence_interval(self, var_id):
        '''
        因为每个featurew维护了evidence_interval属性，所以每标记一个变量之后，需要更新这个属性
        :param var_id:
        :return:
        '''
        var_index = self.find_in_variables(var_id)
        feature_set = self.variables[var_index]['feature_set']
        for kv in feature_set.items():
            for interval_index in range(0, len(self.evidence_interval)):
                if kv[1][1] >= self.evidence_interval[interval_index][0] and kv[1][1] < \
                        self.evidence_interval[interval_index][1]:
                    self.features[kv[0]]['evidence_interval'][interval_index].add(var_id)
                    self.features[kv[0]]['evidence_count'] += 1

    def label(self, var_id_list):
        '''比较k个隐变量的熵，选熵最小的一个打上标签，并把此图学习出的参数写回self.features
        输入：k个id的列表，每个变量对应的概率从variables中拿
        输出：无输出，直接更新vairables中的label和entropy，顺便可以更新一下observed_variables_id和poential_variables_id
        '''
        entropy_list = list()
        if len(var_id_list) > 1:  # 如果传入的变量个数大于1,就每次选熵最小的进行标记
            for var_id in var_id_list:
                var_index = self.find_in_variables(var_id)
                self.variables[var_index]['entropy'] = General.entropy(
                    self.variables[var_index]['inference_probability'])
                entropy_list.append([var_id, self.variables[var_index]['entropy']])
            min_var = heapq.nsmallest(1, entropy_list, key=lambda x: x[1])  # 选出熵最小的变量
            var = min_var[0][0]
        else:
            var = var_id_list[0]
        var_index = self.find_in_variables(var)  # 如果传入的只有1个变量，直接进行标记即可
        self.variables[var_index]['label'] = 1 if self.variables[var_index]['inference_probability'] >= 0.5 else 0
        self.variables[var_index]['probability'] = self.variables[var_index]['inference_probability']
        self.variables[var_index]['is_evidence'] = True
        logging.info('var-' + str(var) + " labeled succeed---------------------------------------------")
        self.poential_variables_id.remove(var)
        self.observed_variables_id.add(var)
        self.labeled_variables_id.add(var)
        self.write_labeled_var_to_evidence_interval(var)
        with open(self.datapath+self.dataname+'_result.txt', 'a') as f:
            f.write(str(var) + " " + str(self.variables[var_index]['label']) + ' ' + str(
                self.variables[var_index]['probability']) + '\n')
        return var
        # #将被标记的图学得的参数写回self.features
        # weight, variable, factor, fmap, domain_mask, edges_num, var_map, feature_map_weight = self.subgraph_cache[min_var[0][0]]
        # weight_map_feature = dict([val, key] for key, val in feature_map_weight.items())  # 键/值互换
        # for index in range(len(weight)):
        #     self.features[weight_map_feature[index]]['tau'] = weight[index]['a']
        #     self.features[weight_map_feature[index]]['alpha'] = weight[index]['b']

    def inference(self):
        '''主流程'''
        update_cache = int(
            self.update_proportion * len(self.poential_variables_id))  # 每推理update_cache个变量后需要重新计算evidential support
        labeled_var = 0
        labeled_count = 0
        var = 0  # 每轮标记的变量id
        update_feature_set = set()  # 存储一轮更新期间证据支持发生变化的feature
        inferenced_variables_id = set()  # 一轮更新期间已经建立过因子图并推理的隐变量
        for feature in self.features:
            update_feature_set.add(feature['feature_id'])
        self.influence_modeling(update_feature_set)
        update_feature_set.clear()
        self.evidential_support()
        self.approximate_probability_estimation(list(self.poential_variables_id))
        logging.info("approximate_probability calculate finished")
        while len(self.poential_variables_id) > 0:
            # 当标记的变量数目达到update_cache时，重新回归并计算evidential support
            if labeled_var == update_cache:
                for var_id in self.labeled_variables_id:
                    var_index = self.find_in_variables(var_id)
                    for feature_id in self.variables[var_index]['feature_set'].keys():
                        update_feature_set.add(feature_id)
                self.influence_modeling(update_feature_set)
                self.evidential_support()
                self.approximate_probability_estimation(list(self.poential_variables_id))
                logging.info("approximate_probability calculate finished")
                labeled_var = 0
                update_feature_set.clear()
                self.labeled_variables_id.clear()
                inferenced_variables_id.clear()
            if len(self.poential_variables_id) >= self.top_m:  # 如果隐变量数目不足topm个，就不需要再选topm了,并且需要从topm实时移除已经标记的变量
                m_list = self.select_top_m_by_es(self.top_m)
            else:
                m_list.remove(var)
            if len(self.poential_variables_id) >= self.top_k:  # 如果隐变量数目不足topk个，就不需要再选topk了,并且需要从topk中实时移除已经标记的变量
                k_list = self.select_top_k_by_entropy(m_list, self.top_k)
            else:
                k_list.remove(var)
            # 只要没有进行更新,就每次只推理新增的变量
            add_list = [x for x in k_list if x not in inferenced_variables_id]
            if len(add_list) > 0:
                for var_id in add_list:
                    # if var_id not in inferenced_variables_id:
                    self.inference_subgraph(var_id)
                    # 每轮更新期间推理过的变量，因为参数没有更新，所以无需再进行推理。
                    inferenced_variables_id.add(var_id)
            var = self.label(k_list)
            labeled_var += 1
            labeled_count += 1
            logging.info("label_num=" + str(labeled_count))

    # def results(self):
    #     '''计算所有相关结果，precison,recall等
    #     暂时先返回所有变量的标签
    #     '''
    #     label_list = list()
    #     for var in self.variables:
    #         label_list.append([var['var_id'],var['label']])
    #     return label_list


if __name__ == '__main__':
    # variables, features, edges, easys = data_pre.get_data()
    # FeatureExtract()
    warnings.filterwarnings('ignore')  # 过滤掉warning输出
    # begin_time = time.time()
    dataname = 'songs'
    datapath = 'ProcessedCache/'
    # with open(datapath+dataname+'_variables.pkl', 'rb') as v:
    #     variables = pickle.load(v)
    # with open(datapath+dataname+'_features.pkl', 'rb') as f:
    #     features = pickle.load(f)
    # with open(datapath+dataname+'_edges.pkl', 'rb') as e:
    #     edges = pickle.load(e)
    # easys = EasyInstanceLabeling.load_easy_instance_from_file(datapath+dataname+'_easys.csv')
    graph = GML(dataname, datapath,variables, features, edges, easys, top_m=2000, top_k=10, update_proportion=0.01,
                tau_and_regression_bound=10,balance = False)
    graph.zh_test()
    print("test over!")
    for i in range(len(variables)):
        print("----------", i)
        print(graph.variables[i]["evidential_support"])
    # graph.init()
    # graph.inference()
    # General.print_results(dataname,datapath)
    # end_time = time.time()
    # print('Running time: %s Seconds' % (end_time - begin_time))
