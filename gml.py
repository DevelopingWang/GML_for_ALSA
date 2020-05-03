import heapq
import math
import pickle
from collections import namedtuple
from copy import copy
from scipy.sparse import *
from scipy.stats import t
from sklearn.linear_model import LinearRegression
import numbskull
from numbskull.numbskulltypes import *
import random
import logging
import time
import warnings
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics
import gml_utils
from evidential_support import EvidentialSupport
from easy_instance_labeling import EasyInstanceLabeling
from evidence_select import EvidenceSelect


class GML:
    '''GML大类: 包括计算Evidential Support，Approximate Estimation of Inferred Probability，
    Construction of Inference Subgraph等；不包括Feature Extract和Easy Instance Labeling
    在实现过程中，注意区分实例变量和类变量
    '''
    def __init__(self,variables, features, es_method, ev_method,top_m=2000, top_k=10, update_proportion=0.01,
                 balance = False):
        '''
        '''
        self.variables = variables
        self.features = features
        self.es_method = es_method
        self.ev_method = ev_method
        self.labeled_variables_set = set()  # 所有新标记变量集合
        self.top_m = top_m
        self.top_k = top_k
        self.update_proportion = update_proportion
        self.balance = balance
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables()
        self.es = EvidentialSupport(self.variables, self.features)
        self.ev = EvidenceSelect(self.variables, self.features)
        logging.basicConfig(
            level=logging.INFO,  # 设置输出信息等级
            format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s'  # 设置输出格式
        )


    def evidential_support(self,update_feature_set):
        if self.es_method == 'interval':
            self.es.evidential_support_by_regression(update_feature_set)
        if self.es_method == 'realtion':
            self.es.evidential_support_by_relation(update_feature_set)

    def select_top_m_by_es(self, m):
        '''根据计算出的Evidential Support(从大到小)选前m个隐变量
        输入：
        1.  m----需要选出的隐变量的个数
        输出： 一个包含m个变量id的列表
        '''
        # 此处选只能在所有隐变量里面选
        poential_var_list = list()
        m_id_list = list()
        for var_id in self.poential_variables_set:
            poential_var_list.append([var_id, self.variables[var_id]['evidential_support']])
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
            var_index = var_id
            self.variables[var_index]['entropy'] = gml_utils.entropy(self.variables[var_index]['probability'])
            m_list.append(self.variables[var_index])
        k_list = heapq.nsmallest(k, m_list, key=lambda x: x['entropy'])
        for var in k_list:
            k_id_list.append(var['var_id'])
        logging.info('select k finished')
        return k_id_list


    def construct_subgraph(self,var_id,evidences):
        '''在选出topk个隐变量之后建立子图
        输入：一个隐变量的id
        输出：按照numbskull的要求因子图,返回weight, variable, factor, fmap, domain_mask, edges
        '''
        var_index = var_id
        feature_set = self.variables[var_index]['feature_set']
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
            new_partial_edges = set()
            new_connected_feature_set = set()
            for edge in partial_edges:
                if edge[1] in new_evidence_set:
                    new_partial_edges.add(edge)
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
            var_index = evidence_id
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
            var_index = elem[1]
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

        if self.ev_method == 'interval':
            evidences = self.ev.select_evidence_by_interval(var_id,10)
        elif self.ev_method == 'relation':
            evidences = self.ev.select_evidence_by_realtion(k_id_list, subgraph_limit_num=1000, k_hop=2)
        weight, variable, factor, fmap, domain_mask, edges_num, var_map, feature_map_weight = self.construct_subgraph(
            evidences,var_id)
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
        ns.inference()
        logging.info("var-" + str(var_id) + " inference finished")
        # 写回概率到self.variables
        self.variables[var_id]['probability'] = ns.factorGraphs[0].marginals[
            var_map[var_id]]
        logging.info("var-" + str(var_id) + " probability recored")


    def label(self, var_id_list):
        '''比较k个隐变量的熵，选熵最小的一个打上标签，并把此图学习出的参数写回self.features
        输入：k个id的列表，每个变量对应的概率从variables中拿
        输出：无输出，直接更新vairables中的label和entropy，顺便可以更新一下observed_variables_id和poential_variables_id
        '''
        entropy_list = list()
        if len(var_id_list) > 1:  # 如果传入的变量个数大于1,就每次选熵最小的进行标记
            for var_id in var_id_list:
                var_index = var_id
                self.variables[var_index]['entropy'] = gml_utils.entropy(
                    self.variables[var_index]['probability'])
                entropy_list.append([var_id, self.variables[var_index]['entropy']])
            min_var = heapq.nsmallest(1, entropy_list, key=lambda x: x[1])  # 选出熵最小的变量
            var = min_var[0][0]
        else:
            var = var_id_list[0]
        var_index = var # 如果传入的只有1个变量，直接进行标记即可
        self.variables[var_index]['label'] = 1 if self.variables[var_index]['probability'] >= 0.5 else 0
        self.variables[var_index]['is_evidence'] = True
        logging.info('var-' + str(var) + " labeled succeed---------------------------------------------")
        self.poential_variables_set.remove(var)
        self.observed_variables_set.add(var)
        self.labeled_variables_set.add(var)
        with open(self.dataname+'/result.txt', 'a') as f:
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
            self.update_proportion * len(self.poential_variables_set))  # 每推理update_cache个变量后需要重新计算evidential support
        labeled_var = 0
        labeled_count = 0
        var = 0  # 每轮标记的变量id
        update_feature_set = set()  # 存储一轮更新期间证据支持发生变化的feature
        inferenced_variables_id = set()  # 一轮更新期间已经建立过因子图并推理的隐变量
        for feature in self.features:
            update_feature_set.add(feature['feature_id'])
        self.evidential_support(update_feature_set)
        update_feature_set.clear()
        self.approximate_probability_estimation(list(self.poential_variables_set))
        logging.info("approximate_probability calculate finished")
        while len(self.poential_variables_set) > 0:
            # 当标记的变量数目达到update_cache时，重新回归并计算evidential support
            if labeled_var == update_cache:
                for var_id in self.labeled_variables_set:
                    var_index = var_id
                    for feature_id in self.variables[var_index]['feature_set'].keys():
                        update_feature_set.add(feature_id)
                self.evidential_support(update_feature_set)
                self.approximate_probability_estimation(list(self.poential_variables_set))
                logging.info("approximate_probability calculate finished")
                labeled_var = 0
                update_feature_set.clear()
                self.labeled_variables_set.clear()
                inferenced_variables_id.clear()
            if len(self.poential_variables_set) >= self.top_m:  # 如果隐变量数目不足topm个，就不需要再选topm了,并且需要从topm实时移除已经标记的变量
                m_list = self.select_top_m_by_es(self.top_m)
            else:
                m_list.remove(var)
            if len(self.poential_variables_set) >= self.top_k:  # 如果隐变量数目不足topk个，就不需要再选topk了,并且需要从topk中实时移除已经标记的变量
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

    def score(self):
        # 计算推理结果的精确率，召回率，f1值
        easys_pred_label = list()
        easys_true_label = list()
        hards_pred_label = list()
        hards_true_label = list()
        for var in self.variables:
            if var['is_easy'] == True:
                easys_true_label.append(var['true_label'])
                easys_pred_label.append(var['label'])
            else:
                hards_true_label.append(var['true_label'])
                hards_pred_label.append(var['label'])

        all_true_label = easys_true_label + hards_true_label
        all_pred_label = easys_pred_label + hards_pred_label

        print("--------------------------------------------")
        print("total:")
        print("--------------------------------------------")
        print("total precision_score: " + str(metrics.precision_score(all_true_label, all_pred_label)))
        print("total recall_score: " + str(metrics.recall_score(all_true_label, all_pred_label)))
        print("total f1_score: " + str(metrics.f1_score(all_true_label, all_pred_label)))
        print("--------------------------------------------")
        print("easys:")
        print("--------------------------------------------")
        print("easys precision_score:" + str(metrics.precision_score(easys_true_label, easys_pred_label)))
        print("easys recall_score:" + str(metrics.recall_score(easys_true_label, easys_pred_label)))
        print("easys f1_score: " + str(metrics.f1_score(easys_true_label, easys_pred_label)))
        print("--------------------------------------------")
        print("hards:")
        print("--------------------------------------------")
        print("hards precision_score: " + str(metrics.precision_score(hards_true_label, hards_pred_label)))
        print("hards recall_score: " + str(metrics.recall_score(hards_true_label, hards_pred_label)))
        print("hards f1_score: " + str(metrics.f1_score(hards_true_label, hards_pred_label)))

if __name__ == '__main__':

    warnings.filterwarnings('ignore')  #过滤掉warning输出
    begin_time = time.time()
    easys = list()
    easys_element = dict()
    easy_file = pd.read_csv('data/abtbuy_easys.csv')
    for easy in easy_file:
        easys_element[easy['id']] = easy['label']
        easys.append(easys_element)
        easys_element.clear()

    with open('data/abtbuy_variables.pkl', 'rb') as v:
        variables = pickle.load(v)
        v.close()
    with open('data/abtbuy_features.pkl', 'rb') as f:
        features = pickle.load(f)
        f.close()
    EasyInstanceLabeling(variables,features,easys).label_easy_by_file()
    graph = GML(variables, features, es_method = 'interval', ev_method = 'interval', top_m = 2000, top_k=10, update_proportion=0.01,balance = False)
    graph.inference()
    graph.score()
    end_time = time.time()
    print('Running time: %s Seconds' % (end_time - begin_time))
