import numpy as np 
import logging
import numbskull
from numbskull.numbskulltypes import *
from evidence_select import EvidenceSelect
    
class ConstructSubgraph:

    def __init__(self, variables, features):
        self.variables = variables
        self.features = features

    def construct_subgraph_for_ALSA(self, k_id_list):
        evidences = EvidenceSelect.select_evidence_by_realtion(k_id_list)
        connected_var_set, connected_edge_set, connected_feature_set = evidences

        var_map = dict()   #用来记录self.variables与numbskull的variable变量的映射-(self,numbskull)
        #初始化变量
        var_num = len(connected_var_set)
        variable = np.zeros(var_num, Variable)
        i = 0
        for var_id in connected_var_set:
            var_index = var_id
            variable[i]["isEvidence"] = self.variables[var_index]['is_evidence']
            variable[i]["initialValue"] = self.variables[var_index]['label']
            variable[i]["dataType"] = 0  
            variable[i]["cardinality"] = 2
            var_map[var_id] = i     #一一记录
            i += 1

        binary = []  #双因子数组，元素为 [(var_id, var_id), weight]
        unary = []   #单因子数组，元素为 [feature_id, weight]
        unary_factor_num = 0    #单因子个数
        for fea_id in connected_feature_set:
            if self.features[fea_id]['feature_type'] == 'binary_feature':
                for (feature_id, id) in connected_edge_set:
                    if fea_id == feature_id and type(id) == tuple:
                        binary.append([id, self.features[fea_id]['weight'][id][0]])
            else:
                w = list(self.features[fea_id]['weight'].values())[0][0]  #[[2.0, '["\'m", \'pleas\']'], [2.0, '["\'m", \'pleas\']']] 得到初始权重2.0
                unary_factor_num += len(self.features[fea_id]['weight'])
                unary.append([fea_id, w])

        #初始化weight,多个单因子可以共享同一个weight
        weight = np.zeros(len(unary) + len(binary), Weight)  #weight的数目单因子数目+双因子数目
        weight_index = 0
        for unary_factor in unary:
            weight[weight_index]["isFixed"] = False
            weight[weight_index]["parameterize"] = False
            weight[weight_index]["initialValue"] = unary_factor[1]
            weight_index += 1
        for binary_factor in binary:
            weight[weight_index]["isFixed"] = False
            weight[weight_index]["parameterize"] = False
            weight[weight_index]["initialValue"] = binary_factor[1]
            weight_index += 1

        # 按照numbskull要求初始化factor,fmap,edges
        factor_num = len(binary) + unary_factor_num
        factor = np.zeros(factor_num, Factor) 
        edges_num = unary_factor_num + 2 * len(binary)
        fmap = np.zeros(edges_num, FactorToVar)
        domain_mask = np.zeros(var_num, np.bool)

        factor_index = 0
        fmp_index = 0
        weight_id = 0
        for [feature_id, w] in unary:
            for var_id in self.features[feature_id]['weight'].keys():
                factor[factor_index]["factorFunction"] = 18
                factor[factor_index]["weightId"] = weight_id
                factor[factor_index]["featureValue"] = 1
                factor[factor_index]["arity"] = 1  # 单因子度为1
                factor[factor_index]["ftv_offset"] = fmp_index  # 偏移量每次加1

                fmap[fmp_index]["vid"] = var_map[var_id]
                fmap[fmp_index]["theta"] = 1
                fmp_index += 1
                factor_index += 1
            weight_id += 1
        for [(var_id1, var_id2), w] in binary:
            factor[factor_index]["factorFunction"] = 9
            factor[factor_index]["weightId"] = weight_id
            factor[factor_index]["featureValue"] = 1
            factor[factor_index]["arity"] = 2  # 双因子度为2
            factor[factor_index]["ftv_offset"] = fmp_index  # 偏移量每次加2

            fmap[fmp_index]["vid"] = var_map[var_id1]
            fmap[fmp_index]["theta"] = 1
            fmap[fmp_index + 1]["vid"] = var_map[var_id2]
            fmap[fmp_index + 1]["theta"] = 1
            fmp_index += 2
            factor_index += 1
            weight_id += 1

        logging.info("construct subgraph for ALSA succeed")

        return weight, variable, factor, fmap, domain_mask, edges_num, var_map



    def construct_subgraph_for_ER(self,var_id):
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
        evidence_set,partial_edges,connected_feature_set = EvidenceSelect.select_evidence_by_interval(var_id)
        #平衡化
        if self.balance:
            label0_var = set()
            label1_var = set()
            for varid in evidence_set:
                if variables[varid]['label'] == 1:
                    label1_var.add(varid)
                elif variables[varid]['label'] == 0:
                    label0_var.add(varid)
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
            if var_id == elem[1]:
                continue
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
        return weight, variable, factor, fmap, domain_mask, edges_num, var_map

    def construct_subgraph_for_ER2(self, var_id):
        '''第二种建图方式
        输入：一个隐变量的id
        输出：按照numbskull的要求因子图,返回weight, variable, factor, fmap, domain_mask, edges
        '''
        index = var_id
        feature_set = self.variables[index]['feature_set']
        evidences = EvidenceSelect.select_evidence_by_interval(var_id)
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
            index = evidence_id
            variable[var_index]["isEvidence"] = True  # self.variables[var_index]['is_evidence']
            variable[var_index]["initialValue"] = self.variables[index]['label']
            variable[var_index]["dataType"] = 0  # datatype=0表示是bool变量，为1时表示非布尔变量。
            variable[var_index]["cardinality"] = 2
            var_map[index] = var_index  # 一一记录
            var_index += 1
        # 4.初始化证据变量及其相关的factor和fmap
        for elem in partial_edges:  # [feature_id,var_id]
            if var_id == elem[1]:
                continue
            index = elem[1]
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
        return weight, variable, factor, fmap, domain_mask, edges_num, var_map