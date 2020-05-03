#提供挑选证据的一些方法
import random


class EvidenceSelect:
    def __init__(self,variables,features):
        self.variables = variables
        self.features = features

    def select_evidence_by_interval(self, var_id,interval_evidence_count):
        '''
        按照feature_value的区间为指定的隐变量挑一定数量的证据变量,适用于ER
        目前是每个feature划分evidence_interval_count个区间，每个区间挑不超过interval_evidence_count个
        输入：var_id -- 隐变量id
             interval_evidence_count  -- 每个区间挑选的证据个数

        输出：
        connected_var_set --证据变量的id集合
        connected_edge_set -- 边的集合
        connected_feature_set --能用得上的feature的集合
        '''

        connected_var_set = set()
        connected_edge_set = set()
        connected_feature_set = set()  # 记录此隐变量上建因子图时实际保留了哪些feature
        feature_set = self.variables[var_id]['feature_set']
        for feature_id in feature_set.keys():
            if self.features[feature_id]['evidence_count'] > 0:  # 有些feature上没有连接证据变量，就不用再加进来
                connected_feature_set.add(feature_id)
                evidence_interval = self.features[feature_id]['evidence_interval']
                for interval in evidence_interval:
                    # 如果这个区间的证据变量小于200，就全加进来
                    if len(interval) <= interval_evidence_count:
                        connected_var_set = connected_var_set.union(interval)
                        for id in interval:
                            connected_edge_set.add((feature_id, id))
                    else:
                        # 如果大于200,就随机采样200个
                        sample = random.sample(list(interval), interval_evidence_count)
                        connected_var_set = connected_var_set.union(sample)
                        for id in sample:
                            connected_edge_set.add((feature_id, id))

        print("var-" + str(var_id) + " select evidence by interval finished")
        return connected_var_set, connected_edge_set, connected_feature_set

    def select_evidence_by_realtion(self, var_id_list, subgraph_limit_num=1000, k_hop=2):
        '''为选出的top_k个隐变量挑选证据，适用于ALSA
        输入：
        var_id_list --- k个变量id的列表
        subgraph_limit_num  --子图允许的最大变量个数
        k_hop   -- 找相邻变量的跳数

        输出：
        connected_var_set --证据变量的id集合
        connected_edge_set -- 边的集合
        connected_feature_set --能用得上的feature的集合
        '''
        connected_var_set = set()
        connected_edge_set = set()
        connected_feature_set = set()  # 记录此隐变量上建因子图时实际保留了哪些feature
        connected_var_set = connected_var_set.union(set(var_id_list))
        current_var_set = connected_var_set
        next_var_set = set()
        # 先找relation型特征的k-hop跳的证据变量(需确定此处是否只添加证据变量，不包括隐变量)
        for k in range(k_hop):
            for var_id in current_var_set:
                feature_set = self.variables[var_id]['feature_set']
                for feature_id in feature_set.keys():
                    if self.features[feature_id]['feature_type'] == 'binary_feature':
                        weight = self.features[feature_id]['weight']
                        for id in weight.keys():
                            if type(id) == tuple and var_id in id:
                                another_var_id = id[0] if id[0] != var_id else id[1]
                                if self.variables[another_var_id]['is_evidence'] == True:
                                    next_var_set.add(another_var_id)
                                    connected_feature_set.add(feature_id)
                                    connected_edge_set.add((feature_id,id))
                connected_var_set = connected_var_set.union(next_var_set)
                current_var_set = next_var_set
                next_var_set.clear()
        #再找和这k个变量共享word型feature的变量（先加证据变量，如果没有超过最大变量限制，再加隐变量）
        subgraph_capacity = subgraph_limit_num - len(connected_var_set)
        unary_connected_unlabeled_var = list()
        unary_connected_unlabeled_edge = list()
        unary_connected_unlabeled_feature = list()
        unary_connected_evidence_var = list()
        unary_connected_evidence_edge = list()
        unary_connected_evidence_feature = list()
        for var_id in var_id_list:
            feature_set = self.variables[var_id]['feature_set']
            for feature_id in feature_set.keys():
                if self.features[feature_id]['feature_type'] == 'unary_feature':
                    weight = self.features[feature_id]['weight']
                    for id in weight.keys():
                        if self.variables[id]['is_evidence'] == True:
                            unary_connected_evidence_var.append(id)
                            unary_connected_evidence_feature.append(feature_id)
                            unary_connected_evidence_edge.append((feature_id,id))
                        else:
                            unary_connected_unlabeled_var.append(id)
                            unary_connected_unlabeled_feature.append(feature_id)
                            unary_connected_unlabeled_edge.append((feature_id, id))
        #限制子图规模大小
        if(len(unary_connected_evidence_var) <= subgraph_capacity ):
            connected_var_set = connected_var_set.union(set(unary_connected_evidence_var))
            connected_feature_set = connected_feature_set.union((set(unary_connected_evidence_feature)))
            connected_edge_set = connected_edge_set.union(set(unary_connected_evidence_edge))
            if(len(unary_connected_unlabeled_var) <= (subgraph_capacity-len(unary_connected_evidence_var))):
                connected_var_set = connected_var_set.union(set(unary_connected_unlabeled_var))
                connected_feature_set = connected_feature_set.union((set(unary_connected_unlabeled_feature)))
                connected_edge_set = connected_edge_set.union(set(unary_connected_unlabeled_edge))
            else:
                connected_var_set = connected_var_set.union(set(unary_connected_unlabeled_var[:subgraph_capacity-len(unary_connected_evidence_var)]))
                connected_feature_set = connected_feature_set.union(set(unary_connected_unlabeled_feature[:subgraph_capacity-len(unary_connected_evidence_var)]))
                connected_edge_set = connected_edge_set.union(set(unary_connected_unlabeled_edge[:subgraph_capacity-len(unary_connected_evidence_var)]))
        else:
            connected_var_set = connected_var_set.union(set(unary_connected_evidence_var[:subgraph_capacity]))
            connected_feature_set = connected_feature_set.union((set(unary_connected_evidence_feature[:subgraph_capacity])))
            connected_edge_set = connected_edge_set.union(set(unary_connected_evidence_edge[:subgraph_capacity]))
        print("select evidece by relation finished")
        return connected_var_set, connected_edge_set, connected_feature_set