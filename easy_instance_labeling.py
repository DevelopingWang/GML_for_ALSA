

class EasyInstanceLabeling:
    '''
    用于简单实例标注的类，提供了一些用于标记easy的方法
    return easys
    '''
    def __init__(self,variables,features, easys = None):

        self.variables = variables
        self.features  = features
        self.easys = easys

    def label_easy_by_file(self):
        '''根据提供的easy列表标出variables中的Easy'''
        if self.easys != None and type(self.easys) == list:
            for var in self.variables:
                var['is_easy'] = False
                var['is_evidence'] = False
            for easy in self.easys:
                var_index = easy['var_id']
                self.variables[var_index]['is_easy'] = True
                self.variables[var_index]['is_evidence'] = True

    def label_easy_by_clustering(self,easy_proportion = 0.3):
        #通过聚类标easy
        pass
    def label_easy_by_custom(self):
        pass
