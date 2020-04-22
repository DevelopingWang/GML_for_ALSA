#Data Structure
'''
变量
variables=list() : 元素类型为dict:variable
variable
{
    'var_id':1,                  #变量ID
    'is_easy':False,         #是否是Easy
    'is_evidence':True,     #是否是证据变量
    'label': 1，             #推理出的标签:0为unmatch,1为unmatch,-1为不知道
    'true_label':1,          #真实标签，用于后续计算精度等
    'probability':0.99，     #匹配概率,不知道的全初始化为0或者nan等
    'approximate_weight':0.3   #近似权重
    'evidential_support':0.3,    #evidential_support，不知道的全初始化为0或者nan等
    'feature_set':                    #此变量所有的feature
    { feature_id1:[theta1,feature_value1],
      feature_id2:[theta2,feature_value2],
        ....                   
    },
    'entropy': 0.4,                #熵,不知道的全初始化为0
}

特征
features  = list() ： 元素类型为dict:feature
feature
{
    'feature_id':1,
    'feature_type':sim.name.cal_Monge_Eklan,   #区分此feature是哪个属性的相似度还是token
    'feature_name': good,                      #区分是哪个token
    'tau':0,
    'alpha':0,
    'regerssion'： object
    'weight':  #从这可以找到有这个特征的所有变量，因为不同变量的feature_value不同，所以计算出来的weight也不同
    {
      var_id1: [weight_value1,feature_value1]
      var_id2: [weight_value2,feature_value2]
      ...
    }
    #针对relation feature,需要存两个id
    'weight':
    {
      (varid1,varid2): [weight_value1,feature_value1]
      (varid3,varid4]):[weight_value1,feature_value1]
      ...
    }
}

边
edges = list():元素为dict:edge
edge
{
    'var_id':0,
    'feature_id':1,
    'feature_value':0.5
    'theta': 0.2
}
'''

#徐驳
class FeatureExtract:     
    '''
    包括数据准备，提取特征等。
    对于以上数据结构中不知道的项可先赋值为Nan或者inf等。
    return variables,features,edges
    '''
    pass
 

#王晨宇
class EasyInstanceLabeling:
    '''
    标Easy
    return easys
    easy的元素结构为:
    {'var_id':0,
      'label':1
    }
    '''
    pass

#贺兴隆，张晗，陈安琪
class GML:
    '''
    GML大类: 包括计算Evidential Support，Approximate Estimation of Inferred Probability，
    Construction of Inference Subgraph等；不包括Feature Extract和Easy Instance Labeling
    '''





    

