#Data Structure
'''
变量
variables=list()   #元素类型为dict:variable
variable
{
    #由用户提供的属性
    'var_id':1,                  #变量ID   int类型，从0开始
    'is_easy': False,            #是否是Easy
    'is_evidence':True,          #是否是证据变量
    'true_label':1,              #真实标签，用于后续计算精度等
    'label': 1，                 #推理出的标签:0为negative,1为为positive, -1为不知道
    'feature_set':               #此变量拥有的所有feature
    { feature_id1: [theta1,feature_value1],
      feature_id2: [theta2,feature_value2],
      ...
    },
    #代码运行期间可能会自动生成的属性
    'probability':0.99，          #推理出的概率
    'evidential_support':0.3,     #evidential_support
    'entropy': 0.4,               #熵
    'approximate_weight':0.3,     #近似权重
     ...
}

特征
features  = list()      #元素类型为dict:feature
feature
{
    #需要用户提供的属性
    'feature_id':1,                                       #特征id, int类型，从0开始
    'feature_type': unary_feature/binary_feature,         #区分此特征是单因子特征还是双因子特征，目前支持unary_feature和binary_feature两种
    'feature_name': good,                                 #特征名，如果是token类型，就是token具体的词，如果是其他类型的特征，就是特征的具体类型
    'weight':                                             #拥有此特征的所有变量的集合
    {
      var_id1:        [weight_value1,feature_value1],     #unary_feature
     (varid3,varid4): [weight_value2,feature_value2],     #binary_feature
      ...
    }
    #代码运行期间可能会自动生成的属性
    'tau':0,
    'alpha':0,
    'regerssion'： object,
    ...
}
'''


