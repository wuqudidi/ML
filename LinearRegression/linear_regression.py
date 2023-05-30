import numpy as np
#数据的预处理
from utils.features import prepare_for_training
#记录一个报错：TypeError: 'module' object is not callable
# from XXX import YYY 是从包里导出模块，当包下方有__init__文件，且文件中写了from .prepare_for_training import prepare_for_training
#就是可以理解为，包外的文件下次导出时可以直接使用该文件的函数，导出的模块有且仅有一个函数否则没有__init__文件的标注，则导出的是模块，引用函数时要 使用 .
#即prepare_for_training.prepare_for_training()
#定义一个类
class LinearRegression:
    #初始化，传进数据、标签、 、 、预处理标记为真即数据需要进行预处理
    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data=True):
        #对数据进行预处理操作
        #拿到预处理后的数据，即标准化的结果、mean值和std标准差
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data,polynomial_degree = 0,sinusoid_degree =0,normalize_data=True)
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        #先得到所有的特征个数
        #shape读取矩阵长度，shape[0]d读取行数，shape[1]读取列数
        num_features  = self.data.shape[1]
        #初始化参数矩阵
        #theta个数与特征个数即x的个数一一对应,np.zeros()初始化一个矩阵，num_features的行数，1列
        self.theta = np.zeros((num_features,1))


    #训练函数，传进来一个学习率即步长，越小较为合理；再传一个迭代次数，一般为64、128、256等，这用500
    def train(self,alpha,num_iterations = 500):
        #x训练模块，执行梯度下降
        cost_history = self.gradient_descent(alpha,num_iterations)
        return self.theta,cost_history



    #执行梯度下降、参数更新和梯度计算
    def gradient_descent(self,alpha,num_iterations):
        #实际迭代模块， 会迭代num_iterations次
        # 每一次损失的变化，损失值函数，指定一个list
        cost_history = []
        #迭代进行
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            #append 向列表末尾添加元素
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history



    #执行参数更新,梯度下降参数更新计算方法，注意是矩阵计算
    def gradient_step(self,alpha):
        #需要学习率，样本个数，预测值，真实值
        #样本个数
        num_examples = self.data.shape[0]
        #预测值,真实的数据（样本）乘以参数
        #直接调用，使用类.函数
        prediction = LinearRegression.hypothesis(self.data,self.theta)
        #参差，真实值即标签
        delta = prediction - self.labels
        theta = self.theta
        #参数更新,这里的delta是shape（num_features,）即n行1列
        #需要将theta转置，虽然直接乘不会报错但含义错了，直接乘即是第一个theta值乘所有的样本了
        #但要的是每一个theta值乘每一个data值，转成行才能如此操作即delta.T，然后后边的self.data会由于前面的转置成1行n列后
        #由np.dot函数自动再把self.data转成n行1列
        #最后(np.dot(delta,self.data )）这个结果是个列矩阵，再转成行矩阵即(np.dot(delta.T,self.data ).T
        theta = theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
        self.theta = theta


    #损失函数，损失计算方法
    def cost_function(self,data,labels):
        num_examples  = data.shape[0]
        delta = LinearRegression.hypothesis(self.data,self.theta)-labels
        cost = 1/2*np.dot(delta.T,delta)/num_examples
        return cost[0][0]



    #调用得到预测值
    #预测值的函数,为了方便，使用静态方法声明，无需实例化即可调用
    @staticmethod
    def hypothesis(data,theta):
        # 预测值,真实的数据（样本）乘以参数
        #np.dot函数相乘运算时向量是否需要转置，计算时已经自动转置了。
        #后者列与前者列相同 ，但乘法是需要后者的行与前者列数相同，则自动转置
        #这里theta由shape(，num_features)自动转成shape(num_features,)
        predictions = np.dot(data,theta)
        return predictions


    #得到损失值
    def get_cost(self,data,labels):
       data_processed= prepare_for_training (data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data
         )[0]
       return self.cost_function(data_processed,labels)


    #用训练的参数模型，去预测得到回归值结果
    def predict(self,data):
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]
        predictions = LinearRegression.hypothesis(data_processed,self.theta)
        return predictions