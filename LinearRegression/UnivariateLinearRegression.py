import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression
#debug,右边代码标红点，需要测试的每行都标红点，再点击下方的debug中的像播放键一样的，一行一行的看变量的赋值，直到有问题的出现，看变量时也可以使用
#一份数据导入12列后，取80%用作训练，后20%用作测试，划分数据集后，调用csv的某列的值用data.[[cloumn_name]].values
#导入数据集
data = pd.read_csv('../data/world-happiness-report-2017.csv')
#指定比例
#sample()参数frac是要返回的比例，比如df中有10行数据，我只想返回其中的80%,那么frac=0.8原数据是dataframe，可以使用sample函数进行打乱
#得到训练和测试数据
train_data = data.sample( frac = 0.8 )
#drop()函数的功能是通过指定的索引或标签名称，也就是行名称或者列名称进行删除数据。
#在划分数据集的时候，生成了训练集，把被分到训练集的样本剔除掉，剩下的就是测试集了。
test_data = data.drop(train_data.index)

#标注要取出的列的名字，此处的input列即样本值，output列即标签值1
input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

print（train_data[input_param_name]）
#train_data是个csv的数据集，视为矩阵，train_data[]则把其中的多维元素
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

plt.scatter(x_train,y_train,label='Train data')
plt.scatter(x_test,y_test,label='test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()

num_iterations = 500
learning_rate =  0.01
#实例化
linear_regression =  LinearRegression(x_train,y_train)
(theta,cost_history) = linear_regression.train(learning_rate,num_iterations)

print('开始时的损失：',cost_history[0])
print('训练后的损失：',cost_history[-1])

plt.plot(range(num_iterations),cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()

predictions_num = 100
x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
y_predictions = linear_regression.predict((x_predictions))

plt.scatter(x_train,y_train,label='Train data')
plt.scatter(x_test,y_test,label='test data')
plt.plot(x_predictions,y_predictions,'r',label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()
