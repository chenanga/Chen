#!/usr/bin/env python
# coding: utf-8


# **Question 1 (understanding overfitting, 40%)**: this question is about linear regression and overfitting phenomenon. Please follow the following step:
# 
# - import numpy and matplotlib.pyplot. These two packages are needed.

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D


# - Generating data. In this simulation, there is only one feature whose value is between 0 and 10. The linear model is $y = 3x+2$. However, the there are noises in the collected data. Number of samples is 100. np.random.seed is to used to control the generated random variables are the same across computers.

# In[11]:


np.random.seed(3)
x = np.linspace(0,10,100)
y = np.ones((100))*2 + 3*x + np.random.randn(100)*5 # standard deviation of the noise is 5


# - Training, validation and testing data are drawn from the data

# In[12]:


n = 100
index = np.arange(0,100, 1) # index of samples running from 0 to 99
index = np.random.permutation(index) # randomly permute the index
train = index[0:70]
test = index[70:90]
val = index[90:100]


xTrain = x[train]
yTrain = y[train]

xTest = x[test]
yTest = y[test]

xVal = x[val]
yVal = y[val]


# - If the model is $y = \alpha_0 + \alpha_1x + \alpha_2x^2+\cdots+\alpha_kx^k$, features are $x, x^2, \cdots, x^k$. $x$ being given, we need to create features $x^2, x^3, \cdots, x^k$. feature method is to create features. Inputs are a feature vector and an order k and the return value is an numpy array with $i$th column being $k$th order of the input feature.

# In[13]:


def feature(x, order):
    temp_arr = np.zeros((order + 1, len(x)), dtype=np.float_)
    for i in range((order+1)):     #转置前的行
        for j in range(len(x)):            #转置前的列
           temp_arr[i][j]=math.pow(x[j],i)
    #print(temp_arr.T)
    return temp_arr.T            #返回转置后的数组


# - In the class, parameter for linear regression model is estimated as $(X^TX)^{-1}X^Ty$ with $X$ and $y$ the data and labels in training data. LR method is to compute the coefficients for linear regression model. There are three input arguments, which are the training data, training labels and order, respectively. In LR method, we firstly compute features needed and then compute the coefficients.

# In[14]:


def LR(x, y, order):
    # input: x is a vector
    #        y represents labels for x
    #        order is an positive integer k
    # output: a numpy array
    X=feature(x, order)   #矩阵X
    #temp=((X.T*X).I)*X.T*y
    temp=np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)),X.T ),y.T)
    return temp


# - Once coefficients for linear regression model are available, we could use linear regression model for prediction. The job of prediction is to predict labels for the given feature vector or data. prediction method does the prediction job. There are three input arguments, which are feature vector, estimated coefficients and the order k. The returned value is a vector representing the predicted labels for the input feature vector.

# In[15]:


def prediction(x, para, order):
    X=feature(x, order)   #矩阵X
    temp=np.dot(X,para.T)
    return temp  


# - Loss function is to evaluate the discrenpency between the predicted label and the true label. In the class loss function based on two norm square is defined as
# \begin{align}
# L = \frac{1}{n}\|\boldsymbol{y} - \hat{\boldsymbol{y}}\|_2^2
# \end{align}
# , where n represents the number of samples. loss method is to compute the loss. Note that np.linalg.norm method is to compute the norm of a vector.

# In[19]:


def loss(y, yhat):
    n=len(y)
    L=(math.pow(np.linalg.norm(y-yhat),2))/n
    return L


# - plot method is to plot the predicted curve and the testing pionts. You do not need to do anything in plot method.

# In[20]:


def plot(x, y, para, order, title):
    X = feature(np.linspace(0,10,100), order)
    plt.figure()
    plt.plot(x, y,'r*', label = 'observed data')
    plt.plot(np.linspace(0,10,100), X.dot(para), label = 'predicted regression line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(title)
    plt.show()


# - In the following code, we create different models with k varying from 1 to 6. 

# In[25]:


compare_loss_arr=np.zeros((3,9),dtype=np.float_) #后加入，统计用
for order in range(1,10):  
    compare_loss_arr[0][order-1]=order              #后加入，统计用
    print('k = ', order)
    para = LR(xTrain, yTrain, order)
    X = feature(xTrain, order)
    compare_loss_arr[1][order-1]=loss(yTrain, X.dot(para))#后加入，统计用
    #print("woshida",X)
    print("Training Loss: %.2f"%loss(yTrain, X.dot(para)))
    X = feature(xTest, order)
    compare_loss_arr[2][order-1]=round(loss(yTest, X.dot(para)),2)#后加入，统计用
    print("Testing Loss: %.2f"%loss(yTest, X.dot(para)))
    plot(xTrain, yTrain, para, order, 'Training data')
    plot(xTest, yTest, para, order, 'Testing data')  
print(compare_loss_arr)
plt.figure()
plt.plot(compare_loss_arr[0],compare_loss_arr[1],color = 'r',label = 'Training Loss')         #训练集的颜色 红色
plt.plot(compare_loss_arr[0],compare_loss_arr[2],'-.', color = 'b', label = 'Testing Loss')
plt.xlabel('K')
plt.ylabel('Training Loss:red     Testing Loss:blue')
plt.legend()
plt.show()


    


# - Comparing the losses for training data and testing data with k from 1 to 9, what do you observe and what conclusion do you get?

# 当k不断增大时候，Training Loss不断减小，在达到某一个数值之后，Training Loss减小的速度开始变缓。
# 当k不断增大时候，Testing Loss不断增大，在达到某一个数值之后，Testing Loss减小，减小到一定程度时候又开始增大。
# 说明k过大时候会出现过拟合，模型更好的贴合训练样本，但是对于测试样本效果较差。而k较小时候，模型不那么贴近样本，训练样本和测试样本效果都较差。k在Training Loss不断减小时候，当遇到Testing Loss低谷拐点时候，整个模型的误差是相对比较小的。

# **Question 2 (simulation of Monte Hall problem, 30%)**: If you are the player in Monte Hall problem, what should you do? The followings are some facts and analysis
# 
# - there are in total three doors. One door hides a car and the other two hide a goat for each.
# We do not know which door hides the car, but Monte knows.
# 
# - you randomly pick one doors in three. 
# 
# - The probability that your selection is one of the two doors with a goat is 2/3.
# In this case, with probability 1 Monte will open the door with the other goat, because he cannot open the door with the car.
# You can only win a car by switching doors. 
# 
# - The probability that your selection is the door with the car is 1/3. 
# In this case, with probability 1/2 Monte can open either of the other doors, since they both contain goats.
# But, if you switch doors at this time, you will win a goat. 
# 
# You could also create code to simulate the Monte Hall problem. 
# 
# 1. Create a wrapper function that allows you to run the simulation n = 1000 times, with a switch or not switch strategy. 
# 
# 2. Create a function named random_door, which uses numpy.random.choice to Bernoulli sample 1 door randomly from a list of integer door indices (1-3 in this case). 
# Use this function to ramdomly select the door the car is behind and the contestant’s initial choice of doors. 
# 
# 3. Create a function monte_choice, which chooses the door Monte opens, conditional on the contestant’s choice of doors and the door with the car. 
# For the case where the contestant has selected the door with the car, select the door to open by simulating the flip of a fair coin using the np.random.binomial function with n = 1. 
# 
# 4. Create a function win_car, which determines if the contestant wins the car, conditional on the strategy selected, {switch, noswitch}, the door the contestant selected, the door with the car, and the door Monte opened. 
# 
# 5. Execute your simulation for each possible strategy. 
# For the two strategies, plot side by side bar charts showing the numbers of successes and failures for each strategy. 
# 
# 6. Describe the strategy a constestant should adopt for this game. 
# How much will the chosen strategy change the probability of winning a car? Is this result consistent with the conditonal probability of this problem.

# In[26]:


import numpy as np
import matplotlib.pyplot as plt
# 初始化三个选项，声明为全局变量
car_select = 0
player_select = 0
Monte_select = 0
# global Number_of_experiments   #实验次数
# global car_Number_of_experiments  #获得汽车次数
Number_of_experiments=0
car_Number_of_experiments=0

def random_door(door_list):
    # 随机门函数
    global car_select
    global player_select
    global Monte_select
    car_select = np.random.choice(door_list)
    player_select = np.random.choice(door_list)
    return [car_select, player_select]


def monte_choice(door_list):
    global car_select
    global player_select
    global Monte_select
    temp = random_door(door_list)
    # 1,参赛者选择的门，带有汽车的门
    if (temp[0] == temp[1]):
        car_index = door_list.index(temp[0])
        Monte = np.random.binomial(1, 0.5, size=None)  # 此时还剩两扇门，如果为0代表选择剩下两扇的第一扇门，如果为1代表选择剩下另一扇门
        if (Monte == 0):
            if (car_index == 0):
                Monte_select = 2  # 代表当汽车在第1扇门时候，这个时候根据binomial的值为0，Monte应该选择剩下两扇门中的第一扇门，也就是第2扇门
            elif (car_index == 1):
                Monte_select = 1  # 代表当汽车在第2扇门时候，这个时候根据binomial的值为0，Monte应该选择剩下两扇门中的第一扇门，也就是第1扇门
            elif (car_index == 2):
                Monte_select = 1  # 代表当汽车在第3扇门时候，这个时候根据binomial的值为0，Monte应该选择剩下两扇门中的第一扇门，也就是第1扇门
        elif (Monte == 1):
            if (car_index == 0):
                Monte_select = 2  # 代表当汽车在第1扇门时候，这个时候根据binomial的值为1，Monte应该选择剩下两扇门中的第2扇门，也就是第3扇门
            elif (car_index == 1):
                Monte_select = 2  # 代表当汽车在第2扇门时候，这个时候根据binomial的值为1，Monte应该选择剩下两扇门中的第2扇门，也就是第3扇门
            elif (car_index == 2):
                Monte_select = 0  # 代表当汽车在第3扇门时候，这个时候根据binomial的值为1，Monte应该选择剩下两扇门中的第2扇门，也就是第二扇门
    # 2、参赛者选择的门，带有羊的门，这个时候Monte会选择剩下的最后一扇门
    else:
        for i in range(3):  # 把剩下没有选择的那扇门索引给Monte_select，
            if (i+1)!= temp[0] and (i+1)!= temp[1]:
                Monte_select = (i+1)

    return Monte_select


def Strategy_select(select):
    global car_select
    global player_select
    global Monte_select
    if select == 1:  # 玩家切换门
        for i in range(3):  # 把剩下那扇门给玩家，
            if (i + 1) != player_select and (i + 1) != Monte_select:
                player_select = i + 1
                break
    return 0

    # selcet=0 不切换策略 ，selcet=1 切换策略 ，


def win_car():
    global car_select
    global player_select
    global Monte_select
    global Number_of_experiments   #实验次数
    global car_Number_of_experiments  #获得汽车次数
    Number_of_experiments=Number_of_experiments+1
    if (player_select == car_select):
        #print("恭喜你获得汽车")
        car_Number_of_experiments = car_Number_of_experiments + 1
    return 0
    
def simulation_experiments(select):
    global car_select
    global player_select
    global Monte_select
    global Number_of_experiments   #实验次数
    global car_Number_of_experiments  #获得汽车次数
    Number_of_experiments = 0
    car_Number_of_experiments = 0

    for i in range(1000):
        door_list = [1, 2, 3]  # 定义门

        # random_door(door_list)
        monte_choice(door_list)
        Strategy_select(select)
        win_car()
    if select == 0:
        print("不更换门时候，获得汽车的概率", car_Number_of_experiments / Number_of_experiments)
        print("不更换门时候，1000次试验获得汽车的次数", car_Number_of_experiments)
    if select == 1:
        print("更换门时候，获得汽车的概率", car_Number_of_experiments / Number_of_experiments)
        print("更换门时候，1000次试验获得汽车的次数", car_Number_of_experiments)
def plot_tu():
    global Number_of_experiments   #实验次数
    global car_Number_of_experiments  #获得汽车次数
    # 这两行代码解决 plt 中文显示的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 输入统计数据
    select1 = ('玩家不更换门', '玩家更换门')

    simulation_experiments(0)
    number_successes = [car_Number_of_experiments]
    number_failures = [Number_of_experiments-car_Number_of_experiments]
    simulation_experiments(1)
    number_successes.append(car_Number_of_experiments)
    number_failures.append(Number_of_experiments-car_Number_of_experiments)

    bar_width = 0.3  # 条形宽度
    index_male = np.arange(len(select1))  # 男生条形图的横坐标
    index_female = index_male + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_male, height=number_successes, width=bar_width, color='r', label='获得汽车次数')
    plt.bar(index_female, height=number_failures, width=bar_width, color='black', label='失败次数')

    plt.legend()  # 显示图例
    plt.xticks(index_male + bar_width / 2, select1)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('次数')  # 纵坐标轴标题
    plt.title('1000次实验中，换门和不换门各自成功和失败的次数')  # 图形标题
    plt.ylim(0, 1000)
    plt.show()
plot_tu()


# $$当更换门时候，获得汽车的概率会变为大概\frac{2}{3} ，不更换门时候，获得汽车的概率大概只有 \frac{1}{3} $$  

# **Quesion 3 (understanding posterior, 20%)**: Suppose a certain disease has an incidence rate of 0.1% (that is, it afflicts 0.1% of the population). A test has been devised to detect this disease. The test does not produce false negatives (that is, anyone who has the disease will test positive for it), but the false positive rate is 5% (that is, about 5% of people who take the test will test positive, even though they do not have the disease). Suppose a randomly selected person takes the test and tests positive. What is the probability that this person actually has the disease?

# P(A1)  代表 该人患病概率 P(A1)=0.001 
# 
# P(A2)  代表 该人不患病概率 P(A2)=0.999
# 
# 
# P(B|A1)代表 该人患病被检测出来阳性的概率P(B|A1)=1
# 
# P(B|A2)代表 该人不患病被检测出来阳性的概率P(B|A2)=0.05
# 
# P(B )  代表 不考虑其他因素，此人被检测为阳性的概率
# 
# P(B)=P(B|A1)P(A1)+P(B|A2)P(A2)=1\*0.001+0.05*0.999=0.05095
# 
# P(A1|B)代表 当此人被检测为阳性时候，患病的概率
# 
# 综上，$$P(A1|B)=\frac{P(B|A1)P(A1)}{P(B)}=\frac{P(B|A1)P(A1)}{P(B|A1)P(A1)+P(B|A2)P(A2)}=0.019627=1.9627\%$$
# 
# 答:此人被检测为阳性时候，患病的概率是$$P(A1|B)=1.9621\%$$
# 

# **Question 4 (10%)**: What is your opinion about overfitting? Please give no more than 5 sentences.

# 过度拟合会过于依赖训练集样本，并且当训练样本中有一些数据噪声或者把一些样本个例当成是普遍存在的时候，会导致测试集样本误差过大，导致模型不够泛化。

# **Question 5 (0%)**: write down any suggestions or questions you want to ask.

# 可能英文的题目阅读理解起来有一些难度或者偏差，要是在一些关键句子上有中文意思就更好了
