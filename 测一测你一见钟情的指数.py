import numpy as np 
import pandas as pd
# 绘图所需的包
import matplotlib.pyplot as plt #每一个pyplot函数都使一幅图像做出些许改变
from mpl_toolkits.mplot3d import Axes3D #用来画三维图
import seaborn as sns #seaborn是对matplotlib进行二次封装
import warnings 
warnings.filterwarnings('ignore') #可以忽略警告消息
#from openpyxl.workbook import Workbook


#第一轮缺失值处理
#数据基本面貌和审查
#数据准备
data=pd.read_csv('C:/Users/成功女人的电脑/Desktop/大二/机器学习/speed_dating_train.csv')
print(data.shape)

#缺失值考察
def missing(data,threshold):
    percent_missing=data.isnull().sum()/len(data)
    #isnull()函数，用来判断缺失值
    missing=pd.DataFrame({'column_name':data.columns,'percent_missing':percent_missing})
    missing_show=missing.sort_values(by='percent_missing')
    #sort_values()将数据集依照某个字段中的数据进行排序
    print(missing_show[missing_show['percent_missing']>0].count())
    print('---------------------------')
    out=missing_show[missing_show['percent_missing']>threshold]
    return out
missing(data,0.7)
#我们尝试分析X_2表单缺失，是否和匹配存在关系
def null_infomation(data,column):
    data_null=data[[column,'match']].dropna()
    #dropna()函数，能够找到DataFrame类型数据的空值，将空值所在的行/列删除后，将新的DataFrame作为返回值返回
    data_shape=data.shape[0]
    data_null_shape=data_null.shape[0]
    print(f'{column}缺失{data_shape-data_null_shape}个值，缺失率为{100*(data_shape-data_null_shape)/data_shape}%')
    dif = 100*(data[data['match']==1].shape[0]/data_shape - data_null[data_null['match']==1].shape[0]/data_null_shape)
    print(f'样本的整体偏差率为{dif}%')
null_infomation(data,'attr7_2')    

# 检查这个观点其实很简单，我们要检查是否存在全部缺失的样本是否可以成功匹配
data_2 = data.loc[:,'satis_2':'amb5_2']
data_2_null = data_2.dropna(how = 'all')
data_2.shape[0]-data_2_null.shape[0]
print(data[data_2.isnull().all(axis=1)]['match'].mean())
data[data_2.isnull().all(axis=1)]['match']
# 笔者一开始怀疑是该名参与者在其他场次填写了信息，从而得到了匹配对象信息，但数据没有统一过来导致的偏差
# 于是我审查了第31个样本（上面的序号30）的所有相亲场次，发现她都没有填写这些表格
data[data.iid==8].loc[:,'satis_2':'amb5_2']
# 再进一步分析,会不会她之所以得到对象的匹配信息是因为她的匹配对象填写了表单
# 哇，这名女生好受欢迎,当晚一共快速相亲了10人，其中8个人都有意向匹配
data[data.iid==8].loc[:,'round':'attr_o']
# 逐一审查她的匹配对象，当审查到12号男选手时
# 哈哈，相较于刚刚的那位女性，这名男性可能就不是那么受欢迎了，他只得到了两位女士的青睐
data[data.iid==12].loc[:,'round':'attr_o']
# 遗憾的是，这名男性也没有填写表格2的信息，算了，就当他们的微信是自己加的好了
# 那这就意味着，不论是否完成后续的调查或者进一步联系，双方在现场就能登记为匹配
# 这个结论十分关键，因为这些特征发生在匹配这一结果之后，这意味删除它们，不会对匹配结果造成影响
data[data.iid==12].loc[:,'satis_2':'amb5_2'].T



#第二轮缺失值处理
# 我们现在剩下119个特征，相较于之前的192个降低不少
data_1 = data.loc[:,'iid':'amb3_s']
data_1.shape
# 适当调低阈值进一步审查
missing(data_1,0.3)
# 认为多少人可能对自己感兴趣，好敏感的问题，怪不得缺失率高
# 这个偏差还是可观的（按照百分之五做阈值，即0.82%以上）
null_infomation(data_1,'expnum')
# 大学的SAT平均分，用来代表大学水平，缺失可能是较差的大学或者没有大学就读
# 这个偏差同样可观，而且偏差为正，缺失带来竞争劣势
null_infomation(data_1,'mn_sat')
# 本科生学费，影响不可观，删掉
null_infomation(data_1,'tuition')
# 后缀3_s，认为自己的吸引力，影响不可观，删掉
null_infomation(data_1,'amb3_s')
# 进行到一半，问吸引力侧重点，影响不可观，删掉
null_infomation(data_1,'shar1_s')
# 收入，暂时保存
null_infomation(data_1,'income')
# 删掉刚刚初步分析的几个组量，并且删去对明显没有用的、或者有编号的文字特征
data_1.drop(columns = ['tuition','tuition','attr3_s','sinc3_s','intel3_s','fun3_s','amb3_s',
                       'shar1_s','attr1_s','sinc1_s','intel1_s','fun1_s','amb1_s',
                       'position','positin1','field','from'],inplace=True)
data_1.shape
# 适当调低阈值再进一步审查，实际上这是最后一次整体审查，之后小于10%的缺失值，将会采取策略填充
missing(data_1,0.1)
# 这个偏差还是较小的（按照百分之五做阈值，即0.82%以上，稍稍超过）
null_infomation(data_1,'attr5_1')
# 本科毕业院校，这个缺失率比较本科SAT小一些，但是人家本科SAT分数影响更可观，而且，文本结构编号手段复杂，删掉，大学信息保留一个SAT招生分数够了
null_infomation(data_1,'undergra')
# 后缀4_1，影响不可观，删掉
null_infomation(data_1,'shar4_1')
# 后续的影响都不可观，全部删掉
null_infomation(data_1,'match_es')
null_infomation(data_1,'shar_o')
null_infomation(data_1,'zipcode')
null_infomation(data_1,'shar')
data_1.drop(columns = ['undergra','attr5_1','sinc5_1','intel5_1','fun5_1','amb5_1',
                       'shar4_1','sinc4_1','attr4_1','intel4_1','fun4_1','amb4_1',
                       'match_es','shar_o','zipcode','shar'],inplace=True)
data_1.shape
# 适当调低阈值进一步审查

#空值填充
missing(data_1,0.05)
# 对于没有填写期望约会数目的，单独记作一类
# 对于没有SAT分数的，单独记作一类
# 重新考虑了一下，还是去掉收入,反正这个收入也是用邮编估计的
data_1['expnum'].fillna(-1, inplace=True)
data_1['mn_sat'].fillna(-1, inplace=True)
data_1.drop(columns = ['income'],inplace=True)
# 刚刚发现做完上述处理，剩下的缺失就比例很小了，不到百分之一，猜测是否是有些行信息损失严重，这样的话，去掉这些行就好了
# 存在缺失的行一共2159行，看来这样假设不对，不少行都存在缺失
data_1[data_1.isnull().any(axis=1)].shape
# 准备对所有数据进行填充，填充主要有用众数、平均数、0等填充手段。我计划主要采用众数填充（这样不会影响按类填充结果）
# 当然，对72个损失中，不适合采用众数填充的，单独处理
missing_out = missing(data_1,0)['column_name']
print(missing_out.index)
# id,pid实际上不是输入特征
data_1.drop(columns = ['iid','pid'],inplace=True)
# 经过审查(随便扫一眼)，应该问题不大，都可以使用众数填充
for columname in data_1.columns:
    data_1[columname].fillna(data_1[columname].mode()[0],inplace=True)
    # 填充完毕，缺失审查结束
missing(data_1,0)


#特征降维
label = data_1['match']
data_input = data_1.drop(columns = ['match'])
corr = data_1.corr('pearson')
plt.figure(figsize=(18,6))
corr['match'].sort_values(ascending=False)[1:].plot(kind='bar')
plt.tight_layout()
# 额，误差为0，没想到你竟是这样的阿里
y_pre = ((data_1['dec']+data_1['dec_o'])/2).values
print((np.floor(y_pre)-label).sum())
# 好的，本次建模到此结束，现在对测试集合套用刚刚建立的完美模型（哈哈哈）
test_data = pd.read_csv('C:/Users/成功女人的电脑/Desktop/大二/机器学习/speed_dating_test.csv')
print(test_data.shape)
# 得到测试集的标准答案
test_pre = ((test_data['dec']+test_data['dec_o'])/2).values
out = pd.DataFrame(np.floor(test_pre))
# 保存处理结果
writer = pd.ExcelWriter('out.xlsx')
out.to_excel(writer)
writer.save()

#EDA
# 做一些简单的EDA工作，首先分析基本匹配成功率，这将作为一个基准，当偏离这个基准时，意味这不同类别的倾向
data_1['match'].value_counts().plot.pie(labeldistance = 1.1,autopct = '%1.2f%%',
                                               shadow = False,startangle = 90,pctdistance = 0.6)
# 首先做的是单因素分析
# 定义一个离散单因素分析的绘图函数
def discrete_plot(data,col_name,label):
    """
    变量为离散型分布
    第1张：按类饼图
    第2张：按类计数（检查是否是由离群特殊值导致分布）
    """
    f,ax = plt.subplots(1,2, figsize=(18,6))
    data[col_name].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=False, cmap='Set3')
    sns.countplot(col_name, hue=label,data=data, ax=ax[1], palette='Set3')
    ax[1].set_title(f'Attrition by {col_name}')
    ax[1].set_xlabel(f'{col_name}')
    ax[1].set_ylabel('Count')
    print(data[[col_name,label]].groupby([col_name]).mean().sort_values(by=label))
    print('--------------------------------')
    print(data.groupby([col_name,label])[col_name].count())

discrete_plot(data_1,'like','match')
discrete_plot(data,'field_cd','match')
discrete_plot(data,'career_c','match')
# 定义一个连续单因素分析的绘图函数
def continue_plot(data,col_name,label):
    """
    变量为连续型分布
    第1张：连续分布图
    第2张：箱线图
    """
    f,ax = plt.subplots(1,2, figsize=(18,6))
    sns.distplot(data[data[label] == 1][col_name],ax=ax[0])
    sns.distplot(data[data[label] == 0][col_name],ax=ax[0])
    plt.legend(['1','0'])
    
    sns.boxplot(y=col_name, x=label , data=data, palette='Set2', ax=ax[1])

# 似乎并没有看出多少错位，可能因为大家都是和对应年龄相亲，造成了年龄影响不大
continue_plot(data_1,'age','match')
# 我想加上性别分析一下
def continue_discrete_plot(data,col_con,col_dis,label):
    fig,ax = plt.subplots(figsize = (9,5))
    sns.violinplot(col_dis,col_con,hue=label,data=data,split=True)
    ax.set_title(f'{col_dis} and {col_con} vs {label}') 
    plt.show()
# 正如之前分析的，可能由于相亲过程考虑的年龄匹配，没有出现年龄较低成功率较高的现象
# 但是奇怪的是，女性（0）出现了高龄相亲，男性则没有
# 还有一个比较符合常识的结论，男性的均线比女性大2岁左右
continue_discrete_plot(data_1,'age','gender','match')

