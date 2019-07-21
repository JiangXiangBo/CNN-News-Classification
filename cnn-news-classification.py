import pickle
import os
import time
import re

# 1、获取数据
# 1.1、获取文件文本路径
def getFilePathList(rootDir):
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list
filePath_list = getFilePathList('THUCNews')
len(filePath_list)
# 文件路径列表中共有836075元素，即在THUCNews文件夹中总共有836075文本文件。

# 1.2、获取所有的样本标签
label_list = []
for filePath in filePath_list:
    label = filePath.split('\\')[1]
    label_list.append(label)
len(label_list)
# 836075

# 1.4、调用pickle库保存label_list标签
with open('label_list.pickle', 'wb') as file:
    pickle.dump(label_list, file)


# 1.5、保存所有的样本内容，保存content_list内容
# 避免内存溢出，每读取一定数量的文件就利用pickle库的dump方法保存。
# 因为有80多万个文本文件，读取时间较长
def getFile(filePath):
    with open(filePath, encoding='utf8') as file:
        fileStr = ''.join(file.readlines(1000))
    return fileStr

interval = 20000
n_samples = len(label_list)
startTime = time.time()
directory_name = 'content_list'
if not os.path.isdir(directory_name):
    os.mkdir(directory_name)
for i in range(0, n_samples, interval):
    startIndex = i
    endIndex = i + interval
    content_list = []
    print('%06d-%06d start' %(startIndex, endIndex))
    for filePath in filePath_list[startIndex:endIndex]:
        fileStr = getFile(filePath)
        content = re.sub('\s+', ' ', fileStr)
        content_list.append(content)
    save_fileName = directory_name + '/%06d-%06d.pickle' %(startIndex, endIndex)
    with open(save_fileName, 'wb') as file:
        pickle.dump(content_list, file)
    used_time = time.time() - startTime
    print('%06d-%06d used time: %.2f seconds' %(startIndex, endIndex, used_time))


# 2、加载数据集
def getFilePathList(rootDir):
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list

contentListPath_list = getFilePathList('content_list')
content_list = []
for filePath in contentListPath_list:
    with open(filePath, 'rb') as file:
        part_content_list = pickle.load(file)
    content_list.extend(part_content_list)
with open('label_list.pickle', 'rb') as file:
    label_list = pickle.load(file)

sample_size = len(content_list)
print('length of content_list，mean sample size: %d' %sample_size)


# 3、词汇表
# 3.1 制作词汇表
# 内容列表content_list中的元素是每篇文章内容，数据类型为字符串。
# 对所有文章内容中的字做统计计数，出现次数排名前10000的字赋值给变量vocabulary_list。
from collections import Counter
def getVocabularyList(content_list, vocabulary_size):
    allContent_str = ''.join(content_list)
    counter = Counter(allContent_str)
    vocabulary_list = [k[0] for k in counter.most_common(vocabulary_size)]
    return ['PAD'] + vocabulary_list

vocabulary_list = getVocabularyList(content_list, 10000)

# 3.2、保存词汇表
with open('vocabulary_list.pickle', 'wb') as file:
    pickle.dump(vocabulary_list, file)


# 4.1、加载词汇表
with open('vocabulary_list.pickle', 'rb') as file:
    vocabulary_list = pickle.load(file)

# 4.2、数据准备
# 划分训练集、测试集；
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(content_list, label_list)

# 训练集文本内容列表train_content_list，训练集标签列表train_label_list，
# 测试集文本内容列表test_content_list，测试集标签列表test_label_list
train_content_list = train_X
train_label_list = train_y
test_content_list = test_X
test_label_list = test_y


# 5 参数配置
vocabulary_size = 10000  # 词汇表达小
sequence_length = 600    # 序列长度
embedding_size = 64      # 词向量维度
num_filters = 256        # 卷积核数目
filter_size = 5          # 卷积核尺寸
num_fc_units = 128       # 全连接层神经元
dropout_keep_probability = 0.5  # dropout保留比例
learning_rate = 1e-3     # 学习率
batch_size = 64          # 每批训练大小

# 使用列表推导式得到词汇及其id对应的列表，并调用dict方法将列表强制转换为字典。
word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
# 打印变量word2id_dict的前5项
list(word2id_dict.items())[:5]

# 使用列表推导式和匿名函数定义函数content2idlist，函数作用是将文章中的每个字转换为id
content2idList = lambda content : [word2id_dict[word] for word in content if word in word2id_dict]
# 使用列表推导式得到的结果是列表的列表，
# 总列表train_idlist_list中的元素是每篇文章中的字对应的id列表；
train_idlist_list = [content2idList(content) for content in train_content_list]


import numpy as np
# 新闻类别是14种，
num_classes = np.unique(label_list).shape[0]

# 获得能够用于模型训练的特征矩阵和预测目标值；
import tensorflow.contrib.keras as kr
# 每个样本统一长度为seq_length，即600
train_X = kr.preprocessing.sequence.pad_sequences(train_idlist_list, sequence_length)
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
# 调用LabelEncoder对象的fit_transform方法做标签编码；
# 调用keras.untils库的to_categorical方法将标签编码的结果再做Ont-Hot编码。
train_y = labelEncoder.fit_transform(train_label_list)
train_Y = kr.utils.to_categorical(train_y, num_classes)

import tensorflow as tf
tf.reset_default_graph()
# 数据占位符准备
X_holder = tf.placeholder(tf.int32, [None, sequence_length])
Y_holder = tf.placeholder(tf.float32, [None, num_classes])
used_time = time.time() - startTime


# 5、搭建神经网络
# 可以更新的模型参数embedding，矩阵形状为vocab_size*embedding_size，即10000*64；
embedding = tf.get_variable('embedding', [vocabulary_size, embedding_size])
# 调用tf.nn库的embedding_lookup方法将输入数据做词嵌入，
# embedding_inputs的形状为batch_size*sequence_length*embedding_size，即64*600*64
embedding_inputs = tf.nn.embedding_lookup(embedding, X_holder)

# 第1个参数是输入数据，第2个参数是卷积核数量num_filters，
# 第3个参数是卷积核大小filter_size。
# 方法结果赋值给变量conv，形状为batch_size*596*num_filters，596是600-5+1的结果
conv = tf.layers.conv1d(embedding_inputs, num_filters, filter_size)
# 变量conv的第1个维度做求最大值操作。
# 方法结果赋值给变量max_pooling，形状为batch_size*num_filters，即64*256
max_pooling = tf.reduce_max(conv, [1])

# 全连接层1，方法结果赋值给变量full_connect，
# 形状为batch_size*num_fc_units，即64*128；
full_connect = tf.layers.dense(max_pooling,
                               num_fc_units)
# dropout
full_connect_dropout = tf.contrib.layers.dropout(full_connect,
                                                 keep_prob=dropout_keep_probability)
full_connect_activate = tf.nn.relu(full_connect_dropout)

# 全连接层2，结果赋值给变量softmax_before，
# 形状为batch_size*num_classes，即64*14
softmax_before = tf.layers.dense(full_connect_activate,
                                 num_classes)
# softmax预测概率值
predict_Y = tf.nn.softmax(softmax_before)
# 交叉熵损失
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_holder,
                                                         logits=softmax_before)
loss = tf.reduce_mean(cross_entropy)
# 优化器"Adam"
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
isCorrect = tf.equal(tf.argmax(Y_holder, 1), tf.argmax(predict_Y, 1))
# 准确率
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))


# 6、参数初始化
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

# 7、模型训练
# 获取测试集中的数据
test_idlist_list = [content2idList(content) for content in test_content_list]
test_X = kr.preprocessing.sequence.pad_sequences(test_idlist_list, sequence_length)
test_y = labelEncoder.transform(test_label_list)
test_Y = kr.utils.to_categorical(test_y, num_classes)

saver = tf.train.Saver()

import random

for i in range(20000):
    # 从训练集中选取batch_size大小，即64个样本做批量梯度下降
    selected_index = random.sample(list(range(len(train_y))), k=batch_size)
    batch_X = train_X[selected_index]
    batch_Y = train_Y[selected_index]
    session.run(train, {X_holder: batch_X, Y_holder: batch_Y})

    step = i + 1
    if step % 200 == 0:
        # 从测试集中随机选取200个样本
        selected_index = random.sample(list(range(len(test_y))), k=200)
        batch_X = test_X[selected_index]
        batch_Y = test_Y[selected_index]
        # 计算损失值loss_value、准确率accuracy_value
        loss_value, accuracy_value = session.run([loss, accuracy], {X_holder: batch_X, Y_holder: batch_Y})
        print('step:%d loss:%.4f accuracy:%.4f' % (step, loss_value, accuracy_value))

# 模型保存
saver.save(session, "model/text_model")


# 9.报告表
# 此段代码主要是调用sklearn.metrics库的precision_recall_fscore_support方法得出报告表。
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def eval_model(y_true, y_pred, labels):
    # 计算每个分类的Precision, Recall, f1, support
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    # 计算总体的平均Precision, Recall, f1, support
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': p,
        u'Recall': r,
        u'F1': f1,
        u'Support': s
    })
    res2 = pd.DataFrame({
        u'Label': ['总体'],
        u'Precision': [tot_p],
        u'Recall': [tot_r],
        u'F1': [tot_f1],
        u'Support': [tot_s]
    })
    res2.index = [999]
    res = pd.concat([res1, res2])
    return res[['Label', 'Precision', 'Recall', 'F1', 'Support']]

eval_model(test_label_list, predict_label_list, labelEncoder.classes_)