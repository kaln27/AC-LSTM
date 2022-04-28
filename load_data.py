import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf



# 载入数据
train_fd0001 = pd.read_csv("./CMAPSSData/train_FD001.txt", sep=' ', header=None)
test_fd0001 = pd.read_csv("./CMAPSSData/test_FD001.txt", sep=' ', header=None)
label = pd.read_csv("./CMAPSSData/RUL_FD001.txt", header=None)

# 丢弃NAN数据
train_fd0001.drop(labels=[26, 27], axis=1, inplace=True)
test_fd0001.drop(labels=[26, 27], axis=1, inplace=True)

# 丢弃方差为0的数据
train_fd0001.drop(labels=[4, 5, 22 ,23], axis=1, inplace=True)
test_fd0001.drop(labels=[4, 5, 22, 23], axis=1, inplace=True)

# 实例化 PCA 和 StandardScaler
pca = PCA(n_components=16)
mms = MinMaxScaler()


# 载入数据并进行PCA 和 StandardScaler
def load_data(train_fd0001, test_fd0001, label):
    # 实例化 PCA 和 StandardScaler
    # TODO：n_components 需根据不同需求来设置
    pca = PCA(n_components=16)
    mms = MinMaxScaler()

    min_case = train_fd0001[0].min()
    max_case = train_fd0001[0].max()
    train_seq = list()
    train_len = 0
    for i in range(min_case, max_case+1):
        data = train_fd0001[train_fd0001[0]==i].drop(columns=[0, 1]).values.tolist()
        train_len += train_fd0001[train_fd0001[0]==i][1].max()
        data = pca.fit_transform(data)
        data = mms.fit_transform(data)
        train_seq.append(data)


    test_seq = list()
    test_len = 0
    for i in range(min_case, max_case+1):
        data = test_fd0001[test_fd0001[0]==i].drop(columns=[0, 1]).values.tolist()
        test_len += test_fd0001[test_fd0001[0]==i][1].max()
        data = pca.fit_transform(data)
        data = mms.fit_transform(data)
        test_seq.append(data)

    train_len = int(train_len/len(train_seq))
    test_len = int(test_len/len(test_seq))
    max_len = int(0.7*train_len + 0.3*test_len)
    train_seq = pad_sequences(train_seq, maxlen=max_len)
    test_seq = pad_sequences(test_seq, maxlen=max_len)

    return tf.convert_to_tensor(train_seq), tf.convert_to_tensor(test_seq), tf.convert_to_tensor(label.values)


def load():
    # 载入数据
    train_fd0001 = pd.read_csv("./CMAPSSData/train_FD001.txt", sep=' ', header=None)
    test_fd0001 = pd.read_csv("./CMAPSSData/test_FD001.txt", sep=' ', header=None)
    label = pd.read_csv("./CMAPSSData/RUL_FD001.txt", header=None)

    # 丢弃NAN数据
    train_fd0001.drop(labels=[26, 27], axis=1, inplace=True)
    test_fd0001.drop(labels=[26, 27], axis=1, inplace=True)

    # 丢弃方差为0的数据
    train_fd0001.drop(labels=[4, 5, 22 ,23], axis=1, inplace=True)
    test_fd0001.drop(labels=[4, 5, 22, 23], axis=1, inplace=True)

    train_seq, test_seq, label = load_data(train_fd0001, test_fd0001, label)
    return train_seq, test_seq, label
