import tensorflow as tf
import numpy as np
from tensorflow import keras
from load_data import load
from model import Model
from matplotlib import pyplot as plt


# 超参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 1000

# 数据预处理
def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    return x, y


# 学习率衰减 
def lr_scheduler(epoch, lr):
    if epoch < EPOCHS*0.2:
        return lr
    else:
        return lr*tf.math.exp(-1e-4*(epoch-EPOCHS*0.2))


# 载入数据
train_seq, test_seq, label = load()


# 获取时间序列长度 和 特征数
train_time_step, train_feature_num = train_seq.shape[1], train_seq.shape[2]
test_time_step, test_feature_num = test_seq.shape[1], test_seq.shape[2]

# 数据集加载
train_dataset = tf.data.Dataset.from_tensor_slices((train_seq, label)).shuffle(100).map(preprocess).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_seq, label)).map(preprocess).batch(BATCH_SIZE)

# 模型实例化
model = Model([None, train_time_step, train_feature_num])
# 载入权重
model.load_weights('./tmp/checkpoint')

# 优化器
optm = keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

# learning rate scheduler
lr_scheduler_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
checkpoint_filepath = './tmp/checkpoint'
# 模型保存
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
# Tensorboard 可视化
tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)


model.compile(optimizer=optm, loss='mse', metrics=['mse'])


his = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, callbacks=[lr_scheduler_callback, model_checkpoint_callback, tb_callback])

pre = model.evaluate(test_dataset)



# 可视化数据
plt.figure(figsize=(16, 8))
plt.plot(label, label='label', color='red')
plt.plot(pre, label='pre', color='blue')
plt.legend()
plt.show()
