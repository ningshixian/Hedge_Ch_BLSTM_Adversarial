import math
import pickle as pkl
import string
import os
import time
from tqdm import tqdm
from decimal import Decimal
from keras.utils import to_categorical
import tensorflow as tf
from keras.layers import *
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model, load_model, Sequential
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import plot_model
from keras.regularizers import l2
import keras.backend as K
import numpy as np
from collections import OrderedDict
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

config =tf.ConfigProto()
config.gpu_options.allow_growth =True
sess =tf.Session(config=config)
K.set_session(sess)


def _shared_layer(concat_input):
    output = Bidirectional(CuDNNLSTM(units=100, return_sequences=True,
                                     kernel_initializer='uniform',
                                     kernel_regularizer=l2(1e-4),
                                     bias_regularizer=l2(1e-4)),
                           name='shared_lstm')(concat_input)
    return output


def _private_layer(input_data, output_shared, modelName=None):
    # # 利用private + shared进行分类
    # private = Bidirectional(CuDNNLSTM(units=100, return_sequences=False, kernel_regularizer=l2(1e-4),
    #                             bias_regularizer=l2(1e-4)), name='private')(input_data)
    # private = concatenate([private, output_shared], axis=-1)
    # private = Dropout(0.5)(private)
    # output = Dense(100, activation='tanh')(private)

    # 仅利用shared进行分类
    output = GlobalAveragePooling1D(name=modelName+'pool')(output_shared)
    output = Dense(100, activation='tanh', name=modelName+'dense100')(output)
    output = Dense(2, activation='softmax', name=modelName+'dense2')(output)
    return output


# adv_loss：共享LSTM模块损失，跟判别器进行对抗，让其预测不准
def adv_loss2(y_true, y_pred):
    current_loss = -K.categorical_crossentropy(y_true, y_pred)
    return current_loss

def class_loss(y_true, y_pred):
    current_loss = K.sum(K.categorical_crossentropy(y_true, y_pred))
    return current_loss


adv_weight = 0.05
loss_function = 0.0
task_list = ['main', 'aux']
def custom_loss(y_actual, y_predicted):
    total_loss= K.constant(value=K.epsilon(), dtype=K.floatx())
    for task_name in task_list:
        total_loss = total_loss + class_loss(y_actual, y_predicted)
    total_loss = total_loss + adv_weight * adv_loss2(y_actual, y_predicted)
    return total_loss


def CNN(seq_length, length, feature_maps, kernels, x):

    concat_input = []
    for feature_map, kernel in zip(feature_maps, kernels):
        reduced_l = length - kernel + 1
        conv = Conv2D(feature_map, (1, kernel), activation='tanh', data_format="channels_last")(x)
        maxp = MaxPooling2D((1, reduced_l), data_format="channels_last")(conv)
        concat_input.append(maxp)

    x = Concatenate()(concat_input)
    x = Reshape((seq_length, sum(feature_maps)))(x)
    return x


def buildModel(embedding_matrix):
    tokens_input = Input(shape=(max_word,),  # 若为None则代表输入序列是变长序列
                         name='tokens_input', dtype='int32')
    tokens = Embedding(input_dim=embedding_matrix.shape[0],  # 索引字典大小
                          output_dim=embedding_matrix.shape[1],  # 词向量的维度
                          weights=[embedding_matrix],
                          trainable=True,
                          name='token_emd')(tokens_input)

    pos_input = Input(shape=(max_word,), name='pos_input')
    pos = Embedding(input_dim=len(pos_index),  # 索引字典大小
                    output_dim=50,  # 词向量的维度
                    trainable=True,
                    name='pos_emd')(pos_input)

    mergeLayers = [tokens, pos]

    gx_input = Input(shape=(max_word,), name='gx_input')
    gx = Embedding(input_dim=2,  # 索引字典大小
                    output_dim=10,  # 词向量的维度
                    trainable=True)(gx_input)
    mergeLayers.append(gx)

    concat_input = concatenate(mergeLayers, axis=-1)  # (none, none, 230)

    # Dropout on final input
    concat_input = Dropout(0.5)(concat_input)

    # shared layer
    output_pub = _shared_layer(concat_input)    # (none, none, 200)
    output_pub_d = Dropout(0.5)(output_pub)

    # discriminator
    pool = GlobalAveragePooling1D()(output_pub_d)
    output1 = Dense(100, activation='tanh')(pool)
    output2 = Dense(2, activation='softmax')(output1)
    # Freeze weights
    pool.trainable = False
    output1.trainable = False
    output2.trainable = False

    # CWS Classifier
    models = {}
    for modelName in ['main', 'aux']:
        output_task = _private_layer(concat_input, output_pub_d, modelName)
        model = Model(inputs=[tokens_input, pos_input, gx_input], outputs=[output_task, output2])
        if modelName=='main':
            # optimizer = RMSprop(lr=1e-3, clipnorm=5.)
            optimizer = RMSprop(lr=1e-3, clipvalue=1., decay=3e-8)
        else:
            # 由于辅助任务的数据较少（200），需要较大的学习率
            optimizer = RMSprop(lr=1e-2, clipvalue=1., decay=3e-8)
        model.compile(loss=['categorical_crossentropy', adv_loss2], # adv_loss2
                      loss_weights=[1, 0.05],
                        metrics=['acc'],   # accuracy
                        optimizer=optimizer)
        models[modelName] = model
    models['main'].summary()

    '''
    discriminator: 尽可能准确地判断共享特征向量来自于哪个领域
    '''
    pool.trainable = True
    output1.trainable = True
    output2.trainable = True
    output_pub.trainable = False
    adv_model = Model(inputs=[tokens_input, pos_input, gx_input], outputs=output2)
    rmsprop = RMSprop(lr=2e-4, clipvalue=1., decay=6e-8)
    adv_model.compile(loss='categorical_crossentropy',
                metrics=['acc'],
                optimizer=rmsprop)
    models['discriminator'] = adv_model

    '''保存模型为图片
    pip3 install pydot-ng 
    sudo apt-get install graphviz'''
    plot_model(models['discriminator'], to_file='discriminator.png', show_shapes=True)
    plot_model(models['main'], to_file='model.png', show_shapes=True)
    return models


# 该回调函数将在每个epoch后保存概率文件
from keras.callbacks import Callback
class WritePRF(Callback):
    def __init__(self, max_f, X_test, y_test, p, r, f):
        super(WritePRF, self).__init__()
        self.test = X_test
        self.y_true = y_test
        self.p, self.r, self.f = p, r, f

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(x=self.test)  # 测试
        y_pred = predictions[0].argmax(axis=-1)  # Predict classes
        pre, rec, f1 = predictLabels2(y_pred, self.y_true)

        self.p.append(pre)
        self.r.append(rec)
        self.f.append(f1)



def predictLabels2(y_pred, y_true):
    # y_true = np.squeeze(y_true, -1)
    lable_pred = list(y_pred)
    lable_true = list(y_true)
    # print(lable_pred)
    # print(lable_true)

    print('\n计算PRF...')
    # import BIOF1Validation
    # pre, rec, f1 = BIOF1Validation.compute_f1(lable_pred, lable_true, idx2label, 'O', 'OBI')
    pre, rec, f1 = prf(lable_pred, lable_true, idx2label)
    print('precision: {:.2f}%'.format(100.*pre))
    print('recall: {:.2f}%'.format(100.*rec))
    print('f1: {:.2f}%'.format(100.*f1))

    return round(Decimal(100.*pre), 2), round(Decimal(100.*rec), 2), round(Decimal(100.*f1), 2)


def prf(lable_pred, lable_true, idx2label):
    '''
    数据中1的个数为a，预测1的次数为b，预测1命中的次数为c
    准确率 precision = c / b
    召回率 recall = c /
    f1_score = 2 * precision * recall / (precision + recall)
    '''
    assert len(lable_pred)==len(lable_true)

    a = 0.
    for i in range(len(lable_true)):
        if lable_true[i]==1:
            a+=1
    b = 1.
    for i in range(len(lable_pred)):
        if lable_pred[i] == 1:
            b += 1

    c=1.
    for i in range(len(lable_true)):
        if lable_pred[i]==1 and lable_true[i]==1:
            c+=1

    precision = c/b
    recall = c/a
    f1 = 2*precision*recall / (precision+recall)
    return precision, recall, f1


if __name__ == '__main__':

    # load data
    fold = '1'
    root1 = r'data/wiki_by_discuss/'+fold+'/'
    label2idx = {'0': 0, '1': 1}
    idx2label = {0: '0', 1: '1'}

    if not os.path.exists(root1 + 'pkl'):
        from preprocess_MT import main
        main(root1)

    with open(root1 + 'pkl/train.pkl', "rb") as f:
        train_x, train_y, train_pos, train_gx = pkl.load(f)
    with open(root1 + 'pkl/aux.pkl', "rb") as f:
        aux_x, aux_y, aux_pos, aux_gx = pkl.load(f)
    with open(root1 + 'pkl/test.pkl', "rb") as f:
        test_x, test_y, test_pos, test_gx = pkl.load(f)
    with open(root1 + 'pkl/emb.pkl', "rb") as f:
        embedding_matrix, pos_index, max_word = pkl.load(f)

    # print(len(train_x), len(train_y), len(train_pos), len(train_gx))  # 2890
    # print(len(test_x), len(test_y), len(test_pos), len(test_gx))  # 2890
    # print(train_y[0])

    epochs=30
    batch_size = 64
    max_f = 0.

    test_y = np.asarray(test_y).argmax(axis=-1)  # Predict classes
    train = [np.asarray(train_x), np.asarray(train_pos), np.asarray(train_gx)]
    aux_train = [np.asarray(aux_x), np.asarray(aux_pos), np.asarray(aux_gx)]
    test = [np.asarray(test_x), np.asarray(test_pos), np.asarray(test_gx)]

    models = buildModel(embedding_matrix)

    e = []
    p = []
    r = []
    f = []
    # 该回调函数将在每个epoch后保存概率文件
    write_prob = WritePRF(max_f, test, test_y, p, r, f)

    y1 = np.ones([len(train[0]), 1])
    y2 = np.zeros([len(aux_train[0]), 1])
    y1 = to_categorical(y1, num_classes=2)
    y2 = to_categorical(y2, num_classes=2)

    for epoch in range(epochs):
        print("\n--------- Epoch %d -----------" % (epoch + 1))
        e.append(epoch)
        models['main'].fit(x=train, y=[np.asarray(train_y), y1],
                            epochs=1, batch_size=batch_size,
                           callbacks=[write_prob]
                            )
        models['aux'].fit(x=aux_train, y=[np.asarray(aux_y), y2],
                          epochs=1, batch_size=batch_size)
                          # callbacks=[write_prob])

        # # 网格搜索进行超参数优化
        # clssifier1 = KerasClassifier(models['main'], batch_size=batch_size)
        # validator = GridSearchCV(clssifier1,
        #                          param_grid={
        #                              'epochs': [15, 25],
        #                          },
        #                          scoring='neg_log_loss',
        #                          n_jobs=1)
        # validator.fit(train, [np.asarray(train_y), y2])

        for i in range(5):
            # 一次迭代过程中，对D的参数更新5次后再对G的参数更新1次
            models['discriminator'].fit(x=train, y=y1,
                              epochs=1, batch_size=batch_size)
            models['discriminator'].fit(x=aux_train, y=y2,
                              epochs=1, batch_size=batch_size)

    print(f)
    with open('prf' + fold + '.txt', 'a') as pf:
        print('write prf...... ')
        index = f.index(max(f))
        pf.write(str(e[index]) + '\n')
        pf.write(str(p[index]) + '\n')
        pf.write(str(r[index]) + '\n')
        pf.write(str(f[index]) + '\n')
        print('do saving')
