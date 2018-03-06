import pickle as pkl
import string
from collections import OrderedDict
import numpy as np
np.random.seed(1337)
from tqdm import tqdm
import codecs
import word2vec
from nltk.corpus import stopwords
stop_word = stopwords.words('english')

nb_word = 0 # 用于计算临时的最长句子的长度
EMBEDDING_DIM = 100 # 词向量的维
MAX_NB_WORDS = 1000000
# max_word = 0
label2idx = {'0': 0, '1': 1}
# idx2label = {0:'O', 1:'B', 2:'I'}
# label2idx = {'O': 0, 'B':1, 'I':2}
embeddingFile = 'embedding/vec-100.txt'

def get_word_index(datasDic):
    """
    1、建立索引字典- word_index
    :param trainFile:
    :param testFile:
    :return:
    """
    print('\n获取索引字典- word_index \n')
    word_counts = {}
    trainFile = datasDic['train']
    auxFile = datasDic['aux']
    testFile = datasDic['test']

    for f in [trainFile, auxFile, testFile]:
        for line in f:
            for w in line:
                if w in word_counts:
                    word_counts[w] += 1
                else:
                    word_counts[w] = 1

    # 根据词频来确定每个词的索引
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts]
    # note that index 0 is reserved, never assigned to an existing word
    word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

    # 加入未登陆新词和填充词
    word_index['retain-padding'] = 0
    word_index['retain-unknown'] = len(word_index)
    print('word_index 长度：%d' % len(word_index))
    return word_index


def pos2index(pos, pos_index, max_word):
    """
    将一行转换为转化为词索引序列
    :param sent:
    :param word_index:
    :return:
    """
    charVec = []
    for p in pos:
        charVec.append(pos_index[p])
    while len(charVec)<max_word:
        charVec.append(0)
    return charVec[:]


def sent2vec2(sent, word_index, max_word):
    """
    将一行转换为转化为词索引序列
    :param sent:
    :param word_index:
    :return:
    """
    charVec = []
    for char in sent:
        if char in word_index:
            charVec.append(word_index[char])
        else:
            print(char)
            charVec.append(word_index['retain-unknown'])
    while len(charVec) < max_word:
        charVec.append(0)
    return [i for i in charVec]


def doc2vec(train, label1, pos1, test, label2, pos2, word_index, pos_index):
    """
    2、将全部训练和测试语料 转化为词索引序列
    :param ftrain:
    :param ftrain_label:
    :param ftest:
    :param ftest_label:
    :param word_index:
    :return:
    """
    x_train, y_train, x_test, y_test = [], [], [], []
    pos_train, pos_test = [], []
    # tags = ['-1', '+1']  # 标注统计信息对应 [ 1.  0.] [ 0.  1.]

    for line in train:
        index_line = sent2vec2(line, word_index)
        print(index_line)
        x_train.append(index_line)
    for line in test:
        index_line = sent2vec2(line, word_index)
        x_test.append(index_line)
    for line in pos1:
        index_line = pos2index(line, pos_index)
        pos_train.append(index_line)
    for line in pos2:
        index_line = pos2index(line, pos_index)
        pos_test.append(index_line)
    for line in label1:
        index = label2idx.get(line)
        index_line = [0, 0]
        index_line[index]=1
        y_train.append(index_line)
    for line in label2:
        index = label2idx.get(line)
        index_line = [0, 0]
        index_line[index] = 1
        y_test.append(index_line)

    return x_train, y_train, pos_train, x_test, y_test, pos_test


def process_data(datasDic, labelsDic, posDic, gxDic, word_index, pos_index, max_word):
    """
    3、将转化后的 词索引序列 转化为神经网络训练所用的张量
    :param data_label:
    :param word_index:
    :param max_len:
    :return:
    """
    x_train, y_train, x_aux, y_aux, x_test, y_test = [], [], [], [], [], []
    pos_train, pos_aux, pos_test = [], [], []
    gx_train, gx_aux, gx_test = [], [], []
    a = ['train', 'aux', 'test']
    data = [x_train, x_aux, x_test]
    pos = [pos_train, pos_aux, pos_test]
    label = [y_train, y_aux, y_test]
    gx = [gx_train, gx_aux, gx_test]

    for i in range(len(a)):
        s = a[i]
        train = datasDic[s]
        label1 = labelsDic[s]
        pos1 = posDic[s]
        gx1 = gxDic[s]
        for line in train:
            index_line = sent2vec2(line, word_index, max_word)
            data[i].append(index_line)
        for line in pos1:
            index_line = pos2index(line, pos_index, max_word)
            pos[i].append(index_line)
        for line in label1:
            index = label2idx.get(line)
            index_line = [0, 0]
            index_line[index] = 1
            label[i].append(index_line)
        for line in gx1:
            gx[i].append([line] * max_word)
    # for i in range(len(a)):
    #     print(len(gx[i]))

    return data, pos, label, gx



def readEmbedFile(embFile):
    """
    读取预训练的词向量文件，引入外部知识
    """
    print("\nProcessing Embedding File...")
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(EMBEDDING_DIM)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.001, 0.001, EMBEDDING_DIM)
    embeddings["NUMBER"] = np.random.uniform(-0.001, 0.001, EMBEDDING_DIM)

    with codecs.open(embFile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        if len(line.split())<=2:
            continue
        values = line.strip().split()
        word = str(values[0])
        # word = wordNormalize(word)
        vector = np.asarray(values[1:], dtype=np.float32)
        embeddings[word] = vector

    print('Found %s word vectors.' % len(embeddings))  # 693537
    return embeddings


def produce_matrix(word_index):
    miss_num=0
    num=0

    embeddingsDic = readEmbedFile(embFile=embeddingFile)

    num_words = min(MAX_NB_WORDS, len(word_index)+1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddingsDic.get(word)
        if embedding_vector is not None:
            num=num+1
            embedding_matrix[i] = embedding_vector
        else:
            # words not found in embedding index will be all-random.
            vec = np.random.uniform(-0.001, 0.001, size=EMBEDDING_DIM)  # 随机初始化
            embedding_matrix[i] = vec
            miss_num=miss_num+1
    print('missnum',miss_num)    # 10604
    print('num',num)    # 56293
    return embedding_matrix


def main(root):

    datasDic = {'train': [], 'aux': [], 'test': []}
    labelsDic = {'train': [], 'aux': [], 'test': []}
    posDic = {'train': [], 'aux': [], 'test': []}
    gxDic = {'train': [], 'aux': [], 'test': []}

    max_word = 0
    num = 1
    pos_index = OrderedDict()
    for t in ['train', 'aux', 'test']:
        with open(root + t + '.data', encoding='utf-8') as f:
            for line in f:
                token = line.strip('\n').split(' ')
                max_word = max(max_word, len(token))
                datasDic[t].append(token)

        with open(root + t + '.label', encoding='utf-8') as f:
            for line in f:
                token = line.strip('\n')
                labelsDic[t].append(token)

        with open(root + t + '.pos', encoding='utf-8') as f:
            for line in f:
                pos = line.strip('\n').split(' ')
                posDic[t].append(pos)
                for p in pos:
                    if p in pos_index:
                        continue
                    else:
                        pos_index[p] = num
                        num += 1

        with open(root + t + '.data', encoding='utf-8') as f:
            for line in f:
                token = line.strip('\n')
                gxDic[t].append(0 if token == 'N' else 1)

    print('max_word: ', max_word)  # 13

    # # 取其中一份的 200 个实例作为训练数据
    # positive=0
    # negative=0
    # for i in range(len(labelsDic['aux'][300:])):
    #     label = labelsDic['aux'][300+i]
    #     if label=='1' and positive<50:
    #         positive+=1
    #         datasDic['train'].append(datasDic['aux'][300+i])
    #         labelsDic['train'].append(labelsDic['aux'][300+i])
    #         posDic['train'].append(posDic['aux'][300+i])
    #     elif label=='0' and negative<150:
    #         negative+=1
    #         datasDic['train'].append(datasDic['aux'][300+i])
    #         labelsDic['train'].append(labelsDic['aux'][300+i])
    #         posDic['train'].append(posDic['aux'][300+i])
    #     elif positive==50 and negative==150:
    #         break

    # 取其中一份的 200 个实例作为辅助训练数据
    positive = 0
    negative = 0
    a = []
    b = []
    c = []
    d = []
    for i in range(len(labelsDic['aux'][300:])):
        label = labelsDic['aux'][300 + i]
        if label == '1' and positive < 50:
            positive += 1
            a.append(datasDic['aux'][300 + i])
            b.append(labelsDic['aux'][300 + i])
            c.append(posDic['aux'][300 + i])
            d.append(gxDic['aux'][300 + i])
        elif label == '0' and negative < 150:
            negative += 1
            a.append(datasDic['aux'][300 + i])
            b.append(labelsDic['aux'][300 + i])
            c.append(posDic['aux'][300 + i])
            d.append(gxDic['aux'][300 + i])
        elif positive == 50 and negative == 150:
            datasDic['aux'] = a
            labelsDic['aux'] = b
            posDic['aux'] = c
            gxDic['aux'] = d
            break

    # print(labelsDic['aux'])

    print('pos 个数：{}'.format(len(pos_index)))  # 32
    print("pos_index['Blank']: ", pos_index['Blank'])

    word_index = get_word_index(datasDic)
    print('Found %s unique tokens.\n' % len(word_index))  # 9962

    data, pos, label, gx = process_data(datasDic, labelsDic, posDic, gxDic, word_index, pos_index, max_word)
    x_train, x_aux, x_test = data[0], data[1], data[2]
    pos_train, pos_aux, pos_test = pos[0], pos[1], pos[2]
    y_train, y_aux, y_test = label[0], label[1], label[2]
    gx_train, gx_aux, gx_test = gx[0], gx[1], gx[2]
    print(np.asarray(x_train).shape)
    print(np.asarray(y_train).shape)
    print(np.asarray(pos_train).shape)
    print(np.asarray(gx_train).shape)

    print(np.asarray(gx_aux).shape)

    import os
    if not os.path.exists(root + 'pkl'):
        os.makedirs(root + 'pkl')

    with open(root + 'pkl/train.pkl', "wb") as f:
        pkl.dump((x_train, y_train, pos_train, gx_train), f, -1)
    with open(root + 'pkl/aux.pkl', "wb") as f:
        pkl.dump((x_aux, y_aux, pos_aux, gx_aux), f, -1)
    with open(root + 'pkl/test.pkl', "wb") as f:
        pkl.dump((x_test, y_test, pos_test, gx_test), f, -1)

    embedding_matrix = produce_matrix(word_index)
    with open(root + 'pkl/emb.pkl', "wb") as f:
        pkl.dump((embedding_matrix, pos_index, max_word), f, -1)
    embedding_matrix = {}

    print('\n保存成功')


# if __name__ == '__main__':
#     main(r'corpus_extracted/wiki4result/')

