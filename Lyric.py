# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 9:59
# @Author  : Jason
# @FileName: Lyric.py

import re
import gluonbook as gb
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import random
import zipfile

# 解压文件
with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('./data/')
# 打开文件
with open('./data/jaychou_lyrics.txt', encoding='utf-8') as f:
    all_chars = f.read()

# 处理数据集，将换行符替换成空格且只保留中文字符，使用前两万个字符当作训练集
all_chars = all_chars.replace('\n', ' ').replace('\r', ' ').replace('\\\\', '')
all_chars = re.sub('[A-Za-z0-9\.\*\+\?\]\[＞＜<】〇〗〖\\\\【>!?>><<~/\u3000》,☆。！《》、`,～？…]', '', all_chars)
# print(len(all_chars))  # 63282
train_set = all_chars[0:20000]
# print(train_set[0:50])

# 将数据集中所有不同字符提取出来做成字典
index_to_char = list(set(all_chars))  # 不同字符列表
# print(index_to_char)
char_to_index = dict([(char, i) for i, char in enumerate(index_to_char)])  # 字符:频次  字典
vocab_size = len(char_to_index)
# print(char_to_index)
# print(vocab_size)  # 一共有2514个不同的汉字

# 将每个汉字转成从0开始的索引
chars_indices = [char_to_index[char] for char in all_chars]  # 索引
sample = chars_indices[:50]
# print('chars:\n', ''.join([index_to_char[idx] for idx in sample]))
# print('\nindices:\n', sample)


# 随机采样
def data_iter_random(chars_indices, batch_size, num_steps, ctx=None):
    num_examples = (len(chars_indices) - 1) // num_steps  # 一共有多少个时间步
    epoch_size = num_examples // batch_size  # 一共循环多少次
    example_indices = list(range(num_examples))  # 时间布列表
    random.shuffle(example_indices)  # 打乱

    def _data(pos):
        return chars_indices[pos:pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i:i + batch_size]
        X = nd.array([_data(j * num_steps) for j in batch_indices], ctx=ctx)
        Y = nd.array([_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
        yield X, Y


my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=3):
    print('X:', X, '\nY:', Y, '\n')


# 顺序采样
def data_iter_consecutive(chars_indices, batch_size, num_steps, ctx=None):
    chars_indices = nd.array(chars_indices, ctx=None)
    data_len = len(chars_indices)
    batch_len = data_len // batch_size
    indices = chars_indices[0:data_len].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i:i + num_steps]
        Y = indices[:, i + 1:i + num_steps + 1]
        yield X, Y


for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=3):
    print('X:', X, '\nY:', Y)


def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]


get_inputs = to_onehot
inputs = get_inputs(X, vocab_size)
print(len(inputs), inputs[0].shape)

# 初始化模型
ctx = gb.try_gpu()
print('will use ', ctx)

num_inputs = vocab_size
num_hiddens = 256  # 隐藏单元个数
num_outputs = vocab_size


def get_params():
    # 隐藏层参数
    W_xh = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens), ctx=ctx)
    W_hh = nd.random.normal(scale=0.01, shape=(num_hiddens, num_hiddens), ctx=ctx)
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # 输出层参数
    W_hy = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs), ctx=ctx)
    b_y = nd.zeros(num_outputs, ctx=ctx)
    params = [W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params


def rnn(inputs, state, *params):
    H = state
    W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return outputs, H


# 做个简单的测试
state = nd.zeros(shape=(X.shape[0], num_hiddens), ctx=ctx)
params = get_params()
outputs, state_new = rnn(get_inputs(X.as_in_context(ctx), vocab_size), state, *params)
print(len(outputs), outputs[0].shape, state_new.shape)


# 定义预测函数,预测基于前缀prefix接下来的num_chars个字符，用它根据训练得到的循环神经网络rnn来创作歌词
def predict_rnn(rnn, prefix, num_chars, params, num_hiddens, vocab_size, ctx, index_to_char, char_to_index, get_inputs,
                is_lstm=False):
    prefix = prefix.lower()  # 前缀字符，可以是单个字符，也可以是词语字符集
    state_h = nd.zeros(shape=(1, num_hiddens), ctx=ctx)
    if is_lstm:
        # 当rnn使用lstm时会用到
        state_c = nd.zeros(shape=(1, num_hiddens), ctx=ctx)
    outputs = [char_to_index[prefix[0]]]
    for i in range(num_chars + len(prefix)):
        X = nd.array([outputs[-1]], ctx=ctx)
        # 在序列中循环迭代隐藏状态。
        if is_lstm:
            # 当 RNN 使用 LSTM 时才会用到（后面章节会介绍），本节可以忽略。
            Y, state_h, state_c = rnn(get_inputs(X, vocab_size), state_h,
                                      state_c, *params)
        else:
            Y, state_h = rnn(get_inputs(X, vocab_size), state_h, *params)
        if i < len(prefix) - 1:
            next_input = char_to_index[prefix[i + 1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        outputs.append(next_input)
    return ''.join([index_to_char[i] for i in outputs])


# 裁剪梯度

def grad_clipping(params, state_h, Y, theta, ctx):
    if theta is not None:
        norm = nd.array([0.0], ctx)
        for param in params:
            norm += (param.grad ** 2).sum()
        norm = norm.sqrt().asscalar()
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm


# 定义模型训练函数
def train_and_predict_rnn(rnn, is_random_iter, num_epochs, num_steps,
                          num_hiddens, lr, clipping_theta, batch_size,
                          vocab_size, pred_period, pred_len, prefixes,
                          get_params, get_inputs, ctx, corpus_indices,
                          idx_to_char, char_to_idx, is_lstm=False):
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        # 如使用相邻采样，隐藏变量只需在该 epoch 开始时初始化。
        if not is_random_iter:
            state_h = nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)
            if is_lstm:
                state_c = nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)
        train_l_sum = nd.array([0], ctx=ctx)
        train_l_cnt = 0
        for X, Y in data_iter(corpus_indices, batch_size, num_steps, ctx):
            # 如使用随机采样，读取每个随机小批量前都需要初始化隐藏变量。
            if is_random_iter:
                state_h = nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)
                if is_lstm:
                    state_c = nd.zeros(shape=(batch_size, num_hiddens),
                                       ctx=ctx)
            # 如使用相邻采样，需要使用 detach 函数从计算图分离隐藏状态变量。
            else:
                state_h = state_h.detach()
                if is_lstm:
                    state_c = state_c.detach()
            with autograd.record():
                # outputs 形状：(batch_size, vocab_size)。
                if is_lstm:
                    outputs, state_h, state_c = rnn(
                        get_inputs(X, vocab_size), state_h, state_c, *params)
                else:
                    outputs, state_h = rnn(
                        get_inputs(X, vocab_size), state_h, *params)
                # 设 t_ib_j 为时间步 i 批量中的元素 j：
                # y 形状：（batch_size * num_steps,）
                # y = [t_0b_0, t_0b_1, ..., t_1b_0, t_1b_1, ..., ]。
                y = Y.T.reshape((-1,))
                # 拼接 outputs，形状：(batch_size * num_steps, vocab_size)。
                outputs = nd.concat(*outputs, dim=0)
                l = loss(outputs, y)
            l.backward()
            # 裁剪梯度。
            grad_clipping(params, state_h, Y, clipping_theta, ctx)
            gb.sgd(params, lr, 1)
            train_l_sum = train_l_sum + l.sum()
            train_l_cnt += l.size
        if epoch % pred_period == 0:
            print("\nepoch %d, perplexity %f"
                  % (epoch, (train_l_sum / train_l_cnt).exp().asscalar()))
            for prefix in prefixes:
                print(' - ', predict_rnn(
                    rnn, prefix, pred_len, params, num_hiddens, vocab_size,
                    ctx, idx_to_char, char_to_idx, get_inputs, is_lstm))


num_epochs = 200
num_steps = 35
batch_size = 32
lr = 0.2
clipping_theta = 5
prefixes = ['分开', '不分开']
pred_period = 40
pred_len = 100

train_and_predict_rnn(rnn, True, num_epochs, num_steps, num_hiddens, lr,
                      clipping_theta, batch_size, vocab_size, pred_period,
                      pred_len, prefixes, get_params, get_inputs, ctx,
                      chars_indices, index_to_char, char_to_index)