# -*- coding: utf-8 -*-
# @Time    : 2018/6/25 10:04
# @Author  : Jason
# @FileName: door.py

from mxnet import nd
import zipfile
import re
import gluonbook as gb

# load data
with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('./data')
with open('./data/jaychou_lyrics.txt', encoding='utf-8') as f:
    all_chars = f.read()

# preprocess data
all_chars = all_chars.replace('\n', ' ').replace('\r', ' ')
all_chars = re.sub('[A-Za-z0-9\.\*\+\?\]\[＞＜<】〇〗〖\\\\【>!?>><<~/\u3000》,☆。！《》、`,～？…]', '', all_chars)
all_chars = all_chars[0:20000]
only_char_list = list(set(all_chars))
only_char_dict = dict([(char, i) for i, char in enumerate(only_char_list)])

chars_frequency = [only_char_dict[char] for char in all_chars]  # the frequency of each character
# print(chars_frequency)
vocab_size = len(chars_frequency)

# initialize the model

ctx = gb.try_gpu()  # use gpu to accelerate speed


num_inputs = vocab_size
num_hiddens = 256  # 256 nodes
num_outputs = vocab_size


def get_params():
    '''
    :return: return parameters
    '''
    # Update Gate Zt=σ(Xt*Wxz+Ht−1*Whz+bz).
    W_xz = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens), ctx=ctx)
    W_hz = nd.random.normal(scale=0.01, shape=(num_hiddens, num_hiddens), ctx=ctx)
    b_z = nd.zeros(num_hiddens, ctx=ctx)
    # Reset Gate Rt=σ(Xt*Wxr+Ht−1*Whr+br)
    W_xr = nd.random.normal(scale=.01, shape=(num_inputs, num_hiddens), ctx=ctx)
    W_hr = nd.random.normal(scale=0.01, shape=(num_hiddens, num_hiddens), ctx=ctx)
    b_r = nd.zeros(num_hiddens, ctx=ctx)
    #  Ht=tanh(Xt*Wxh+Rt⊙Ht−1*Whh+bh)
    W_xh = nd.random.normal(scale=.01, shape=(num_inputs, num_hiddens), ctx=ctx)
    W_hh = nd.random.normal(scale=0.01, shape=(num_hiddens, num_hiddens), ctx=ctx)
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # Output Gate
    W_hy = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs), ctx=ctx)
    b_y = nd.zeros(num_outputs, ctx=ctx)

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params


# 定义模型，根据门控循环单元的计算表达式来定义模型
def gru_rnn(inputs, H, *params):
    '''
    更新门 Zt=σ(Xt*Wxz+Ht−1*Whz+bz)
    重置门 Rt=σ(Xt*Wxr+Ht−1*Whr+br)
    候选隐藏层状态 Ht=tanh(Xt*Wxh+Rt⊙Ht−1*Whh+bh)
    下一状态 Ht=Zt⊙Ht−1+(1−Zt)⊙H_hat
    输出 Y=Ht*Why+by
    :param inputs: 输入
    :param H: 上一个状态
    :param params: 参数对
    :return: 输出
    '''
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:
        Z = nd.sigmoid(nd.dot(X, W_xz) + nd.dot(H, W_hz) + b_z)  # 更新门
        R = nd.sigmoid(nd.dot(X, W_xr) + nd.dot(H, W_hr) + b_r)  # 重置门
        H_hat = nd.tanh(nd.dot(X, W_xh) + R * nd.dot(H, W_hh) + b_h)  # 候选隐藏层状态
        H = Z * H + (1 - Z) * H_hat  # 下一个状态
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return (outputs, H)


# 设置超参，并训练模型来创作歌词
get_inputs = gb.to_onehot
num_epochs = 150  # 训练次数
num_steps = 35  # 时间步长
batch_size = 32
lr = 0.25  # 学习率
clipping_theta = 5  # 裁剪梯度
prefixes = ['分开', '不分开']
pred_peroid = 30
pred_len = 100

print(gb.train_and_predict_rnn(gru_rnn, False, num_epochs, num_steps, num_hiddens, lr, clipping_theta, batch_size,
                               vocab_size, pred_peroid, pred_len, prefixes, get_params, get_inputs, ctx,
                               chars_frequency, only_char_list, only_char_dict))
