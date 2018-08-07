# -*- coding: utf-8 -*-
# @Time    : 2018/6/25 14:54
# @Author  : Jason
# @FileName: LSTM.py

import gluonbook as gb
from mxnet import nd
import zipfile
import re

# 读取数据
with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('./data/')
with open('./data/jaychou_lyrics.txt', encoding='utf-8') as f:
    all_chars = f.read()

# 处理数据,保留中文字符,去掉其他字符
all_chars = all_chars.replace('\n', ' ').replace('\r', ' ')
all_chars = re.sub('[A-Za-z0-9\.\*\+\?\]\[＞＜<】〇〗〖\【\\>!?>><<~/\u3000》,☆。！《》、`,～？…]', '', all_chars)
all_chars = all_chars[0:20000]
only_char_list = list(set(all_chars))
only_char_dict = dict([(char, i) for i, char in enumerate(only_char_list)])
only_char_frequency = [only_char_dict[char] for char in all_chars]
vocab_size = len(only_char_dict)

# 初始化模型

ctx = gb.try_gpu()
input_dim = vocab_size
hiddens_dim = 256
output_dim = vocab_size


def get_params():
    '''
    输入门：It=σ(Xt*Wxi+Ht−1*Whi+bi)
    遗忘门：Ft=σ(Xt*Wxf+Ht−1*Whf+bf)
    输出门：Ot=σ(Xt*Wxo+Ht−1*Who+bo)
    候选记忆细胞：Ct_hat=tanh(Xt*Wxc+Ht−1*Whc+bc)
    :return: 参数对
    '''
    # 输入门参数
    W_xi = nd.random.normal(scale=.01, shape=(input_dim, hiddens_dim), ctx=ctx)
    W_hi = nd.random.normal(scale=.01, shape=(hiddens_dim, hiddens_dim), ctx=ctx)
    b_i = nd.zeros(hiddens_dim, ctx=ctx)
    # 遗忘门参数
    W_xf = nd.random.normal(scale=.01, shape=(input_dim, hiddens_dim), ctx=ctx)
    W_hf = nd.random.normal(scale=.01, shape=(hiddens_dim, hiddens_dim), ctx=ctx)
    b_f = nd.zeros(hiddens_dim, ctx=ctx)
    # 输出门参数
    W_xo = nd.random.normal(scale=.01, shape=(input_dim, hiddens_dim), ctx=ctx)
    W_ho = nd.random.normal(scale=.01, shape=(hiddens_dim, hiddens_dim), ctx=ctx)
    b_o = nd.zeros(hiddens_dim, ctx=ctx)
    # 候选记忆细胞参数
    W_xc = nd.random.normal(scale=.01, shape=(input_dim, hiddens_dim), ctx=ctx)
    W_hc = nd.random.normal(scale=.01, shape=(hiddens_dim, hiddens_dim), ctx=ctx)
    b_c = nd.zeros(hiddens_dim, ctx=ctx)
    # 输出层参数
    W_hy = nd.random.normal(scale=.01, shape=(hiddens_dim, output_dim), ctx=ctx)
    b_y = nd.zeros(output_dim, ctx=ctx)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params


# 定义模型
def lstm_rnn(inputs, state_h, state_c, *params):
    '''

    :param inputs: 输入
    :param state_h: 上一时刻的输出
    :param state_c: 上一时刻的状态
    :param params: 参数对
    :return: 输出
    输入门：It=σ(Xt*Wxi+Ht−1*Whi+bi)
    遗忘门：Ft=σ(Xt*Wxf+Ht−1*Whf+bf)
    输出门：Ot=σ(Xt*Wxo+Ht−1*Who+bo)
    输入状态：I_state=tanh(Xt*Wxc+Ht−1*Whc+bc)
    输出：Y=Why*Ht-1+by
    '''
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hy, b_y] = params
    H = state_h  # 与输入组成一个输入门状态，控制有多少新输入补充到最新记忆里
    C = state_c  # 记录这一时刻的状态，传给下一时刻
    outputs = []
    for X in inputs:
        I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)  # 输入门,就是
        C_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)  # 输入门状态用来控制有多少输入信息补充到最新的记忆
        F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)  # 遗忘门，控制上一刻有多少信息被遗忘
        O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)  # 输出门
        C = F * C + C_tilda * I  # 更新作为下一刻的状态
        H = O * C.tanh()
        Y = nd.dot(H, W_hy) + b_y  # 这一刻的输出作为下一刻的输入
        outputs.append(Y)
    return (outputs, H, C)


# 训练模型
get_inputs = gb.to_onehot
num_epochs = 150
num_steps = 5
batch_size = 10
lr = 0.25
clipping_theta = 5
prefixes = ['我爱', '下雨']
pred_period=5
pred_len=100

gb.train_and_predict_rnn(lstm_rnn,False, num_epochs, num_steps, hiddens_dim,
                         lr, clipping_theta, batch_size, vocab_size,
                         pred_period, pred_len, prefixes, get_params,
                         get_inputs, ctx, only_char_frequency, only_char_list,
                         only_char_dict, is_lstm=True)
