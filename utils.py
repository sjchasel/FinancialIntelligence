#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from copy import deepcopy as copy
from tqdm import tqdm
import torch.optim as optim
import random
import re
import os
import jieba
def seed_torch(seed=1122):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_PATH = 'bert-base-chinese/'


class GetBERT(nn.Module):

    def __init__(self):
        super(GetBERT, self).__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained("chinese-bert-wwm-ext")
        self.bert = BertModel.from_pretrained("chinese-bert-wwm-ext")
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, sentence_lists):
        """
        输入句子列表(去掉了停用词的)
        """
        sentence_lists = [' '.join(x) for x in sentence_lists]
        # print('sentence_lists:'+str(sentence_lists))
        ids = self.bert_tokenizer(sentence_lists, padding=True, return_tensors="pt")
        # print('ids:'+str(ids))
        inputs = ids['input_ids']
        # print('inputs:'+str(inputs))

        embeddings = self.bert(inputs)
        # print(str(embeddings[0].shape))
        x = embeddings[0]  # 1 * 768
        # print(x.shape)
        return x


class Pre:
    def __init__(self, text):
        """
        输入一个文本
        """
        self.puncs_coarse = ['。', '!', '；', '？', '……', '\n', ' ']
        self.text = text
        self.stopwords = self.deal_wrap('dict/stop1205.txt')

    def segment(self, sentence):
        sentence_seged = jieba.cut(sentence.strip())
        outstr = ''
        for word in sentence_seged:
            if word not in self.stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        word_list = outstr.split(' ')
        pattern = '[A-Za-z]*[0-9]*[\'\"\%.\s\@\!\#\$\^\&\*\(\)\-\<\>\?\/\,\~\`\:\;]*[：；”“ ‘’+-——！，。？、~@#￥%……&*（）【】]*'
        t = [re.sub(pattern, "", x.strip()) for x in word_list]
        t = [x for x in t if x != '']
        return ''.join(t)

    def deal_wrap(self, filedict):
        temp = []
        for x in open(filedict, 'r', encoding='utf-8').readlines():
            temp.append(x.strip())
        return temp

    def split_sentence_coarse(self):
        """
        按照。！？“”等中文完整句子语义来分句
        1. 去除换行符、多余的空格、百分号
        2. 分句，存入列表
        :return:装着每个句子的列表（包括标点符号）
        """

        text = self.text
        sentences = []
        start = 0
        for i in range(len(text)):
            if text[i] in self.puncs_coarse:
                sentences.append(text[start:i + 1])
                start = i + 1
        if start == 0:
            sentences.append(text)
        return sentences

    def get_keywords(self, data):
        """
        如果句子太长，就进行关键词提取
        """
        from jieba import analyse
        textrank = analyse.textrank
        keywords = textrank(data, topK=8)
        return ''.join(keywords)

    def preprocess(self):
        # 分句
        sentences = self.split_sentence_coarse()
        # 对每个句子，去除里面的停用词，再连起来
        # 对每个句子，如果句子太长，长度大于20（我随便定的），就抽取八个关键词连起来
        new_sent = []
        for i in sentences:
            if len(i) < 5:
                new_sent.append(i)
                continue
            i = self.segment(i)
            if len(i) > 25:
                i = self.get_keywords(i)
            if i != '':
                new_sent.append(i)
        return new_sent


class GetData():
    def __init__(self, pos=4000, neg=3600):
        data = pd.read_excel('sentiment_classify_data/comments_raw_v1.xls')
        data = data[data['score'] != 3].reset_index()
        data['label'] = data['score'].map(lambda a: 1 if a in [4, 5] else 0)
        data.drop(['id', 'score'], inplace=True, axis=1)
        data['content'] = [str(i) for i in list(data['content'])]
        # 原数据标签为0（负向情感）的数据有3632条，正向情感的有57262条
        data1 = data[data['label'] == 1].sample(pos)
        data0 = data[data['label'] == 0].sample(neg)
        data = pd.concat([data1, data0], axis=0, ignore_index=True)
        self.data = data

    def split_sen(self):
        x = []
        y = []
        for i in trange(len(self.data)):
            p = Pre(self.data['content'][i])
            sen_lst = p.preprocess()
            if sen_lst == []:
                continue
            x.append(sen_lst)
            y.append(self.data['label'][i])
        print(len(x))
        print(y.count(1))
        print(y.count(0))
        return x, y


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm_layer = nn.LSTM(input_size=768, hidden_size=128, batch_first=True)
        self.linear_layer = nn.Linear(in_features=128, out_features=2, bias=True)

    def forward(self, x):
        out1, (h_n, h_c) = self.lstm_layer(x)
        a, b, c = h_n.shape
        out = self.linear_layer(h_n.reshape(a * b, c))
        out = F.log_softmax(out, dim=1)
        return out


def train_model(epoch, train_dataLoader, test_dataLoader):
    # 训练模型
    best_model = None
    train_loss = 0
    test_loss = 0
    best_loss = 100
    epoch_cnt = 0
    for _ in range(epoch):
        total_train_loss = 0
        total_train_num = 0
        total_test_loss = 0
        total_test_num = 0
        for x, y in tqdm(train_dataLoader,
                         desc='Epoch: {}| Train Loss: {}| Test Loss: {}'.format(_, train_loss, test_loss)):
            # for x, y in train_dataLoader:
            x_num = len(x)
            p = model(x)
            loss = loss_func(p, y.long())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            total_train_loss += loss.item()
            total_train_num += x_num
        train_loss = total_train_loss / total_train_num
        train_loss_list.append(train_loss)
        for x, y in test_dataLoader:
            x_num = len(x)
            p = model(x)
            loss = loss_func(p, y.long())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            total_test_loss += loss.item()
            total_test_num += x_num
        test_loss = total_test_loss / total_test_num
        test_loss_list.append(test_loss)

        # early stop
        if best_loss > test_loss:
            best_loss = test_loss
            best_model = copy(model)
            torch.save(best_model.state_dict(), 'lstm_.pth')
            epoch_cnt = 0
        else:
            epoch_cnt += 1

        if epoch_cnt > early_stop:
            torch.save(best_model.state_dict(), 'lstm_.pth')
            print("保存模型")
            # print(best_model.state_dict())
            break
def test_model(test_dataLoader_):
    pred = []
    label = []
    model_.load_state_dict(torch.load("lstm_.pth"))
    model_.eval()
    total_test_loss = 0
    total_test_num = 0
    for x, y in test_dataLoader_:
        x_num = len(x)
        p = model_(x)
#         print('##', len(p), len(y))
        loss = loss_func(p, y.long())
        total_test_loss += loss.item()
        total_test_num += x_num
        pred.extend(p.data.squeeze(1).tolist())
        label.extend(y.tolist())
    test_loss = total_test_loss / total_test_num
    # print('##', len(pred), len(label))
    return pred, label, test_loss, test_loss_list