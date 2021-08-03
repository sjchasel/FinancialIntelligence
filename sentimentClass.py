import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel
import pandas as pd
import re
import jieba

BERT_PATH = 'bert-base-chinese/'


class GetBERT(nn.Module):

    def __init__(self):
        super(GetBERT, self).__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained("C:/Users/12968/Desktop/chinese-bert-wwm-ext")
        self.bert = BertModel.from_pretrained("C:/Users/12968/Desktop/chinese-bert-wwm-ext")
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
        x = embeddings[0].mean(dim=1)  # 1 * 768
        # print(x.shape)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        out = F.softmax(x)
        return out

    def predict(self, x):
        pred = torch.tensor(model.forward(x)).squeeze(dim=1)
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return ans


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
            i = self.segment(i)
            if len(i) > 20:
                i = self.get_keywords(i)
            if i != '':
                new_sent.append(i)
        return new_sent

data = pd.read_excel('E:/FinancialIntelligence/sentiment_classify_data/comments_raw_v1.xls')
data = data[data['score']!=3].reset_index()
data['label'] = data['score'].map(lambda a : 1 if a in [4,5] else 0)
data.drop(['id','score'],inplace=True,axis=1)
#data['content'].astypes(str)
data['content'] = [str(i) for i in list(data['content'])]
print(data['label'].value_counts())

data1 = data[data['label']==1].sample(3600)
data0 = data[data['label']==0].sample(3600)
data = pd.concat([data1,data0],axis=0,ignore_index=True)
x = []
y = []
errors = []
model = GetBERT()
for i in trange(len(data)):
    p = Pre(data['content'][i])
    sen_lst = p.preprocess()
    if sen_lst == []:
        errors.append(i)
        continue
    x.append(model(sen_lst))
    y.append(data['label'][i])

criterion = nn.CrossEntropyLoss()  #交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #Adam梯度优化器
epochs = 10000
losses = []
model = Model()
x = torch.stack(x,0).requires_grad_()
for i in range(epochs):
    y_pred = torch.tensor(model.forward(x)).squeeze(dim=1).requires_grad_()
    loss = criterion(y_pred,torch.tensor(y))
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

from sklearn.metrics import accuracy_score
print(accuracy_score(model.predict(x),y))