import seaborn as sns
import matplotlib.pyplot as plt
import jieba
import pandas as pd
import random



class Sentiment():

    def __init__(self, data):
        self.data = data
        self.stopwords = []
        self.posdic = []
        self.negdic = []
        self.mostdict = []
        self.verydict = []
        self.moredict = []
        self.ishdict = []
        self.insufficientdict = []
        self.inversedict = []

#     def read_file(self):
#         """
#         提取csv文件中的研报内容，存入list并返回
#         :param filename:
#         :return:
#         """
#         data = pd.read_csv(self.filename)
#         return list(data['评论'])

    def dict_load(self, path):
        dict = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip() != '':  # 养成去空好习惯
                    dict.append(line.strip())
        return dict

    def load_dicts(self):
        stop = 'dict/stop1205.txt'
        pos = 'dict/pos_all_dict.txt'
        neg =  'dict/neg_all_dict.txt'
        most =  'dict/most.txt'
        very = 'dict/very.txt'
        more = 'dict/more.txt'
        ish = 'dict/ish.txt'
        insufficient = 'dict/insufficiently.txt'
        inverse = 'dict/inverse.txt'
        self.stopwords = self.dict_load(stop)
        self.posdict = self.dict_load(pos)
        self.negdict = self.dict_load(neg)
        self.mostdict = self.dict_load(most)  # 权值为2
        self.verydict = self.dict_load(very)  # 权值为1.5
        self.moredict = self.dict_load(more)  # 权值为1.25
        self.ishdict = self.dict_load(ish)  # 权值为0.5
        self.insufficientdict = self.dict_load(insufficient)  # 权值为0.25
        self.inversedict = self.dict_load(inverse)  # 权值为-1

    def seg_sentence(self, sentence):
        """
        输入字符串，返回分词后的列表
        :param sentence:
        :return:
        """
        sentence_seged = jieba.cut(sentence.strip())
        outstr = ''
        for word in sentence_seged:
            if word not in self.stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr.split(' ')

    def match_adverb(self, word, sentiment_value):
        """
        对不同种类的词赋予不同的权重
        :param sentiment_value:
        :return:
        """
        # 最高级权重为
        if word in self.mostdict:
            sentiment_value *= 2  # 2/8
        # 比较级权重
        elif word in self.verydict:
            sentiment_value *= 1.75  # 1.75/6
        # 比较级权重
        elif word in self.moredict:
            sentiment_value *= 1.5  # 1.5/4
        # 轻微程度词权重
        elif word in self.ishdict:
            sentiment_value *= 1.2  # 1.2/2
        # 相对程度词权重
        elif word in self.insufficientdict:
            sentiment_value *= 0.5
        # 否定词权重
        elif word in self.inversedict:
            sentiment_value *= -1
        else:
            sentiment_value *= 1
        return sentiment_value

    def cal_score(self, words_list):

        # i，s 记录情感词和程度词出现的位置
        i = 0  # 记录扫描到的词位子
        s = 0  # 记录情感词的位置
        poscount = 0  # 记录积极情感词数目
        negcount = 0  # 记录消极情感词数目
        # 逐个查找情感词
        for word in words_list:
            # 如果为积极词
            if word in self.posdict:
                poscount += 1  # 情感词数目加1
                # 在情感词前面寻找程度副词
                for w in words_list[s:i]:
                    poscount = self.match_adverb(w, poscount)
                s = i + 1  # 记录情感词位置
            # 如果是消极情感词
            elif word in self.negdict:
                negcount += 1
                for w in words_list[s:i]:
                    negcount = self.match_adverb(w, negcount)
                s = i + 1
            # 如果结尾为感叹号或者问号，表示句子结束，并且倒序查找感叹号前的情感词，权重+4
            elif word == '!' or word == '！' or word == '?' or word == '？':
                for w2 in words_list[::-1]:
                    # 如果为积极词，poscount+2
                    if w2 in self.posdict:
                        poscount += 4
                        break
                    # 如果是消极词，negcount+2
                    elif w2 in self.negdict:
                        negcount += 4
                        break
            i += 1  # 定位情感词的位置
        # 计算情感值
        sentiment_score = poscount - negcount
        return sentiment_score

    def res(self, sentiment_score):
        # print('情感分值：', sentiment_score)
        if sentiment_score < 0:
            # print('情感倾向：消极')
            res = -1
        elif sentiment_score == 0:
            # print('情感倾向：中性')
            res = 1 # 中性标记为积极
        else:
            # print('情感倾向：积极')
            res = 1
        return res

    def run(self):
        """
        :return: 两个列表，一个列表存放分数，一个列表存放结果
        """
        content_list = list(self.data['评论'])
        self.load_dicts()
        data = []
        for content in content_list:
            data.append(self.seg_sentence(content))
        scores = []
        result = []
        for i in data:
            scores.append(self.cal_score(i))
        for score in scores:
            result.append(self.res(score))
        return scores, result