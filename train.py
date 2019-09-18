# -*- coding:utf-8 -*-

import re
import numpy as np
import pandas as pd
import jieba
import time
import gc


def word_cut(data): 
    stoplist = [line.strip() for line in open('../data/stopword.txt').readlines()]    
    # jieba.load_userdict('userdict.txt')
    def get_cut(text):
        r1 = '（）'  # 应该考虑多种括号，【】，[], (), （）
        r2 = '[a-zA-Z0-9’!"#$%&\'()（）；‘’*+,-./：:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        r3 = '\s+'
        seg="\n".join(text)
        seg_list = jieba.cut(re.sub(r3, "", re.sub(r2, "", re.sub(r1, "", seg))), cut_all=False)
        words = ' '.join(seg_list)
        words = [word for word in words.split() if word not in stoplist and len(word)>=2]
        words = ' '.join(words)
        if len(words) == 0:
            return False
        else:
            return words
        
    user_id = [item.split(',')[0] for item in open(data)]
    texts = [get_cut(item.split(',')[1:]) for item in open(data) if get_cut(item)]

    df_data = pd.concat([pd.DataFrame(user_id), pd.DataFrame(texts)], axis=1)
    df_data.columns = ['id', 'text']
    return df_data



def main():
    start = time.clock()

    # get train data
    train = word_cut("../data/Train_DataSet.csv")
    label = pd.read_csv("../data/Train_DataSet_Label.csv")
    df_data = pd.merge(train, label, on='id')
    df_data.to_csv("train.csv", index=False)

    # train model

    # predict
    
    end = time.clock()
    print('Program over, time cost is %s' % (end-start))



if __name__ == "__main__":
    main()
