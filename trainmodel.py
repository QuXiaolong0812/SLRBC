# -*- coding:utf-8 -*-
import time
import sys
import argparse
import random
import copy
import torch
import gc
import pickle
import os
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.gazlstm import GazLSTM as SeqModel
from utils.data import Data
from tqdm import tqdm
import datetime
import io
import flask, json
from flask import request,g, session

import main
from transformers.models.bert.tokenization_bert import BertTokenizer
max_sent_length = 250
NULLKEY = "-null-"


def normalize_word(word):#单词序列化，所有的数字字符转换成'0'
    new_word = ""
    for char in word:
        if char.isdigit():#判断字符是否为数字
            new_word += '0'
        else:
            new_word += char
    return new_word
# words = ['0', '年', '前', '开', '始', '患', '者', '无', '明', '显', '诱', '因', '出', '现', '中', '上', '腹', '疼', '痛', '，', '每', '次', '持', '续', '时', '间', '不', '等', '，', '饥', '饿', '时', '较', '明', '显', '，', '进', '食', '后', '有', '加', '重', '，', '偶', '有', '打', '嗝', '，', '无', '反', '酸', '、', '烧', '心', '、', '嗳', '气', '，', '无', '恶', '心', '、', '呕', '吐', '，', '无', '呕', '血', '、', '黑', '便', '，', '自', '行', '服', '用', '药', '物', '（', '具', '体', '不', '详', '）', '后', '缓', '解', '，', '未', '予', '以', '重', '视', '。']
def getinstance(words,data):
    if data.use_robert == True:
        tokenizer = BertTokenizer.from_pretrained('./NER/chinese_roberta_wwm_large_ext', do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained('./NER/bert-base-chinese', do_lower_case=True)
    instence_texts = []
    instence_Ids = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    char_padding_size = -1
    char_padding_symbol = '</pad>'
    data.gaz_count[1] = 0 #把 UNKNOWN 标签的词频设置为0,原来的代码中没有UNKNOWN类型标签的词频，遇到UNKNOWN类型词汇代码会崩溃
    for idx in range(len(words)):
        word = words[idx]
        if data.number_normalized:
            word = normalize_word(word)
        label = "O"
        if idx < len(words) -1 and len(words[idx+1]) > 2:
            biword = word + words[idx+1].strip().split()[0]
        else:
            biword = word + NULLKEY
        biwords.append(biword)
        # words.append(word)
        labels.append(label)
        word_Ids.append(data.word_alphabet.get_index(word))
        biword_index = data.biword_alphabet.get_index(biword)
        biword_Ids.append(biword_index)
        label_Ids.append(data.label_alphabet.get_index(label))
        char_list = []
        char_Id = []
        for char in word:
            char_list.append(char)
        if char_padding_size > 0:
            char_number = len(char_list)
            if char_number < char_padding_size:
                char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
            assert(len(char_list) == char_padding_size)
        else:
            ### not padding
            pass
        for char in char_list:
            char_Id.append(data.char_alphabet.get_index(char))
        chars.append(char_list)
        char_Ids.append(char_Id)



    if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (
            len(words) > 0):  # max_sent_length<0代表没有长度限制，否则words的个数应该在0到max_sent_length之间
        gaz_Ids = []
        layergazmasks = []
        gazchar_masks = []
        w_length = len(words)

        gazs = [[[] for i in range(4)] for _ in
                range(w_length)]  # gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[w_id1,w_id2,...]  None:0
        gazs_count = [[[] for i in range(4)] for _ in range(w_length)]

        gaz_char_Id = [[[] for i in range(4)] for _ in
                       range(w_length)]  ## gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[[w1c1,w1c2,...],[],...]

        max_gazlist = 0
        max_gazcharlen = 0
        for idx in range(w_length):

            matched_list = data.gaz.enumerateMatchList(words[idx:])
            matched_length = [len(a) for a in matched_list]
            matched_Id = [data.gaz_alphabet.get_index(entity) for entity in matched_list]

            if matched_length:
                max_gazcharlen = max(max(matched_length), max_gazcharlen)

            for w in range(len(matched_Id)):
                gaz_chars = []
                g = matched_list[w]
                for c in g:
                    gaz_chars.append(data.word_alphabet.get_index(c))

                if matched_length[w] == 1:  ## Single
                    gazs[idx][3].append(matched_Id[w])
                    gazs_count[idx][3].append(1)
                    gaz_char_Id[idx][3].append(gaz_chars)
                else:
                    gazs[idx][0].append(matched_Id[w])  ## Begin
                    gazs_count[idx][0].append(data.gaz_count[matched_Id[w]])
                    gaz_char_Id[idx][0].append(gaz_chars)
                    wlen = matched_length[w]
                    gazs[idx + wlen - 1][2].append(matched_Id[w])  ## End
                    gazs_count[idx + wlen - 1][2].append(data.gaz_count[matched_Id[w]])
                    gaz_char_Id[idx + wlen - 1][2].append(gaz_chars)
                    for l in range(wlen - 2):
                        gazs[idx + l + 1][1].append(matched_Id[w])  ## Middle
                        gazs_count[idx + l + 1][1].append(data.gaz_count[matched_Id[w]])
                        gaz_char_Id[idx + l + 1][1].append(gaz_chars)

            for label in range(4):
                if not gazs[idx][label]:
                    gazs[idx][label].append(0)
                    gazs_count[idx][label].append(1)
                    gaz_char_Id[idx][label].append([0])

                max_gazlist = max(len(gazs[idx][label]), max_gazlist)

            matched_Id = [data.gaz_alphabet.get_index(entity) for entity in matched_list]  # 词号
            if matched_Id:
                gaz_Ids.append([matched_Id, matched_length])
            else:
                gaz_Ids.append([])

        ## batch_size = 1
        for idx in range(w_length):
            gazmask = []
            gazcharmask = []

            for label in range(4):
                label_len = len(gazs[idx][label])
                count_set = set(gazs_count[idx][label])
                if len(count_set) == 1 and 0 in count_set:
                    gazs_count[idx][label] = [1] * label_len

                mask = label_len * [0]
                mask += (max_gazlist - label_len) * [1]

                gazs[idx][label] += (max_gazlist - label_len) * [0]  ## padding
                gazs_count[idx][label] += (max_gazlist - label_len) * [0]  ## padding

                char_mask = []
                for g in range(len(gaz_char_Id[idx][label])):
                    glen = len(gaz_char_Id[idx][label][g])
                    charmask = glen * [0]
                    charmask += (max_gazcharlen - glen) * [1]
                    char_mask.append(charmask)
                    gaz_char_Id[idx][label][g] += (max_gazcharlen - glen) * [0]
                gaz_char_Id[idx][label] += (max_gazlist - label_len) * [[0 for i in range(max_gazcharlen)]]
                char_mask += (max_gazlist - label_len) * [[1 for i in range(max_gazcharlen)]]

                gazmask.append(mask)
                gazcharmask.append(char_mask)
            layergazmasks.append(gazmask)
            gazchar_masks.append(gazcharmask)

        texts = ['[CLS]'] + words + ['[SEP]']
        bert_text_ids = tokenizer.convert_tokens_to_ids(texts)

        instence_texts.append([words, biwords, chars, gazs, labels])
        instence_Ids.append(
            [word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids, gazs, gazs_count, gaz_char_Id, layergazmasks,
             gazchar_masks, bert_text_ids])
        return  instence_texts,instence_Ids

def trainmodel(modelname,modeltype, dropout, hidden_dim, epoch, lr):
    #将model args写入文件
    model_args = {}
    model_args["lr"] = lr
    model_args["drop"] = dropout
    model_args["hidden_dim"] = hidden_dim
    model_args["epoch"] = epoch
    save_args_dir = './save_model/' + modelname + "_args"
    if modeltype == "SLRBC":
        model_args["use_bert"] = True
        model_args["use_robert"] = True
        model_args["use_seftlexion"] = True
    elif  modeltype == "SoftLexicon-BERT":
        model_args["use_bert"] = True
        model_args["use_robert"] = False
        model_args["use_seftlexion"] = True
    elif modeltype == "BiLSTM-CRF":
        model_args["use_bert"] = False
        model_args["use_robert"] = False
        model_args["use_seftlexion"] = False
    output = open(save_args_dir, 'wb')
    pickle.dump(model_args, output)
    output.close()
    save_data_name = "./data/save.dset"
    save_model_dir = './save_model/' + modelname + "_model"
    logfile_path = "./log/" + datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S') + ".txt"
    logfile = open(logfile_path, "a")
    with open(save_data_name, 'rb') as fp:
        data = pickle.load(fp)
        gpu = torch.cuda.is_available()
        data.HP_gpu = gpu
        data.HP_lr =  model_args["lr"]
        data.HP_dropout =  model_args["drop"]
        data.HP_hidden_dim =  model_args["hidden_dim"]
        data.HP_iteration =  model_args["epoch"]
        data.use_bert = model_args["use_bert"]
        data.use_robert = model_args["use_robert"]
        data.HP_use_seftlexion = model_args["use_seftlexion"]
        main.train(data, save_model_dir, logfile)  # 模型训练
    f = open(logfile_path, encoding="utf-8")
    log_text = f.read()
    return  log_text

def loadmodel(modelname):
    model_dir = './save_model/' + modelname + "_model"
    save_args_dir = './save_model/' + modelname + "_args"
    save_data_name = "./data/save.dset"  # common
    with open(save_data_name, 'rb') as fp:
        data = pickle.load(fp)
        gpu = torch.cuda.is_available()
        data.HP_gpu = gpu
        with open(save_args_dir, 'rb') as fp_args:
            model_args = pickle.load(fp_args)
            data.HP_lr = model_args["lr"]
            data.HP_dropout = model_args["drop"]
            data.HP_hidden_dim = model_args["hidden_dim"]
            data.HP_iteration = model_args["epoch"]
            data.use_bert = model_args["use_bert"]
            data.use_robert = model_args["use_robert"]
            data.HP_use_seftlexion = model_args["use_seftlexion"]
        # data.HP_dropout = 0
        model = SeqModel(data)
        model.eval()
        model.load_state_dict(torch.load(model_dir))
    return model,data

def testmodel(words,model,data):
    # if(len(words) < max_sent_length):
    #     return  test_one_text(words, model, data)
    # else:
    ret = []
    textlist = words.split("。")
    # print(textlist)
    for text in textlist:
        if text != '':
            text += "。"
            label = test_one_text(text, model, data)
            ret = ret + label
            # for i in range(len(text)):
            #     print(text[i], "  ", label[i])
    # for i in range(len(words)):
    #     print(words[i], "  ", ret[i])
    return ret
def test_one_text(words,model,data):
    ret = []
    words = list(words)#把srt类型的输入转成list类型
    with torch.no_grad():
        text, instance = getinstance(words, data)
        gaz_list, batch_word, batch_biword, batch_wordlen, batch_label, layer_gaz, gaz_count, gaz_chars, gaz_mask, gazchar_mask, mask, batch_bert, bert_mask = main.batchify_with_label(
            instance, data.HP_gpu, data.HP_num_layer, True)
        tag_seq, gaz_match = model(gaz_list, batch_word, batch_biword, batch_wordlen, layer_gaz, gaz_count, gaz_chars,
                                   gaz_mask, gazchar_mask, mask, batch_bert, bert_mask)
        for i in range(len(tag_seq[0])):
            label = data.label_alphabet.get_instance(tag_seq[0][i])
            ret.append(label)
        # for i in range(len(words)):
        #     print(words[i], "  ", ret[i])
        return ret

#使用模型示例代码
def fun():
    # trainmodel("aa", "SLRBC",0.4, 300, 1, 0.015)
    text = "患者3月前因“直肠癌”于在我院于全麻下行直肠癌根治术（DIXON术），手术过程顺利，术后给予抗感染及营养支持治疗，患者恢复好，切口愈合良好。，术后病理示：直肠腺癌（中低度分化），浸润溃疡型，面积3.5*2CM，侵达外膜。两端切线另送“近端”、“远端”及环周底部切除面未查见癌。肠壁一站（10个）、中间组（8个）淋巴结未查见癌。，免疫组化染色示：ERCC1弥漫（+）、TS少部分弱（+）、SYN（-）、CGA（-）。术后查无化疗禁忌后给予3周期化疗，，方案为：奥沙利铂150MG D1，亚叶酸钙0.3G+替加氟1.0G D2-D6，同时给与升白细胞、护肝、止吐、免疫增强治疗，患者副反应轻。院外期间患者一般情况好，无恶心，无腹痛腹胀胀不适，无现患者为行复查及化疗再次来院就诊，门诊以“直肠癌术后”收入院。   近期患者精神可，饮食可，大便正常，小便正常，近期体重无明显变化。"
    model, data = loadmodel("aa") #load的modelname必须是训练过的modelname
    labellist = testmodel(text,model,data)
    print(labellist)
    return labellist

server = flask.Flask(__name__, static_url_path='')
@server.route("/ner", methods=['GET', "POST"])
def NER():#向前端展示的接口
    text = request.values.get("text")
    print(text)
    # model_name = request.values.get("model")
    global model
    global data
    labellist = testmodel(text,model,data)
    data2 = {"text": text, "labellist": labellist}
    return json.dumps(data2, ensure_ascii=False)

@server.route("/loadmodel", methods=['GET', "POST"])
def LoadModel():#向前端展示的接口
    global model
    global data
    # del model
    # del data
    modelname = request.values.get("modelname")
    model, data = loadmodel(modelname)
    return json.dumps({"a":1}, ensure_ascii=False)

@server.route("/trainmodel", methods=['GET', "POST"])
def trainModel():#向前端展示的接口
    global model
    global data
    # del model
    # del data
    modelname = request.values.get("modelname")#模型训练好保存的模型名称
    modeltype = request.values.get("modeltype")#对何种模型训练
    dropout = request.values.get("drop")
    hidden_dim = request.values.get("hidden_dim")
    epoch = request.values.get("epoch")
    lr = request.values.get("lr")
    dropout = float(dropout)
    hidden_dim = int(hidden_dim)
    epoch = int(epoch)
    lr = float(lr)
    log_text = trainmodel(modelname,modeltype,dropout,hidden_dim,epoch,lr)
    return json.dumps({"log_text":log_text}, ensure_ascii=False)


if __name__ == '__main__':
    server.run(debug=True, port=80, host='0.0.0.0')
