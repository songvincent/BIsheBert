#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import random
from collections import Counter
from tqdm import tqdm
# from util.Logginger import init_logger
# import config.args as args
import operator


# In[ ]:


import jieba


# In[ ]:


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer


# In[ ]:


def train_val_split(X, y, valid_size=0.2, random_state=2020, shuffle=True):
    """
    训练集验证集分割
    :param X: sentences
    :param y: labels
    :param random_state: 随机种子
    """
#     logger.info('Train val split')

    data = []
    for data_x, data_y in tqdm(zip(X, y), desc='Merge'):
        data.append((data_x, data_y))
    del X, y

    N = len(data)
    test_size = int(N * valid_size)

    if shuffle:
        random.seed(random_state)
        random.shuffle(data)

    valid = data[:test_size]
    train = data[test_size:]

    return train, valid


# In[ ]:



def sent2char(line):
    """
    句子处理成单词
    :param line: 原始行
    :return: 单词， 标签
    """
    res = line.strip('\n').split()
    return res


# In[ ]:


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None,others =None):
        """创建一个输入实例
        Args:
            guid: 每个example拥有唯一的id
            text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
            text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
            label: example对应的标签，对于训练集和验证集应非None，测试集为None
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.others = others


class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, output_mask,others):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.output_mask = output_mask
        self.others = others


# In[ ]:


class DataProcessor(object):
    """数据预处理的基类，自定义的MyPro继承该类"""
    def get_train_examples(self, data_dir):
        """读取训练集 Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """读取验证集 Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """读取标签 Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as fr:
            lines = []
            for line in fr:
                _line = line.strip('\n')
                lines.append(_line)
            return lines


# In[ ]:


class MyPro(DataProcessor):
    """将数据构造成example格式"""
    """输入为json文件，每一行一个dumps后的字符串"""
    def _create_example(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%d" % (set_type, i)
            line = json.loads(line)
            text_a = line["source"] ##文字内容
            label = line["target"]  ##list
            relations = line["relations"]
            #assert len(label) == len(text_a.split())  利用split是无法进行中文分词的，需要用到list()函数
            assert len(label) == len(list(label))
            example = InputExample(guid=guid, text_a=text_a, label=label,others = relations)
            examples.append(example)
        return examples

    def get_train_examples(self, data_dir):
        lines = self._read_json(data_dir)
        examples = self._create_example(lines, "train")
        return examples

    def get_dev_examples(self, data_dir):
        lines = self._read_json(data_dir)
        examples = self._create_example(lines, "dev")
        return examples

    def get_labels(self):
        return args.labels


# In[ ]:


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    # 标签转换为数字
    label_map = {label: i for i, label in enumerate(label_list)}

    # load sub_vocab
    sub_vocab = {}
#     with open(VOCAB_FILE, 'r') as fr:
#         for line in fr:
#             _line = line.strip('\n')
#             if "##" in _line and sub_vocab.get(_line) is None:
#                 sub_vocab[_line] = 1

    features = []
    for ex_index, example in enumerate(examples):
#         tokens_a = tokenizer.tokenize(example.text_a)  这种分词，会将“2015”分为一个词
                                                        ## encode方法 则是在上述的基础上，前后加结束标志（[CLS][SEP]）
    
        ### 整理关系，
        others = example.others
        
        
        tokens_a = list(example.text_a)
#         print(tokens_a)
        labels = example.label
#         print(tokens_a)
#         print(labels)
        assert len(tokens_a)== len(labels)
        # labels = example.label.split()
        
        if len(tokens_a)==0 or len(labels)==0:
            continue
            
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length-2)]
            labels = labels[:(max_seq_length-2)]
        
        
        assert len(tokens_a)== len(labels)  ### 第一个监测点
        
        # ----------------处理source--------------
        ## 句子首尾加入标示符
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        ## 词转换成数字
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # ---------------处理target----------------
        ## Notes: label_id中不包括[CLS]和[SEP]
        label_id = [label_map[l] for l in labels]
        
        label_id = [0] + label_id + [0]
        label_padding = [0] * (max_seq_length-len(label_id))  ## 由 -1 --> 0
        label_id += label_padding

        ## output_mask用来过滤bert输出中sub_word的输出,只保留单词的第一个输出(As recommended by jocob in his paper)
        ## 此外，也是为了适应crf
        output_mask = [0 if sub_vocab.get(t) is not None else 1 for t in tokens_a]
        output_mask = [0] + output_mask + [0]
        output_mask += padding

        # ----------------处理后结果-------------------------
        # for example, in the case of max_seq_length=10:
        # raw_data:          春 秋 忽 代 谢le
        # token:       [CLS] 春 秋 忽 代 谢 ##le [SEP]
        # input_ids:     101 2  12 13 16 14 15   102   0 0 0
        # input_mask:      1 1  1  1  1  1   1     1   0 0 0
        # label_id:          T  T  O  O  O
        # output_mask:     0 1  1  1  1  1   0     0   0 0 0
        # --------------看结果是否合理------------------------

        feature = InputFeature(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               label_id=label_id,
                               output_mask=output_mask,
                              others = others)
        features.append(feature)

    return features


# In[ ]:


def create_batch_iter(mode,data_dir,train_batch_size,label_list):
    """构造迭代器"""
#     processor, tokenizer = init_params()
    processor = MyPro()
    tokenizer = BertTokenizer('../bert-base-chinese/vocab.txt')
    if mode == "train":
        examples = processor.get_train_examples(data_dir)

#         num_train_steps = int(
#             len(examples) /train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        batch_size = train_batch_size

#         logger.info("  Num steps = %d", num_train_steps)

    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
        batch_size = args.eval_batch_size
    else:
        raise ValueError("Invalid mode %s" % mode)

#     label_list = processor.get_labels()

    # 特征
    features = convert_examples_to_features(examples, label_list, 512, tokenizer)

#     logger.info("  Num examples = %d", len(examples))
#     logger.info("  Batch size = %d", batch_size)


    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
#     for 
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    
    max_len = 0
    for f in features:
        if len(f.others) > max_len:
            max_len = len(f.others)
    pad = (0,0,0)
    otherss = []
    for f in features:
        tmp = f.others
        pad_len = max_len-len(tmp)
        for i in range(pad_len):
            tmp.append(pad)
        otherss.append(tmp)
    all_others = torch.tensor(otherss)
    
#     all_others = torch.tensor([f.others for f in features], dtype=torch.long)
#     all_output_mask = torch.tensor([f.output_mask for f in features], dtype=torch.long)

    # 数据集
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_others)# , all_output_mask)

    if mode == "train":
        sampler = RandomSampler(data)
    elif mode == "dev":
        sampler = SequentialSampler(data)
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 迭代器
    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if mode == "train":
        return iterator
#         return iterator, num_train_steps
    elif mode == "dev":
        return iterator
    else:
        raise ValueError("Invalid mode %s" % mode)


# In[ ]:


# label_list = ['O', 'B-Nh', 'I-Nh', 'B-NDR', 'I-NDR']
# train_iter = create_batch_iter("train",'../multi_data/multi_trainner.txt',32,label_list)

# for step, batch in enumerate(train_iter):
#     input_ids, input_mask, segment_ids, label_ids = batch
#     print(input_ids.shape)   ##(batch_size,512)
#     break


# In[ ]:


# mypro = MyPro()
# in_exas =  mypro.get_train_examples('../multi_data/multi_trainner.txt')

# raw_label = ['Nh','NDR']
# label_list = ['O']
# for rl in raw_label:
#     label_list.append('B-'+rl)
#     label_list.append('I-'+rl)
# print(label_list)   
# # for ie in in_exas:
# #     print(ie.label)
# #     break

# from transformers import BertTokenizer

# tokenizer = BertTokenizer('../bert-base-chinese/vocab.txt')

# #convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)


# features = convert_examples_to_features(in_exas,label_list,512,tokenizer)

# # features[0]

# # input_ids=input_ids,
# # input_mask=input_mask, attention_mask
# # segment_ids=segment_ids,tokrn_type_id
# # label_id=label_id,
# # output_mask=output_mask

# tokenizer = BertTokenizer('../bert-base-chinese/vocab.txt')

# tokenizer.encode('2015年2月16日我去北京')

# tokenizer.tokenize('2015年2月16日我去北京')


# In[ ]:


# from transformers import BertTokenizer

# tokenizer = BertTokenizer('../bert-base-chinese/vocab.txt')

# #convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)


# features = convert_examples_to_features(in_exas[1:2],label_list,512,tokenizer)


# In[ ]:




