#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch.nn as nn
import torch
import torch.nn.functional as F
# from net.crf import CRF
import numpy as np
# from sklearn.metrics import f1_score, classification_report
# import config.args as args
from transformers import BertPreTrainedModel, BertModel
# from transformers import  BertModel


# In[2]:


class Bert_Softmax(BertPreTrainedModel):
    def __init__(self,
                 config,
                 num_tag):
        super(Bert_Softmax, self).__init__(config)
        # model = BertModel.from_pretrained('../bert-base-chinese/', config=config)
        self.bert = BertModel.from_pretrained('../bert-base-chinese/', config=config) #.cuda()
        # for p in self.bert.parameters():
        #     p.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_tag)
#         self.apply(self.init_bert_weights)

#         self.crf = CRF(num_tag)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask):
        '''
        self.bert
        return tuple length = 2
        tuple[0].shape [batch_size,seq_len+2(CLS+SEP),768]
        '''
        bert_encode, _ = self.bert(input_ids, attention_mask, token_type_ids)  ## 
        bert_encode = bert_encode.cuda()
        output = self.classifier(bert_encode)  ##output_shape [batch_size,seq_len+2,num_tag]
        output = F.log_softmax(output, dim=2)
#         output = torch.softmax(output,dim=2)  ##output_shape [batch_size,seq_len+2,num_tag]
#         output = torch.argmax(output,dim=2)   ##output_shape [batch_size,seq_len+2]
        return output


# In[5]:


class RE_Bert_Softmax(BertPreTrainedModel):
    def __init__(self,
                 config,
                 num_tag):
        super(RE_Bert_Softmax, self).__init__(config)
        # model = BertModel.from_pretrained('../bert-base-chinese/', config=config)
        self.bert = BertModel.from_pretrained('../bert-base-chinese/', config=config) #.cuda()
        # for p in self.bert.parameters():
        #     p.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_tag)
#         self.apply(self.init_bert_weights)

#         self.crf = CRF(num_tag)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask):
        '''
        self.bert
        return tuple length = 2
        tuple[0].shape [batch_size,seq_len+2(CLS+SEP),768]
        '''
        _,bert_encode = self.bert(input_ids, attention_mask, token_type_ids)  ## [CLS]的output
        bert_encode = bert_encode.cuda()
#         bert_encode_re = [for be in bert_encode]
        output = self.classifier(bert_encode)  ##output_shape [batch_size,seq_len+2,num_tag]
#         output = F.log_softmax(output, dim=2)
#         output = torch.softmax(output,dim=2)  ##output_shape [batch_size,seq_len+2,num_tag]
#         output = torch.argmax(output,dim=2)   ##output_shape [batch_size,seq_len+2]
        return output


# In[3]:


# from data_loader import MyPro,convert_examples_to_features,create_batch_iter

# # from bert_ner import Bert_Softmax
# import torch
# from transformers import BertConfig

# label_list = ['O', 'B-Nh', 'I-Nh', 'B-NDR', 'I-NDR']
# train_iter = create_batch_iter("train",'../multi_data/multi_trainner.txt',16,label_list)

# config = BertConfig.from_json_file('../bert-base-chinese/bert_config.json')
# bert_softmax = Bert_Softmax(config,5).cuda()

# # device = torch.device(if torch.cuda.is_available() and not args.no_cuda else "cpu")



# for step, batch in enumerate(train_iter):
    
#     input_ids, input_mask, segment_ids, label_ids = batch   ##(batch_size,512)
#     input_ids = input_ids.cuda()
#     input_mask = input_mask.cuda()
#     segment_ids = segment_ids.cuda()
#     label_ids = label_ids.cuda()
#     out = bert_softmax(input_ids,segment_ids,input_mask)  #[batch_size,seq_len+2(512)]
#     print(out.shape)
#     print(out)
# #     out = bert_softmax(input_ids,token_type_ids,attention_mask)  #[batch_size,seq_len+2(512)]
#     break


# In[2]:


# from transformers import BertConfig
# config = BertConfig.from_json_file('../bert-base-chinese/bert_config.json')
# model = BertModel.from_pretrained('../bert-base-chinese/', config=config)

# model.parameters()

# count = 0
# for e in model.parameters():
#     count += 1
# #     print(type(e))
#     print(e.shape)
#     print(e)
#     break

# # for name, param in model.named_parameters():
# #     print(name,param.size())


# In[ ]:




