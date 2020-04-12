#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


def evaluate(out,label_ids,input_mask):
    total_tp =0.0
    total_tn = 0.0
    total_fp = 0.0
    total_fn = 0.0
    for i in range(len(out)):
        label_true = [label_ids[i][j].item() for j in range(512) if input_mask[i][j] == 1 ]
        y_pre = [out[i][j].tolist() for j in range(512) if input_mask[i][j] == 1 ]
#         y_pre = torch.argmax(torch.tensor(y_pre),dim = 1)

        assert len(label_true) == len(y_pre)
        length = len(label_true)

        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0

        for i in range(length):
            if label_true[i] == y_pre[i]:
                if label_true[i] == 0: ### 负例预测为负例
                    tn += 1
                else:             ### 正例预测为正例
                    tp += 1
            else:
                if label_true[i] == 0:  ### 负例 预测为正例
                    fp += 1
                else:
                    fn += 1         ### 正例预测为负例

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
#         print("tp %d,tn %d,fp %d,fn %d" %(tp,tn,fp,fn))
        if tp == 0:
            continue
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*precision*recall/(precision+recall)
#         print("precision :%.4f,recall :%.4f,f1 :%.4f" %(precision,recall,f1))
    if total_tp != 0:
        batch_precision = total_tp/(total_tp+total_fp)
        batch_recall = total_tp/(total_tp+total_fn)
        batch_f1 = 2*batch_precision*batch_recall/(batch_precision+batch_recall)
    
#     print('*********************')
#     print('batch_precision:%.4f  batch_recall:%.4f  batch_f1: %.4f' %(batch_precision,batch_recall,batch_f1))  
    
    return batch_precision,batch_recall,batch_f1
    


# In[4]:


def get_entitys(label_id):
    valid_pos = torch.nonzero(label_id).squeeze(1)
    
    entitys = []
    #         entitys2 = []
    if len(valid_pos) == 0:
        return []
    flag = valid_pos[0]
    start = flag
    end = start
    for vp in valid_pos[1:]:
        if vp -1 == flag:
            end = vp
        else:
            entity = [start.item(),(end+1).item()]
    #         entity = entity.replace(' ','')
            entitys.append(entity)
            start = vp

        flag = vp
    entity = [start.item(),(end+1).item()]
    # entity = entity.replace(' ','')
    entitys.append(entity)   ### 最后一个实体
    return entitys


# In[3]:


def evaluate_bor(out,label_ids,input_mask):
    total_tp =0.0
    total_tn = 0.0
    total_fp = 0.0
    total_fn = 0.0
    batch_precision,batch_recall,batch_f1 = 0.0,0.0,0.0
    for i in range(len(out)):
        label_true = torch.tensor([label_ids[i][j].item() for j in range(512) if input_mask[i][j] == 1 ])
        y_pre = torch.tensor([out[i][j].item() for j in range(512) if input_mask[i][j] == 1 ])

        assert len(label_true) == len(y_pre)
        length = len(label_true)

        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        
        true_entitys = get_entitys(label_true)
        pre_entitys = get_entitys(y_pre)
        
        for pe in pre_entitys:
            if pe in true_entitys:
                tp += 1
            else:
                fp += 1
        fn = len(true_entitys) - tp

#         for i in range(length):
            
#             if label_true[i] == y_pre[i]:
#                 if label_true[i] == 0: ### 负例预测为负例
#                     tn += 1
#                 else:             ### 正例预测为正例
#                     tp += 1
#             else:
#                 if label_true[i] == 0:  ### 负例 预测为正例
#                     fp += 1
#                 else:
#                     fn += 1         ### 正例预测为负例

        total_tp += tp
#         total_tn += tn
        total_fp += fp
        total_fn += fn
#         print("tp %d,tn %d,fp %d,fn %d" %(tp,tn,fp,fn))
        if tp == 0:
            continue
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*precision*recall/(precision+recall)
#         print("precision :%.4f,recall :%.4f,f1 :%.4f" %(precision,recall,f1))
    if total_tp != 0:
        batch_precision = total_tp/(total_tp+total_fp)
        batch_recall = total_tp/(total_tp+total_fn)
        batch_f1 = 2*batch_precision*batch_recall/(batch_precision+batch_recall)
    
#     print('*********************')
#     print('batch_precision:%.4f  batch_recall:%.4f  batch_f1: %.4f' %(batch_precision,batch_recall,batch_f1))  
    
    return batch_precision,batch_recall,batch_f1


# In[3]:


def re_evaluate(out,label_ids):
    
    out1 = torch.softmax(out,dim=1)
    y_pre = torch.argmax(out1,dim = 1)
    label_true = label_ids
    total_tp =0.0
    total_tn = 0.0
    total_fp = 0.0
    total_fn = 0.0
    for i in range(len(out)):
        
        if label_true[i] == y_pre[i]:
            if label_true[i] == 0: ### 负例预测为负例
                total_tn += 1
            else:             ### 正例预测为正例
                total_tp += 1
        else:
            if label_true[i] == 0:  ### 负例 预测为正例
                total_fp += 1
            else:
                total_fn += 1         ### 正例预测为负例

#         print("tp %d,tn %d,fp %d,fn %d" %(tp,tn,fp,fn))
#         if toal_tp == 0:
#             continue
#         precision = tp/(tp+fp)
#         recall = tp/(tp+fn)
#         f1 = 2*precision*recall/(precision+recall)
# #         print("precision :%.4f,recall :%.4f,f1 :%.4f" %(precision,recall,f1))
    if total_tp != 0:
        batch_precision = total_tp/(total_tp+total_fp)
        batch_recall = total_tp/(total_tp+total_fn)
        batch_f1 = 2*batch_precision*batch_recall/(batch_precision+batch_recall)
    else:
        batch_precision = 0.0
        batch_recall = 0.0
        batch_f1 = 0.0
        
    
#     print('*********************')
#     print('batch_precision:%.4f  batch_recall:%.4f  batch_f1: %.4f' %(batch_precision,batch_recall,batch_f1))  
    
    return batch_precision,batch_recall,batch_f1


# In[2]:


def pipe_evaluate(pre_label,true_labels):
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    
    e1 = (pre_label[1],pre_label[2])
    e2s = []
    for tl in true_labels:
        e2s.append((tl[1],tl[2]))
    
    if e1 in e2s:   #### 被预测实体有关系
        if pre_label in true_labels:
            tp += 1
        else:
            fn += 1   ###正例预测为负例
    else:
        if pre_label[0] == 'O':
            tn += 1
        else:
            fp += 1   ###负例预测为正例
            
    return [tp,tn,fp,fn]


# In[ ]:




