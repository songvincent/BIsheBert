#!/usr/bin/env python
# coding: utf-8

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
    


# In[ ]:




