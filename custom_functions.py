import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (roc_auc_score,roc_curve,precision_recall_curve, auc, mean_squared_error,
                             classification_report, confusion_matrix , average_precision_score, accuracy_score,silhouette_score)
from inspect import signature

def print_classification_performance2class_report(model,X_test,y_test):
    """ 
        Program: print_classification_performance2class_report
        Author: Siraprapa W.
        
        Purpose: print standard 2-class classification metrics report
    """
    sns.set()
    start = datetime.datetime.now()
    y_pred = model.predict(X_test)
    end = datetime.datetime.now()
    y_pred_proba = model.predict_proba(X_test)[:,1]
    conf_mat = confusion_matrix(y_test,y_pred)
    TN = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    TP = conf_mat[1][1]
    PC =  TP/(TP+FP)
    RC = TP/(TP+FN)
    FS = 2 *((PC*RC)/(PC+RC))
    AP = average_precision_score(y_test,y_pred)
    ACC = accuracy_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Accuracy:{:.2%}".format(ACC))
    print("Precision:{:.2%}".format(PC))
    print("Recall:{:.2%}".format(RC))
    print("Fscore:{:.2%}".format(FS))
    print("Average precision:{:.2%}".format(AP))
    print("RMSE:{:.4}".format(rmse))
    
    fig = plt.figure(figsize=(20,3))
    fig.subplots_adjust(hspace=0.2,wspace=0.2)
    
    #heatmap
    plt.subplot(141)
    labels = np.asarray([['True Negative\n{}'.format(TN),'False Positive\n{}'.format(FP)],
                         ['False Negative\n{}'.format(FN),'True Positive\n{}'.format(TP)]])
    sns.heatmap(conf_mat,annot=labels,fmt="",cmap=plt.cm.Blues,xticklabels="",yticklabels="",cbar=False)
    
    #ROC
    plt.subplot(142)
    pfr, tpr, _ = roc_curve(y_test,y_pred_proba)
    roc_auc = auc(pfr, tpr)
    gini = (roc_auc*2)-1
    plt.plot(pfr, tpr, label='ROC Curve (area =  {:.2%})'.format(roc_auc) )
    plt.plot([0,1], [0,1])
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver Operating Charecteristic Curve with Gini {:.2}'.format(gini))
    plt.legend(loc='lower right')
    
    #pr
    plt.subplot(143)
    precision, recall, _ = precision_recall_curve(y_test,y_pred_proba)
    step_kwargs = ({'step':'post'}
                  if 'step'in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall,precision,color='b',alpha=0.2, where='post')
    plt.fill_between(recall,precision,alpha=0.2,color='b',**step_kwargs)
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('2-class Precision-Recall Curve: AP={:.2%}'.format(AP))
    
    plt.show()
    
    return ACC,PC,RC,FS,AP,rmse,roc_auc,gini,(end-start).total_seconds()*1000.0
    

class MetricContainer():
    
    def __init__(self):
        self.i=0
        self.list_model=[]
        self.list_desc=[]
        self.list_acc = []
        self.list_pc = []
        self.list_rc = []
        self.list_fs = []
        self.list_ap = []
        self.list_rmse = []
        self.list_roc_auc = []
        self.list_gini = []
        self.list_timed = []
    
    def add_metric(self,desc,acc,pc,rc,fs,ap,rmse,roc_auc,gini,timed):
        self.i += 1
        self.list_model.append('Model_{}'.format(self.i))        
        self.list_desc.append(desc)        
        self.list_acc.append(acc)
        self.list_pc.append(pc)
        self.list_rc.append(rc)
        self.list_fs.append(fs)
        self.list_ap.append(ap)
        self.list_rmse.append(rmse)
        self.list_roc_auc.append(roc_auc)
        self.list_gini.append(gini)
        self.list_timed.append(timed)

    def get_metric(self):
        compare = pd.DataFrame([self.list_model,self.list_desc,self.list_acc,self.list_pc,self.list_rc,self.list_fs,self.list_ap,self.list_rmse,self.list_roc_auc,self.list_gini,self.list_timed]).transpose()
        compare.columns=['model','description','accuracy','precision','recall','fscore','average_precision','rmse','auc','gini','timed']
        compare.set_index('model')

        return compare