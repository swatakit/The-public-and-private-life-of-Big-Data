import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve, auc
from sklearn.metrics import classification_report, confusion_matrix , average_precision_score, accuracy_score,silhouette_score
from sklearn.utils.fixes import signature

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
    print("Accuracy:{:.2%}".format(ACC))
    print("Precision:{:.2%}".format(PC))
    print("Recall:{:.2%}".format(RC))
    print("Fscore:{:.2%}".format(FS))
    print("Average precision:{:.2%}".format(AP))
    
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
    
    #hist
    plt.subplot(144)
    tmp = pd.DataFrame(data=[y_test,y_pred_proba]).transpose()
    tmp.columns=['class','proba']
    mask_c0 = tmp['class']==0
    mask_c1 = tmp['class']==1
    plt.hist(tmp.loc[mask_c0,'proba'].dropna(),density=True,alpha=0.5,label='0',bins=20)
    plt.hist(tmp.loc[mask_c1,'proba'].dropna(),density=True,alpha=0.5,label='1',bins=20)
    plt.ylabel('Density')
    plt.xlabel('Probability')
    plt.title('2-class Distribution' )
    plt.legend(loc='upper right')
    
    plt.show()
    
    return ACC,PC,RC,FS,AP,roc_auc,gini,end-start
    

def print_kmeans_clusters_report(model,X_test):
    """ 
        Program: print_kmeans_clusters_report
        Author: Siraprapa W.
        
        Purpose:Print KMeans distplot and histogram, All and by Cluster.
                Assume that input X is pre-processed and ready to be predicted by the input model.
                Outliers are tagged at p95th,p97th,p99th
    """
    sns.set()
    print("Model informations:\n")
    print("\t algorithm=",model.algorithm)
    print("\t init:",model.init)
    print("\t random_state:",model.random_state)
    print("\t n_clusters:",model.n_clusters)
    
    start = datetime.datetime.now()
    X_test_clusters = model.predict(X_test)
    end = datetime.datetime.now()
    
    X_test_clusters_centers = model.cluster_centers_

    #Calculate dist from each centroid
    dist = [np.linalg.norm(x-y) for x, y in zip(X_test.values,X_test_clusters_centers[X_test_clusters])]
    km_y_pred = pd.DataFrame([pd.Series(X_test_clusters),pd.Series(dist)]).transpose()
    km_y_pred.columns = ['cluster','dist_cls_cent']
    p95=km_y_pred.groupby('cluster').agg(lambda x: np.percentile(x['dist_cls_cent'],95))
    p95.columns=['p95']
    p97=km_y_pred.groupby('cluster').agg(lambda x: np.percentile(x['dist_cls_cent'],97))
    p97.columns=['p97']
    p99=km_y_pred.groupby('cluster').agg(lambda x: np.percentile(x['dist_cls_cent'],99))
    p99.columns=['p99']
    ptats = pd.concat([p95,p97,p99],axis=1)
    km_y_pred_pstats = pd.merge(km_y_pred,ptats,how='left',left_on=['cluster'],right_on=['cluster'])

    #tag EV
    km_y_pred['p95th'] = np.int64(km_y_pred_pstats['dist_cls_cent']>=km_y_pred_pstats['p95'])
    km_y_pred['p97th'] = np.int64(km_y_pred_pstats['dist_cls_cent']>=km_y_pred_pstats['p97'])
    km_y_pred['p99th'] = np.int64(km_y_pred_pstats['dist_cls_cent']>=km_y_pred_pstats['p99'])
    
    mask_p95 = (km_y_pred['p95th']==1)
    mask_p97 = (km_y_pred['p97th']==1)
    mask_p99 = (km_y_pred['p99th']==1)
    mask_none = ~((mask_p95) & (mask_p97) & (mask_p99))
    
    fig = plt.figure(figsize=(9,3))
    fig.subplots_adjust(hspace=0.2,wspace=0.2)
    plt.subplot(1,2,1)
    sns.distplot(km_y_pred.loc[mask_none,'dist_cls_cent'].dropna() ,hist=False,label='None')
    sns.distplot(km_y_pred.loc[mask_p95,'dist_cls_cent'].dropna() ,hist=False,label='p95th')
    sns.distplot(km_y_pred.loc[mask_p97,'dist_cls_cent'].dropna() ,hist=False,label='p97th')
    sns.distplot(km_y_pred.loc[mask_p99,'dist_cls_cent'].dropna() ,hist=False,label='p99th')
    plt.ylabel('Density')
    plt.xlabel('Distant from Centroid')
    plt.title('Cluster = *All*')
    plt.legend(loc='upper right')
    
    plt.subplot(1,2,2)
    plt.hist(km_y_pred.loc[mask_none,'dist_cls_cent'].dropna() ,density=True,alpha=0.5,
             label='None - {}'.format(np.count_nonzero(mask_none)))
    plt.hist(km_y_pred.loc[mask_p95,'dist_cls_cent'].dropna() ,density=True,alpha=0.5,
             label='p95th - {}'.format(np.count_nonzero(mask_p95)))
    plt.hist(km_y_pred.loc[mask_p97,'dist_cls_cent'].dropna() ,density=True,alpha=0.5,
             label='p97th - {}'.format(np.count_nonzero(mask_p97)))
    plt.hist(km_y_pred.loc[mask_p99,'dist_cls_cent'].dropna() ,density=True,alpha=0.5,
             label='p99th - {}'.format(np.count_nonzero(mask_p99)))
    plt.ylabel('Percentage')
    plt.xlabel('Frequency Distribution')
    plt.title('Cluster = *All*')
    plt.legend(loc='upper right')
    
    plt.show()
    
    n_clusters = model.n_clusters
   
    list_n_clusters = []
    list_n_clusters_none = []
    list_n_clusters_p95th = []
    list_n_clusters_p97th = []
    list_n_clusters_p99th = []

    for i in range(n_clusters):
        fig = plt.figure(figsize=(n_clusters*10,3))
        fig.subplots_adjust(hspace=0.2,wspace=0.2)
        mask = km_y_pred['cluster']==i
        toplot = km_y_pred.loc[mask]
        mask_p95 = (toplot['p95th']==1)
        mask_p97 = (toplot['p97th']==1)
        mask_p99 = (toplot['p99th']==1)
        mask_none = ~((mask_p95) & (mask_p97) & (mask_p99))

        list_n_clusters.append(i)
        list_n_clusters_none.append(np.count_nonzero(mask_none))
        list_n_clusters_p95th.append(np.count_nonzero(mask_p95))
        list_n_clusters_p97th.append(np.count_nonzero(mask_p97))
        list_n_clusters_p99th.append(np.count_nonzero(mask_p99))
        
        plt.subplot(1,n_clusters*2,i+1)
        sns.distplot(toplot.loc[mask_none,'dist_cls_cent'].dropna() ,hist=False,label='None')
        sns.distplot(toplot.loc[mask_p95,'dist_cls_cent'].dropna() ,hist=False,label='p95th')
        sns.distplot(toplot.loc[mask_p97,'dist_cls_cent'].dropna() ,hist=False,label='p97th')
        sns.distplot(toplot.loc[mask_p99,'dist_cls_cent'].dropna() ,hist=False,label='p99th')
        plt.ylabel('Density')
        plt.xlabel('Distant from Centroid')
        plt.title('Cluster = {}'.format(i))
        plt.legend(loc='upper right')
        
        plt.subplot(1,n_clusters*2,i+2)
        plt.hist(toplot.loc[mask_none,'dist_cls_cent'].dropna() ,density=True,alpha=0.5,
                 label='None - {}'.format(np.count_nonzero(mask_none)))
        plt.hist(toplot.loc[mask_p95,'dist_cls_cent'].dropna() ,density=True,alpha=0.5,
                 label='p95th - {}'.format(np.count_nonzero(mask_p95)))
        plt.hist(toplot.loc[mask_p97,'dist_cls_cent'].dropna() ,density=True,alpha=0.5,
                 label='p97th - {}'.format(np.count_nonzero(mask_p97)))
        plt.hist(toplot.loc[mask_p99,'dist_cls_cent'].dropna() ,density=True,alpha=0.5,
                 label='p99th - {}'.format(np.count_nonzero(mask_p99)))
        plt.ylabel('Percentage')
        plt.xlabel('Frequency Distribution')
        plt.title('Cluster = {}'.format(i))
        plt.legend(loc='upper right')
        
        plt.show()
        
    all_clusters = pd.DataFrame([list_n_clusters,
                                 list_n_clusters_none,
                                 list_n_clusters_p95th,
                                 list_n_clusters_p97th,
                                 list_n_clusters_p99th]).transpose()
    
    all_clusters.columns=['cluster','None','p95th','p97th','p99th']    
    print(all_clusters)
    
    return all_clusters,end-start


def print_elbows(X_train,n):
    """
        Program: print_elbows
        Author: Siraprapa W.
        
        Purpose: plot elbows by input xtrain and n with Fast KMeans
    """
    clustno = range(1,n)
    score = []
    sse = []
    for i in clustno:
        try:
            km = MiniBatchKMeans(n_clusters=i)
            km.fit(X_train)
            score.append(km.score(X_train))
            sse.append(km.inertia_)        
        except:
            pass

    #PLotting
    fig = plt.figure(figsize=(9,3))
    fig.subplots_adjust(hspace=0.2,wspace=0.2)

    plt.subplot(1,2,1)
    plt.plot(clustno,score,'-o')
    plt.ylabel('Score')
    plt.xlabel('Number of Clusters')
    plt.title('Elbow Curve')

    plt.subplot(1,2,2)
    plt.plot(clustno,sse,'-o')
    plt.ylabel('Sum squared distances')
    plt.xlabel('Number of Clusters')
    plt.title('Elbow Curve')

    plt.show()
    
    return


def print_dbscan_report(model,X_train,first_n_small=10):
    """
        Program: print_dbscan_report
        Author: Siraprapa W.
        
        Purpose: print and plot output from dbscan
    """
    core_sample_mask = np.zeros_like(model.labels_,dtype=bool)
    core_sample_mask[model.core_sample_indices_] = True
    labels = model.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0 )
    n_outliers_ = list(labels).count(-1)
    silhouette = silhouette_score(X_train,labels,sample_size=5000) #to avoid memory error

    print("Estimated number of clusters: {}".format(n_clusters_))
    print("Estimated number of outliers: {}".format(n_outliers_))
    print("Silhouette: {:.2%}".format(silhouette))

    """ not informative, commment out for now
    
    counts = np.bincount(labels[labels>=0])
    smallest_clusters = np.argsort(counts)[:first_n_small]
    print("The first {} smallest clusters:".format(first_n_small))
    print(smallest_clusters)
    print("Theirs counts are:")
    print(counts[smallest_clusters])
    

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0,1,len(unique_labels))]
    
    for k,col in zip(unique_labels,colors):
        if k==-1:
            col = [0,0,0,1]

        class_member_mask = (labels == k)

        xy = X_train[class_member_mask & core_sample_mask]
        plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor=tuple(col),markeredgecolor='k',markersize=14)
        xy = X_train[class_member_mask & ~core_sample_mask]
        plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor=tuple(col),markeredgecolor='k',markersize=6)

    plt.show()
    
    """
    
    return n_clusters_, n_outliers_,silhouette


def print_isolation_forest_report(model,X_test,y_test):
    """ 
        Program: print_isolation_forest_report
        Author: Siraprapa W.
        
        Purpose: print standard 2-class classification metrics report, note that for iso, outlier is regards as abnormal
    """
    sns.set()
    start = datetime.datetime.now()
    y_pred = model.predict(X_test)
    end = datetime.datetime.now()
    y_pred = [1 if x ==-1 else 0 for x in y_pred]
    #y_pred_proba = model.predict_proba(X_test)[:,1]
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
    print("Accuracy:{:.2%}".format(ACC))
    print("Precision:{:.2%}".format(PC))
    print("Recall:{:.2%}".format(RC))
    print("Fscore:{:.2%}".format(FS))
    print("Average precision:{:.2%}".format(AP))
    
    fig = plt.figure(figsize=(8,3))
    fig.subplots_adjust(hspace=0.2,wspace=0.2)
    
    #heatmap
    plt.subplot(121)
    labels = np.asarray([['True Negative\n{}'.format(TN),'False Positive\n{}'.format(FP)],
                         ['False Positive\n{}'.format(FN),'False Positive\n{}'.format(TP)]])
    sns.heatmap(conf_mat,annot=labels,fmt="",cmap=plt.cm.Blues,xticklabels="",yticklabels="",cbar=False)
    
    #ROC
    plt.subplot(122)
    pfr, tpr, _ = roc_curve(y_test,y_pred)
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
    
    """ To be fix for iso later
    
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
    
    #hist
    plt.subplot(144)
    tmp = pd.DataFrame(data=[y_test,y_pred_proba]).transpose()
    tmp.columns=['class','proba']
    mask_c0 = tmp['class']==0
    mask_c1 = tmp['class']==1
    plt.hist(tmp.loc[mask_c0,'proba'].dropna(),density=True,alpha=0.5,label='0',bins=20)
    plt.hist(tmp.loc[mask_c1,'proba'].dropna(),density=True,alpha=0.5,label='1',bins=20)
    plt.ylabel('Density')
    plt.xlabel('Probability')
    plt.title('2-class Distribution' )
    plt.legend(loc='upper right')
    
    """
    plt.show()
    
    return ACC,PC,RC,FS ,AP ,roc_auc, gini,end-start
    
    
def print_isolation_forest_trees_report(model,X_trains):
    """
        Program: print_isolation_forest_trees_report
        Author: Siraprapa W.
        
        Purpose:Print number of trees and identified outliers
    """
    sns.set()
    n_estimators_ = len(model.estimators_)
    start = datetime.datetime.now()
    anomaly_score = model.decision_function(X_trains)
    end = datetime.datetime.now()
    
    predicted =  model.predict(X_trains)
    outliers = np.sum(predicted<0)
    print("Model informations:\n")
    print("\t number of trees: ",n_estimators_)
    print("\t number of outliers:",outliers)
    print("\t random_state:",model.random_state)
    fig = plt.figure(figsize=(9,3))
    plt.hist(anomaly_score)
    
    return n_estimators_,outliers,end-start

