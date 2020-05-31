
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
folderpath =  r'C:\Users\ender\Desktop\IE413_Assignment'
filename ='OnlineRetailerDataset.csv'
#filepath = folderpath + '\\' filename
filepath= os.path.join(folderpath,filename)
filepath


# In[2]:


column_names = ['User_ID','Gender','Age','Marital_Status','Website_Activity','Browsed_Electronics_12Mo','Bought_Electronics_12Mo','Bought_Digital_Media_18Mo','Bought_Digital_Books','Payment_Method','eReader_Adoption'
]
df= pd.read_csv(filepath,header=0, names=column_names,index_col=['User_ID'])


# In[3]:


drop_add=df['Age']


# In[4]:


df=df.drop(columns='Age')


# In[5]:


df=df.apply(LabelEncoder().fit_transform)


# In[6]:


df['Age']=drop_add


# In[7]:


df = df[['Gender','Age','Marital_Status','Website_Activity','Browsed_Electronics_12Mo','Bought_Electronics_12Mo','Bought_Digital_Media_18Mo','Bought_Digital_Books','Payment_Method','eReader_Adoption'
]]


# In[8]:


X=df.drop(columns='eReader_Adoption')
y=df['eReader_Adoption']


# In[9]:


def evaluatemodels (numfold,classifiers):
    skf = StratifiedKFold(n_splits=numfold) #number of splits

    clf_performances1=[]
    clf_performances2=[]
    clf_performances3=[]
    
    for clf in classifiers:

        accuracy_results1=[]
        accuracy_results2=[]
        accuracy_results3=[]
        for train_index, test_index in skf.split(X, y):  #train and test indexes 4th
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                
                acc=accuracy_score(y_test,y_pred)
                accuracy_results1.append(acc)
                
                acc2=recall_score(y_test, y_pred,average=None)
                accuracy_results2.append(acc2)
                
                acc3=precision_score(y_test,y_pred,average=None)
                accuracy_results3.append(acc3)
                
                
        
        clf_performances1.append(np.average(accuracy_results1))
        clf_performances2.append(np.average(accuracy_results2))
        clf_performances3.append(np.average(accuracy_results3))
    
    return [clf_performances1,clf_performances2,clf_performances3]
    


# In[10]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier

clf_dt= DecisionTreeClassifier()
clf_logreg= LogisticRegression()
clf_mlp=MLPClassifier()
clf_nb=GaussianNB()
clf_sv=SVC()
clf_gaussian=GaussianProcessClassifier()



classifiers = [clf_dt, clf_logreg, clf_mlp,clf_nb,clf_sv,clf_gaussian]

k=10

clf_per=evaluatemodels(k,classifiers)


# In[11]:


esa=['1- Accuracy_score','2- Recall Score','3-Precision Score']
es=['Decision Tree','Logistics Regression','MLP','Gaussian Naive Bayes','SVC','GaussianProcessClassifier']


# In[12]:


for i in range(3):
    print("*********************")
    print(esa[i])
    print("**********************")
    for j in range(6):
        print(clf_per[i][j],es[j])

