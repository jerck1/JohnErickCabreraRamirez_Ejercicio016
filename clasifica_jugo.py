#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import f1_score
#from sklearn import cross_validation
#from sklearn.model_selection import cross_validate
import sklearn.tree


# In[30]:


# Carga datos
data = pd.read_csv('OJ.csv')

# Remueve datos que no se van a utilizar
data = data.drop(['Store7', 'PctDiscCH','PctDiscMM'],axis=1)

# Crea un nuevo array que sera el target, 0 si MM, 1 si CH
purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['Purchase']=='MM')
purchasebin[ii] = 0

data['Target'] = purchasebin

# Borra la columna Purchase
data = data.drop(['Purchase'],axis=1)

# Crea un dataframe con los predictores
predictors = list(data.keys())
predictors.remove('Target')
predictors.remove('Unnamed: 0')
#print(predictors)
#print(np.shape(data['Target']))
#print(data)


# In[31]:


train, test, y_train, y_test = train_test_split(data, data["Target"], train_size=0.5)


# In[50]:


def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample


# In[65]:


train_ran=bootstrap_resample(np.array(train))
print(np.shape(train))
print(np.shape(train_ran))


# In[86]:


F1_score=[]
Fea_imp=[]


# In[87]:


for i in range(1,11):
    clf = sklearn.tree.DecisionTreeClassifier(max_depth=i)
#    print(clf)
    clf.fit(train_ran,y_train)
#    plt.figure(figsize=(10,10))
#    _= sklearn.tree.plot_tree(clf)
    clf.predict(train_ran)
    F1_score=np.append(F1_score,sklearn.metrics.f1_score(y_train, clf.predict(X)))
#    print("f1: ",sklearn.metrics.f1_score(y_train, clf.predict(X)))
    Fea_imp=np.append(Fea_imp,clf.feature_importances_)
#    print("imp ",clf.feature_importances_)


# In[ ]:




