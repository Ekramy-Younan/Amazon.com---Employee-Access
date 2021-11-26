#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv(r"train.csv")


# In[3]:


df.head()


# # Column Name	Description
# ACTION	ACTION is 1 if the resource was approved, 0 if the resource was not
# RESOURCE	An ID for each resource
# MGR_ID	The EMPLOYEE ID of the manager of the current EMPLOYEE ID record; an employee may have only one manager at a time
# ROLE_ROLLUP_1	Company role grouping category id 1 (e.g. US Engineering)
# ROLE_ROLLUP_2	Company role grouping category id 2 (e.g. US Retail)
# ROLE_DEPTNAME	Company role department description (e.g. Retail)
# ROLE_TITLE	Company role business title description (e.g. Senior Engineering Retail Manager)
# ROLE_FAMILY_DESC	Company role family extended description (e.g. Retail Manager, Software Engineering)
# ROLE_FAMILY	Company role family description (e.g. Retail Manager)
# ROLE_CODE	Company role code; this code is unique to each role (e.g. Manager)

# In[4]:


df.info()


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


allcolumns=df.columns
for item in allcolumns:
    print(df[item].nunique())


# In[7]:


correl=df.corr()


# In[8]:


correl


# In[9]:


sns.countplot(df["ACTION"])


# In[10]:


df.ACTION.nunique()


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


x=df.drop("ACTION",axis=1)


# In[13]:


y=df["ACTION"]


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=101)


# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[16]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[17]:


model = LogisticRegression()
model.fit(x_train, y_train)
predictedvalues=model.predict(x_test)
print(accuracy_score(y_test,predictedvalues))
print(confusion_matrix(y_test, predictedvalues))


# # Logistic Regression give accuracy of 93.9% but if we look into its confusion matrix, then we can reach to the conclusion that its not good model as it predicts everything as class 1 and has not predicted any itema as class 0, so it is affected by the biasness of model

# In[18]:


#Lets try random forest classifier.
model = RandomForestClassifier()
model.fit(x_train, y_train)
predictedvalues=model.predict(x_test)
print(accuracy_score(y_test,predictedvalues))
print(confusion_matrix(y_test, predictedvalues))


# # even though improvement in accuracy from logistic regression to random forest is little but here we can see that confusion metrics shows that it has also classfied well better the the class which is having less number of examples in dataset which makes it really good method

# In[19]:


model = AdaBoostClassifier()
model.fit(x_train, y_train)
predictedvalues=model.predict(x_test)
print(accuracy_score(y_test,predictedvalues))
print(confusion_matrix(y_test, predictedvalues))


# # Here also problem is same as in logistic regression that it can't work well with biased class

# In[20]:


model = GradientBoostingClassifier()
model.fit(x_train, y_train)
predictedvalues=model.predict(x_test)
print(accuracy_score(y_test,predictedvalues))
print(confusion_matrix(y_test, predictedvalues))


# # This is also not giving good result compare to random forest as here also it is able to predict corectly only 7 data points of class 0 whose elements are so less.

# # Lets execute random forest on test data as we have choosen random forest as the final model

# In[21]:


model = RandomForestClassifier()
model.fit(x_train, y_train)
predictedvalues=model.predict(x_test)
print(accuracy_score(y_test,predictedvalues))
print(confusion_matrix(y_test, predictedvalues))


# In[24]:


test_data = pd.read_csv(r"test.csv")
print (x_train.shape)
print (test_data.shape)
print (test_data.columns)
test_data.drop("id",axis=1, inplace=True)
predictedoutput = model.predict(test_data)
print (predictedoutput)


# In[ ]:




