#!/usr/bin/env python
# coding: utf-8

# # Dhaka City Rainfall Prediction

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[43]:


df = pd.read_csv("C:\\Users\\jubair pantho\\Desktop\\Dhaka Rainfall prediction\\dhaka rainfall.csv")
df
df.head()


# In[44]:


df.shape


# In[4]:


plt.scatter(df['avg_wind_speed'], df['rainfall'])


# In[5]:


plt.scatter(df['humidity'], df['rainfall'])


# In[6]:


plt.scatter(df['mean_max_temp'], df['rainfall'])


# In[7]:


plt.scatter(df['Mean_min_temp'], df['rainfall'])


# In[8]:


plt.xlabel('total_month')
plt.ylabel('Total Monthly average wind speed(knots)')

plt.bar(df.month_total,df.avg_wind_speed, color='purple',  width = 0.4, label = 'Avg wind speed')

plt.legend()


# In[9]:


plt.xlabel('total_month')
plt.ylabel('Total Monthly Rainfall(mm)')


plt.bar(df.month_total,df.rainfall, color='Green',  width = 0.4, label = 'Rainfall')

plt.legend()


# In[10]:


plt.xlabel('total month')
plt.ylabel('Monthly Average Humidity(%)')

plt.bar(df.month_total,df.humidity, color='Red',  width = 0.4, label = 'Humidity')

plt.legend()


# In[11]:


plt.xlabel('total month')
plt.ylabel('Mean Maximum Temperature')

plt.bar(df.month_total,df.mean_max_temp, color='Blue',  width = 0.4, label = 'Mean Maximum Temperature')

plt.legend()


# In[12]:


plt.xlabel('Total Month')
plt.ylabel('Mean Minimum Temperature')


plt.bar(df.month_total,df.Mean_min_temp, color='black',  width = 0.4, label = 'Mean Minimum Temperature')
plt.legend()


# ## Independent And Dependent Variables

# In[13]:


x = df[['humidity', 'mean_max_temp','Mean_min_temp','avg_wind_speed']]
y = df[['rainfall']]


# ## By Using Test Train split 

# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state = 10)


# ## Calling different ML models

# In[15]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()


# In[16]:


from sklearn.svm import SVC
svm = SVC()


# In[17]:


from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()


# In[18]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[19]:


clf.fit(x_train, y_train)


# In[20]:


clf.score(x_test,y_test)


# In[21]:


clf.predict(x_test)


# ## K Fold Cross Validation

# In[22]:


from sklearn.model_selection import cross_val_score


# In[23]:


cross_val_score(LinearRegression(), x, y)


# In[24]:


cross_val_score(SVC(), x, y)


# In[25]:


cross_val_score(RandomForestClassifier(n_estimators= 80), x, y)


# In[26]:


cross_val_score(LogisticRegression(), x, y)


# # K means Clustering for Humidity vs rainfall

# In[27]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# In[28]:


scaler = MinMaxScaler()
scaler.fit(df[['humidity']])
df[['humidity']] = scaler.transform(df[['humidity']])

scaler.fit(df[['rainfall']])
df[['rainfall']] = scaler.transform(df[['rainfall']])

df


# In[29]:


km = KMeans(n_clusters= 3)


# In[30]:


y_predicted = km.fit_predict(df[['humidity','rainfall']])
y_predicted


# In[31]:


df['cluster'] = y_predicted
df.head()


# In[32]:


km.cluster_centers_


# In[33]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.humidity, df1['rainfall'], color ='green')
plt.scatter(df2.humidity, df2['rainfall'], color ='red')
plt.scatter(df3.humidity, df3['rainfall'], color ='black')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='purple', marker='*', label='centroids')

plt.xlabel('humidity')
plt.ylabel('rainfall')
plt.legend()


# In[34]:


k_range = range(1,10)
sse = []
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df[['humidity', 'rainfall']])
    sse.append(km.inertia_)


# In[35]:


sse


# In[36]:


plt.xlabel('K')
plt.ylabel('Sume of squared error')


plt.plot(k_range,sse)


# # Hyper parameter Tuning

# In[37]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.5)


# In[38]:


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


# In[39]:


model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
    'naive_bayes_multinomial': {
        'model': MultinomialNB(),
        'params': {}
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
            
        }
    },
    'linear_regression': {
        'model': LinearRegression(),
        'params': {}
            
    }
}


# In[40]:


from sklearn.model_selection import GridSearchCV
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x, y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# ## End
