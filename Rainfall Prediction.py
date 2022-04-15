#!/usr/bin/env python
# coding: utf-8

# # Sylhet Rainfall Prediction

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[19]:


df = pd.read_csv("C:\\Users\\jubair pantho\\Desktop\\Dhaka Rainfall prediction\\shylhel rainfall.csv")
df
df.head()


# In[20]:


plt.scatter(df['avg_wind_speed'], df['rainfall'])


# In[21]:


plt.scatter(df['humidity'], df['rainfall'])


# In[22]:


plt.scatter(df['mean_max_temp'], df['rainfall'])


# In[23]:


plt.scatter(df['Mean_min_temp'], df['rainfall'])


# In[24]:


plt.xlabel('total_month')
plt.ylabel('Total Monthly average wind speed(knots)')

plt.bar(df.month_total,df.avg_wind_speed, color='purple',  width = 0.4, label = 'Avg wind speed')

plt.legend()


# In[25]:


plt.xlabel('total_month')
plt.ylabel('Total Monthly Rainfall(mm)')


plt.bar(df.month_total,df.rainfall, color='Green',  width = 0.4, label = 'Rainfall')

plt.legend()


# In[26]:


plt.xlabel('total month')
plt.ylabel('Monthly Average Humidity(%)')

plt.bar(df.month_total,df.humidity, color='Red',  width = 0.4, label = 'Humidity')

plt.legend()


# In[27]:


plt.xlabel('total month')
plt.ylabel('Mean Maximum Temperature')

plt.bar(df.month_total,df.mean_max_temp, color='Blue',  width = 0.4, label = 'Mean Maximum Temperature')

plt.legend()


# In[28]:


plt.xlabel('Total Month')
plt.ylabel('Mean Minimum Temperature')


plt.bar(df.month_total,df.Mean_min_temp, color='black',  width = 0.4, label = 'Mean Minimum Temperature')
plt.legend()


# In[29]:


x = df[['humidity', 'mean_max_temp','Mean_min_temp','avg_wind_speed']]
y = df[['rainfall']]


# In[30]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# In[31]:


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


# In[32]:


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


# In[33]:


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


# # Khulna Rainfall Prediction
# 

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[35]:


df = pd.read_csv("C:\\Users\\jubair pantho\\Desktop\\Dhaka Rainfall prediction\\khulna rainfall.csv")
df
df.head()


# In[36]:


plt.scatter(df['avg_wind_speed'], df['rainfall'])


# In[37]:


plt.scatter(df['humidity'], df['rainfall'])


# In[38]:


plt.scatter(df['mean_max_temp'], df['rainfall'])


# In[39]:


plt.scatter(df['Mean_min_temp'], df['rainfall'])


# In[40]:


plt.xlabel('total_month')
plt.ylabel('Total Monthly average wind speed(knots)')

plt.bar(df.month_total,df.avg_wind_speed, color='purple',  width = 0.4, label = 'Avg wind speed')

plt.legend()


# In[41]:


plt.xlabel('total_month')
plt.ylabel('Total Monthly Rainfall(mm)')


plt.bar(df.month_total,df.rainfall, color='Green',  width = 0.4, label = 'Rainfall')

plt.legend()


# In[42]:


plt.xlabel('total month')
plt.ylabel('Monthly Average Humidity(%)')

plt.bar(df.month_total,df.humidity, color='Red',  width = 0.4, label = 'Humidity')

plt.legend()


# In[43]:


plt.xlabel('total month')
plt.ylabel('Mean Maximum Temperature')

plt.bar(df.month_total,df.mean_max_temp, color='Blue',  width = 0.4, label = 'Mean Maximum Temperature')

plt.legend()


# In[44]:


plt.xlabel('Total Month')
plt.ylabel('Mean Minimum Temperature')


plt.bar(df.month_total,df.Mean_min_temp, color='black',  width = 0.4, label = 'Mean Minimum Temperature')
plt.legend()


# In[45]:


x = df[['humidity', 'mean_max_temp','Mean_min_temp','avg_wind_speed']]
y = df[['rainfall']]


# In[46]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# In[47]:


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


# In[48]:


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


# In[49]:


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


# # Dinajpur Rainfall Prediction

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[51]:


df = pd.read_csv("C:\\Users\\jubair pantho\\Desktop\\Dhaka Rainfall prediction\\dinajpur rainfall.csv")
df
df.head()


# In[52]:


plt.scatter(df['avg_wind_speed'], df['rainfall'])


# In[53]:


plt.scatter(df['humidity'], df['rainfall'])


# In[54]:


plt.scatter(df['mean_max_temp'], df['rainfall'])


# In[55]:


plt.scatter(df['Mean_min_temp'], df['rainfall'])


# In[56]:


plt.xlabel('total_month')
plt.ylabel('Total Monthly average wind speed(knots)')

plt.bar(df.month_total,df.avg_wind_speed, color='purple',  width = 0.4, label = 'Avg wind speed')

plt.legend()


# In[57]:


plt.xlabel('total_month')
plt.ylabel('Total Monthly Rainfall(mm)')


plt.bar(df.month_total,df.rainfall, color='Green',  width = 0.4, label = 'Rainfall')

plt.legend()


# In[58]:


plt.xlabel('total month')
plt.ylabel('Monthly Average Humidity(%)')

plt.bar(df.month_total,df.humidity, color='Red',  width = 0.4, label = 'Humidity')

plt.legend()


# In[59]:


plt.xlabel('total month')
plt.ylabel('Mean Maximum Temperature')

plt.bar(df.month_total,df.mean_max_temp, color='Blue',  width = 0.4, label = 'Mean Maximum Temperature')

plt.legend()


# In[60]:


plt.xlabel('Total Month')
plt.ylabel('Mean Minimum Temperature')


plt.bar(df.month_total,df.Mean_min_temp, color='black',  width = 0.4, label = 'Mean Minimum Temperature')
plt.legend()


# In[61]:


x = df[['humidity', 'mean_max_temp','Mean_min_temp','avg_wind_speed']]
y = df[['rainfall']]


# In[62]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# In[63]:


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


# In[64]:


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


# In[65]:


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


# # Chittagong Rainfall

# In[66]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[67]:


df = pd.read_csv("C:\\Users\\jubair pantho\\Desktop\\Dhaka Rainfall prediction\\chittagong rainfall.csv")
df
df.head()


# In[68]:


plt.scatter(df['avg_wind_speed'], df['rainfall'])


# In[69]:


plt.scatter(df['humidity'], df['rainfall'])


# In[70]:


plt.scatter(df['mean_max_temp'], df['rainfall'])


# In[71]:


plt.scatter(df['Mean_min_temp'], df['rainfall'])


# In[72]:


plt.xlabel('total_month')
plt.ylabel('Total Monthly average wind speed(knots)')

plt.bar(df.month_total,df.avg_wind_speed, color='purple',  width = 0.4, label = 'Avg wind speed')

plt.legend()


# In[73]:


plt.xlabel('total_month')
plt.ylabel('Total Monthly Rainfall(mm)')


plt.bar(df.month_total,df.rainfall, color='Green',  width = 0.4, label = 'Rainfall')

plt.legend()


# In[74]:


plt.xlabel('total month')
plt.ylabel('Monthly Average Humidity(%)')

plt.bar(df.month_total,df.humidity, color='Red',  width = 0.4, label = 'Humidity')

plt.legend()


# In[75]:


plt.xlabel('total month')
plt.ylabel('Mean Maximum Temperature')

plt.bar(df.month_total,df.mean_max_temp, color='Blue',  width = 0.4, label = 'Mean Maximum Temperature')

plt.legend()


# In[76]:


plt.xlabel('Total Month')
plt.ylabel('Mean Minimum Temperature')


plt.bar(df.month_total,df.Mean_min_temp, color='black',  width = 0.4, label = 'Mean Minimum Temperature')
plt.legend()


# In[77]:


x = df[['humidity', 'mean_max_temp','Mean_min_temp','avg_wind_speed']]
y = df[['rainfall']]


# In[78]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# In[79]:


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


# In[80]:


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


# In[81]:


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


# In[ ]:




