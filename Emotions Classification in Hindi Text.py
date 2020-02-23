#!/usr/bin/env python
# coding: utf-8

# ### Importing Important Libraries and Data Set

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import os


# In[2]:


text = []
cat = []


# In[3]:


angry=[]
base_path = "C:/Users/Saurav/Desktop/Python Practise/emotions/angry"
for i in range(130):
    filename = str(i)+".txt"
    path_to_file = os.path.join(base_path, filename)
    fd = pd.read_csv(path_to_file , 'r')
    angry.append(list(fd.columns))    
    
    
for item in angry:
    text.append(item[0]) 
    cat.append(0)


# In[4]:


happy=[]
base_path = "C:/Users/Saurav/Desktop/Python Practise/emotions/happy"
for i in range(151):
    filename = str(i)+".txt"
    path_to_file = os.path.join(base_path, filename)
    fd = pd.read_csv(path_to_file , 'r')
    happy.append(list(fd.columns))    

for item in happy:
    text.append(item[0]) 
    cat.append(1)    


# In[5]:


neutral=[]
base_path = "C:/Users/Saurav/Desktop/Python Practise/emotions/neutral"
for i in range(128):
    filename = str(i)+".txt"
    path_to_file = os.path.join(base_path, filename)
    fd = pd.read_csv(path_to_file , 'r')
    neutral.append(list(fd.columns))    

for item in neutral:
    text.append(item[0]) 
    cat.append(2)        


# In[6]:


sad=[]
base_path = "C:/Users/Saurav/Desktop/Python Practise/emotions/sad"
for i in range(104):
    filename = str(i)+".txt"
    path_to_file = os.path.join(base_path, filename)
    fd = pd.read_csv(path_to_file , 'r')
    sad.append(list(fd.columns))    

for item in sad:
    text.append(item[0]) 
    cat.append(3)    


# In[7]:


print(len(text))
print(len(cat))


# In[8]:


data = pd.DataFrame(data=[text, cat])
data = data.T
data.rename(columns={0:'Text', 1:'Class'}, inplace=True)

data['Class'] = data['Class'].astype(int)


# In[9]:


data.head()


# In[10]:


data.isnull().sum()


# In[11]:


data.dtypes


# In[12]:


data.head()


# In[13]:


data.groupby('Class').Class.count()


# In[14]:


X = data['Text']
y = data['Class']


# ### Train Test Split

# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=48)


# ### Count Vectorizer (We can also use Tf-idf vectorizer)

# In[16]:


stop_words = ['',' ',' ','!','! ','!  ','! !','! ! ','! ! !','?','ही','तुमसे','बार','आप','तुम्हारे','तु','रहा','कुछ','कभी','एक','तुम','होता','नहीं','कितनी','पर','तू','हो','है','क्यों','एप','कर','काम','रहे','बातें','लग','आता','ये चैनल्स','करनी','अपना','पैक्स','चीज़','क्या','अरे ये','करा','मैं']
def my_tokenizer(s):
    return s.split(' ')

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=2, ngram_range=(1, 3), encoding='ISCII',tokenizer=my_tokenizer,stop_words=stop_words).fit(X_train)


# In[17]:


print(len(vect.get_feature_names()))         # Printing length of Vocabulary


# In[18]:


vect.get_feature_names()             ## Printing Vocabulary


# In[19]:


X_train_vectorized = vect.transform(X_train)    # Getting Bag of words representation for all the documents
X_train_vectorized


# In[20]:


X_train_vectorized.shape


# ### Logistic Regression Model (It works nice for sparse Matrix)

# In[21]:


from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(C=0.05, max_iter=10000, solver='newton-cg', multi_class='multinomial')
model1.fit(X_train_vectorized, y_train)


# In[22]:


from sklearn.metrics import accuracy_score
X_test_transformed = vect.transform(X_test)
y_pred_train = model1.predict(X_train_vectorized)
y_pred_test = model1.predict(X_test_transformed)
print('Train accuracy = ', accuracy_score(y_train, y_pred_train))
print('Test accuracy = ', accuracy_score(y_test, y_pred_test))


# ### Looking into 50 top and bottom learned features

# In[23]:


feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model1.coef_[0].argsort()

print('Largest Coeff')
print(feature_names[sorted_coef_index[:-50:-1]])

print('Smallest Coeff')
print(feature_names[sorted_coef_index[:50]])


# In[24]:


vect.get_stop_words


# ### Training on the whole Data Set and 10 fold Cross Validation Score

# In[25]:


from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=2, ngram_range=(1, 3), encoding='ISCII',tokenizer=my_tokenizer,stop_words=stop_words).fit(X)

X_vectorized = vect.transform(X)    # Getting Bag of words representation for all the documents
X_vectorized


# In[26]:


from sklearn.linear_model import LogisticRegression
model2 = LogisticRegression(C=0.085, max_iter=10000, solver='newton-cg', multi_class='multinomial')

c=cross_val_score(model2, X_vectorized, y, cv=10)
count = 1
for item in c:
    print('cross validation score '+str(count)+' =', item)
    count=count+1


# In[29]:


print('Final cross validation score = ', np.mean(c))


# In[ ]:




