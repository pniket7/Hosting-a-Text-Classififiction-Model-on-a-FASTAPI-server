#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform, randint
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn


# In[2]:


app = FastAPI()


# In[3]:


data = pd.read_csv('/home/niket/Downloads/Normalized_Data.csv')


# In[4]:


data.head()


# In[5]:


X = data['Text'].values
y = data['Non_Acceptance'].values


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# In[8]:


lr_model = LogisticRegression()


# In[9]:


param_dist = {
    'penalty': ['l1', 'l2'],
    'C': loguniform(1e-4, 100),
    'fit_intercept': [True, False],
    'solver': ['liblinear', 'saga', 'lbfgs'],
    'max_iter': randint(100, 1000),
}


# In[10]:


random_search = RandomizedSearchCV(
    estimator=lr_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=10,
    n_jobs=-1,
    random_state=42
)


# In[11]:


random_search.fit(X_train_vectorized, y_train)


# In[12]:


print("Logistic Regression:")
print("Best Parameters:", random_search.best_params_)
print("Accuracy:", accuracy_score(y_test, random_search.predict(X_test_vectorized)))     


# In[13]:


joblib.dump(random_search, "model.joblib")


# In[14]:


class InputData(BaseModel):
    text: str


# In[15]:


model = joblib.load("model.joblib")


# In[16]:


@app.post("/predict")
def predict(input_data: InputData):
    text = [input_data.text]
    text_vectorized = vectorizer.transform(text)
    prediction = int(model.predict(text_vectorized)[0])
    return {"prediction": prediction}


# In[ ]:




