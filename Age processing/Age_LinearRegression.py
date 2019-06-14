import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")
data_address = pd.DataFrame(np.zeros((data.shape[0],6),dtype=int),columns=["Mr","Mrs","Miss","Master","Dr","Other"])
data = pd.concat([data,data_address],axis=1)
data.head()

data_MrAge = data[0:0]
data_MrsAge = data[0:0]
data_MissAge = data[0:0]
data_MasterAge = data[0:0]
data_DrAge = data[0:0]
data_OtherAge = data[0:0]

import math

def append_age(row,axis=0):    
    global data_MrAge
    global data_MrsAge
    global data_MissAge
    global data_MasterAge
    global data_DrAge
    global data_OtherAge
    
    i = int(row["PassengerId"])-1
    name = row["Name"]
    if "Mrs" in name:
            data_MrsAge = pd.concat([data_MrsAge,data[i:i+1]],axis=0)
            row["Mrs"] = 1.0
    elif "Mr" in name:
            data_MrAge = pd.concat([data_MrAge,data[i:i+1]],axis=0)
            row["Mr"] = 1.0
    elif "Miss" in name:
            data_MissAge = pd.concat([data_MissAge,data[i:i+1]],axis=0)
            row["Miss"] = 1.0
    elif "Master" in name:
            data_MasterAge = pd.concat([data_MasterAge,data[i:i+1]],axis=0)
            row["Master"] = 1.0
    elif "Dr" in name:
            data_DrAge = pd.concat([data_DrAge,data[i:i+1]],axis=0)
            row["Dr"] = 1.0
    else:
            data_OtherAge = pd.concat([data_OtherAge,data[i:i+1]],axis=0)
            row["Other"] = 1.0
    return row


# In[8]:


def sep_age(data):
    data = data.apply(append_age,axis=1)
    return data
expanded_data = sep_age(data)


# In[9]:


processed_data = expanded_data.drop(["Name","Ticket","Cabin","Fare"],axis=1)
processed_data = pd.get_dummies(processed_data)
processed_data.head()


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


data_withAge = processed_data[pd.isna(processed_data["Age"]) == False]
data_withoutAge = processed_data[pd.isna(processed_data["Age"])]


# In[12]:


LR = LinearRegression()
LR.fit(data_withAge[["Pclass","SibSp","Parch","Mr","Mrs","Miss","Master","Dr","Other","Sex_female","Sex_male","Embarked_C","Embarked_Q","Embarked_S"]],data_withAge["Age"])


# In[13]:


data_withoutAge["Age"] = LR.predict(data_withoutAge[["Pclass","SibSp","Parch","Mr","Mrs","Miss","Master","Dr","Other","Sex_female","Sex_male","Embarked_C","Embarked_Q","Embarked_S"]])


# In[14]:


def round_age(row):
    age = row["Age"]
    rounded_age = round(age)
    row["Age"] = rounded_age
    return row

data_withoutAge = data_withoutAge.apply(round_age,axis=1)


# In[15]:


data_ageFilled = pd.concat([data_withAge,data_withoutAge],axis=0)


# In[17]:


data["Age"] = data_ageFilled["Age"]


# In[19]:


data.to_csv("train_AgeFilled.csv")


# ### Fill the test file

# In[20]:


submit_data = pd.read_csv("test.csv")


# In[ ]:




