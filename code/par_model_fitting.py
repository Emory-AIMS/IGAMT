
# coding: utf-8

# In[1]:


import sdv
import pandas as pd
import os

# In[2]:


raw_data = pd.read_pickle("./data_preprocessing/raw_data.pkl")


# In[3]:


raw_data.shape


# In[4]:


raw_data = raw_data.fillna(0)

raw_data = raw_data.loc[:80000]

# In[5]:


rae_data = raw_data.sort_values(by=['ID', 'DATEINYEARS'])


# In[ ]:


entity_columns = ["ID"]
context_columns = ["Male", "White", "Black", "OtherRace"] # structural feature
#sequence_index = 'DATEINYEARS'


# In[ ]:





# In[ ]:


# https://sdv.dev/SDV/user_guides/timeseries/par.html
from sdv.timeseries import PAR

model = PAR(
        entity_columns=entity_columns,
        context_columns=context_columns,
        #sequence_index=sequence_index
    )
model.fit(raw_data)


# In[ ]:

if not os.path.exists("./models/par/"):
    os.mkdir("./models/par/")
model.save('./models/par/par_model.pkl')

if not os.path.exists("./results/par/"):
    os.mkdir("./results/par/")
new_data = model.sample(1000)
new_data.to_csv("./results/par/par_generated_patients.csv", index=False)
