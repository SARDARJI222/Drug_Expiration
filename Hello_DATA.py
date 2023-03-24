#!/usr/bin/env python
# coding: utf-8

# # Importing the required libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans


# # Load the dataset

# In[2]:


data = pd.read_csv('medicines.csv')


# In[3]:


data= data[['Drug Name', 'Drug Dosage Expiry Date']]


# # Preprocess the dataset

# In[4]:


dates = pd.to_datetime(data['Drug Dosage Expiry Date'])
data['expiration_month'] = dates.dt.month
data['expiration_year'] = dates.dt.year


# In[5]:


data.shape


# In[6]:


data.head()


# # REMOVING NON-USEFUL DATA

# In[7]:


data = data[data['expiration_year'] == 2018]


# In[8]:


data.shape


# In[9]:


data.head()


# # CHECKING FOR ANY NULL VALUES 

# In[10]:


data['Drug Dosage Expiry Date'].isna().sum()


# In[11]:


data['expiration_month'].isna().sum()


# In[12]:


data['expiration_year'].isna().sum()


# # CHECKING UNIQUE VALUES FOR MONTH AND YEAR

# In[13]:


unique_years = data['expiration_year'].unique()

# Print the unique values
print(unique_years)


# In[14]:


unique_month = data['expiration_month'].unique()

# Print the unique values
print(unique_month)


# In[15]:


# Get the unique values in the expiration_year column in descending order
unique_month_desc = data['expiration_month'].sort_values(ascending=False).unique()

# Print the unique values in descending order
print(unique_month_desc)


# # APPLYING ONE HOT ENCODING

# In[16]:


# Use the pandas get_dummies() function to one-hot encode the "expiration_month" column
one_hot = pd.get_dummies(data['expiration_month'])


# In[17]:


# Rename the columns to be more descriptive
one_hot.columns = ['month_'+str(i) for i in range(1,13)]


# In[18]:


# Concatenate the one-hot encoded data back onto the original dataframe
data = pd.concat([data, one_hot], axis=1)


# In[19]:


data.head()


# In[20]:


from sklearn.preprocessing import LabelEncoder


# In[21]:


le = LabelEncoder()


# In[22]:


le.fit(data['Drug Name'])


# # TRANSFORMING THE DATASET

# In[23]:


data['Drug_Id'] = le.transform(data['Drug Name'])


# In[24]:


data.head()


# In[25]:


data = data.drop('Drug Name', axis=1)


# In[26]:


data = data.drop('Drug Dosage Expiry Date', axis=1)


# In[27]:


data


# # APPLYING CLUSTERING

# In[32]:


kmeans = KMeans(n_clusters=12)
kmeans.fit(data)


# In[33]:


labels = kmeans.labels_


# In[34]:


labels


# # PLOTTING 

# In[35]:


# define colors for each cluster
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'olive', 'cyan', 'magenta']

# assign colors to each point based on cluster labels
point_colors = [colors[label] for label in labels]

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(data['expiration_month'], data['expiration_year'], c=point_colors)
ax.set_xlabel('Month')
ax.set_ylabel('Year')
ax.set_title('Month-wise Clustering of Medicines by Expiration Date')
ax.set_ylim(2017, 2019) # set y-axis limits
plt.show()


# In[36]:


data = data.reset_index(drop=True)


# In[37]:


count = data['Drug_Id'].nunique()
print(count)


# In[38]:


data


# In[28]:


groups = data.groupby('expiration_month')


# In[29]:


groups


# In[30]:


data


# In[31]:


# Group the medicines based on expiration month
groups = data.groupby('expiration_month')

# Assign a group number to each group
group_num = 0
for name, group in groups:
    data.loc[group.index, 'group'] = group_num
    group_num += 1


# In[57]:


# saving the dataframe
data.to_csv('data1.csv')


# # REPLOTTING THE PLOT

# In[39]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(data['expiration_month'], data['group'], c=data['group'], cmap='rainbow')
ax.set_xlabel('Expiration Month')
ax.set_ylabel('Group')
ax.set_title('Grouped Medicines by Expiration Month')
plt.show()


# # GETTING EACH  CLUSTER VALUES

# In[40]:


group_1_drugs = data[data['group'] == 1]['Drug_Id'].tolist()
print(group_1_drugs)


# # EVALUATION

# In[41]:


from sklearn.metrics import silhouette_score

# calculate the silhouette score for the model
silhouette_avg = silhouette_score(data, kmeans.labels_)

print("The average silhouette score is :", silhouette_avg*100)


# In[42]:


from sklearn.metrics import calinski_harabasz_score
score = calinski_harabasz_score(data, kmeans.labels_)
print(score)


# In[50]:


import joblib

# save the KMeans model
joblib.dump(kmeans, 'kmeans_model.pkl')


# In[51]:


import joblib

# load the saved KMeans model
kmeans_model = joblib.load('kmeans_model.pkl')

# use the loaded model for prediction
labels = kmeans_model.predict(data)


# In[47]:


data1 = data[['expiration_month','expiration_year','Drug_Id','group']]


# In[48]:


data1


# In[49]:


data1.to_csv('data2.csv')


# In[52]:


import streamlit as st
import joblib
import pandas as pd

# Load the model and the dataset
model = joblib.load("kmeans_model.pkl")
data = pd.read_csv("data2.csv")

# Get the unique groups from the dataset
groups = data["group"].unique()

# Create a list of months
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# Create dropdown menus to select the month and group
selected_month = st.selectbox("Select a month", months)
selected_group = st.selectbox("Select a group", groups)

# Create a button to initiate the search for expired drugs
if st.button("Find expired drugs"):
    # Filter the drugs by the selected month and group
    filtered_drugs = data[(data["expiration_month"] == months.index(selected_month) + 1) & (data["Group"] == selected_group)]

    # Extract the drug IDs and display them on the app interface
    drug_ids = filtered_drugs["Drug_Id"].tolist()
    st.write("Expired drug IDs:")
    for Drug_Id in drug_ids:
        st.write(Drug_Id)


# In[55]:


get_ipython().system('streamlit run Hello_DATA.py')


# In[ ]:




