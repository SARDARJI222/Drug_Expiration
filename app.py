#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import joblib
import pandas as pd

# Load the model and the dataset
model = joblib.load("kmeans_model.pkl")
data = pd.read_csv("data1.csv")

# Define the options for the selectboxes
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
years = range(2018, 2020)

st.title("Expired Drugs Finder")
# Define the selectboxes for month, year and group
selected_month = st.selectbox("Select a month", months)
selected_year = st.selectbox("Select a year", years)


# Create a button to initiate the search for expired drugs
if st.button("Find expired drugs"):
    # Filter the drugs by the selected month, year and group
    filtered_drugs = data[(data["expiration_month"] == months.index(selected_month) + 1) & (data["expiration_year"] == selected_year)]

    # Extract the drug IDs and display them on the app interface
    drug_ids = filtered_drugs["Drug_Id"].tolist()
    st.write("Expired drug IDs:")
    for Drug_Id in drug_ids:
        st.write(Drug_Id)


# In[6]:


get_ipython().system('streamlit run app.py')


# In[ ]:




