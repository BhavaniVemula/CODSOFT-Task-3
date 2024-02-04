#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary libraries


# In[5]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


# In[6]:


# Loading the Iris flower dataset


# In[7]:


iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})


# In[8]:


# Training the K-Nearest Neighbors classifier


# In[29]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(iris.data, iris.target)


# In[30]:


# Predicting the species of the Iris flowers in the dataset


# In[31]:


y_pred = knn.predict(iris.data)
iris_df['predicted_species'] = [iris.target_names[pred] for pred in y_pred]


# In[32]:


# Formatting the predicted species of the Iris flowers


# In[33]:


output = "FLOWER  SPECIES\n"
for i in range(len(y_pred)):
    if y_pred[i] == 0:
        output += f"Iris {iris.target_names[0]}\n"
    elif y_pred[i] == 1:
        output += f"Iris {iris.target_names[1]}\n"
    else:
        output += f"Iris {iris.target_names[2]}\n"


# In[34]:


print(output)


# In[35]:


sns.histplot(data=iris_df[iris_df['species'] == 'setosa'], x='sepal length (cm)', bins=10, kde=True, color='blue')
plt.title('Histogram of Sepal Length for Setosa Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()


# In[36]:


sns.scatterplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='predicted_species', palette='viridis')
plt.title('Scatterplot of Sepal Length vs Sepal Width with Predicted Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()


# In[37]:


species_counts = iris_df['species'].value_counts()
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B2FF', '#99FF99'])
plt.title('Distribution of Flower Species in Iris Dataset')
plt.axis('equal')  
plt.show()


# In[ ]:




