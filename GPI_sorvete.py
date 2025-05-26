#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("sakshisatre/ice-cream-sales-dataset")

print("Path to dataset files:", path)


# In[2]:


# prompt: LER DADOS ICE CREAM E COLOCAR DATAFRAME

import pandas as pd

# Construct the path to the CSV file
csv_file_path = f"{path}/Ice Cream.csv"

# Read the CSV into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Display the first few rows of the DataFrame
df.head()


# In[3]:


# @title Temperature vs Revenue

from matplotlib import pyplot as plt
df.plot(kind='scatter', x='Temperature', y='Revenue', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[4]:


# prompt: gerar modelo de regressão linear para prever a revenue em função da Temperature

from sklearn.linear_model import LinearRegression

# Extract the features (Temperature) and the target (Revenue)
X = df[['Temperature']]
y = df['Revenue']

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Print the coefficients (slope and intercept)
print("Intercept:", model.intercept_)
print("Coefficient (Temperature):", model.coef_[0])

# Now you can use the model to make predictions
# For example, predict the revenue for a temperature of 25 degrees
predicted_revenue = model.predict([[25]])
print(f"Predicted revenue for 25 degrees: {predicted_revenue[0]}")

# Plot the regression line on the scatter plot
plt.figure()
df.plot(kind='scatter', x='Temperature', y='Revenue', s=32, alpha=.8)
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.title('Temperature vs Revenue with Linear Regression')
plt.show()

