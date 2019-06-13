#!/usr/bin/env python
# coding: utf-8

# # Board Game Review Prediction

# We git clone https://github.com/ThaWeatherman/scrapers.git
#     
# And copy the games.csv file to current directory

# In[1]:


import sys
import pandas
import matplotlib
import seaborn 
import sklearn 

print(sys.version)
print(pandas.__version__)
print(matplotlib.__version__)
print(seaborn.__version__)
print(sklearn.__version__)


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[3]:


#load our dataset

#pandas' dataframe

games = pandas.read_csv('games.csv')


# In[4]:


#print the names of the columns in games
#know the shape

print(games.columns)
print(games.shape)


# In[5]:


#make a histogram of all the ratings of in the average_rating column
#our target is the 'average_rating' column

plt.hist(games['average_rating'])


# In[6]:


#print the first row of all the games with zero scores
print(games[games['average_rating'] == 0].iloc[0])

#print the first row of all the games with scores greater than zero
print(games[games['average_rating'] > 0].iloc[0])


# In[7]:


#remove any rows with user review

games = games[games['users_rated'] > 0]

#remove any rows with missing values

games = games.dropna(axis=0)


#make a new histogram
plt.hist(games["average_rating"])
plt.show()


# In[8]:


print(games.columns)


# In[10]:


#correlation matrix

corrmat = games.corr()

fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat, vmax = 0.8, square = True)

plt.show()


# In[11]:


#part od dataset preprocessing
#get all the columns from the dataframe
columns = games.columns.tolist()

#filter the columns to remove data we do not want
columns = [c for c in columns if c not in ['bayes_average_rating','average_rating','type','name','id']]

#store the variable we'll be predicting on
target = 'average_rating'


# In[16]:


#split the dataset and generate training and test datasets
from sklearn.model_selection import train_test_split

#generate traning sets
train = games.sample(frac = 0.8, random_state = 1)

#select anything not in the training set and put it in test dataset
test = games.loc[~games.index.isin(train.index)]

#print shapes
print(train.shape)
print(test.shape)


# In[17]:


#import linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#initialize the model class
LR = LinearRegression()

#fit the data in the model
LR.fit(train[columns], train[target])


# In[18]:


#generate predictions for test dataset
predictions = LR.predict(test[columns])

#compute error b/w test predictions and actual values
mean_squared_error(predictions, test[target])


# In[19]:


#import the random forest model
from sklearn.ensemble import RandomForestRegressor

#initialize the model
RFR = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 10, random_state = 1)

#fit the model to the data
RFR.fit(train[columns], train[target])


# In[23]:


#make predictions
predictions = RFR.predict(test[columns])

#compute error
mean_squared_error(predictions, test[target])


# In[25]:


test[columns].iloc[0]


# In[26]:


#predictions
rating_LR = LR.predict(test[columns].iloc[0].values.reshape(1, -1))
rating_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1, -1))

#print the predictions
print(rating_LR)
print(rating_RFR)


# In[27]:


test[target].iloc[0]

