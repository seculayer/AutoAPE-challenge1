#!/usr/bin/env python
# coding: utf-8

# In[108]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
np.set_printoptions(threshold=sys.maxsize)


# In[109]:


train_kaggle = '/kaggle/input/covid19-global-forecasting-week-1/train.csv'
test_kaggle = '/kaggle/input/covid19-global-forecasting-week-1/test.csv'
submit_kaggle = '/kaggle/input/covid19-global-forecasting-week-1/submission.csv'

df_train = pd.read_csv(train_kaggle)
df_test = pd.read_csv(test_kaggle)
sample = pd.read_csv(submit_kaggle)


# In[110]:


df_train.info()


# In[111]:


df_train.head(5)


# In[112]:


df_test.info()


# In[113]:


df_test.head(5)


# In[114]:


sample.head(5)


# ## EDA

# ## About the Data 
# 1. Contains Daily Reports of Number of Cases and Fatalities for countries.
# 2. [Missing Data]Contains some entries with Province/State Information Missing - Dropped.
# 3. Contains latitude and longitude for entries, Can Plot on map.
# 4. Date - 22nd Feb to 23nd March.
# 5. Country/Region - 163

# In[115]:


# Dataset Dimesnions
print('Train shape', df_train.shape)
print('Test shape', df_test.shape)
# Missing/Null Values
print('\nTrain Missing\n', df_train.isnull().sum())
print('\nTest Missing\n', df_test.isnull().sum())


# ### Unique countries in the dataset 

# In[116]:


lst = df_train['Country/Region'].unique()
print('Total_Countries\n:', len(lst))
for i in lst:
    print(i)


# ### Date Range for the Dataset 

# In[117]:


print(df_train['Date'].min(), ' - ', df_train['Date'].max())


# ### Checking Daily Worldwide Confirmed Cases and Fatalities 

# In[118]:


# GroupBy syntax (columns to group by in list)[Columns to aggregate, apply function to] . aggregation functions on it 
train_cases_conf = df_train.groupby(['Date'])['ConfirmedCases'].sum()
train_cases_conf


# In[119]:


train_cases_conf.plot(figsize = (10,8), title = 'Worldwide Confirmed Cases')


# In[120]:


train_fatal = df_train.groupby(['Date'])['Fatalities'].sum()
train_fatal


# In[121]:


train_fatal.plot(figsize = (10,8), title = 'Worldwide Fatalaties')


# ### Check Confirmed cases and fatalities for a country 
# scale = "linear", "log"

# In[122]:


def country_stats(country, df):
    country_filt = (df['Country/Region'] == country)
    df_cases = df.loc[country_filt].groupby(['Date'])['ConfirmedCases'].sum()
    df_fatal = df.loc[country_filt].groupby(['Date'])['Fatalities'].sum()
    fig, axes = plt.subplots(nrows = 2, ncols= 1, figsize=(15,15))
    df_cases.plot(ax = axes[0])
    df_fatal.plot(ax = axes[1])
    
country_stats('US', df_train)


# #### Fatalities and Confirmed Cases by Country (Log Scale)

# In[123]:


# grouping using same Country filter to get fatalities on each date (grouped by date)
# groupby([list of columns to groupby]) [which columns to apply aggregate fx to ]. (aggregate function)

def country_stats_log(country, df):
    count_filt =(df_train['Country/Region'] == country)
    df_count_case = df_train.loc[count_filt].groupby(['Date'])['ConfirmedCases'].sum()
    df_count_fatal = df_train.loc[count_filt].groupby(['Date'])['Fatalities'].sum()
    plt.figure(figsize=(15,10))
    plt.axes(yscale = 'log')
    plt.plot(df_count_case.index, df_count_case.tolist(), 'b', label = country +' Total Confirmed Cases')
    plt.plot(df_count_fatal.index, df_count_fatal.tolist(), 'r', label = country +' Total Fatalities')
    plt.title(country +' COVID Cases and Fatalities (Log Scale)')
    plt.legend()
    

country_stats_log('US', df_train)


# ###  Most Affected Countries

# In[124]:


# as_index = False to not make the grouping column the index, creates a df here instead of series, preserves
# Confirmedcases column

train_case_country = df_train.groupby(['Country/Region'], as_index=False)['ConfirmedCases'].max()

# Sorting by number of cases
train_case_country.sort_values('ConfirmedCases', ascending=False, inplace = True)
train_case_country


# In[125]:


plt.figure(figsize=(8,6))
plt.bar(train_case_country['Country/Region'][:5], train_case_country['ConfirmedCases'][:5], color = ['red', 'yellow','black','blue','green'])


# #### No. of Cases on final date(2020/03/23), (Not Increase, Cumulative)

# In[126]:


# Confirmed Cases till a particular day by country

def case_day_country (Date, df):
    df = df.groupby(['Country/Region', 'Date'], as_index = False)['ConfirmedCases'].sum()
    date_filter = (df['Date'] == Date)
    df = df.loc[date_filter]
    df.sort_values('ConfirmedCases', ascending = False, inplace = True)
    sns.catplot(x = 'Country/Region', y = 'ConfirmedCases' , data = df.head(10), height=5,aspect=3, kind = 'bar')
    
    
case_day_country('2020-03-23', df_train)


# # Data Wrangling/ Pre-processing/ Cleaning 
# 1. Identifying and Handling missing values.
# 2. Data Formating.
# 3. Data Normalization(centering and scaling).
# 4. Data bining.
# 5. Turning categorical values into numerical values.

# ### Need to Exclude Leaky Data, the same Dates are in both train and test set.
# 1. First convert object to python datetime type <br>
# Using pd.to_datetime() <br>
# Check Getting converted to float, because haven't converted to date before comparison, still object.

# In[127]:


df_train.Date = pd.to_datetime(df_train['Date'])
print(df_train['Date'].max())
print(df_test['Date'].min())


# ### Truncate df_train by date < df_test['Date'].min()

# In[128]:


date_filter = df_train['Date'] < df_test['Date'].min()
df_train = df_train.loc[date_filter]


# In[129]:


# Dropping ID and getting rid of Province/State with NULL values 
df_train.info()


# In[130]:


# lets get Cumulative sum of ConfirmedCases and Fatalities for each country on each data (same as original data)
# Doing to create copy without ID and 

train_country_date = df_train.groupby(['Country/Region', 'Date', 'Lat', 'Long'], as_index=False)['ConfirmedCases', 'Fatalities'].sum()


# In[131]:


print(train_country_date.info())
print(train_country_date.isnull().sum())


# ### Feature Engineering
# Splitting Date into day, month, day of week. <br>
# Check if Date is in python datetime format. Else, convert object to python datetime type <br>
# Using pd.to_datetime()

# In[132]:


train_country_date.info()


# #### Using Pandas Series.dt.month
# The month as January=1, December=12.

# In[133]:


# Adding day, month, day of week columns 

train_country_date['Month'] = train_country_date['Date'].dt.month
train_country_date['Day'] = train_country_date['Date'].dt.day
train_country_date['Day_Week'] = train_country_date['Date'].dt.dayofweek
train_country_date['quarter'] = train_country_date['Date'].dt.quarter
train_country_date['dayofyear'] = train_country_date['Date'].dt.dayofyear
train_country_date['weekofyear'] = train_country_date['Date'].dt.weekofyear


# In[134]:


train_country_date.head()


# In[135]:


train_country_date.info()


# #### Same Feature Engineering for Test Set

# In[136]:


# First drop Province/State
df_test.drop('Province/State', axis = 1, inplace = True)

# Converting Date Object to Datetime type

df_test.Date = pd.to_datetime(df_test['Date'])
df_test.head(2)


# In[137]:


# adding Month, DAy, Day_week columns Using Pandas Series.dt.month

df_test['Month'] = df_test['Date'].dt.month
df_test['Day'] = df_test['Date'].dt.day
df_test['Day_Week'] = df_test['Date'].dt.dayofweek
df_test['quarter'] = df_test['Date'].dt.quarter
df_test['dayofyear'] = df_test['Date'].dt.dayofyear
df_test['weekofyear'] = df_test['Date'].dt.weekofyear


# In[138]:


df_test.info()


# #### Councatenating Train-Test to Label encode Country/Region Categorical Variable.
# 1. Make copy of train data without Confirmed Cases and Fatalities. Index - 0 to 17608
# 2. Concatenate train, test.
# 3. Label Encode Countries.
# 4. Add back Cofirmed Cases, Fatalities columns to clean_train_data.
# 5. Modelling
# 6. Saving Predicted Values with ForecastID

# In[139]:


# train_country_date
# df_test
# Lets select the Common Labels and concatenate.

labels = ['Country/Region', 'Lat', 'Long', 'Date', 'Month', 'Day', 'Day_Week','quarter', 'dayofyear', 'weekofyear']

df_train_clean = train_country_date[labels]
df_test_clean = df_test[labels]

data_clean = pd.concat([df_train_clean, df_test_clean], axis = 0)


# In[140]:


data_clean.info()


# ## Preparing Data For Models - LabelEncode Country

# In[141]:


from sklearn.preprocessing import LabelEncoder


# In[142]:


# Label Encoder for Countries 

enc = LabelEncoder()
data_clean['Country'] = enc.fit_transform(data_clean['Country/Region'])
data_clean


# In[143]:


# Dropping Country/Region and Date

data_clean.drop(['Country/Region', 'Date'], axis = 1, inplace=True)


# ### Splitting Back into Train and Test

# In[144]:


index_split = df_train.shape[0]
data_train_clean = data_clean[:index_split]


# In[145]:


data_test_clean = data_clean[index_split:]


# ### Adding Back Confirmed Cases and Fatalities
# Using original df_train, check shape is same, head, tail have same values. ORDER NEEDS TO BE SAME.

# In[146]:


data_train_clean.tail(5)


# ### Creating Features and Two Labels

# In[147]:


x = data_train_clean[['Lat', 'Long', 'Month', 'Day', 'Day_Week','quarter', 'dayofyear', 'weekofyear', 'Country']]
y_case = df_train['ConfirmedCases']
y_fatal = df_train['Fatalities']


# ### Train-Test Split - Confirmed Cases

# In[148]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y_case, test_size = 0.3, random_state = 42)


# ### Train-Test Split - Fatalities

# In[149]:


from sklearn.model_selection import train_test_split

x_train_fatal, x_test_fatal, y_train_fatal, y_test_fatal = train_test_split(x, y_fatal, test_size = 0.3, random_state = 42)


# ## Modeling - Regression Problem 
# Using features Country/Region, Lat, Long, Month, Day, Day_week, quarter, dayofyear, weekofyear.<br>
# To predict ConfirmedCases, Fatalities.
# ### To predict 2 Different Target Variables, Train two classifiers, one for each.

# # Modelling

# ## RandomForest Regressor

# In[150]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# #### For ConfirmedCases

# In[151]:


rf = RandomForestRegressor(n_estimators =100)
rf.fit(x_train, y_train.values.ravel())


# In[152]:


rf.score(x_train, y_train)


# In[153]:


rf.score(x_test, y_test)


# In[154]:


# Predicted Values and MSE
y_pred_train = rf.predict(x_train)
print(mean_squared_error(y_train, y_pred_train))


# In[155]:


# Training on entire set and predict values.

rf.fit(x, y_case.values.ravel())


# In[156]:


# Predicted ConfirmedCases
rf_pred_case = rf.predict(data_test_clean)


# In[157]:


plt.figure(figsize=(15,8))
plt.plot(rf_pred_case)


# #### For Fatalities

# In[158]:


rf.fit(x, y_fatal.values.ravel())


# In[159]:


rf_pred_fatal = rf.predict(data_test_clean)


# In[160]:


plt.figure(figsize=(20,8))
plt.plot(rf_pred_fatal)


# In[161]:


# Saving to Submission.csv

submission = pd.read_csv(submit_kaggle)
submission['ConfirmedCases'] = rf_pred_case
submission['Fatalities'] = rf_pred_fatal

submission.to_csv('submission.csv', index = False)

