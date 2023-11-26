#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib as plt
import numpy as np
import math
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure 
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score


# ### Loading Data

# In[2]:


train_df= pd.read_csv("D:\ASDC internship\healthcare//train_data.csv")
test_df = pd.read_csv("D:\ASDC internship\healthcare//test_data.csv")


# In[3]:


train_df.head(5)


# In[4]:


test_df.head()


# ### EDA

# In[5]:


train_df.shape


# In[6]:


train_df.columns


# In[7]:


train_df.describe()


# In[8]:


train_df.corr()


# In[9]:


train_df.info()


# ### Concat the DataSets

# In[10]:


df = pd.concat([train_df, test_df])


# In[11]:


df.dtypes


# In[12]:


df.shape


# In[13]:


df.corr()


# ### Checking Duplicates

# In[14]:


df.duplicated().sum()


# ### Check Missing

# In[15]:


df.isna().sum()


# In[16]:


for col in df.columns:
    prct_missing = np.mean(df[col].isnull()).round(5)*100
    print(f'{col} ->  {prct_missing }%')


# <h5> Categorical Features

# In[17]:


Categorical=['Hospital_code','City_Code_Hospital','Hospital_type_code','Hospital_region_code','Department','Ward_Type','Ward_Facility_Code','Bed Grade','City_Code_Patient','Type of Admission','Severity of Illness','Stay','Age']
for col in Categorical :
    print(train_df[col].unique())
    print('--------------------------------------------')
    print(f'num of {col}->',len(train_df[col].unique()))
    print('--------------------------------------------')
    print(pd.DataFrame(train_df[col].value_counts()))
    print('____________________________________________ \n')


# <h5> numerical features

# In[18]:


numerical = ['Available Extra Rooms in Hospital','Visitors with Patient','Admission_Deposit']
for col in numerical:
    print(pd.DataFrame(train_df[col].describe()))
    print('\n ====================================== \n')


# In[19]:


pd.DataFrame(train_df['Available Extra Rooms in Hospital'].value_counts())


# In[20]:


train_df['Available Extra Rooms in Hospital'].describe().round(0)


# ### Deticting Some insights About Rooms 
#      - AVG num of rooms aVailable in "Hospitals" -> 3
#      - the highest number available  rooms -> 24

# In[21]:


train_df['Visitors with Patient'].describe().round(0)


# ### Deticting Some insights About Visitors with Patient
#     - AVG number of Visitors with Patient-> 3
#     - the highest number of Visitors with Patient -> 32

# In[22]:


train_df['Admission_Deposit'].describe().round()


# ### Deticting Some insights About Deposits
#     - The lowest Deposit -> 1800
#     - AVG Deposits -> 4877
#     - the highest Deposit -> 11920

# In[23]:


highDep=df['Admission_Deposit'][df['Admission_Deposit']>df['Admission_Deposit'].mean()].count()
lowDep = df['Admission_Deposit'][df['Admission_Deposit']<df['Admission_Deposit'].mean()].count()
print(f"Num of Patients With Deposit Above {df['Admission_Deposit'].min()} is -> {highDep} \n")
print(f"Num of Patients With Deposit Below {df['Admission_Deposit'].min()} is -> {lowDep}")


# ### Insighting - Reporting 

# In[24]:


pd.DataFrame(train_df.groupby('Hospital_code')['Hospital_type_code','City_Code_Hospital','Hospital_region_code'].value_counts())


# In[25]:


pd.DataFrame(train_df.groupby('Ward_Facility_Code')['Department'].value_counts())


# In[26]:


pd.DataFrame(train_df.groupby('Department')['Ward_Type'].value_counts())


# In[27]:


pd.DataFrame(train_df.groupby('Department')['Type of Admission'].value_counts())


# In[28]:


pd.DataFrame(train_df.groupby('Severity of Illness')['Type of Admission'].value_counts())


# In[29]:


pd.DataFrame(train_df.groupby('Type of Admission')['Severity of Illness','Department'].value_counts())


# In[30]:


pd.DataFrame(train_df.groupby('Type of Admission')['Bed Grade','Severity of Illness'].value_counts())


# In[31]:


pd.DataFrame(train_df.groupby('Age')['Ward_Type'].value_counts())


# In[32]:


pd.DataFrame(train_df.groupby('Stay')['Severity of Illness'].value_counts())


# In[33]:


pd.DataFrame(train_df.groupby('Severity of Illness')['Bed Grade'].value_counts())


# In[34]:


train_df


# In[35]:


pd.DataFrame(train_df.groupby('Department')['Stay'].value_counts())


# In[36]:


pd.DataFrame(train_df.groupby('Department')['Hospital_type_code'].value_counts())


# In[37]:


pd.DataFrame(df.groupby('Hospital_region_code')['Admission_Deposit'].sum())


# In[38]:


pd.DataFrame(df.groupby('Hospital_code')['Hospital_type_code','City_Code_Hospital','Hospital_region_code'].value_counts())


# In[39]:


pd.DataFrame(df.groupby('Ward_Facility_Code')['Department'].value_counts())


# ## Data preprocessing

# <h3>missing values

# In[40]:


mode_Bed_Grad=df['Bed Grade'].mode()[0]
df['Bed Grade'].replace(np.nan,mode_Bed_Grad,inplace=True)


# In[41]:


median_City_Code_Patient=df['City_Code_Patient'].median()
#median_City_Code_Patient
df['City_Code_Patient'].replace(np.nan,median_City_Code_Patient,inplace=True)


# In[42]:


mode_Stay=df['Stay'].mode()[0]
#mode_Stay
df['Stay'].replace(np.nan,mode_Stay,inplace=True)


# ## Checking the outlaiers

# In[43]:


for col in numerical:
    df.boxplot(column=[col])
    plt.show()


# In[44]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range=Q1-(1.5*IQR)
    upper_range=Q3+(1.5*IQR)
    return lower_range , upper_range


# In[45]:


for col in numerical:
    lowincome,uppincome=remove_outlier(df[col])
    df[col]=np.where(df[col]>uppincome,uppincome,df[col])
    df[col]=np.where(df[col]<lowincome,lowincome,df[col])
    df.boxplot(column=[col])
    plt.show()
   


# ## Data Visualization 

# In[46]:


train_df.head(5)


# In[47]:


plt.figure(figsize=(16,6))
sns.barplot(x="Hospital_code", y="City_Code_Hospital", data=train_df.sort_values("Age"))


# In[48]:


sns.distplot(df['Admission_Deposit'])


# In[49]:


sns.displot(data=train_df, x="Ward_Facility_Code")


# In[50]:


plt.figure(figsize=(16,6))
sns.barplot(x="Stay", y="Bed Grade", data=train_df.sort_values("Age"))


# In[51]:


sns.displot(data=train_df, x="City_Code_Hospital", hue=None, kind="kde")


# In[52]:


plt.figure(figsize=(20, 6))
sns.countplot(data=train_df.dropna().sort_values("Age").reset_index(drop=True), x="Age")
plt.show()


# In[53]:


train_df


# In[54]:


sns.stripplot(data=train_df, y="Department", x="Severity of Illness", hue="Type of Admission")


# In[72]:


plt.figure(figsize=(10, 6))
sns.countplot(y='Hospital_code', data=train_df, order=df['Hospital_type_code'].value_counts().index)
plt.title('Party Distribution')
plt.xlabel('Count')
plt.ylabel('Hospital_code')
plt.tight_layout()
plt.show()


# In[56]:


train_df


# In[57]:


import plotly.express as px
df = px.data.tips()
fig = px.histogram(train_df, x="Department",color = 'Ward_Type')
fig.show()


# In[58]:


import plotly.express as px

df = px.data.tips()

fig = px.box(train_df, x="Age", y="Ward_Type", color="Severity of Illness")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()


# ### Scaling Num Features

# In[59]:


df = pd.concat([train_df, test_df])
for col in numerical:
    scaler = MinMaxScaler()
    df[col] = scaler.fit_transform(df[[col]])


# ## Encoding Data 

# In[60]:


df


# In[61]:


df.describe()


# In[62]:


features=['Hospital_type_code','Hospital_region_code','Department','Ward_Type','Ward_Facility_Code','Type of Admission','Severity of Illness','Age','Stay']
le = preprocessing.LabelEncoder()
for col in features:
    df[col]= le.fit_transform(df[col])


# In[63]:


df


# ## splitting the Data 

# In[64]:


df.drop(['case_id', 'patientid'], axis=1,inplace=True)


# In[65]:


y=df['Stay']
x = df.drop(['Stay'], axis=1)


# In[66]:


train_x, test_x, train_y, test_y = train_test_split(x, y, random_state = 0)


# In[67]:


model = XGBClassifier()
model.fit(train_x, train_y)
predictions = model.predict(test_x)
mse=mean_absolute_error(test_y,predictions)
print("Mean Absolute Error: " + str(mse))


# In[68]:


df


# In[69]:


x


# In[70]:


y


# In[71]:


perm = PermutationImportance(model, random_state=1).fit(test_x, test_y)
eli5.show_weights(perm, feature_names = test_x.columns.tolist())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




