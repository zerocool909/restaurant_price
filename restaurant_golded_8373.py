# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:53:48 2019

@author: adraj
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:23:32 2019

@author: adraj
"""



import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error
import pandas_profiling


def drop_outliers_for_two(df, field_name1,field_name2):
   df.drop((df[(df[field_name1] == 3) & (df[field_name2] < 3 )]).index, inplace=True) 
   print(df)
   # print(df)
    #df2 = df[(df[field_name1] == 2)]
    #df3 = df2[(df2[field_name2] > 4)]
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    print('Encoding Text Dummy Complete for {}'.format(name)) 
    
    
def get_trimmed(topic):
    topic = topic.replace(" ", "")
    return topic      
    
    
#encode_text_dummy(df, 'OPEN_ON_DAYS')



df = pd.read_csv('C:\\Users\\adraj\\Desktop\\MacchineLearningPython\\ReataurantRates\\fulldatav0405_sans_topic_corrected.csv',encoding='iso-8859-1')
training_set = df.copy()


#not useful
df_topic = pd.DataFrame
df_topic['topics_combined'] = df.apply(lambda x: get_trimmed(x.TOPIC), axis=1)
print(get_trimmed('CASUAL DINING,BAR'))
df_topic = df.apply(lambda x: get_trimmed(x.TOPIC), axis=1)
df['re_topic'] = df_topic


"""

from math import radians, cos, sin, asin, sqrt 
def distance(lat1, lat2, lon1, lon2): 
      
    # The math module contains a function named 
    # radians which converts from degrees to radians. 
    lon1 = radians(lon1) 
    lon2 = radians(lon2) 
    lat1 = radians(lat1) 
    lat2 = radians(lat2) 
       
    # Haversine formula  
    dlon = lon2 - lon1  
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  
    c = 2 * asin(sqrt(a))       
    # Radius of earth in kilometers. Use 3956 for miles 
    r = 6371       
    # calculate the result 
    return(c * r) 
      
      
training_set['Totaldistance'] = training_set.apply(lambda x: distance(x.CITY_LOC_LATITUDE, 
            x.RESTAURANT_LOC_LATITUDE,x.CITY_LOC_LONGITUDE,x.RESTAURANT_LOC_LONGITUDE), axis=1)
"""


profile = pandas_profiling.ProfileReport(df)
profile.to_file(outputfile="output_recombined.html")

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['CITY'] = le.fit_transform(df['CITY'])
df['CITY'].fillna('no_city',inplace=True)
#df_train.dropna(subset = ['LOCALITY'],inplace = True)



from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
x = v.fit_transform(df['CUISINES'])

df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
df1 = pd.DataFrame(x.toarray(), columns="cuisines_"+v.get_feature_names())



pd.DataFrame(training_set, columns=['COST']).to_csv('prediction_restaurant_XGBCV_02.csv')
training_set.to_csv('training_set_geolocation.csv',index=True)

training_set =df

training_set.isnull().sum()
training_set['VOTES'].fillna(training_set.VOTES.median(), inplace=True)
training_set.RATING = pd.to_numeric(training_set.RATING, errors='coerce')
training_set['RATING'].replace(to_replace ="-", value =training_set.RATING.median(),inplace=True) 
training_set['RATING'].fillna(training_set.RATING.mean(), inplace=True)
training_set.isnull().sum()

df_train=df.loc[:12583]
df_test=df.loc[12584:]
"""Not working as IDs are getting repeated. VLOOKUP instead
definitive = df[['TITLE','RESTAURANT_ID','CUISINES','LOCALITY','CITY','VOTES']]
train_locations = pd.read_csv('C:\\Users\\adraj\\.spyder-py3\\df_train_final_locations.csv',encoding='iso-8859-1')
train_loc = train_locations[['RESTAURANT_ID','TITLE','Location']]

pd.merge(definitive,train_loc,left_on='RESTAURANT_ID',how='inner')
all_user_data = pd.merge(definitive,train_locations, on=['RESTAURANT_ID','TITLE','CUISINES','LOCALITY'],how='left')
new_df = pd.merge(definitive, train_loc,  how='left', left_on=['RESTAURANT_ID','TITLE'], right_on = ['RESTAURANT_ID','TITLE'])
"""

del df_train['sr.no.']
del df_train['LOCALITY']
del df_train['CITY']
del df_train['TOPIC']
del df_train['CUISINES']
del df_train['TIME']
del df_train['EST_TYPE']

del df_test['sr.no.']
del df_test['LOCALITY']
del df_test['COST']
del df_test['CITY']
del df_test['TOPIC']
del df_test['CUISINES']
del df_test['TIME']
del df_test['EST_TYPE']


df_train.columns


#df_train['VOTES'] = df_train['VOTES'].apply(lambda x: [x.mean() for i in x if str(i) == "nan"])

df_train['VOTES'].fillna(df_train.VOTES.mean(), inplace=True)
df_train['RATING'].fillna(df_train.RATING.mode(), inplace=True)
df_train['RATING'].replace(to_replace ="-", value ='nan',inplace=True) 
df_train.RATING = pd.to_numeric(df_train.RATING, errors='coerce')
df_train['RATING'].fillna(df_train.RATING.median(), inplace=True)


df_test['VOTES'].fillna(df_test.VOTES.mean(), inplace=True)
df_test['RATING'].fillna(df_test.RATING.mode(), inplace=True)
df_test['RATING'].replace(to_replace ="-", value ='nan',inplace=True) 
df_test.RATING = pd.to_numeric(df_test.RATING, errors='coerce')
df_test['RATING'].fillna(df_test.RATING.median(), inplace=True)




print(df_train.isnull().sum())

X = df_train.copy()
y = X.pop('COST')


train, test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

params = {
    'n_estimators':4000,
    'colsample_bytree': 0.9,
    'objective': 'reg:linear',
    'max_depth': 8,
    'min_child_weight': 3,
    'eta': 0.01,
    'subsample': 0.8,
    'seed' : 6,
    'reg_alpha' : 0.7,
    'reg_lambda' : 1
}

dtrain = xgb.DMatrix(train, y_train)
dtest = xgb.DMatrix(test, y_test)

model = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=800,
    nfold=15,
    early_stopping_rounds=200
)

# Fit
final_gb = xgb.train(params, dtrain, num_boost_round=len(model))

preds = final_gb.predict(dtest)
#ppp=preds
#for val in ppp:
 #   ppp.
  #  if (ppp[val]):
   #     print("value changing {}".format(ppp[val]))
    #    preds[val]=0

print("Root mean square error for test dataset: {}".format(np.round(np.sqrt(mean_squared_error(y_test, preds)), 4)))

print(y_test[0:10],preds[0:10])

#for submission test data
####################################################################### 
dtest = xgb.DMatrix(df_test)

#test_data = np.array(dtest)

predictions = final_gb.predict(dtest)
pd.DataFrame(predictions, columns=['COST']).to_csv('prediction_restaurant_XGBCV_sans_03.csv')

########################################################################


params = {
    'n_estimators': 1000,
    'colsample_bytree': 0.95, #0.89
    'objective': 'reg:linear',
    'max_depth': 30,
    'min_child_weight': 12,
    'learning_rate': 0.02, #0.01
    'subsample': 0.85,
    'seed' : 2
}

dtrain = xgb.DMatrix(train, y_train)
dtest = xgb.DMatrix(test, y_test)

model = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=800,#800
    nfold=10,#20
    early_stopping_rounds=50
)

# Fit
final_gb = xgb.train(params, dtrain, num_boost_round=len(model))

preds = final_gb.predict(dtest)
#ppp=preds
#for val in ppp:
 #   ppp.
  #  if (ppp[val]):
   #     print("value changing {}".format(ppp[val]))
    #    preds[val]=0

print("Root mean square error for test dataset: {}".format(np.round(np.sqrt(mean_squared_error(y_test, preds)), 4)))
#162.1785 - 0.72832711 
#167.2049 - 0.72972630
#166.0772 - 0.73143831
#166.8254 - 0.73341420
#170.5689 - 0.73400
print(y_test[0:10],preds[0:10])

def rmsle(ypred, ytest) : 
    assert len(ytest) == len(ypred)
    return np.sqrt(np.mean((np.log1p(ypred) - np.log1p(ytest))**2))

print(rmsle(preds,y_test))


dtest = xgb.DMatrix(test)

predictions = final_gb.predict(dtest)
pd.DataFrame(predictions, columns=['Fees']).to_csv('prediction_DOC_XGBCV_14.csv')

############################################################################
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred =(model_lgb.predict(test.values))

lgb_pred =(model_lgb.predict(df_test.values))


print("Root mean square error for test dataset: {}".format(np.round(np.sqrt(mean_squared_error(y_test, lgb_pred)), 4)))
print(rmsle(y_train, lgb_train_pred))

##############################################################################

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

#Read trainig and testing files


df_cb = pd.read_csv('C:\\Users\\adraj\\Desktop\\MacchineLearningPython\\ReataurantRates\\fulldatav0405.csv',encoding='iso-8859-1')

df_cb['CITY'].fillna('no_city',inplace=True)

train = df_cb.loc[:12584]
test = df_cb.loc[12585:]


#Identify the datatype of variables
del train['sr.no.']
del train['CUISINES']
del train['TIME']
del train['LOCALITY']

del test['sr.no.']
del test['CUISINES']
del test['TIME']
del test['LOCALITY']
del test['COST']

train.dtypes
train.isnull().sum()
X = train.drop(['COST'], axis=1)
y = train.COST

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)

categorical_features_indices = np.where(X_train.dtypes == object)[0]

#importing library and building model
from catboost import CatBoostRegressor
model=CatBoostRegressor(iterations=500, depth=3, learning_rate=0.1, loss_function='RMSE')
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)

submission = pd.DataFrame()
submission['Item_Identifier'] = test['Item_Identifier']
submission['Outlet_Identifier'] = test['Outlet_Identifier']
submission['Item_Outlet_Sales'] = model.predict(test)
submission.to_csv("Submission.csv")



