# -*- coding: utf-8 -*-
"""
Created on Tue May  7 22:22:25 2019

@author: adraj
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import KFold
#from sklearn.linear_model import LassoCV , RidgeCV, ElasticNet, Lasso,Ridge
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor  
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import datetime
from mlxtend.regressor import StackingCVRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


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



df = pd.read_csv('C:\\Users\\adraj\\Desktop\\MacchineLearningPython\\ReataurantRates\\fulldatav0405_sans_topic_corrected_maha.csv',encoding='iso-8859-1')
training_set = df.copy()

training_set['CITY'].fillna('no_city',inplace=True)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
training_set['CITY'] = le.fit_transform(training_set['CITY'])
#df_train.dropna(subset = ['LOCALITY'],inplace = True)

del training_set['sr.no.']
del training_set['LOCALITY']
del training_set['CITY']
del training_set['TOPIC']
del training_set['CUISINES']
del training_set['TIME']

#boxcox transformation
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in training_set.columns:
    if i!='COST':
        if training_set[i].dtype in numeric_dtypes: 
            numerics2.append(i)

skew_features = training_set[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
skews = pd.DataFrame({'skew':skew_features})
print(skews)

high_skew = skew_features[skew_features > 0.3]
high_skew = high_skew
skew_index = high_skew.index

for i in skew_index:
     if i!='=COST':
        training_set[i]= boxcox1p(training_set[i], boxcox_normmax(training_set[i]+1))

        
skew_features2 = training_set[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
skews2 = pd.DataFrame({'skew':skew_features2})
skews2


training_set.COST = np.log(training_set.COST)


df_train=training_set.loc[:12582]
df_test=training_set.loc[12583:]


del df_test['COST']

dp=pd.get_dummies(training_set)

"""#log and convert
train=np.log(2500)
print(np.exp(train))
"""
print(df_train.isnull().sum())



X = df_train.copy()
y = X.pop('COST')



train, test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

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
predictions=np.exp(predictions)
pd.DataFrame(predictions, columns=['COST']).to_csv('prediction_restaurant_XGBCV_sans_07_box.csv')

df_sub = pd.DataFrame(data=predictions, columns=['Price'])
writer = pd.ExcelWriter('prediction_restaurant_XGBCV_sans_07_box.xlsx', engine='xlsxwriter')
df_sub.to_excel(writer,sheet_name='Sheet1', index=False)
writer.save()

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

lightgbm=LGBMRegressor(objective='regression',num_leaves=450,
                              learning_rate=0.1, n_estimators=1200,
                              max_bin = 30, bagging_fraction = 0.8,
                              bagging_freq = 9, feature_fraction = 0.129,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =3, min_sum_hessian_in_leaf = 6, random_state=10)

xgb=XGBRegressor(learning_rate =0.1, 
      n_estimators=1500, max_depth=30, min_child_weight=12,gamma=0, reg_alpha=2e-5,
      subsample=0.8,colsample_bytree=0.8,
      nthread=4,scale_pos_weight=1,seed=27,verbose=True,random_state=10)

grb=GradientBoostingRegressor(learning_rate=0.1,n_estimators=400, max_depth=12
                              ,subsample=0.8,
                              verbose=False,random_state=10)

svr = Pipeline([('Scaler',RobustScaler()), ('SVR',SVR(C= 10000, epsilon= 0.008, gamma=0.009))])

krr=KernelRidge(alpha=1, kernel='polynomial', gamma=0.001,degree=3,coef0=5)

rf=RandomForestRegressor(n_estimators=3000, oob_score = False, n_jobs = -1,random_state =50,
                         max_features = "auto", min_samples_leaf = 2,warm_start=True,criterion='mse',max_depth=50)

avg=StackingCVRegressor(regressors=(lightgbm,grb,svr,krr,rf),meta_regressor=xgb, use_features_in_secondary=True)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


x=np.array(X)
Y=np.array(y)

avg.fit(x,Y)
y_pred=avg.predict(x)

rmsle(y,y_pred)
Predict=avg.predict(np.array(test_df.drop('Price',axis=1)))


# Converting price back to original scale and making it integer
Predict=np.exp(Predict)
Predict=Predict.astype(int)







#########################################################################3#
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
print(rmsle(y_test, lgb_pred))

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

###############################################################3
#aint workinh
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import datetime
from sklearn.model_selection import GridSearchCV

xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.02, 0.03, 0.05], #so called `eta` value
              'max_depth': [27, 30, 33],
              'min_child_weight': [9,12,15],
              'silent': [1],
              'subsample': [0.7,0.8,0.9],
              'colsample_bytree': [0.7,0.75,0.8],
              'n_estimators': [500,1000,1200]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(train, y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

