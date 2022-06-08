import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
train=pd.read_csv("dataset/train.csv")
test=pd.read_csv("dataset/test.csv")
meal=pd.read_csv("dataset/meal_info.csv")
center_info=pd.read_csv("dataset/fulfilment_center_info.csv")
test['num_orders']=123456
print(train['num_orders'].describe(include='all',exclude=None))
trainfinal=pd.merge(train, meal,on="meal_id",how="outer")
trainfinal=pd.merge(trainfinal,center_info,on="center_id",how="outer")
trainfinal.head()
trainfinal=trainfinal.drop(['center_id','meal_id'],axis=1)
cols=trainfinal.columns.tolist()
cols=cols[:2]+cols[9:]+cols[7:9]+cols[2:7]
print(cols)
trainfinal=trainfinal[cols]
from sklearn import preprocessing
lb1=preprocessing.LabelEncoder()
trainfinal['center_type']=lb1.fit_transform(trainfinal['center_type'])
lb2=preprocessing.LabelEncoder()
trainfinal['category']=lb2.fit_transform(trainfinal['category'])
lb3=preprocessing.LabelEncoder()
trainfinal['cuisine']=lb3.fit_transform(trainfinal['cuisine'])
trainfinal.head()
trainfinal.info()
train=train[train['week'].isin(range(1,146))]
test=train[train['week'].isin(range(146,156))]
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,7))
sns.distplot(trainfinal.num_orders,bins=25)
plt.xlabel('num_orders')
plt.ylabel('number of buyers')
plt.title('num_orders distribution')
trainfinal2=trainfinal.drop(['id'],axis=1)
plt.show()
correlation=trainfinal2.corr(method='pearson')
columns=correlation.nlargest(8,'num_orders').index
print(columns)
corelation_map=np.corrcoef(trainfinal2[columns].values.T)
sns.set(font_scale=0.1)
heatmap=sns.heatmap(corelation_map,cbar=True,annot=True,square=True,fmt='.2f',yticklabels=columns.values,xticklabels=columns.values)
plt.show()
features=columns.drop(['num_orders'])
trainfinal3=trainfinal[features]
x=trainfinal3.values
y=trainfinal['num_orders'].values
trainfinal3.head()
from sklearn.model_selection import train_test_split
x_train, x_val , y_train, y_val = train_test_split(x,y,test_size=0.25)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor 
from sklearn.preprocessing import StandardScaler
XG = XGBRegressor()
XG.fit(x_train, y_train)
y_pred = XG.predict(x_val)
y_pred[y_pred<0] = 0
from sklearn import metrics 
print('XG RMSLE:',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))

LR=LinearRegression()
LR.fit(x_train, y_train)
y_pred = LR.predict(x_val)
y_pred[y_pred<0] = 0
from sklearn import metrics 
print('LR RMSLE:',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))

L=Lasso()
L.fit(x_train, y_train)
y_pred = L.predict(x_val)
y_pred[y_pred<0] = 0
from sklearn import metrics 
print('L RMSLE:',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))

EN=ElasticNet()
EN.fit(x_train, y_train)
y_pred = EN.predict(x_val)
y_pred[y_pred<0] = 0
from sklearn import metrics 
print('EN RMSLE:',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))

DT=DecisionTreeRegressor()
DT.fit(x_train, y_train)
y_pred = DT.predict(x_val)
y_pred[y_pred<0] = 0
from sklearn import metrics 
print('DT RMSLE:',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))
KNN=KNeighborsRegressor()
KNN.fit(x_train, y_train)
y_pred = KNN.predict(x_val)
y_pred[y_pred<0] = 0
from sklearn import metrics 
print('KNN RMSLE:',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))

GB=GradientBoostingRegressor()
GB.fit(x_train, y_train)
y_pred = GB.predict(x_val)
y_pred[y_pred<0] = 0
from sklearn import metrics 
print('GB RMSLE:',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))

pickle.dump(XG,open('Flask/fdemand.pkl','wb'))

testfinal=pd.merge(test,meal,on='meal_id',how='outer')
testfinal=pd.merge(testfinal,center_info,on='center_id',how='outer')
testfinal=testfinal.drop(['meal_id','center_id'], axis=1)
tcols=testfinal.columns.tolist()
tcols=tcols[:2]+tcols[8:]+tcols[6:8]+tcols[2:6]
testfinal=testfinal[tcols]
lb1=preprocessing.LabelEncoder()
testfinal['center_type']=lb1.fit_transform(testfinal['center_type'])
lb2=preprocessing.LabelEncoder()
testfinal['category']=lb2.fit_transform(testfinal['category'])
lb3=preprocessing.LabelEncoder()
testfinal['cuisine']=lb3.fit_transform(testfinal['cuisine'])
testfinal.info()
x_test=testfinal[features].values
pred= XG.predict(x_test)
pred[pred<0]=0
submit = pd.DataFrame({'id':testfinal['id'],'num_order':pred})
submit.to_csv("dataset/submission.csv",index=False)
print(submit.describe())
sc=StandardScaler()

cat=train.drop(['checkout_price','base_price'],axis=1)
num=train[['checkout_price','base_price']]
scal= pd.DataFrame(sc.fit_transform(num),columns=num.columns)
trains=pd.concat([scal,cat],axis=1)

train=trains[trains['week'].isin(range(1,136))]
test=trains[trains['week'].isin(range(136,146))]



x_train=train.drop(['id','num_orders','week'],axis=1)
y_train=train['num_orders']

x_test=test.drop(['id','num_orders','week'],axis=1)
y_test=test['num_orders']

xgb = XGBRegressor(max_depth = 9,
    learning_rate=0.5,
        silent= 1, 
        objective= 'reg:linear',
        eval_metric= 'rmse',
        seed= 4)

xgb.fit(x_train,y_train)
print('Train Score :',xgb.score(x_train,y_train))
print('Test Score :',xgb.score(x_test,y_test))

predictions = xgb.predict(x_test)
print('Explained Variance :',metrics.explained_variance_score(predictions,y_test))
print("XGB RMSLE :",np.sqrt(metrics.mean_squared_error(y_test,predictions)))




