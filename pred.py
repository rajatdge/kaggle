import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier

train =pd.read_csv('D:/train.csv')
test =pd.read_csv('D:/test.csv')
count=0
for i in train['Name']:
    if(train['Survived'][count]==1):
        print(i.split(",")[0])
    count=count+1

X=train.drop(['Survived','Name','PassengerId','Ticket'],1)
Y=train['Survived']

X1=test.drop(['Name','PassengerId','Ticket'],1)

t=X.fillna(value={'Cabin':0})
X1=X1.fillna(value={'Cabin':0})

avg_train=X['Age'].mean()
avg_test=X1['Age'].mean()
#print(t.head())


t2 = pd.DataFrame(index=t.index)
for col,col_data in t.iteritems():
    if col_data.dtype == object:
        col_data=pd.get_dummies(col_data,prefix=col)
    t2 = t2.join(col_data)


for i in range(t.shape[0]):
    if t2.Age[i]==np.nan and t2.Parch[i]==2:
        t2.Age[i]=2

t2=t2.fillna(value={'Age':avg_train})

for i in ['Age','Fare']:
    t2[i]=scale(t2[i])
#print(t2.describe())

t3 = pd.DataFrame(index=X1.index)
for col,col_data in X1.iteritems():
    if col_data.dtype == object:
        col_data=pd.get_dummies(col_data,prefix=col)
    t3 = t3.join(col_data)


for i in range(X1.shape[0]):
    if t3.Age[i]==np.nan and t3.Parch[i]==2:
        t3.Age[i]=2

t3=t3.fillna(value={'Age':avg_test,'Fare':X1['Fare'].mean()})

for i in ['Age','Fare']:
    t3[i]=scale(t3[i])

for col in t2.columns:
    if col not in t3.columns:
        t3.join(t2[col])
        t3[col]=0

for col in t3.columns:
    if col not in t2.columns:
        t2.join(t3[col])
        t2[col]=0

clf1 = SVC(C=100.0)
clf1.fit(t2,Y)
print ("SVC")
print(str(clf1.score(t2,Y)*100)+"%")

clf2 = RandomForestClassifier()
clf2.fit(t2,Y)
print ("Random Forest Classifier")
print (clf2.score(t2,Y))


result=clf1.predict(t3)
#print(t3.describe())
df=pd.DataFrame(result,index=test['PassengerId'],columns=['Survived'])
df.to_csv('D:\Output\prediction_svc.csv')