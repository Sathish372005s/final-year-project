#======================= IMPORT PACKAGES =============================

import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing


#===================== 1. DATA SELECTION ==============================

#=== READ A DATASET ====

data_frame=pd.read_csv("Synthetic_Financial_datasets_log.csv")
data_frame=data_frame[0:20000]

print("------------------------------------")
print(" 1.Data Selection ")
print("------------------------------------")
print()
print(data_frame.head(20))


#=====================  2.DATA PREPROCESSING ==========================


#=== CHECK MISSING VALUES ===

print("-------------------------------------------------------")
print("                    2.Preprocessing                  ")
print("-------------------------------------------------------")
print()
print("-------------------------------------------------------------")
print("Before Checking missing values ")
print("-------------------------------------------------------------")
print()
print(data_frame.isnull().sum())



#=== LABEL ENCODING ===


label_encoder = preprocessing.LabelEncoder()

object_columns = data_frame.select_dtypes(include=['object', 'string']).columns

print("-------------------------------------------------------------")
print(" Label Encoding for Object/String Columns ")
print("-------------------------------------------------------------")

# Loop through each object column and apply label encoding
for column in object_columns:
    print(f"\nBefore label encoding for column: '{column}'")
    print(data_frame[column].head(15))

    data_frame[column] = label_encoder.fit_transform(data_frame[column])

    print(f"\nAfter label encoding for column: '{column}'")
    print(data_frame[column].head(15))
    print("-------------------------------------------------------------")



#=============================== 3. DATA SPLITTING ============================

X=data_frame.drop('isFraud',axis=1)
y=data_frame['isFraud']


X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

print("-------------------------------------------------------------")
print(" Data Splitting ")
print("-------------------------------------------------------------")
print()
print("Total No.of data's in dataset: ", data_frame.shape[0])
print()
print("Total No.of training data's  : ", X_train.shape[0])
print()
print("Total No.of testing data's  : ", X_test.shape[0])





#-------------------------- CLASSIFICATION  --------------------------------

# ---- RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


rf = RandomForestClassifier()

rf.fit(X_train,y_train)

pred_rf = rf.predict(X_test)

acc_rf = metrics.accuracy_score(pred_rf,y_test)*100

error_rf = 100 - acc_rf

print("---------------------------------------------")
print("   Classification - RandomForest Classifier  ")
print("---------------------------------------------")

print()


print("1) Accuracy = ", acc_rf )
print()
print("2) Error Rate = ", error_rf)
print()
print("3) Classification Report =")
print()
print(metrics.classification_report(pred_rf,y_test))


import pickle
with open('model_rf.pickle', 'wb') as f:
    pickle.dump(rf, f)



# ---- LOGISTIC REGRESSION

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

pred_dt = dt.predict(X_test)

acc_dt = metrics.accuracy_score(pred_dt,y_test)*100

error_dt = 100 - acc_dt

print("---------------------------------------------------")
print("   Classification - Decision Tree Classifier  ")
print("----------------------------------------------------")

print()


print("1) Accuracy = ", acc_dt )
print()
print("2) Error Rate = ", error_dt)
print()
print("3) Classification Report =")
print()
print(metrics.classification_report(pred_dt,y_test))


import pickle
with open('model_dt.pickle', 'wb') as f:
    pickle.dump(dt, f)















