import pandas as pd
dataset=pd.read_csv('test_scores.csv')

X=dataset[['school_setting', 'school_type',
       'teaching_method', 'n_student', 'student_id', 'gender', 'lunch',
       'pretest']]
y=dataset['posttest']

# For school_setting column
school_setting=X['school_setting']
school_setting=pd.get_dummies(school_setting,drop_first=True)
X=pd.concat([X,school_setting],axis=1)
X.drop('school_setting',axis=1,inplace=True)

# For school_type column
school_type=X['school_type']
school_type=pd.get_dummies(school_type,drop_first=True)
X=pd.concat([X,school_type],axis=1)
X.drop('school_type',axis=1,inplace=True)

# For teaching_method column
teaching_method=X['teaching_method']
teaching_method=pd.get_dummies(teaching_method,drop_first=True)
X=pd.concat([X,teaching_method],axis=1)
X.drop('teaching_method',axis=1,inplace=True)

# For gender column
gender=X['gender']
gender=pd.get_dummies(gender,drop_first=True)
X=pd.concat([X,gender],axis=1)
X.drop('gender',axis=1,inplace=True)

# For lunch column
lunch=X['lunch']
lunch=pd.get_dummies(lunch,drop_first=True)
X=pd.concat([X,lunch],axis=1)
X.drop('lunch',axis=1,inplace=True)

X.drop('student_id', axis=1, inplace=True)

# For Feature Selection
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
mind=SelectFromModel(Lasso())
mind.fit(X,y)
mind.get_support()
# For Feature Elimination
feature_el=['Suburban','Urban','Public','Male','Qualifies for reduced/free lunch']
for feature in feature_el:
    X.drop(feature, axis=1, inplace=True)

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)

# Creating LinearRegression Model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

import joblib
# Dumping the model
joblib.dump(model,'score.pk1')
mind=joblib.load('score.pk1')

print("    Please Enter the below details!")
n_student=input("Enter No. of students in that class: ")
pre_test_marks=input("Enter pre-test marks: ")
teaching_method=input("Is teaching method standard(yes/no): ")
if teaching_method=="yes":
    teaching_method=1    
else:
    teaching_method=0

# Predicting Value
y_pred=mind.predict([[float(n_student) , float(pre_test_marks) , int(teaching_method)]])
print(y_pred)