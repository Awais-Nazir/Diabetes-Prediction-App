import pandas as pd
# import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# Loading Data 
data = pd.read_csv("pima-indians-diabetes.csv")

X = data.drop('class',axis=1)     # Predictor feature columns (8 X m)
Y = data['class']   # Predicted class (1=True, 0=False) (1 X m)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


rep_0 = SimpleImputer(missing_values=0, strategy="mean")
cols=x_train.columns
x_train = pd.DataFrame(rep_0.fit_transform(x_train))
x_test = pd.DataFrame(rep_0.fit_transform(x_test))

x_train.columns = cols
x_test.columns = cols


model = LogisticRegression(solver="liblinear")

model.fit(x_train,y_train)

def model_prediction(Preg,Plas,Pres,skin,test,mass,pedi,age):
    # ls =  np.array([6,171,97,2984,14.5,75,1,0,0]).reshape(1,-1)
    # ls = np.array([Preg,Plas,Pres,skin,test,mass,pedi,age])
    # ls = ls.reshape(-1,1)
    res= model.predict(np.array([Preg,Plas,Pres,skin,test,mass,pedi,age]).reshape(1,-1))
    if res[0]==1:
        return True
    else:
        return False
# print(model.score(x_test,y_test))

# res = model_prediction(9,102,76,37,0,32.9,0.665,46)
# print(res)

if __name__=="__main__":
    print(model.predict(x_test))

print(x_test.tail())