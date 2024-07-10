
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

features = ['district_encoder',\
        'area',\
        'new_num_floors',\
        'new_bedrooms',\
        'houseTypes_Bán Luxury home',\
        'houseTypes_Bán Nhà',\
        'houseTypes_Bán Nhà cổ',\
        'houseTypes_Bán Nhà mặt phố',\
        'houseTypes_Bán Nhà riêng']

df = pd.read_excel('HCM_data.xlsx')
df_tranform = pd.DataFrame(data = StandardScaler().fit_transform(df.loc[:, features].values), columns = features)

class splitData():
    def __init__(self):
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
    
    def split(self, x, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.1, random_state=42)

class LinearModel():
    def __init__(self, data):
        self.model = LinearRegression()
        self.data = data
    
    def linear(self):
        self.model.fit(self.data.X_train, self.data.y_train)
        
    def predict(self):
        mse, bias, var = bias_variance_decomp(self.model, self.data.X_train, self.data.y_train, self.data.X_test, self.data.y_test, loss='mse', num_rounds=200, random_seed=1)
        print("MSE:", mse , '\n','Bias:', bias, '\n', 'Variance:', var)
    
class LassoModel():
    def __init__(self, data):
        self.model = LassoCV(cv =5)
        self.data = data
    
    def linear(self):
        self.model.fit(self.data.X_train, self.data.y_train)
        
    def predict(self):
        mse, bias, var = bias_variance_decomp(self.model, self.data.X_train, self.data.y_train, self.data.X_test, self.data.y_test, loss='mse', num_rounds=200, random_seed=1)
        print("MSE:", mse , '\n','Bias:', bias, '\n', 'Variance:', var)
        
class RidgeModel():
    def __init__(self, data):
        self.model = RidgeCV(cv =5)
        self.data = data
    
    def linear(self):
        self.model.fit(self.data.X_train, self.data.y_train)
        
    def predict(self):
        mse, bias, var = bias_variance_decomp(self.model, self.data.X_train, self.data.y_train, self.data.X_test, self.data.y_test, loss='mse', num_rounds=200, random_seed=1)
        print("MSE:", mse , '\n','Bias:', bias, '\n', 'Variance:', var)
    
if __name__ =="__main__":
    
    y = df['price'].values
    x = df_tranform[features].values
    data = splitData()
    data.split(x, y)
    
    linear_model = LinearModel(data)
    lasso_model = LassoModel(data)
    ridge_model = RidgeModel(data)
    
    linear_model.linear()
    linear_model.predict()
    
    lasso_model.linear()
    lasso_model.predict()
    
    ridge_model.linear()
    ridge_model.predict()
    
    
    
    
# python model/linear/linear_hcm.py