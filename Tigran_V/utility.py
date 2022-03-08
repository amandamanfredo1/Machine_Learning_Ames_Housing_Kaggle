
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.experimental import enable_iterative_imputer #enables iterative imputer 
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer,MissingIndicator
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, HuberRegressor
from sklearn.metrics import r2_score
import statsmodels.api as sm




#**************************************** Visualization and EDA *********************************************************










#**************************************** Preprocessing DataFrame for ML modeling ***************************************

homes=pd.read_csv(r"/Users/tiko/Documents/Machine_Learning_Ames_Housing_Kaggle/Ames_HousePrice.csv")

def preprocessed_DF(df):
    
    df = homes.copy()
    
    
    ## How features will be classified based ond EDA and preprocessing 
    
    num_feat = ['GrLivArea','LotFrontage', 'LotArea','YearBuilt','GarageArea', 'TotSft','YearRemodAdd','BsmtFinSF1',
           'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','BsmtFullBath','BsmtHalfBath','FullBath',
           'HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','WoodDeckSF',]

    ord_cat = ['Condition1','Condition2', 'OverallQual','OverallCond','ExterQual','ExterCond',
          'BsmtQual','BsmtCond','KitchenQual','GarageQual','GarageCond', 'HeatingQC', 'Neighborhood','GarageFinish',
          'GarageCars',]


    nom_cat = ['MSSubClass','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
          'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
          'Foundation','BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'Heating','CentralAir',
          'Electrical','Functional', 'GarageType', 'GarageFinish']
    
    '''
    *** Feature Engineering based on EDA ***
    '''
    # homes that were built after 1980 and redomeled after 2000 displayed most correlation with increasing sale price
    df['newBltRmd'] = np.where((homes.YearBuilt > 1980) & (homes.YearRemodAdd > 2000), 1, 0) 
    # MasVnrType has minimal correlation with 
    df['hasMasVnr'] = np.where((homes.MasVnrArea > 0) & (homes.MasVnrType != 'None'), 1, 0) 
    df['TotSft'] = df['TotalBsmtSF']+df['1stFlrSF'] +df['2ndFlrSF']
    df['UnfBsmt'] = np.where(df.BsmtUnfSF>0, 1,0)
    df['NewGarage'] = np.where(df['GarageYrBlt']>2000,1,0)
    df['HasWoodDk'] = np.where(df['WoodDeckSF']>0,1,0)
    df['HasPorch'] = np.where((homes.OpenPorchSF > 0) | (homes.EnclosedPorch > 0) 
                             |(homes['3SsnPorch'] > 0) |(homes.ScreenPorch > 0) , 1, 0) 
    df['HasPool'] = np.where(df.PoolArea>0,1,0)
    df['PavedDrWay'] = np.where(df.PavedDrive  == 'Y',1,0) #combined minority classes
    df.drop(['Unnamed: 0','Alley', 'FireplaceQu','PoolQC','Fence','MiscFeature','MiscVal',
           'MoSold','YrSold','SaleType','SaleCondition'], axis = 1, inplace = True)
    df['HasBsmntType2'] = np.where(df.BsmtFinSF2 >0, 1,0)
    num_feat = ['GrLivArea','LotFrontage', 'LotArea','YearBuilt','GarageArea', 'TotSft','YearRemodAdd','BsmtFinSF1',
           'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','BsmtFullBath','BsmtHalfBath','FullBath',
           'HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','WoodDeckSF',]

    ord_cat = ['Condition1','Condition2', 'OverallQual','OverallCond','ExterQual','ExterCond',
          'BsmtQual','BsmtCond','KitchenQual','GarageQual','GarageCond', 'HeatingQC', 'Neighborhood','GarageFinish',
          'GarageCars',]


    nom_cat = ['MSSubClass','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
          'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
          'Foundation','BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'Heating','CentralAir',
          'Electrical','Functional', 'GarageType', 'GarageFinish']
    impute = IterativeImputer(n_nearest_features=10,initial_strategy='median',max_iter=200,random_state=101)
    df[num_feat] = impute.fit_transform(df[num_feat])
    scaler = RobustScaler()
    df[num_feat] = scaler.fit_transform(df[num_feat])
    ordEncode = OrdinalEncoder()
    df[ord_cat] = ordEncode.fit_transform(df[ord_cat])
    impute_nom = SimpleImputer(strategy='most_frequent')
    df[nom_cat] = impute_nom.fit_transform(df[nom_cat])
    dumm = pd.get_dummies(df[nom_cat],prefix='_is', drop_first=True)
    df.drop(nom_cat,axis=1, inplace=True)
    df = df.join(dumm)
    df.drop(['MasVnrType', 'PavedDrive'], axis = 1, inplace=True)
    df.fillna(1.0,inplace=True)
    return df
    
    
    
