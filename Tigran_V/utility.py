'''

All imports for all notebooks. Included in requirements.txt

'''
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


homes=pd.read_csv(r"/Users/tiko/Documents/Machine_Learning_Ames_Housing_Kaggle/Ames_HousePrice.csv")


#**************************************** Visualization and EDA *********************************************************
def target_dist(df):
    '''
    
    Analyzing how our target column (SalePrice) is distributed. For linear models, we need our target variable to 
    follow the gaussian distribution.
    
    '''

    fig, (ax1, ax2) = plt.subplots(1,2, sharex=False, sharey=True)

    sns.histplot(df['SalePrice'],ax=ax1,bins=30);
    sns.histplot(np.log(df['SalePrice']),ax=ax2,bins = 30);

    ax1.set_title("Sale Price ($) - Right Skewed")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Sale Price in thousands ($)")

    ax2.set_title("Sale Price - Log Transformation")
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Sale Price - Logged")
    

def check_class_imbalance(df, nrow, ncol):
    '''
    Function creates subplots of all 'categorical' columns in our dataframe and outputs count of occurance for all 
    classes (categorical) in the dataframe. 
    
    '''

    fig, ax  = plt.subplots(figsize = (15, 50))
    for n, col in enumerate(df.columns):
        plt.subplot(nrow, ncol,n+1) #specify number of rows and number of columns for graph output
        df[col].value_counts().plot.bar()
        plt.title(str(col))
    plt.tight_layout()







'''
Based on the subplots output, we can see that we have many categorical input variables that have imbalanced 
classes. One way to solve this problem is to combine  minority classes into one. 

This is tricky as we also do not want to lose additional information detrimental to our output. 


'''    


def check_dist(df, nrow, ncol):
    '''
    Function creates subplots of all 'numerical' columns in our dataframe and outputs distribution
    of all numerical datasets. Normally distributed input variables help improve our ML model.
    The more normally distributed, the better. 
    
    
    ***USE Discretization Transforms if necessry ***
    
    '''

    fig, ax  = plt.subplots(figsize = (15, 50))
    for n, col in enumerate(df.columns):
        plt.subplot(nrow, ncol,n+1) #specify number of rows and number of columns for graph output
        df[col].plot.hist(bins = 30)
        plt.title(str(col))
    plt.tight_layout()



def target_correlation(df, nrow, ncol):
    '''
    creating subplots of all numerical columns to find correlation between the input variables and our
    target, 'SalePrice'. 
    
    
    '''
    fig, ax  = plt.subplots(figsize = (15, 50))

    for i, col in enumerate(df.columns):
        plt.subplot(nrow,ncol,i+1) 
        plt.title(str(col) + ' & Sales Price Correlation')
        sns.scatterplot(data = df, x = df[col], y = df['SalePrice'])
    plt.tight_layout()

def NA_visualized(df):
    '''
    NA Values Visualized
    darker palette ==  more na values
    '''
    plt.figure(figsize=(20,15));
    return sns.heatmap(df.isna(),cmap="BuPu",cbar=False);

def bar_plot(data,x,hue = None, y = 'SalePrice', orient = 'v'):
    plt.figure(figsize=(20,12));
    plt.title('Average Sales Price ($) & '+ str(x))

    return sns.barplot(data = data, x = x, hue = hue, y = y, orient = orient);

def box_plot(data,x,hue = None, y = 'SalePrice', orient = 'v'):
    plt.figure(figsize=(20,12));
    plt.title('Average Sales Price($) & '+ str(x))
    return sns.boxplot(data = data, x = x, hue = hue, y = y, orient = orient);

def scatter_rel(data, x):
    plt.title('Sales Price ($) and '+ str(x))
    sns.scatterplot(data = data, x = x, y = 'SalePrice')
#**************************************** Preprocessing DataFrame for ML modeling ***************************************


## How features will be classified based ond EDA and preprocessing (refer to EDA & preprocessing notebook)
    
num_cols = ['GrLivArea','LotFrontage', 'LotArea','YearBuilt','GarageArea', 'TotSft','YearRemodAdd','BsmtFinSF1',
           'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','BsmtFullBath','BsmtHalfBath','FullBath',
           'HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','WoodDeckSF',]

ordinal_cols = ['Condition1','Condition2', 'OverallQual','OverallCond','ExterQual','ExterCond',
          'BsmtQual','BsmtCond','KitchenQual','GarageQual','GarageCond', 'HeatingQC', 'Neighborhood','GarageFinish',
          'GarageCars',]


nominal_cols = ['MSSubClass','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
          'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
          'Foundation','BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'Heating','CentralAir',
          'Electrical','Functional', 'GarageType', 'GarageFinish']

cols_to_drop = ['PID','Unnamed: 0','Alley', 'FireplaceQu','PoolQC','Fence','MiscFeature','MiscVal',
           'MoSold','YrSold','SaleType','SaleCondition']
    
def preprocessed_DF(num_cols, ordinal_cols, nominal_cols,  cols_to_drop, scaleType =None):
    
    '''
    *** Feature Engineering based on EDA ***
    '''
    homes=pd.read_csv(r"/Users/tiko/Documents/Machine_Learning_Ames_Housing_Kaggle/Ames_HousePrice.csv")

    df = homes.copy()
        
    # Feature Engineering
    df['newBltRmd'] = np.where((df.YearBuilt > 1980) & (df.YearRemodAdd > 2000), 1, 0) 
    # MasVnrType has minimal correlation with 
    df['hasMasVnr'] = np.where((df.MasVnrArea > 0) & (df.MasVnrType != 'None'), 1, 0) 
    df['TotSft'] = df['TotalBsmtSF']+df['1stFlrSF'] +df['2ndFlrSF']
    df['UnfBsmt'] = np.where(df.BsmtUnfSF>0, 1,0)
    df['NewGarage'] = np.where(df['GarageYrBlt']>2000,1,0)
    df['HasWoodDk'] = np.where(df['WoodDeckSF']>0,1,0)
    df['HasPorch'] = np.where((df.OpenPorchSF > 0) | (df.EnclosedPorch > 0) 
                             |(df['3SsnPorch'] > 0) |(df.ScreenPorch > 0) , 1, 0) 
    df['HasPool'] = np.where(df.PoolArea>0,1,0)
    df['PavedDrWay'] = np.where(df.PavedDrive  == 'Y',1,0) #combined minority classes
    df['HasBsmntType2'] = np.where(df.BsmtFinSF2 >0, 1,0)
    
   
    # Dropping columns that  provide no significance (ID, index, columns w/t more than 90% NA values)
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis = 1, inplace = True)
            
   
    # Imputing, encoding (ordinal, non-ordinal) & scaling based on column types
    
    impute = IterativeImputer(n_nearest_features=10,initial_strategy='median',max_iter=200,random_state=101)
    df[num_cols] = impute.fit_transform(df[num_cols])
    ordEncode = OrdinalEncoder()
    df[ordinal_cols] = ordEncode.fit_transform(df[ordinal_cols])
    impute_nom = SimpleImputer(strategy='most_frequent')
    df[nominal_cols] = impute_nom.fit_transform(df[nominal_cols])
    dumm = pd.get_dummies(df[nominal_cols],prefix='_is', drop_first=True)
    df.drop(nominal_cols,axis=1, inplace=True)
    df = df.join(dumm)
    df.drop(['MasVnrType', 'PavedDrive'], axis = 1, inplace=True) #new columns were created for this information in feature engineering section
    df.fillna(1.0,inplace=True) #1.0 are minority classes (will not impact model since na count low)
    
        #Scaling columns based on type of scaler 
    while scaleType != None:
        if scaleType=='standard':
            from sklearn.preprocessing import StandardScaler
            scaler=StandardScaler()
        elif scaleType=='MinMax':
            from sklearn.preprocessing import MinMaxScaler
            scaler=MinMaxScaler()
        elif scaleType=='MaxAbs':
            from sklearn.preprocessing import MaxAbsScaler
            scaler=MaxAbsScaler()
        elif scaleType == 'Robust':
        
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
    
        
        # Scaling columns specified by the function.
        ct = ColumnTransformer(
        [("scale", scaler, num_cols)]) # num_feat -- > numerical features (EDA)
        df[num_cols] = ct.fit_transform(df)
        
        scaleType = None #ending while loop

    return df