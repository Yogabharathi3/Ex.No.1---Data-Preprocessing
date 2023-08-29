# Ex.No.1---Data-Preprocessing
## AIM:
To perform Data preprocessing in a data set downloaded from Kaggle

## REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
# STEP 1:
Importing the libraries
# STEP 2:
Importing the dataset
# STEP 3:
Taking care of missing data
# STEP 4:
Encoding categorical data
# STEP 5:
Normalizing the data
# STEP 6:
Splitting the data into test and train
# STEP 7:

## PROGRAM:
# DEVELOPED BY:YOGABHARATHI S
# REGISTER NO:212222230179
```
import pandas as pd

df=pd.read_csv("/content/Churn_Modelling.csv")

df.head()

df.isnull().sum()

df.drop(["RowNumber","Age","Gender","Geography","Surname"],inplace=True,axis=1)

print(df)

x=df.iloc[:,:-1].values

y=df.iloc[:,-1].values

print(x)

print(y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1 = pd.DataFrame(scaler.fit_transform(df))

print(df1)

from sklearn.model_selection import train_test_split

xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

print(xtrain)

print(len(xtrain))

print(xtest)

print(len(xtest))

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df1 = sc.fit_transform(df)

print(df1)
```

## OUTPUT:
# df.head():
![Screenshot 2023-08-29 171824](https://github.com/Yogabharathi3/Ex.No.1---Data-Preprocessing/assets/118899387/3b04fcd6-dc55-48b0-8a11-ada8254cccb4)

# df.isnull().sum():
![Screenshot 2023-08-29 171834](https://github.com/Yogabharathi3/Ex.No.1---Data-Preprocessing/assets/118899387/0a8495d0-05fc-4a88-b715-e493ff688c98)

# df value:
![Screenshot 2023-08-29 171845](https://github.com/Yogabharathi3/Ex.No.1---Data-Preprocessing/assets/118899387/c98f6143-2a58-4408-8f31-2dfbd21c8502)

# VALUES OF INPUT AND OUTPUT DATA ON VAR X AND Y:
![Screenshot 2023-08-29 171852](https://github.com/Yogabharathi3/Ex.No.1---Data-Preprocessing/assets/118899387/6a7dfffd-e550-426a-a038-b53af5bd5f3a)
![Screenshot 2023-08-29 171858](https://github.com/Yogabharathi3/Ex.No.1---Data-Preprocessing/assets/118899387/a438074f-a221-4359-88d7-20f43b1c1e70)

# NORMALIZING DATA:
![Screenshot 2023-08-29 171906](https://github.com/Yogabharathi3/Ex.No.1---Data-Preprocessing/assets/118899387/8d3e9c48-6201-4121-8d43-aa114429fa87)

# X_TRAIN AND Y_TRAIN VALUES:
![Screenshot 2023-08-29 171914](https://github.com/Yogabharathi3/Ex.No.1---Data-Preprocessing/assets/118899387/fcf123ca-9b0f-41ce-8a57-4ecea71f32e6)

# X AND Y VALUES:
![Screenshot 2023-08-29 171924](https://github.com/Yogabharathi3/Ex.No.1---Data-Preprocessing/assets/118899387/47e3bb7b-3f0b-407f-b74b-10183101179c)

# X_TEST AND Y_TEST VALUES:
![Screenshot 2023-08-29 171935](https://github.com/Yogabharathi3/Ex.No.1---Data-Preprocessing/assets/118899387/7bf56ecf-33c6-466e-bb2a-5369ff931a47)

## RESULT:
Thus,the program to perform Data preprocessing in a data set downloaded from Kaggle is implemented successfully .

