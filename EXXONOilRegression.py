import pandas as pd
import numpy as np
# import math
import sklearn
import pickle
from sklearn.linear_model import LinearRegression

# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def modelcreation():
    path = r"C:\Users\Arun\Desktop\pytonscr\oil_exxon.xlsx"
    price_data = pd.read_excel(path)

    testpath = r"C:\Users\Arun\Desktop\pytonscr\oil_exxon_Test.xlsx"
    test_data = pd.read_excel(testpath)
    # set the index equal to the date column & then drop the old date column
    price_data.index = pd.to_datetime(price_data['date'])
    price_data = price_data.drop(['date'], axis=1)
    # print the first five rows
    #print(price_data.head())
    #print(price_data.dtypes)

    new_column_names = {'exon_price': 'exxon_price'}
    # rename the column
    price_data = price_data.rename(columns=new_column_names)
    #print(price_data.head())
    # check for missing values
    price_data.isna().any()
    # drop any missing values
    price_data = price_data.dropna()
    # let's check to make sure they've all been removed.
    price_data.isna().any()
    #print(price_data.describe())
    # y will have exxon price and X will have oil price
    #Y = price_data.drop('oil_price', axis=1)
    y = price_data[['exxon_price']]
    X = price_data[['oil_price']]



    #loop and run thourgh X and Y

    # Split X and y into X_
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=1)

    # create a Linear Regression model object.
    regression_model = LinearRegression()
    # pass through the X_train & y_train data set.
    regression_model.fit(X_train, y_train)
    intercept = regression_model.intercept_[0]
    coefficient = regression_model.coef_[0][0]
    print("The Coefficient for our model is {:.2}".format(coefficient))
    print("The intercept for our model is {:.4}".format(intercept))
    acc = regression_model.score(X_test, y_test)
    print("accuracy : {:.2}".format(acc))
    with open("ExxonOil.pickle", "wb") as f:
        pickle.dump(regression_model, f)



def predictExxon(oil):
    pickle_in = open("ExxonOil.pickle", "rb")
    linear = pickle.load(pickle_in)
    prediction = linear.predict([[oil]])
    predicted_value = prediction[0][0]
    print("For the oil price of",oil ," the predicted Exxon Value value is {:.4}".format(predicted_value))

#modelcreation()

oil = 67.33
predictExxon(oil)