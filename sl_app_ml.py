import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Adding Title
st.title("Car price prediction")

# writing text in app
st.write("""
# Using Machine Learning Making a Car Price prediction Model

""")

data=pd.read_csv("car data.csv")
data["Transmission"].replace({"Manual":1, "Automatic":0},inplace=True)
data["Seller_Type"].replace({"Dealer":1, "Individual":0},inplace=True)
data["Fuel_Type"].replace({"Petrol":1, "CNG":2 ,"Diesel":0},inplace=True)


x=data[["Age","Transmission","Seller_Type","Fuel_Type","Kms_Driven","Present_Price"]]
y=data["Selling_Price"]
st.write("Size of X", x.shape)
st.write("size of Y", y.shape)

# Creating model
model = LinearRegression()

# Train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=1)
st.write(x_train.shape)
st.write(x_test.shape)
st.write(y_train.shape)
st.write(y_test.shape)

# fitting the model
model.fit(x_train,y_train)

# parameter estimator
st.write("slope value(m)",model.intercept_)
st.write("constant value(C)", model.coef_)

# making Prediction

y_pred = model.predict(x_test)
y_pred = pd.DataFrame(y_pred,columns=["Predicted"])

st.write("test data size", y_test.shape)
st.write("predicted data size", y_pred.shape)

# manual prediction


# plotting the least Squres line
sns.pairplot(data, x_vars=["Age","Transmission","Seller_Type","Fuel_Type","Kms_Driven","Present_Price"],
             y_vars=["Selling_Price"], size=7, aspect=0.7, kind= "reg")
st.pyplot()
from sklearn.metrics import mean_squared_error
st.write("Mean squred matrics",mean_squared_error(y_test,y_pred))

st.write("R2 metrics",model.score(x_test,y_test))



transmission = st.sidebar.selectbox("Select transmission type",("manual","auto"))
def get_trans(transmission):
    if transmission=="manual":
        return 1
    else:
        return 0


Seller_Type = st.sidebar.selectbox("Select seller type",("Dealer","owner"))
def get_seller(seller):
    if seller=="Dealer":
        return 1
    else:
        return 0

Fuel_Type = st.sidebar.selectbox("Select fuel type",("petrol","diesel","CNG"))
def get_fuel(fuel):
    if fuel == "petrol":
        return 1
    elif fuel == "diesel":
        return 0
    else:
        return 2

age =st.sidebar.slider("age", 1, 20)

Kms_Driven = st.sidebar.slider("Kms_Driven", 1000, 200000)

Present_Price = st.sidebar.slider("Present_price", 1, 40)

st.write(get_trans(transmission))


new_pr = np.array([[age, get_trans(transmission), get_seller(Seller_Type), get_fuel(Fuel_Type), Kms_Driven, Present_Price]])
#new_pr=np.array([10,])
new_pr = new_pr.reshape(1, -1)
price= model.predict(new_pr)
if st.button("Predict"):
    st.write("predicted price of the car:", price)

residuals = y_test - y_pred
