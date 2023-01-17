import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Spliting the Dataset into training and testing data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)

# Creating the linear regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the training set
y_pred = regressor.predict(x_test)

# Visualising the training set
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Salary vs Experience (Training Dataset)")
plt.xlabel("Year of Experience")
plt.ylabel("Salary")
plt.show()

# Visualising the testing set
plt.scatter(x_test, y_test, color='red')
# Same line if training or testing data is plotted
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Salary vs Experience (Testing Dataset)")
plt.xlabel("Year of Experience")
plt.ylabel("Salary")
plt.show()
