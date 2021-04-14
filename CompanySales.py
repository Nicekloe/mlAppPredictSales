import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("Company_data.csv")
#use required features
cdf = df[['TV','Radio','Newspaper','Sales']]

#Training Data and predictor Variable
#Use all data for training
x = cdf.iloc[:, :3]
y = cdf.iloc[:, -1]
regressor = LinearRegression()

#Fitting model with training data
regressor.fit(x, y)

#Saving model to current directory
#Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))

#Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print("Le montant des ventes est de :", model.predict([[8.7, 48.9, 75]]))