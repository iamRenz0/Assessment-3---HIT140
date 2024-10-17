#libraries
import pandas as pd
import math
import numpy as np

#visualization
import seaborn as sns
import matplotlib.pyplot as plt

#regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Reading the CSV dataset into a dataframe (df)
df_demographic = pd.read_csv("dataset1.csv")
df_screen_time = pd.read_csv("dataset2.csv")
df_well_being = pd.read_csv("dataset3.csv")

print(df_demographic.info())
print(df_screen_time.info())
print(df_well_being.info())

#sum and calculate total screen time
df_screen_time['total_screen_time'] = (
    df_screen_time['C_we'] + df_screen_time['C_wk'] +
    df_screen_time['G_we'] + df_screen_time['G_wk'] +
    df_screen_time['S_we'] + df_screen_time['S_wk'] +
    df_screen_time['T_we'] + df_screen_time['T_wk']
)

#Merge Datasets based on ID
merged_data = pd.merge(df_demographic, df_screen_time, on='ID')
merged_data = pd.merge(merged_data, df_well_being, on='ID')


#check for any inconsistencies with original source
print(merged_data.columns)

print("\nData Info: ")
print(merged_data.info())

print("\nFirst few rows of the data:")
print(merged_data.head())

print("\nMissingValues in each column: ")
print(merged_data.isnull().sum())


#Perform Data cleaning / Data Wrangling.

#remove rows with missing values
cleaned_data = merged_data.dropna()

#check data types again after the cleaning
print("\nData Types After Cleaning: ")
print(cleaned_data.dtypes)

#convert Categorical Data into Binary
gender_dummies = pd.get_dummies(cleaned_data["gender"], prefix = "gender", drop_first = True)
cleaned_data = pd.concat([cleaned_data, gender_dummies], axis = 1)
cleaned_data.drop("gender", axis = 1, inplace = True)

#Splitting data for training the model
X = merged_data[["gender", "minority", "deprived", "total_screen_time"]]
y = merged_data["Optm"] #ALTER IF YOU WANT OTHER WELL-BEING SCORES *****
#splitting the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Linear Regression Model
model = LinearRegression()

#fit the model using the training data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#show the predicted values of y next to the actual values of y
df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_pred)

#calculate standard performance metrics of linear regression
#MAE (Mean Absolute Error) - Avg of absolute difference between actual and predicted values
mae = metrics.mean_absolute_error(y_test, y_pred)
#MSE (Mean Squared Eror) - Average of Squared differences between actual and predicted values
mse = metrics.mean_squared_error(y_test, y_pred)
#Root MSE - Square root of MSE, giving error in same units as target var
rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
#Normalised Root MSE - Measure of how well the Explanatory (independent) variables explain variance in the Response (dependent) variable
r_2 = metrics.r2_score(y_test, y_pred)

#Print evaluation metrics
print("\nModel Performance Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Square Error (RMSE):", rmse)
print("R-squared (R²):", r_2)


#print coefficients and intercept
print("\nCoefficients: ", model.coef_)
print("\nIntercept: ", model.intercept_) #tells us the value of the response var if explanatory was 0


#VISUALIZATIONS

#Scatterplot Design
sns.set(style="whitegrid")
#scatter plot
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(x=cleaned_data['total_screen_time'], y=cleaned_data['Optm'], alpha=0.7, label='Data Points')
#regression line
regression_line = sns.regplot(x=cleaned_data['total_screen_time'], y=cleaned_data['Optm'], scatter=False, color='red', label='Regression Line')
#titles and labels
plt.title('Relationship between Total Screen Time and Well-Being Score (Optm)', fontsize=16)
plt.xlabel('Total Screen Time (Hours)', fontsize=14)
plt.ylabel('Well-Being Score (Optm)', fontsize=14) #*** ALTER NAME IF GOING FOR A DIFF VARIABLE
plt.xlim(0, cleaned_data['total_screen_time'].max() + 1)  # Adjust x-axis limit for better visibility
plt.ylim(0, 6)  # Since well-being scores range from 1 to 5 / added +1 for viewing.
#legend location
plt.legend(loc='upper left')
#show the plot
plt.show()


#BAR GRAPH
#average optm well-being scores by gender
average_scores = cleaned_data.groupby('gender_1').Optm.mean()

#bar chart
plt.figure(figsize=(8, 5))
average_scores.plot(kind='bar', color=['blue', 'orange'])
plt.title('Average Well-Being Scores by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Well-Being Score')
plt.xticks(ticks=[0, 1], labels=['Non-Male', 'Male'], rotation=0)  # Set x-axis labels to "non-male" and "Male"
plt.ylim(1, 5)  # Set y-axis limits for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)

#display the bar graph/chart
plt.show()




#BOX PLOT
#box plot
plt.figure(figsize=(8, 5))
sns.boxplot(x='gender_1', y='Optm', data=cleaned_data, palette='Set2')

#title and labels
plt.title('Box Plot of Well-Being Scores by Gender')
plt.xlabel('Gender')
plt.ylabel('Well-Being Score')

#x-axis labels
plt.xticks(ticks=[0, 1], labels=['Non-Male', 'Male'])

#Display the box plot
plt.show()





