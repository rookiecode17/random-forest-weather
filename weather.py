import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import pydot


# import data
features = pd.read_csv('data/temps.csv')
# print(features.head(5))
# print('data dimension', features.shape)
# print(features.describe())

years = features['year']
months = features['month']
days = features['day']

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# print(dates[:5])
plt.style.use('fivethirtyeight')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize= (10, 10))
fig.autofmt_xdate(rotation = 45)

ax1.plot(dates, features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature');ax1.set_title('Max Temp')

ax2.plot(dates, features['temp_1'])
ax2.set_xlabel('');ax2.set_ylabel('Temperature');ax2.set_title('Max Temp Yesterday')

ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date');ax3.set_ylabel('Temperature');ax3.set_title("Max Temp Two Days Ago")

ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date');ax4.set_ylabel('Temperature');ax4.set_title("Friend's Estimate")

plt.tight_layout(pad=2)
# plt.show()

# Data Preprocessing

#one-hot encoding
features = pd.get_dummies(features)
print(features.head(5))

# Labeling and Format Conversion

labels = np.array(features['actual'])
features = features.drop('actual', axis = 1)
feature_list = list(features.columns)
features = np.array(features)

# data splitting
train_feature, test_feature, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state=42)
# print("training features:", train_feature.shape)
# print("training labels", train_labels.shape)
# print("test features", test_feature.shape)
# print("test labels", test_labels.shape)

# choose model random froest regressor
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train_feature, train_labels)

# data predict
predictions = rf.predict(test_feature)

# error calculations
errors = abs(predictions - test_labels)

# mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# print('MAPE:', np.mean(mape))

# Tree Model Visualization

# tree = rf.estimators_[5]
# export_graphviz(tree, out_file='tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# graph.write_png('tree.png')

# feature importances
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Imporatnces: {}'.format(*pair)) for pair in feature_importances]

# the most important features
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)

important_indices = [feature_list.index('temp_1'), feature_list.index('average')]
train_important = train_feature[:, important_indices]
test_important = test_feature[:, important_indices]

rf_most_important.fit(train_important, train_labels)

predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)

mape = np.mean(100 * (errors /test_labels))
print('mape', mape)

plt.close('all')  
plt.figure(figsize=(10, 6)) 
x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation = 'vertical')
plt.xticks(x_values, feature_list, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable');plt.title('Variable Importances')
plt.tight_layout()
plt.show()

plt.close('all')

months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})

months = test_feature[:, feature_list.index('month')]
days = test_feature[:, feature_list.index('day')]
years = test_feature[:, feature_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]

test_dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in test_dates]
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})

plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')

plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = 60)
plt.legend()

plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values')
plt.tight_layout()
plt.show()