import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import time
from pprint import pprint

features = pd.read_csv('data/temps_extended.csv')
# print(features.head(5))
# print('data dimension', features.shape)

# print(round(features.describe(), 2))

years = features['year']
months = features['month']
days = features['day']

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
plt.style.use('fivethirtyeight')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 10))
fig.autofmt_xdate(rotation = 45)

ax1.plot(dates, features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature (F)'); ax1.set_title('Max Temp')

ax2.plot(dates, features['temp_1'])
ax2.set_xlabel('');ax2.set_ylabel('Temperature (F)'); ax2.set_title('Max Temp Yesterday')

ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature (F)'); ax3.set_title('Max Temp Two Days Ago')

ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature (F)'); ax4.set_title('Friend Estimate')

plt.tight_layout(pad=2)
# plt.show()

plt.close('all')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols = 2, figsize = (15, 10))
fig.autofmt_xdate(rotation = 45)

ax1.plot(dates, features['average'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature (F)'); ax1.set_title('Historical Avg Max Temp')

ax2.plot(dates, features['ws_1'], 'r-')
ax2.set_xlabel(''); ax2.set_ylabel('Wind Speed (mph)'); ax2.set_title('Wind Speed Yesterday')

ax3.plot(dates, features['prcp_1'], 'r-')
ax3.set_xlabel('Date'); ax3.set_ylabel('Precipition (in)'); ax3.set_title('Preciption Yesterday')

ax4.plot(dates, features['snwd_1'], 'ro')
ax4.set_xlabel('Date'); ax4.set_ylabel('Snow Depth (in)'); ax4.set_title('Snow Depth Yesterday')
plt.tight_layout(pad=2)
# plt.show()
plt.close('all')

seasons = []

for month in features['month']:
    if month in [1, 2, 12]:
        seasons.append('winter')
    elif month in [3, 4, 5]:
        seasons.append('spring')
    elif month in [6, 7, 8]:
        seasons.append('summer')
    elif month in [9, 10, 11]:
        seasons.append('fall')

reduced_features = features[['temp_1', 'prcp_1', 'average', 'actual']]
reduced_features['season'] = seasons

sns.set(style='ticks', color_codes=True)
palette = sns.xkcd_palette(['dark blue', 'dark green', 'gold', 'orange'])

sns.pairplot(reduced_features, hue = 'season', diag_kind = 'kde', palette = palette, plot_kws=dict(alpha = 0.7), diag_kws = dict(fill=True));
# plt.show()
plt.close("all")

features = pd.get_dummies(features)

labels = features['actual']
features = features.drop('actual', axis = 1)

feature_list = list(features.columns)

features = np.array(features)
labels = np.array(labels)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 0)

print('training features', train_features.shape)
print('training labels', train_labels.shape)
print('test features', test_features.shape)
print('test labels', test_labels.shape)

original_feature_indices = [feature_list.index(feature) for feature in feature_list if feature not in ['ws_1', 'prcp_1', 'snwd_1']]

original_features = pd.read_csv('data/temps.csv')
original_features = pd.get_dummies(original_features)

original_train_features = train_features[:, original_feature_indices]
original_test_features = test_features[:, original_feature_indices]

rf = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf.fit(original_train_features, train_labels);

baseline_predictions = rf.predict(original_test_features)
baseline_errors = abs(baseline_predictions - test_labels)
print("average temperature errors: ", round(np.mean(baseline_errors), 2), 'degrees.')

baseline_mape = 100 * np.mean((baseline_errors / test_labels))

baseline_accuracy = 100 - baseline_mape
print('Accuracy:', round(baseline_accuracy, 2), '%')

rf_exp = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf_exp.fit(train_features, train_labels)

predictions = rf_exp.predict(test_features)
errors = abs(predictions - test_labels)

print('Average Temperature errors: ', round(np.mean(errors), 2), 'degree.')

mape = np.mean(100 * (errors / test_labels))

improvement_baseline = 100 * abs(mape - baseline_mape) / baseline_mape
print('Model Performance Improvement with Addidtional Features:', round(improvement_baseline, 2), '%')

accuracy = 100 - mape
print('Accuracy', round(accuracy, 2), '%')

importances = list(rf_exp.feature_importances_)

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

[print('Features: {:20} Importances: {}'.format(*pair)) for pair in feature_importances];

plt.style.use('fivethirtyeight')

x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
plt.xticks(x_values, feature_list, rotation= 'vertical')

plt.ylabel('Importance'); plt.xlabel('Variable');plt.title('Variable Importance')
# plt.show()

plt.close("all")

sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]

cumulative_importances = np.cumsum(sorted_importances)

plt.plot(x_values, cumulative_importances, 'g-')
plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
plt.xticks(x_values, sorted_features, rotation = 'vertical')
plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances')
plt.tight_layout()
# plt.show()

plt.close("all")
# print('Number of features for 95% importance:', np.where(cumulative_importances > 0.95)[0][0] + 1)

important_feature_names = [feature[0] for feature in feature_importances[0:5]]
important_indices = [feature_list.index(feature) for feature in important_feature_names]

important_train_features = train_features[:, important_indices]
important_test_features = test_features[:, important_indices]

print('Important train features shape:', important_train_features.shape)
print("Important test features shape:", important_test_features.shape)

rf_exp.fit(important_train_features, train_labels)
predictions = rf_exp.predict(important_test_features)

errors = abs(predictions - test_labels)

print('Average Temperature Errors:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / test_labels)

accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%')

all_features_time = []

for _ in range(10):
    start_time = time.time()
    rf_exp.fit(train_features, train_labels)
    all_features_predictions = rf_exp.predict(test_features)
    end_time = time.time()
    all_features_time.append(end_time - start_time)

all_features_time = np.mean(all_features_time)
print('Average Training and Testing Time Using All Features', round(all_features_time, 2), 'second')

reduced_features_time = []

for _ in range(10):
    start_time = time.time()
    rf_exp.fit(important_train_features, train_labels)
    reduced_features_predictions = rf_exp.predict(important_test_features)
    end_time = time.time()
    reduced_features_time.append(end_time - start_time)

reduced_features_time = np.mean(reduced_features_time)
print('Avearge Training and Testing Time Using Reduced Reatures:', round(reduced_features_time, 2), 'second.')

all_accuracy = 100 * (1 - np.mean(abs(all_features_predictions - test_labels) / test_labels))
reduced_accuracy = 100 * (1 - np.mean(abs(reduced_features_predictions - test_labels) / test_labels))

comparison = pd.DataFrame({'features': ['all (17)', 'reduced (5)'], 
                           'run_time': [round(all_features_time, 2), round(reduced_features_time, 2)],
                           'accuracy': [round(all_accuracy, 2), round(reduced_accuracy, 2)]})

print(comparison[['features', 'accuracy', 'run_time']])

relative_accuracy_decrease = 100 * (all_accuracy - reduced_accuracy) / all_accuracy
print('Relative Decrease in Accuracy:', round(relative_accuracy_decrease, 3), '%.')

relative_runtime_decrease = 100 * (all_features_time - reduced_features_time) / all_features_time
print('Relative Improvement in Time Efficiency:', round(relative_runtime_decrease, 3), '%.')

original_features = pd.read_csv('data/temps.csv')
original_features = pd.get_dummies(original_features)

original_labels = np.array(original_features['actual'])
original_features = original_features.drop('actual', axis = 1)

original_feature_list = list(original_features.columns)
original_features = np.array(original_features)

original_train_features, original_test_features, original_train_labels, original_test_labels = train_test_split(original_features, original_labels)

original_feature_indices = [feature_list.index(feature) for feature in feature_list if feature not in ['ws_1', 'prcp_1', 'snwd_1']]

original_test_features = test_features[:, original_feature_indices]

original_features_time = []

for _ in range(10):
    start_time = time.time()
    rf.fit(original_train_features, original_train_labels)
    original_features_predictions = rf.predict(original_test_features)
    end_time = time.time()
    original_features_time.append(end_time - start_time)

original_features_time = np.mean(original_features_time)

original_mae = np.mean(abs(original_features_predictions - test_labels))
exp_all_mae = np.mean(abs(all_features_predictions - test_labels))
exp_reduced_mae = np.mean(abs(reduced_features_predictions - test_labels))

original_accuracy = 100 * (1 - np.mean(abs(original_features_predictions - test_labels) / test_labels))

model_comparison = pd.DataFrame({'model': ['original', 'exp_all', 'exp_reduced'],
                                 'error (degrees)': [original_mae, exp_all_mae, exp_reduced_mae],
                                 'accuracy': [original_accuracy, all_accuracy, reduced_accuracy],
                                 'run_time (s)': [original_features_time, all_features_time, reduced_features_time]})

model_comparison = model_comparison[['model', 'error (degrees)', 'accuracy', 'run_time (s)']]
print(model_comparison)

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (16, 5), sharex = True)

x_values = [0, 1, 2]
labels = list(model_comparison['model'])
plt.xticks(x_values, labels)

fontdict = {'fontsize': 18}
fontdict_yaxis = {'fontsize': 14}

ax1.bar(x_values, model_comparison['error (degrees)'], color = ['b', 'r', 'g'], edgecolor = 'k', linewidth = 1.5)
ax1.set_ylim(bottom = 3.5, top = 4.5)
ax1.set_ylabel('Error (degrees) (F)', fontdict = fontdict_yaxis);
ax1.set_title('Model Error Comparison', fontdict = fontdict)

ax2.bar(x_values, model_comparison['accuracy'], color = ['b', 'r', 'g'], edgecolor = 'k', linewidth = 1.5)
ax2.set_ylim(bottom = 92, top = 94)
ax2.set_ylabel('Accuracy (%)', fontdict = fontdict_yaxis)
ax2.set_title('Model Accuracy Comparison', fontdict = fontdict)

ax3.bar(x_values, model_comparison['run_time (s)'], color = ['b', 'r', 'g'], edgecolor = 'k', linewidth = 1.5)
ax3.set_ylim(bottom = 0, top = 1)
ax3.set_ylabel('Run Time (sec)', fontdict = fontdict_yaxis)
ax3.set_title('Model Run-Time Comparison', fontdict = fontdict)
# plt.show()

features = pd.read_csv('data/temps_extended.csv')

features = pd.get_dummies(features)

labels = features['actual']
features = features.drop('actual', axis = 1)
feature_list = list(features.columns)

features = np.array(features)
labels = np.array(labels)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Test Labels Shape:', test_labels.shape)
print('{:0.1f} years of data in the trainging set'.format(train_features.shape[0] / 365.))
print('{:0.1f} years of data in the test set'.format(test_features.shape[0] / 365.))

important_feature_names = ['temp_1', 'average', 'ws_1', 'temp_2', 'friend', 'year']
important_indices = [feature_list.index(feature) for feature in important_feature_names]

important_train_features = train_features[:, important_indices]
important_test_features = test_features[:, important_indices]

print('Important train features shape:', important_train_features.shape)
print('Important test features shape:', important_test_features.shape)

train_features = important_train_features[:]
test_features = important_test_features[:]

feature_list = important_feature_names[:]

rf = RandomForestRegressor(random_state = 42)
pprint(rf.get_params())

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = [1.0, 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 20, num = 2)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf' : min_samples_leaf,
               'bootstrap' : bootstrap}
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                               n_iter = 100, scoring = 'neg_mean_absolute_error',
                               cv = 3, verbose = 2, random_state = 42, n_jobs = -1)
rf_random.fit(train_features, train_labels)
print(rf_random.best_params_)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors /test_labels)
    accuracy = 100 - mape

    print('Average temperature errors: ',np.mean(errors))
    print('Accuracy = {:0.2f}%'.format(accuracy))

base_model = RandomForestRegressor(random_state = 42)
base_model.fit(train_features, train_labels)
evaluate(base_model, test_features, test_labels)

best_random = rf_random.best_estimator_
evaluate(best_random, test_features, test_labels)


param_grid = {
    'bootstrap' : [True],
    'max_depth' : [8, 10, 12],
    'max_features' : [1.0],
    'min_samples_leaf' : [2, 3, 4, 5, 6],
    'min_samples_split' : [3, 5, 7],
    'n_estimators' : [800, 900, 1000, 1200]
}

rf = RandomForestRegressor()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                           scoring = 'neg_mean_absolute_error', cv = 3, 
                           n_jobs = -1, verbose = 2)

grid_search.fit(train_features, train_labels)

print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
evaluate(best_grid, test_features, test_labels)

param_grid = {
    'bootstrap' : [True],
    'max_depth' : [12, 15, None],
    'max_features' : [3, 4, 1.0],
    'min_samples_leaf' : [5, 6, 7],
    'min_samples_split' : [7, 10, 13],
    'n_estimators' : [900, 1000, 1200]
}

rf = RandomForestRegressor()

grid_search_ad = GridSearchCV(estimator = rf, param_grid=param_grid,
                              scoring='neg_mean_absolute_error', cv = 3,
                              n_jobs = -1, verbose = 2)

grid_search_ad.fit(train_features, train_labels)
print(grid_search_ad.best_params_)
best_grid_ad = grid_search_ad.best_estimator_
evaluate(best_grid_ad, test_features, test_labels)
print("the best model params:")
pprint(best_grid_ad.get_params())