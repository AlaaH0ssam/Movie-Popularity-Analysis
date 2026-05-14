####################################LIBRARIES###############################
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

################################# READING DATA ################################
df=pd.read_csv('train_data.csv')
print(df.head())
print(df.info())
print(df.shape)

########################### DISTRIBUTION OF POPULARITY ##########################
plt.figure()
plt.hist(df['popularity'], bins=50)
plt.title("Distribution of Popularity")
plt.xlabel("Popularity")
plt.ylabel("Frequency")
plt.show()
frequency_counts = df['popularity'].value_counts()
print(frequency_counts);

#################################### Preprocessing ############################################
########################### Data Cleaning & Handling Missing Target ##########################
df = df.drop_duplicates()
df = df.dropna(subset=['popularity'])

########################### Feature Engineering & Extraction ##########################
df['is_title_changed'] = (df['title'] != df['original_title']).astype(int)
df['title_length'] = [len(str(x)) for x in df['title'].fillna('')]
df['original_title_length'] = [len(str(x)) for x in df['original_title'].fillna('')]
df['theatrical'] = df['theatrical'].astype(int)
df['adult'] = df['adult'].astype(str).str.lower().map({'true':1,'false':0}).fillna(0)
df['has_backdrop'] = df['backdrop_path'].notna().astype(int)
df['has_homepage'] = df['homepage'].notna().astype(int)
df['has_poster'] = df['poster_path'].notna().astype(int)
df['overview_length'] = [len(str(x).split()) for x in df['overview'].fillna('')]
df['has_tagline'] = df['tagline'].notna().astype(int)

########################### Temporal Feature Extraction ################################
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_day'] = df['release_date'].dt.day
df['release_month'] = df['release_date'].dt.month
df['release_year'] = df['release_date'].dt.year

########################### Handling Multi-label Categorical Data ##########################
multiLabelFeatures = ['genres', 'production_companies', 'production_countries', 'spoken_languages']
for col in multiLabelFeatures:
    temp_list = df[col].fillna('').tolist()
    df['numOf_'+col] = [len(x.split(',')) if x != '' else 0 for x in temp_list]
    df['First_'+col] = [x.split(',')[0].strip() if x != '' else 'Unknown' for x in temp_list]

########################## Feature Selection (Manual) / Dropping Useless Feature######################
droppedColumns = [
    'id', 'imdb_id', 'homepage', 'poster_path', 'backdrop_path',
    'title', 'overview', 'tagline', 'release_date', 'original_title',
    'genres', 'production_companies', 'production_countries', 'spoken_languages'
]
df.drop(columns=droppedColumns, inplace=True, errors='ignore')

######################################### Data Splitting ######################################
X = df.drop(columns=['popularity'])
y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##################################### Handling Large Values (Skewness Correction) #################
columnsWithLargeNums=['revenue','budget','vote_count']
for col in columnsWithLargeNums:
    if col in X_train.columns:
        X_train[col] = np.log1p(X_train[col].clip(lower=0))
        X_test[col] = np.log1p(X_test[col].clip(lower=0))

################################### Handling Missing Values (Mean & Mode Imputation)##################
columnsWithNumericValues = X_train.select_dtypes(include='number').columns.tolist()
columnsWithCategoricalValues = X_train.select_dtypes(include='object').columns.tolist()
ms1Means = {}
ms1Modes = {}
ms1Freqs = {}
for col in columnsWithNumericValues:
    mean_value = X_train[col].mean()
    ms1Means[col] = mean_value
    X_train[col] = X_train[col].fillna(mean_value)
    X_test[col] = X_test[col].fillna(mean_value)

for col in columnsWithCategoricalValues:
    mode_value = X_train[col].mode()[0]
    ms1Modes[col] = mode_value
    X_train[col] = X_train[col].fillna(mode_value)
    X_test[col] = X_test[col].fillna(mode_value)

for col in columnsWithCategoricalValues:
    mostFreqCategory = X_train[col].value_counts().nlargest(50).index
    mostFreqCategoryDict = {
        category: category for category in mostFreqCategory
    }
    X_train[col] = X_train[col].map(mostFreqCategoryDict).fillna("Other")
    X_test[col] = X_test[col].map(mostFreqCategoryDict).fillna("Other")
    freqMap = X_train[col].value_counts()
    ms1Freqs[col] = freqMap
    X_train[col] = X_train[col].map(freqMap)
    X_test[col] = X_test[col].map(freqMap).fillna(0)

####################################### Outliers Treatment ################################
totalNumOfNumericalColumns = X_train.columns.tolist()
ms1Outliers = {}
for col in totalNumOfNumericalColumns:
    lower_bound = X_train[col].quantile(0.01)
    upper_bound = X_train[col].quantile(0.99)
    ms1Outliers[col] = {'lower': lower_bound, 'upper': upper_bound}
    X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)
    X_test[col] = X_test[col].clip(lower=lower_bound, upper=upper_bound)

########################### Feature Scaling (Standardization / Z-score Normalization) ##########################
ms1Scaling = {}
for col in totalNumOfNumericalColumns:
    meanValue = X_train[col].mean()
    stdValue = X_train[col].std()
    ms1Scaling[col] = {'mean': meanValue, 'std': stdValue}
    if stdValue != 0:
        X_train[col] = (X_train[col] - meanValue) / stdValue
        X_test[col] = (X_test[col] - meanValue) / stdValue

################################## Feature Selection ######################################################
# selector = SelectKBest(f_regression, k=20)
# X_train_final = selector.fit_transform(X_train, y_train)
# X_test_final = selector.transform(X_test)

#################################### Correlation Heatmap(Matrix) ########################################
temp_train_for_corr = X_train.copy()
temp_train_for_corr['popularity'] = y_train
top_correlated_features = temp_train_for_corr.corr()['popularity'].abs().sort_values(ascending=False).head(15).index
plt.figure(figsize=(12, 10))
sns.heatmap(temp_train_for_corr[top_correlated_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Top Features Correlated with Popularity")
plt.show()

######################################## Regression Models #####################################################################
########## Visualization (Actual vs Predicted) & (Error Distribution) & (Feature importance) & (Residual Plot)##################
def actual_vs_pred(y_true, y_pred, title):
    plt.scatter(y_true, y_pred, alpha=0.3)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)

def residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(title)

def error_distribution(y_true, y_pred, title):
    residuals = y_true - y_pred
    plt.hist(residuals, bins=40)
    plt.title(title)
    plt.xlabel("Error")
    plt.ylabel("Frequency")

def feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), np.array(feature_names)[indices])
    plt.title(title)

####################################### Random Forest Model ##############################
Rforset = RandomForestRegressor(
     n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
    )
Rforset.fit(X_train, y_train)
y_pred_train = Rforset.predict(X_train)
y_pred_test = Rforset.predict(X_test)

def print_metrics(y_true, y_pred, set_name="Test"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{set_name} Set Metrics:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")
    return mae, rmse, r2

print_metrics(y_train, y_pred_train, "Train")
print_metrics(y_test,  y_pred_test,  "Test")

################################# Random Forest Plotting ###########################
y_pred = Rforset.predict(X_test)
plt.figure(figsize=(14, 9))
plt.subplot(2, 2, 1)
actual_vs_pred(y_test, y_pred, "Random Forest - Actual vs Predicted")
plt.subplot(2, 2, 2)
residuals(y_test, y_pred, "Random Forest - Residuals")
plt.subplot(2, 2, 3)
error_distribution(y_test, y_pred, "Random Forest - Errors")
plt.subplot(2, 2, 4)
feature_importance(Rforset, X_train.columns, "Random Forest - Feature Importance")
plt.tight_layout(pad=4.0)
plt.show()

####################################### Decision Tree Model ##############################
dsntree = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=100,
    min_samples_leaf=50,
    random_state=42
)
dsntree.fit(X_train, y_train)
y_pred_train = dsntree.predict(X_train)
y_pred_test = dsntree.predict(X_test)
print_metrics(y_train, y_pred_train, "Train")
print_metrics(y_test,  y_pred_test,  "Test")

####################################### Decision Tree Plotting ##############################
plt.figure(figsize=(14, 9))

plt.subplot(2, 2, 1)
actual_vs_pred(y_test, y_pred_test, "Decision Tree - Actual vs Predicted")

plt.subplot(2, 2, 2)
residuals(y_test, y_pred_test, "Decision Tree - Residuals")

plt.subplot(2, 2, 3)
error_distribution(y_test, y_pred_test, "Decision Tree - Errors")

plt.subplot(2, 2, 4)
feature_importance(dsntree, X_train.columns, "Decision Tree - Feature Importance")

plt.tight_layout(pad=4.0)
plt.show()

############################# Saving Best Model MS1 #########################################
r2RandomForest = r2_score(y_test, Rforset.predict(X_test))
r2DecisionTree = r2_score(y_test, dsntree.predict(X_test))
if r2RandomForest > r2DecisionTree:
    bestRegressionModel = Rforset
    print("Best Regression Model is: Random Forest")
else:
    bestRegressionModel = dsntree
    print("Best Regression Model is: Decision Tree")
ms1SavedData = {
    'means': ms1Means,
    'modes': ms1Modes,
    'freqs': ms1Freqs,
    'outliers': ms1Outliers,
    'scales': ms1Scaling,
    'best_model': bestRegressionModel
}
pickle.dump(ms1SavedData, open("ms1_regression_data.pkl", "wb"))
print("MS1 Data successfully saved for the Test Script")
