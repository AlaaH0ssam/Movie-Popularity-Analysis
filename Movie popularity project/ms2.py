############################### LIBRARIES #################################
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

############################### READING DATA #################################
df = pd.read_csv("train_data.csv",low_memory=False)
print(df.head())
print(df.info())
print(df.shape)

############################### TARGET HANDLING ############################
if 'popularity' in df.columns:
    df.drop(columns=['popularity'], inplace=True)
print(df['popularityLevel'].value_counts())

############################### LABEL ENCODING (TARGET ONLY) ###############################
labelEncoderForTargetOnly = LabelEncoder()
df['popularityLevel'] = labelEncoderForTargetOnly.fit_transform(df['popularityLevel'])

############################### SPLIT DATA ###############################
X = df.drop(columns=['popularityLevel'])
y = df['popularityLevel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

########################### Feature Engineering & Extraction ###########################
for dfONtrainTest in [X_train, X_test]:
    dfONtrainTest['is_title_changed'] = (dfONtrainTest['title'] != dfONtrainTest['original_title']).astype(int)
    dfONtrainTest['title_length'] = [len(str(x)) for x in dfONtrainTest['title'].fillna('')]
    dfONtrainTest['original_title_length'] = [len(str(x)) for x in dfONtrainTest['original_title'].fillna('')]
    dfONtrainTest['theatrical'] = dfONtrainTest['theatrical'].astype(int)
    dfONtrainTest['adult'] = dfONtrainTest['adult'].astype(str).str.lower().map({'true':1,'false':0}).fillna(0)
    dfONtrainTest['has_backdrop'] = dfONtrainTest['backdrop_path'].notna().astype(int)
    dfONtrainTest['has_homepage'] = dfONtrainTest['homepage'].notna().astype(int)
    dfONtrainTest['has_poster'] = dfONtrainTest['poster_path'].notna().astype(int)
    dfONtrainTest['overview_length'] = [len(str(x).split()) for x in dfONtrainTest['overview'].fillna('')]
    dfONtrainTest['has_tagline'] = dfONtrainTest['tagline'].notna().astype(int)
    dfONtrainTest['release_date'] = pd.to_datetime(dfONtrainTest['release_date'], errors='coerce')
    dfONtrainTest['release_day'] = dfONtrainTest['release_date'].dt.day
    dfONtrainTest['release_month'] = dfONtrainTest['release_date'].dt.month
    dfONtrainTest['release_year'] = dfONtrainTest['release_date'].dt.year

######################### Handling Multi-label Categorical Data #####################################
multiLabelFeatures = ['genres', 'production_companies', 'production_countries', 'spoken_languages']
for col in multiLabelFeatures:
    for dfONtrainTest in [X_train, X_test]:
        newGeneratedList = dfONtrainTest[col].fillna('').tolist()
        dfONtrainTest['numOf_'+col] = [len(x.split(',')) if x != '' else 0 for x in newGeneratedList]
        dfONtrainTest['First_'+col] = [x.split(',')[0].strip() if x != '' else 'Unknown' for x in newGeneratedList]

######################### Feature Selection (Manual) / Dropping Useless Features #####################################
droppedColumns = [
    'id','imdb_id','homepage', 'poster_path', 'backdrop_path',
    'title', 'overview', 'tagline', 'release_date', 'original_title',
    'genres', 'production_companies', 'production_countries', 'spoken_languages'
]
X_train.drop(columns=droppedColumns, inplace=True, errors='ignore')
X_test.drop(columns=droppedColumns, inplace=True, errors='ignore')

########################### Handling Missing Values ###########################
columnsWithNumericValues = X_train.select_dtypes(include='number').columns
columnsWithCategoricalValues = X_train.select_dtypes(include='object').columns
meanValues = X_train[columnsWithNumericValues].mean()
modeValues = X_train[columnsWithCategoricalValues].mode().iloc[0]
X_train[columnsWithNumericValues] = X_train[columnsWithNumericValues].fillna(meanValues)
X_test[columnsWithNumericValues] = X_test[columnsWithNumericValues].fillna(meanValues)
X_train[columnsWithCategoricalValues] = X_train[columnsWithCategoricalValues].fillna(modeValues)
X_test[columnsWithCategoricalValues] = X_test[columnsWithCategoricalValues].fillna(modeValues)

########################### Handling Large Values (Skewness Correction) #################
columnsWithLargeNums=['revenue','budget','vote_count']
for col in columnsWithLargeNums:
    if col in X_train.columns:
        X_train[col] = np.log1p(X_train[col].clip(lower=0))
        X_test[col] = np.log1p(X_test[col].clip(lower=0))

########################### Frequancy Encoding ###########################
freqEncodingMaps = {}
for col in columnsWithCategoricalValues:
    freq = X_train[col].value_counts()
    freqEncodingMaps[col] = freq
    X_train[col] = X_train[col].map(freq)
    X_test[col] = X_test[col].map(freq).fillna(0)

########################### Outliers Treatment ###########################
columnsWithNumericValues = X_train.columns
outliersBounds = {}
for col in columnsWithNumericValues:
    lower = X_train[col].quantile(0.01)
    upper = X_train[col].quantile(0.99)
    outliersBounds[col] = {'lower': lower, 'upper': upper}
    X_train[col] = X_train[col].clip(lower, upper)
    X_test[col] = X_test[col].clip(lower, upper)

########################### Feature Scaling (Standardization / Z-score Normalization) ###########################
scaler = StandardScaler()
X_train[columnsWithNumericValues] = scaler.fit_transform(X_train[columnsWithNumericValues])
X_test[columnsWithNumericValues] = scaler.transform(X_test[columnsWithNumericValues])

########################### Feature Selection ###########################
columnsWithConstantValues = [col for col in X_train.columns if X_train[col].nunique() <= 1]
X_train.drop(columns=columnsWithConstantValues, inplace=True)
X_test.drop(columns=columnsWithConstantValues, inplace=True)

selector = SelectKBest(score_func=f_classif, k=20)
X_train_final = selector.fit_transform(X_train, y_train)
X_test_final = selector.transform(X_test)
selectedFeatures = X_train.columns[selector.get_support()]

########################### Saving And Loading Our Steps Using Pickle ###########################
stepsForApplyingPreprocessing = {
    "labelEncoder": labelEncoderForTargetOnly,
    "meanValues": meanValues,
    "modeValues": modeValues,
    "freqMaps": freqEncodingMaps,
    "outlierBounds": outliersBounds,
    "scaler": scaler,
    "selector": selector,
    "selectedFeatures": selectedFeatures,
    "droppedConstantCols": columnsWithConstantValues
}
pickle.dump(stepsForApplyingPreprocessing, open("stepsForPreprocessing.pkl", "wb"))


################################ Classification Models ##########################################
################################## RESULTS STORAGE ##############################################
modelAccuracyResults = {}
trainTimeResults = {}
testTimeResults = {}
################################### Logistic Regression ###############################################
print("\n"+"*"*80 )
print(f"{'Logistic Regression':^80}")
print("*"*80 + "\n")
logisticRegressionModel = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

start_train = time.time()
logisticRegressionModel.fit(X_train_final, y_train)
end_train = time.time()

start_test = time.time()
y_predLogisticRegression = logisticRegressionModel.predict(X_test_final)
end_test = time.time()

accuracyLogisticRegression = accuracy_score(y_test, y_predLogisticRegression)
modelAccuracyResults['Logistic Regression'] = accuracyLogisticRegression
trainTimeResults['Logistic Regression'] = end_train - start_train
testTimeResults['Logistic Regression'] = end_test - start_test

print(f"Training Time: {trainTimeResults['Logistic Regression']:.2f} seconds")
print(f"Test Time: {testTimeResults['Logistic Regression']:.2f} seconds")
print("Accuracy:", accuracyLogisticRegression)
print("\nClassification Report:\n", classification_report(y_test, y_predLogisticRegression))

################################### Linear SVC ###############################################
print("\n"+"*"*80 )
print(f"{'Linear SVC':^80}")
print("*"*80 + "\n")
linearSVC = LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)

start_train = time.time()
linearSVC.fit(X_train_final, y_train)
end_train = time.time()

start_test = time.time()
y_predLinearSVC = linearSVC.predict(X_test_final)
end_test = time.time()

accuracyLinearSVC = accuracy_score(y_test, y_predLinearSVC)
modelAccuracyResults['Linear SVC'] = accuracyLinearSVC
trainTimeResults['Linear SVC'] = end_train - start_train
testTimeResults['Linear SVC'] = end_test - start_test

print(f"Training Time: {trainTimeResults['Linear SVC']:.2f} seconds")
print(f"Test Time: {testTimeResults['Linear SVC']:.2f} seconds")
print("Accuracy:", accuracyLinearSVC)
print("\nClassification Report:\n", classification_report(y_test, y_predLinearSVC))

################################### Linear SVC Confusion Matrix ###############################################
confusionMatrixLinearSVC = confusion_matrix(y_test, y_predLinearSVC)
classNames = labelEncoderForTargetOnly.classes_
plt.figure(figsize=(6, 4))
sns.heatmap(confusionMatrixLinearSVC, annot=True,fmt='d',cmap='Blues', xticklabels=classNames,yticklabels=classNames,cbar=True)
plt.title('Confusion Matrix - Linear SVC', fontsize=14, pad=20)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.tight_layout()
plt.show()

################################### Decision Tree ###############################################
print("\n"+"*"*80 )
print(f"{'Decision Tree':^80}")
print("*"*80 + "\n")
decisonTreeModel = DecisionTreeClassifier(random_state=42,class_weight='balanced')

start_train = time.time()
decisonTreeModel.fit(X_train_final, y_train)
end_train = time.time()

start_test = time.time()
y_predDecisionTree = decisonTreeModel.predict(X_test_final)
end_test = time.time()

accuracyDecisionTree = accuracy_score(y_test, y_predDecisionTree)
modelAccuracyResults['Decision Tree'] = accuracyDecisionTree
trainTimeResults['Decision Tree'] = end_train - start_train
testTimeResults['Decision Tree'] = end_test - start_test

print(f"Training Time: {trainTimeResults['Decision Tree']:.2f} seconds")
print(f"Test Time: {testTimeResults['Decision Tree']:.2f} seconds")
print("Accuracy:", accuracyDecisionTree)
print("\nClassification Report:\n", classification_report(y_test, y_predDecisionTree))

################################### Tuned Decision Tree ###############################################
print("\n"+"*"*80 )
print(f"{'Tuned Decision Tree':^80}")
print("*"*80 + "\n")

#Validation Set
X_tune_dt, _, y_tune_dt, _ = train_test_split(X_train_final, y_train, train_size=0.1, random_state=42, stratify=y_train)

param_grid = {
    'max_depth': [8, 10, 12, 15],
    'min_samples_leaf': [5, 10, 15, 20]
}
tunedDecisionTree = GridSearchCV(DecisionTreeClassifier(random_state=42, class_weight='balanced'), param_grid=param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
tunedDecisionTree.fit(X_tune_dt, y_tune_dt)
cvResults = pd.DataFrame(tunedDecisionTree.cv_results_)
depthImpact = cvResults.groupby('param_max_depth')['mean_test_score'].mean()
print("\nImpact of max_depth on Decision Tree")
print(depthImpact.to_frame(name='Validation Score'))
print("-" * 50)
plt.figure(figsize=(8, 5))
plt.plot(depthImpact.index, depthImpact.values, marker='o', color='purple', linewidth=2, markersize=8)
plt.title('Impact of max_depth on Model Performance (Decision Tree)', fontsize=14)
plt.xlabel('max_depth Value', fontsize=12)
plt.ylabel('Validation Score (F1 Weighted)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
print("Best Parameters for Decision Tree:", tunedDecisionTree.best_params_)

decisionTreeModelEstimator = tunedDecisionTree.best_estimator_
start_train = time.time()
decisionTreeModelEstimator.fit(X_train_final, y_train)
end_train = time.time()

start_test = time.time()
y_predTunningDecisionTree = decisionTreeModelEstimator.predict(X_test_final)
end_test = time.time()

accTuneddecisionTree = accuracy_score(y_test, y_predTunningDecisionTree)
modelAccuracyResults['Tuned Decision Tree'] = accTuneddecisionTree
trainTimeResults['Tuned Decision Tree'] = end_train - start_train
testTimeResults['Tuned Decision Tree'] = end_test - start_test

print(f"Training Time: {trainTimeResults['Tuned Decision Tree']:.2f} seconds")
print(f"Test Time: {testTimeResults['Tuned Decision Tree']:.2f} seconds")
print("Accuracy:",accTuneddecisionTree)
print("\nClassification Report:\n", classification_report(y_test, y_predTunningDecisionTree))

################################### Random Forest ##############################################
print("\n"+"*"*80 )
print(f"{'Random Forest':^80}")
print("*"*80 + "\n")
randomForestModel = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)

start_train = time.time()
randomForestModel.fit(X_train_final, y_train)
end_train = time.time()

start_test = time.time()
y_predRandomForest = randomForestModel.predict(X_test_final)
end_test = time.time()

accuracyRandomForest = accuracy_score(y_test, y_predRandomForest)
modelAccuracyResults['Random Forest'] = accuracyRandomForest
trainTimeResults['Random Forest'] = end_train - start_train
testTimeResults['Random Forest'] = end_test - start_test

print(f"Training Time: {trainTimeResults['Random Forest']:.2f} seconds")
print(f"Test Time: {testTimeResults['Random Forest']:.2f} seconds")
print("Accuracy:",accuracyRandomForest)
print(classification_report(y_test, y_predRandomForest))
print(confusion_matrix(y_test, y_predRandomForest))

################################### Random Forest Confusion Matrix ###############################################
confusionMatrixRandomForest = confusion_matrix(y_test, y_predRandomForest)
classNames = labelEncoderForTargetOnly.classes_
plt.figure(figsize=(6, 4))
sns.heatmap(confusionMatrixRandomForest,annot=True,fmt='d',cmap='Blues',xticklabels=classNames, yticklabels=classNames,cbar=True)
plt.title('Confusion Matrix - Random Forest', fontsize=14, pad=20)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.tight_layout()
plt.show()

################################### Tuned Random Forest ###############################################
print("\n"+"*"*80 )
print(f"{'Tuned Random Forest':^80}")
print("*"*80 + "\n")

#Validation Set
X_tune_rf, _, y_tune_rf, _ = train_test_split(X_train_final, y_train, train_size=0.1, random_state=42, stratify=y_train)

param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [10, 15, 20],
    "min_samples_leaf": [1, 3, 5]
}
tunningRandomForest = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
gridTunningRandomForest = RandomizedSearchCV(tunningRandomForest, param_distributions=param_grid, n_iter=4, cv=2, scoring='f1_weighted', random_state=42)

gridTunningRandomForest.fit(X_tune_rf, y_tune_rf)
cvResultsRandomForest = pd.DataFrame(gridTunningRandomForest.cv_results_)
estimatorsImpact = cvResultsRandomForest.groupby('param_n_estimators')['mean_test_score'].mean()
print("\nImpact of n_estimators on Random Forest")
print(estimatorsImpact.to_frame(name='Validation Score'))
print("-" * 50)
plt.figure(figsize=(8, 5))
plt.plot(estimatorsImpact.index, estimatorsImpact.values, marker='s', color='green', linewidth=2, markersize=8)
plt.title('Impact of n_estimators on Model Performance (Random Forest)', fontsize=14)
plt.xlabel('Number of Trees (n_estimators)', fontsize=12)
plt.ylabel('Validation Score (F1 Weighted)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
print("Best params for Random Forest:", gridTunningRandomForest.best_params_)

bestTunningRandomForestModel = gridTunningRandomForest.best_estimator_

start_train = time.time()
bestTunningRandomForestModel.fit(X_train_final, y_train)
end_train = time.time()

start_test = time.time()
y_predTunningRandomForest = bestTunningRandomForestModel.predict(X_test_final)
end_test = time.time()

accuracyTunedRandomForest = accuracy_score(y_test, y_predTunningRandomForest)
modelAccuracyResults['Tuned Random Forest'] = accuracyTunedRandomForest
trainTimeResults['Tuned Random Forest'] = end_train - start_train
testTimeResults['Tuned Random Forest'] = end_test - start_test

print(f"Training Time: {trainTimeResults['Tuned Random Forest']:.2f} seconds")
print(f"Test Time: {testTimeResults['Tuned Random Forest']:.2f} seconds")
print("Accuracy:", accuracyTunedRandomForest)
print(classification_report(y_test, y_predTunningRandomForest))
print(confusion_matrix(y_test, y_predTunningRandomForest))

################################### Tuned Random Forest Confusion Matrix ###############################################
confusionMatrixTuned = confusion_matrix(y_test, y_predTunningRandomForest)
classNames = labelEncoderForTargetOnly.classes_
plt.figure(figsize=(6, 4))
sns.heatmap(confusionMatrixTuned,annot=True, fmt='d',cmap='Blues', xticklabels=classNames,yticklabels=classNames,cbar=True)
plt.title('Confusion Matrix -  Tuned Random Forest', fontsize=14, pad=20)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.tight_layout()
plt.show()

################################### Visualization Results(3 Bar Graphs) ###############################################
modelNamesVisualization = list(modelAccuracyResults.keys())
colors = ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f', '#edc949']
#################################### Classification Accuracy #############################################################
plt.figure(figsize=(10, 5))
plt.bar(modelNamesVisualization, modelAccuracyResults.values(), color=colors)
plt.title('Classification Accuracy Comparison', fontsize=14)
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(modelAccuracyResults.values()):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#################################### Training Time #############################################################
plt.figure(figsize=(10, 5))
plt.bar(modelNamesVisualization, trainTimeResults.values(), color=colors)
plt.title('Total Training Time Comparison (Seconds)', fontsize=14)
plt.ylabel('Time (Seconds)')
for i, v in enumerate(trainTimeResults.values()):
    plt.text(i, v + 0.1, f"{v:.2f}s", ha='center', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#################################### Testing Time #############################################################
plt.figure(figsize=(10, 5))
plt.bar(modelNamesVisualization, testTimeResults.values(), color=colors)
plt.title('Total Test Time Comparison (Seconds)', fontsize=14)
plt.ylabel('Time (Seconds)')
for i, v in enumerate(testTimeResults.values()):
    plt.text(i, v + 0.01, f"{v:.4f}s", ha='center', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


########################### Saving The Best Model MS2 ###########################
classificationModels = {
    'Logistic Regression': logisticRegressionModel,
    'Linear SVC': linearSVC,
    'Decision Tree': decisonTreeModel,
    'Tuned Decision Tree': decisionTreeModelEstimator,
    'Random Forest': randomForestModel,
    'Tuned Random Forest': bestTunningRandomForestModel
}
bestModelName = max(modelAccuracyResults, key=modelAccuracyResults.get)
bestModelObject = classificationModels[bestModelName]

print("\n"+"*"*80 )
print(f"\t\tThe Best Model is: {bestModelName} with Accuracy: {modelAccuracyResults[bestModelName]:.4f}")
print("*"*80+ "\n")
pickle.dump(bestModelObject, open("best_classification_model.pkl", "wb"))
print("All steps completed successfully Best model saved.")