############################### LIBRARIES #################################
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

############################### READING DATA #################################
testFilePath = "unseenTestSample.csv"
try:
    testScript = pd.read_csv(testFilePath, low_memory=False)
    print(f"\nSuccessfully loaded test data: {testFilePath}")
    print(f"Shape: {testScript.shape}")
except FileNotFoundError:
    print(f"\nError: Test file '{testFilePath}' not found in the current directory.")
    exit()

############################################# Feature Engineering & Extraction #############################
print("\nApplying Feature Engineering")
testScript['is_title_changed'] = (testScript['title'] != testScript['original_title']).astype(int)
testScript['title_length'] = [len(str(x)) for x in testScript['title'].fillna('')]
testScript['original_title_length'] = [len(str(x)) for x in testScript['original_title'].fillna('')]
testScript['theatrical'] = testScript['theatrical'].astype(int)
testScript['adult'] = testScript['adult'].astype(str).str.lower().map({'true':1,'false':0}).fillna(0)
testScript['has_backdrop'] = testScript['backdrop_path'].notna().astype(int)
testScript['has_homepage'] = testScript['homepage'].notna().astype(int)
testScript['has_poster'] = testScript['poster_path'].notna().astype(int)
testScript['overview_length'] = [len(str(x).split()) for x in testScript['overview'].fillna('')]
testScript['has_tagline'] = testScript['tagline'].notna().astype(int)
testScript['release_date'] = pd.to_datetime(testScript['release_date'], errors='coerce')
testScript['release_day'] = testScript['release_date'].dt.day
testScript['release_month'] = testScript['release_date'].dt.month
testScript['release_year'] = testScript['release_date'].dt.year

######################### Handling Multi-label Categorical Data #####################################
multiLabelFeatures = ['genres', 'production_companies', 'production_countries', 'spoken_languages']
for col in multiLabelFeatures:
    tempList = testScript[col].fillna('').tolist()
    testScript['numOf_'+col] = [len(x.split(',')) if x != '' else 0 for x in tempList]
    testScript['First_'+col] = [x.split(',')[0].strip() if x != '' else 'Unknown' for x in tempList]

######################### Feature Selection (Manual) / Dropping Useless Features #####################################
droppedColumns = [
    'id','imdb_id','homepage', 'poster_path', 'backdrop_path',
    'title', 'overview', 'tagline', 'release_date', 'original_title',
    'genres', 'production_companies', 'production_countries', 'spoken_languages'
]
testScript.drop(columns=droppedColumns, inplace=True, errors='ignore')

####################################### Classification Models MS2 ######################################################
if 'popularityLevel' in testScript.columns:
    print("\nClassification Models MS2")
    y_true = testScript['popularityLevel']
    X_test = testScript.drop(columns=['popularityLevel'])
    if 'popularity' in X_test.columns:
        X_test.drop(columns=['popularity'], inplace=True)
    try:
        with open("stepsForPreprocessing.pkl", "rb") as f:
            MS2Steps = pickle.load(f)
        with open("best_classification_model.pkl", "rb") as f:
            classModel = pickle.load(f)
    except FileNotFoundError as e:
         print(f"\nError loading pickle files for MS2: {e}")
         exit()
    print("Loading and applying preprocessing steps")
    y_true_encoded = MS2Steps['labelEncoder'].transform(y_true)

################################# Preproccesing ########################################
############################### Handling Missing Values ################################
    numericalColumnsValues = X_test.select_dtypes(include='number').columns
    categoricalColumnValues = X_test.select_dtypes(include='object').columns
    X_test[numericalColumnsValues] = X_test[numericalColumnsValues].fillna(MS2Steps['meanValues'])
    X_test[categoricalColumnValues] = X_test[categoricalColumnValues].fillna(MS2Steps['modeValues'])

#################################  Skewness Correction ##################################
    columnsWithLargeNums=['revenue','budget','vote_count']
    for col in columnsWithLargeNums:
        if col in X_test.columns:
            X_test[col] = np.log1p(X_test[col].clip(lower=0))

##################################### Frequency Encoding ################################
    for col in categoricalColumnValues:
        if col in MS2Steps['freqMaps']:
            X_test[col] = X_test[col].map(MS2Steps['freqMaps'][col]).fillna(0)

###################################### Outliers Treatment ###############################
    for col in X_test.columns:
        if col in MS2Steps['outlierBounds']:
            lower = MS2Steps['outlierBounds'][col]['lower']
            upper = MS2Steps['outlierBounds'][col]['upper']
            X_test[col] = X_test[col].clip(lower, upper)

##################################### Scaling & Alignment #################################
    expected_scaler_cols = MS2Steps['scaler'].feature_names_in_
    for col in expected_scaler_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test[expected_scaler_cols] = MS2Steps['scaler'].transform(X_test[expected_scaler_cols])

#################################### Feature Selection ####################################
    expected_all_cols = MS2Steps['selector'].feature_names_in_
    for col in expected_all_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test_for_selector = X_test[expected_all_cols]
    X_test_final = MS2Steps['selector'].transform(X_test_for_selector)
    print("Predicting using the best classification model")
    y_pred = classModel.predict(X_test_final)
    print("\n" + "*"*60)
    print(f"\tModel Used: {type(classModel).__name__}")
    print(f"\tClassification Accuracy: {accuracy_score(y_true_encoded, y_pred):.4f}")
    print("*"*60)

#################################### Regression Model MS1 ########################################
if 'popularity' in testScript.columns:
    print("\nRegression Models MS1")
    y_true = testScript['popularity']
    X_test = testScript.drop(columns=['popularity'])
    try:
        with open("ms1_regression_data.pkl", "rb") as f:
            MS1Data = pickle.load(f)
    except FileNotFoundError as e:
         print(f"\n[!] Error loading pickle files for MS1: {e}")
         exit()
    print("Loading and applying preprocessing steps")

########################################## Preproccessing #########################################
############################### Handling Missing Values ################################
    numericalColumnsValues = X_test.select_dtypes(include='number').columns
    categoricalColumnValues = X_test.select_dtypes(include='object').columns
    for col in numericalColumnsValues:
        if col in MS1Data['means']:
            X_test[col] = X_test[col].fillna(MS1Data['means'][col])
    for col in categoricalColumnValues:
        if col in MS1Data['modes']:
            X_test[col] = X_test[col].fillna(MS1Data['modes'][col])

#################################  Skewness Correction ##################################
    columnsWithLargeNums=['revenue','budget','vote_count']
    for col in columnsWithLargeNums:
        if col in X_test.columns:
             X_test[col] = np.log1p(X_test[col].clip(lower=0))

##################################### Frequency Encoding ################################
    for col in categoricalColumnValues:
        if col in MS1Data['freqs']:
            mostFreqCategoryDict = {k:k for k in MS1Data['freqs'][col].index[:50]}
            X_test[col] = X_test[col].map(mostFreqCategoryDict).fillna("Other")
            X_test[col] = X_test[col].map(MS1Data['freqs'][col]).fillna(0)

###################################### Outliers Treatment ##############################
    num_cols_updated = X_test.columns
    for col in num_cols_updated:
        if col in MS1Data['outliers']:
            lower = MS1Data['outliers'][col]['lower']
            upper = MS1Data['outliers'][col]['upper']
            X_test[col] = X_test[col].clip(lower, upper)

##################################### Scaling #############################################
    for col in num_cols_updated:
        if col in MS1Data['scales']:
            meanValue = MS1Data['scales'][col]['mean']
            stdValue = MS1Data['scales'][col]['std']
            if stdValue != 0:
                X_test[col] = (X_test[col] - meanValue) / stdValue

#################################### Feature Selection & Prediction #######################
    print("Predicting using the best regression model")
    regressionModel = MS1Data['best_model']
    modelColumnsSelected = regressionModel.feature_names_in_
    for col in modelColumnsSelected:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test_final = X_test[modelColumnsSelected]
    y_pred = regressionModel.predict(X_test_final)
    print("\n" + "*"*60)
    print(f"\tModel Used: {type(regressionModel).__name__}")
    print(f"\tRegression MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"\tRegression R2 Score: {r2_score(y_true, y_pred):.4f}")
    print("*"*60)
if not ('popularity' in testScript.columns or 'popularityLevel' in testScript.columns):
    print("\nError: Neither 'popularity' nor 'popularityLevel' columns were found in the dataset.")
    print("Cannot determine if the task is Regression or Classification.")