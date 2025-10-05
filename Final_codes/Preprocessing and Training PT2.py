import pandas as pd
import numpy as np
from fontTools.tfmLib import PASSTHROUGH
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer # Import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import matplotlib
import threading
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error


class preprocessing():
    def __init__(self):
        Trainer = training()
        print("Initiating Preprocessing... 1\n")
        self.preprocessing1()
        print("Training Random Forest Model 1... \n")
        Trainer.RandomForest1()
        print("Initiating Preprocessing... 2\n")
        self.preprocessing2()
        print("Training Random Forest Model 2... \n")
        Trainer.RandomForest2()

    def preprocessing1(self):
        data_dir = "../data/processed/"
        listdir = os.listdir(data_dir)

        BATTERY_METADATA = {
            "TEST_1_processed": {"capacity": 85, "charged": 85, "type": "b5"},
            "TEST_2_processed": {"capacity": 81.28, "charged": 81.28, "type": "b1"},
            "TEST_3_processed": {"capacity": 85, "charged": 85, "type": "b5"},
            "TEST_4_processed": {"capacity": 85, "charged": 85, "type": "b2"},
            "TEST_5_processed": {"capacity": 88.81, "charged": 88.81, "type": "b2"},
            "TEST_6_processed": {"capacity": 81.84, "charged": 81.84, "type": "b1"},
            "TEST_7_processed": {"capacity": 81.84, "charged": 36, "type": "b1"},
            "TEST_8_processed": {"capacity": 88.81, "charged": 27, "type": "b2"},
            "TEST_9_processed": {"capacity": 85, "charged": 80, "type": "tn1"},
            "TEST_10_processed": {"capacity": 85, "charged": 54, "type": "tn1"},
            "TEST_11_processed": {"capacity": 85, "charged": 85, "type": "b5"},
            "TEST_12_processed": {"capacity": 85, "charged": 67, "type": "b5"},
            "TEST_13_processed": {"capacity": 85, "charged": 85, "type": "b5"},
            "TEST_14_processed": {"capacity": 88.83, "charged": 52, "type": "b3"},
            "TEST_15_processed": {"capacity": 88.35, "charged": 70, "type": "b3"},
            "TEST_16_processed": {"capacity": 88.35, "charged": 61, "type": "b3"},
            "TEST_17_processed": {"capacity": 88.35, "charged": 88.35, "type": "b3"},
        }
        
        DATASET = pd.DataFrame()
        
        for i in listdir:
            for j in BATTERY_METADATA.keys():
                k = j + '.csv'
                if i == k:
                    dataset = pd.read_csv(data_dir + i)
                    dataset['type'] = BATTERY_METADATA[j]['type']
                    dataset['capacity'] = BATTERY_METADATA[j]['capacity']
                    dataset['charged'] = BATTERY_METADATA[j]['charged']
                    DATASET = pd.concat([DATASET, dataset], ignore_index=True)

        DATASET["discharge_rate"] = DATASET["Current"] / (DATASET["Voltage"] + 1e-6)
        DATASET["discharge_ratio"] = DATASET["Ah Out"] / (DATASET["charged"] + 1e-6)

        
        with open("../Final_codes/DATASET.csv", "w") as f:
            pd.DataFrame.to_csv(DATASET, f, index=False)



    def preprocessing2(self):
        dataset = pd.read_csv("../Final_codes/DATASET.csv")
        data = dataset.copy()
        data = data.drop(['Cumulative Actual Disch Ah', 'Time to Depletion'], axis=1)
        
        model = joblib.load("../Final_codes/battery_random_forest_model1.joblib")

        categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        for col in categorical_features:
            data[col] = categorical_imputer.fit_transform(data[[col]]).ravel()

        for col in categorical_features:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

        pred1 = {}
        pred2 = {}

        def predictor(model, data):
            for index, i in data.iterrows():
                x = pd.DataFrame(i).T
                pred1[index],pred2[index] = model.predict(x)[0]

        print("Initiating Model 1 Prediction over 6 parellel threads...")

        t1 = threading.Thread(target=predictor, args=(model, data.iloc[:1000, :]))
        t2 = threading.Thread(target=predictor, args=(model, data.iloc[1000:2000, :]))
        t3 = threading.Thread(target=predictor, args=(model, data.iloc[2000:3000, :]))
        t4 = threading.Thread(target=predictor, args=(model, data.iloc[3000:4000, :]))
        t5 = threading.Thread(target=predictor, args=(model, data.iloc[4000:5000, :]))
        t6 = threading.Thread(target=predictor, args=(model, data.iloc[5000:5437, :]))

        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t6.start()

        t1.join()
        t2.join()
        t3.join()
        t4.join()
        t5.join()
        t6.join()

        dataset['predicted TOD'] = pred1
        dataset['predicted CDC'] = pred2
        with open("../Final_codes/DATASET2.csv", "w") as f:
            f.write(
                dataset.to_csv(index=False)
            )


class training():

    def RandomForest1(self):
        TARGET_VARIABLE = ['Time to Depletion', 'Cumulative Actual Disch Ah']

        data = pd.read_csv("../Final_codes/DATASET.csv")

        Y = data[TARGET_VARIABLE]
        X = data.drop(TARGET_VARIABLE, axis=1)

        print("NaN locations (before imputation):")
        for column in data.columns:
            if data[column].isna().any():
                print(f"\n{column}:")
                print(data[data[column].isna()].index)


        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print("Numerical Featrures are : ", numerical_features)
        print("Categorical Featrures are : ", categorical_features)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

        numerical_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_features] = numerical_imputer.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = numerical_imputer.transform(X_test[numerical_features])

        categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        for col in categorical_features:
            X_train[col] = categorical_imputer.fit_transform(X_train[[col]]).ravel()
            X_test[col] = categorical_imputer.transform(X_test[[col]]).ravel()

        for col in categorical_features:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

        numerical_transformer = Pipeline(steps=[
            ('pass', 'passthrough') # You might want to scale numerical features
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_features)
        ], remainder='passthrough')

        rf_model = RandomForestRegressor(
            random_state=42,
            bootstrap=True,
            criterion='absolute_error',
            n_jobs=-1,
            n_estimators=200,
            max_depth=14,
            min_samples_split=3,
            min_samples_leaf=1,
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', rf_model)
        ])

        print("Initiating training ...")
        pipeline.fit(X_train, Y_train)

        Y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(Y_test, Y_pred)
        print(f"Mean Absolute Error is : {mae:.2f}")

        joblib.dump(pipeline, "../Final_codes/battery_random_forest_model1.joblib")


    def RandomForest2(self):
        TARGET_VARIABLE = ['Time to Depletion','Cumulative Actual Disch Ah']
        
        data = pd.read_csv("../Final_codes/DATASET2.csv")
        
        Y = data[TARGET_VARIABLE]
        X = data.drop(TARGET_VARIABLE, axis=1)

        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print("Numerical Featrures are : ", numerical_features)
        print("Categorical Featrures are : ", categorical_features)
        
        print("NaN locations (before imputation):")
        for column in data.columns:
            if data[column].isna().any():
                print(f"\n{column}:")
                print(data[data[column].isna()].index)

        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

        numerical_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_features] = numerical_imputer.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = numerical_imputer.transform(X_test[numerical_features])

        categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        for col in categorical_features:
            X_train[col] = categorical_imputer.fit_transform(X_train[[col]]).ravel()
            X_test[col] = categorical_imputer.transform(X_test[[col]]).ravel()

        for col in categorical_features:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler()) # You might want to scale numerical features
        ])
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_features)
        ], remainder='passthrough')
        
        rf_model = RandomForestRegressor(
            random_state=42,
            bootstrap=True,
            criterion='absolute_error',
            n_jobs=-1,
            n_estimators=100
        )
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', rf_model)
        ])

        print("Initiating training ...")
        pipeline.fit(X_train, Y_train)
        print("Training complete!")

        Y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(Y_test, Y_pred)
        print(f"Mean Absolute Error is : {mae:.2f}")

        joblib.dump(pipeline, "../Final_codes/battery_random_forest_model2.joblib")


if __name__ == "__main__":
    preprocessing()