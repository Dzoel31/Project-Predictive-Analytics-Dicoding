# -*- coding: utf-8 -*-
"""passenger_satisfaction_prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_RkPfBM2spJJxjgwDzJCkNc-1weN5V5C

# First Submission Project: Predictive Analysis of Airline Passenger Satisfaction

- Name:  Dzulfikri Adjmal
- Email: dzulfikriadjmal@gmail.com
- ID Dicoding: dzulfikriadjmal

## Import Library
"""

!pip install catboost

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

"""## Load Data"""

train_data = pd.read_csv('airline-passenger-satisfaction/train.csv', index_col=0)
test_data = pd.read_csv('airline-passenger-satisfaction/test.csv', index_col=0)

"""## Data Preprocessing"""

train_data.info()

test_data.info()

train_data.head()

test_data.head()

train_data.describe()

test_data.describe()

"""### Check Null Value"""

print("Null values in train data: ")
print(train_data.isna().sum()[train_data.isna().sum() > 0], "\n")
print("Null values in test data: ")
print(test_data.isna().sum()[test_data.isna().sum() > 0])

"""Terdapat missing value pada dataset, sehingga perlu dilakukan penanganan missing value. Pada data train terdapat missing value pada kolom `Arrival Delay in Minutes` dengan jumlah sebanyak 310, sedangkan pada data test di kolom yang sama terdapat 83 missing value. Jumlah missing value pada kedua dataset tidak terlalu banyak, sehingga dapat dilakukan penghapusan missing value.

### Check Duplicate Value
"""

print("Duplicated value in train data: ", train_data.duplicated().sum())
print("Duplicated value in test data: ", test_data.duplicated().sum())

"""### Check Outlier Value"""

exclude_features = ['id', 'satisfaction']
numerical_features = train_data[train_data.columns.difference(exclude_features)].select_dtypes(include=[np.number]).columns
categorical_features = train_data.select_dtypes(include='object').columns

plt.figure(figsize=(25, 25))
for i, feature in enumerate(numerical_features):
    plt.subplot(5, 4, i + 1)
    sns.boxplot(train_data[feature], orient='v' )
    plt.title(feature)
plt.show()

plt.figure(figsize=(25, 25))
for i, feature in enumerate(numerical_features):
    plt.subplot(5, 4, i + 1)
    sns.boxplot(test_data[feature])
    plt.title(feature)
plt.show()

"""## Data Cleaning"""

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

print("Null values in train data: ")
print(train_data.isna().sum()[train_data.isna().sum() > 0], "\n")
print("Null values in test data: ")
print(test_data.isna().sum()[test_data.isna().sum() > 0])

outlier_column = [
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

for column in outlier_column:
    train_data[column] = np.log(train_data[column] + 1)

plt.figure(figsize=(25, 25))
for i, feature in enumerate(numerical_features):
    plt.subplot(5, 4, i + 1)
    sns.boxplot(train_data[feature], orient="v")
    plt.title(feature)
plt.show()

outlier_column = [
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

for column in outlier_column:
    test_data[column] = np.log(test_data[column] + 1)

plt.figure(figsize=(25, 25))
for i, feature in enumerate(numerical_features):
    plt.subplot(5, 4, i + 1)
    sns.boxplot(test_data[feature])
    plt.title(feature)
plt.show()

"""## EDA (Exploratory Data Analysis)"""

plt.figure(figsize=(25, 25))
sns.pairplot(train_data[numerical_features[:5]])
plt.show()

plt.figure(figsize=(25, 25))
sns.pairplot(train_data[numerical_features[5:10]])
plt.show()

plt.figure(figsize=(25, 25))
sns.pairplot(train_data[numerical_features[10:]])
plt.show()

avg_rating_by_service = train_data.iloc[:, 7:20].mean(axis=0)
highlight_color = ['grey' if (x < avg_rating_by_service.mean()) else 'red' for x in avg_rating_by_service.values]

plt.figure(figsize=(10, 5))
sns.barplot(x=avg_rating_by_service.index, y=avg_rating_by_service.values, palette=highlight_color)
plt.xticks(rotation=90)
plt.title("Average rating by service")
plt.show()

avg_rating_by_service.round(2).sort_values(ascending=False).head()

"""Kode di atas digunakan untuk menghitung nilai rata-rata dari 13 layanan yang diberikan oleh maskapai penerbangan. Dari hasil perhitungan tersebut, dilihat kembali layanan yang mendapatkan rating lebih tinggi dari nilai rata-rata keseluruhan."""

plt.figure(figsize=(20, 15))
sns.heatmap(train_data[numerical_features].corr().round(2), annot=True, cmap='coolwarm')
plt.show()

f, ax = plt.subplots(2, 2, figsize=(15, 5))
sns.histplot(data=train_data, x='Flight Distance', hue='Inflight wifi service', ax=ax[0, 0])
sns.histplot(data=train_data, x='Flight Distance', hue='Inflight entertainment', ax=ax[0, 1])
sns.histplot(data=train_data, x='Flight Distance', hue='On-board service', ax=ax[1, 0])
sns.histplot(data=train_data, x='Flight Distance', hue='Seat comfort', ax=ax[1, 1])
plt.tight_layout()
plt.show()

"""Dari diagram histogram di atas, terlihat bahwa layanan Seat Comfort dan Inflight Entertainment memiliki korelasi yang tinggi dengan kepuasan penumpang. Hal ini dapat dilihat dari distribusi data, dimana penumpang yang menempuh perjalanan dengan jarak yang jauh cenderung memberikan rating yang tinggi pada kedua layanan tersebut.

## Data Preparation
"""

train_data['satisfaction'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title('Distribution of Satisfaction in Training Data')
plt.xlabel('Satisfaction')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Neutral or Dissatisfied', 'Satisfied'], rotation=0)
plt.show()

"""Dari jumlah kelas pada kedua dataset, dapat dilihat terjadi ketidakseimbangan jumlah kelas. Hal ini dapat mempengaruhi performa model yang akan dibuat. Untuk menangani ketidakseimbangan kelas, dapat dilakukan oversampling atau undersampling. Pada kasus ini, akan dilakukan oversampling dengan menggunakan metode SMOTE. Sebelum melakukan oversampling, perlu dilakukan encoding pada kolom kategorikal."""

dict_values= {}

for i, col in enumerate(categorical_features):
    dict_values[col] = train_data[col].unique()

dict_values

"""Sebelum melakukan pelatihan model, perlu dilakukan encoding pada kolom kategorikal. Pada kasus ini, akan digunakan metode Label Encoding dari library sklearn. Proses ini dilakukan pada kedua dataset. Menggunakan metode `copy()` untuk menghindari perubahan pada dataset asli."""

le = LabelEncoder()
train_data_encoded = train_data.copy()
test_data_encoded = test_data.copy()

for i, feature in enumerate(categorical_features):
    train_data_encoded[feature] = le.fit_transform(train_data[feature])
    test_data_encoded[feature] = le.fit_transform(test_data[feature])

dict_values_encode = {}

for i, col in enumerate(categorical_features):
    dict_values_encode[col] = train_data_encoded[col].unique()

dict_values_encode

"""Kode di bawah ini digunakan untuk melakukan oversampling pada data train menggunakan metode SMOTE. Proses ini dilakukan untuk menangani ketidakseimbangan kelas pada data train. Pertama, melakukan import SMOTE dari library imblearn. Kemudian membagi data train menjadi features dan target. Selanjutnya, melakukan inisiasi SMOTE dan melakukan proses oversampling pada data train."""

over_sample = SMOTE()

X = train_data_encoded.drop('satisfaction', axis=1)
y = train_data_encoded['satisfaction']

X, y = over_sample.fit_resample(X, y)

X = pd.DataFrame(X, columns=train_data_encoded.drop('satisfaction', axis=1).columns)
y = pd.Series(y)

train_data_encoded = pd.concat([X, y], axis=1)

train_data_encoded["satisfaction"].value_counts().plot(kind="bar", color=["blue", "orange"])
plt.title("Distribution of Satisfaction in Training Data")
plt.xlabel("Satisfaction")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["Neutral or Dissatisfied", "Satisfied"], rotation=0)
plt.show()

data_train_proportion = train_data_encoded.shape[0] / (train_data_encoded.shape[0] + test_data_encoded.shape[0])
data_test_proportion = test_data_encoded.shape[0] / (train_data_encoded.shape[0] + test_data_encoded.shape[0])

print("Proportion of data train: ", round(data_train_proportion,3))
print("Proportion of data test: ", round(data_test_proportion,3))

# Merge train and test data and split with 80:20 ratio
data = pd.concat([train_data_encoded, test_data_encoded], axis=0)
data.reset_index(drop=True, inplace=True)

X = data.drop('satisfaction', axis=1)
y = data['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Memastikan data train dan test memiliki proporsi yang sama
train_proportion = X_train.shape[0] / (X_train.shape[0] + X_test.shape[0])
test_proportion = X_test.shape[0] / (X_train.shape[0] + X_test.shape[0])

print("Proportion of train data: ", round(train_proportion, 3))
print("Proportion of test data: ", round(test_proportion, 3))

X_train.head()

X_test.head()

dict_model = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'CatBoost Classifier': CatBoostClassifier(verbose=0)
}

df_eval = pd.DataFrame(columns=['Accuracy Train', 'Accuracy Test', 'Precision Train', 'Precision Test'], index=dict_model.keys())

prediction_result = {}

for model_name, model in dict_model.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred)
    precision_train = precision_score(y_train, y_pred_train)
    precision_test = precision_score(y_test, y_pred)

    prediction_result[model_name] = y_pred

    df_eval.loc[model_name] = [accuracy_train, accuracy_test, precision_train, precision_test]

df_eval.reset_index(inplace=True)
df_eval.rename(columns={"index": "Model"}, inplace=True)
df_eval

df_accuracy = df_eval[['Model', 'Accuracy Train', 'Accuracy Test']].melt(id_vars='Model', var_name='Data', value_name='Accuracy')
df_precision = df_eval[['Model', 'Precision Train', 'Precision Test']].melt(id_vars='Model', var_name='Data', value_name='Precision')

plt.figure(figsize=(15, 5))
sns.barplot(x='Model', y='Accuracy', hue='Data', data=df_accuracy)
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

plt.figure(figsize=(15, 5))
sns.barplot(x='Model', y='Precision', hue='Data', data=df_precision)
plt.title('Precision Comparison')
plt.ylabel('Precision')
plt.show()

for key, value in prediction_result.items():
    print(f"Classification Report {key}")
    print(classification_report(y_test, value))

for key, value in prediction_result.items():
    matrix = confusion_matrix(y_test, value)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Neutral or Dissatisfied', 'Satisfied'], yticklabels=['Neutral or Dissatisfied', 'Satisfied'])
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title(f"Confusion Matrix {key}")
    plt.show()