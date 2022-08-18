# -*- coding: utf-8 -*-
"""
Created on Fri May 20 17:32:48 2022

@author: AKalish
"""


import warnings
warnings.filterwarnings('ignore')
import datetime
from functools import partial
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score, silhouette_samples, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

pd.options.mode.use_inf_as_na = True
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

################ Importing the Dataset   ######################

cols = ["age", "gender", "country", "visiting Wuhan", "from Wuhan", "death", "recovered", "location", "symptom_onset", "hosp_visit_date"]

df = pd.read_csv("COVID19_line_list_data.csv", sep=",", usecols=cols, na_values=['nan'])

################ Cleaning of the Dataset   ######################

for i in range(0, len(df.death)):
    temp = df.death[i]
    if df.death[i] != "1" and df.death[i] != "0":
        df.death[i] = "1"

for i in range(0, len(df.recovered)):
    temp = df.recovered[i]
    if df.recovered[i] != "1" and df.recovered[i] != "0":
        df.recovered[i] = "1"

outcome = []

for i in range(0, len(df)):
    if (df.death[i] == "1"):
        if (df.recovered[i] == "0"):
            outcome.append(0)
    elif (df.recovered[i] == "1"):
        if (df.death[i] == "0"):
            outcome.append(1)
    else:
        outcome.append(2)
        
        
onsets = []

df["symptom_onset"].fillna(0, inplace=True)
df["hosp_visit_date"].fillna(0, inplace=True)
df["gender"].fillna("NA", inplace=True)


for i in range(0, len(df)):
    if (df["symptom_onset"][i] != 0):
        df["symptom_onset"][i] = 1
    if (df["hosp_visit_date"][i] != 0):
        df["hosp_visit_date"][i] = 1

df['outcome'] = outcome
    

dataMean = df.mean()
dataMedian = df.median()
dataMode = df.mode()

means = {"age" : dataMean["age"]}
medians = {"age" : dataMedian["age"]}
modes = {"age" : dataMode["age"]}

df["age"].fillna(value=means.get("age"))

for i in range(0, len(df)):
    if (df["gender"][i] == "NA"):
        df = df.drop(labels=i, axis=0)

#################### Preparing the Dataset ##########################

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

enc.fit(df['location'])
df['location'] = enc.transform(df['location'])
enc.fit(df['country'])
df['country'] = enc.transform(df['country'])
enc.fit(df['gender'])
df['gender'] = enc.transform(df['gender'])
enc.fit(df['age'])
df['age'] = enc.transform(df['age'])
enc.fit(df['visiting Wuhan'])
df['visiting Wuhan'] = enc.transform(df['visiting Wuhan'])
enc.fit(df['from Wuhan'])
df['from Wuhan'] = enc.transform(df['from Wuhan'])
enc.fit(df['recovered'])
df['recovered'] = enc.transform(df['recovered'])
enc.fit(df['death'])
df['death'] = enc.transform(df['death'])
enc.fit(df['symptom_onset'])
df['symptom_onset'] = enc.transform(df['symptom_onset'])
enc.fit(df['hosp_visit_date'])
df['hosp_visit_date'] = enc.transform(df['hosp_visit_date'])

############################# Part 1A ###############################

correlation_matrix = df.corr()
print(correlation_matrix)
print()

############################# Part 1B ###############################

X =  df.drop(columns='outcome')
all_features = X.columns
y = df['outcome']

# uncomment for a really cool chart
chart1 = alt.Chart(df).mark_point().encode(x='age', y='country', size='gender:N', color='outcome:N').interactive().properties(
    width=600,
    height=400)
chart1.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

twod_pca = PCA(n_components=2)
X_pca = twod_pca.fit_transform(X_scaled)

df['pca1'] = X_pca[:,0]
df['pca2'] = X_pca[:,1]

df['member'] = 1
df.groupby('outcome')['member'].transform('count').div(df.shape[0])

selection = alt.selection_multi(fields=['outcome'], bind='legend')

df['quality'] = df['outcome'].map(df['outcome'].value_counts(normalize=True).to_dict())

chart2 = alt.Chart(df.sample(900, weights='quality')).mark_circle(size=60).encode(
    x=alt.X('pca1', title='Principal Component 1'),
    y=alt.Y('pca2', title='Principal Component 2'),
    color=alt.Color('outcome:N'),
    tooltip=['outcome'],
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
).properties(
    title='PCA analysis of COVID19 data',
    width=600,
    height=400
).add_selection(
    selection
).interactive()

chart2.show()

sklearn_pca = sk.decomposition.PCA()
sklearn_pca.fit(df[cols])
print('Variance percent explained', sklearn_pca.explained_variance_ratio_)
print('Feature Names in order greatest to least of influence:', sklearn_pca.feature_names_in_)
print()

############################# Part 2A ###############################
df = df.drop(axis=1, labels=['pca1', 'pca2', 'member', 'quality', 'recovered'])

wuhanSymptoms_model = BayesianNetwork([("symptom_onset", "visiting Wuhan")])
wuhanSymptoms_model.fit(df)
inf = VariableElimination(wuhanSymptoms_model)

print(inf.query(variables=['symptom_onset'], evidence={'visiting Wuhan': 1}))
print()
############################# Part 2B ################################

hospitalSymptomWuhan_model = BayesianNetwork([("visiting Wuhan", "symptom_onset")])
hospitalSymptomWuhan_model.add_node("hosp_visit_date")
hospitalSymptomWuhan_model.fit(df)
inf = VariableElimination(hospitalSymptomWuhan_model)
print(inf.query(variables=['hosp_visit_date'], evidence={'symptom_onset': 1, 'visiting Wuhan': 1}))
print()
############################# Part 2C ################################

wuhanDeath_model = BayesianNetwork([("death", "visiting Wuhan")])
wuhanDeath_model.fit(df)
inf = VariableElimination(wuhanDeath_model)

print(inf.query(variables=['death'], evidence={'visiting Wuhan': True}))
print()
############################# Part 2D ################################


df2 = pd.read_csv("COVID19_line_list_data.csv", sep=",", usecols=['recovered', 'hosp_visit_date', 'visiting Wuhan'], na_values=['nan', '0', '12/30/1899'])


indexes = []

for i in range(len(df2)):
    if df2['recovered'][i] == '1':
        df2['recovered'][i] = '12/30/1899'
        indexes.append(i)
 
for i in indexes:
    df2.drop(axis=0, index=i, inplace=True)

df2 = df2[df2['hosp_visit_date'].notna()]
df2 = df2[df2['recovered'].notna()]
df2 = df2[df2['visiting Wuhan'].notna()]


difference = (pd.to_datetime(df2.iloc[:,2]) - pd.to_datetime(df2.iloc[:,0]))
difference = [pd.Timedelta(x) for x in difference]

df2 = df2.assign(time=difference)



wuhanTime_model = BayesianNetwork([("time", "visiting Wuhan")])
wuhanTime_model.fit(df2)
inf = VariableElimination(wuhanTime_model)

table = inf.query(variables=['time'], evidence={'visiting Wuhan': True})

print("Probability Table for recovery time if visited Wuhan:")
print(table)
print()

print("Average Time to recover if visited Wuhan")
print(df2['time'].mean())
print()

############################# Part 3A ################################

X = df
y = df['outcome']
X_train, X_test, Y_train, Y_test = split(X, y)
X_train, X_test, Y_train, Y_test = split(X_train, Y_train, test_size=0.2, random_state=10)

knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

# print(Y_pred)
print("Accuracy for predicting the outcome: " + str(accuracy_score(Y_test, Y_pred)))
print()

############################# Part 3B ################################

print("Correlation Matrix for age: ")
print(correlation_matrix['age'])
print()

withoutAge = df.columns.values.tolist()
withoutAge.remove("age")
withoutAge.remove("symptom_onset")
withoutAge.remove("location")
withoutAge.remove("hosp_visit_date")
withoutAge.remove("visiting Wuhan")
withoutAge.remove("from Wuhan")
withoutAge.remove("outcome")

X = df[withoutAge]
y = df[["age"]]

scaler2 = StandardScaler()
scaler2.fit(X)
scaler2.transform(X)

X_train, X_test, Y_train, Y_test = split(X, y)
X_train, X_test, Y_train, Y_test = split(X_train, Y_train, test_size=0.2, random_state=10)

model = LinearRegression()
model.fit(X_train, Y_train)

l = [x for x in withoutAge]

string = ""

for item in l:
    string += item
    string += ", "
    
string = string[:-1]


print("MSE of predicting age based on " + string[:-1] + ":")
pred_Y = model.predict(X_test)
print(mean_squared_error(Y_test, pred_Y, squared=False))
print()

############################# Part 3C ################################

range_n_clusters = [4, 5, 6, 7, 8]

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

data = df

pca = PCA(2)
 
#Transform the data
df5 = pca.fit_transform(data)
 
#Initialize the class object
kmeans = KMeans(n_clusters= 6)
 
#predict the labels of clusters.
label = kmeans.fit_predict(df)
 
#Getting unique labels
u_labels = np.unique(label)
# centroids = kmeans.cluster_centers_
 
# plotting the results:
for i in u_labels:
    plt.scatter(df5[label == i , 0] , df5[label == i , 1] , label = i)
# plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()
print()


############################# Part 4A ################################

X = df.drop(columns=['outcome'])
y = df['outcome']

X_train, X_test, Y_train, Y_test = split(X, y)
X_train, X_test, Y_train, Y_test = split(X_train, Y_train, test_size=0.2, random_state=10)

svc_model = SVC(class_weight='balanced', probability=True)

svc_model.fit(X_train, Y_train)

svc_predict = svc_model.predict(X_test)# check performance
pred_prob = svc_model.predict_proba(X_test)
roc = roc_auc_score(Y_test, pred_prob, multi_class='ovr')
print('ROCAUC score of balanced model:',roc)
acc = accuracy_score(Y_test, svc_predict)
print('Accuracy score of balanced model:',acc)
f = f1_score(Y_test, svc_predict, average='micro')
print('F1 score of balanced model:',f)
print()


############################# Part 4B ################################

# How to take better care of missing values??

# In the report

############################# Part 4C ################################

X = df.drop(columns=['outcome'])
y = df['outcome']

X_train, X_test, Y_train, Y_test = split(X, y)
X_train, X_test, Y_train, Y_test = split(X_train, Y_train, test_size=0.2, random_state=10)

pipelineRFC = make_pipeline(StandardScaler(), RandomForestClassifier(criterion='gini', random_state=10))

param_grid_rfc = [{
    'randomforestclassifier__max_depth':[2, 3, 4],
    'randomforestclassifier__max_features':[2, 3, 4, 5, 6]
}]

gsRFC = GridSearchCV(estimator=pipelineRFC,
                      param_grid = param_grid_rfc,
                      scoring='balanced_accuracy',
                      cv=10,
                      refit=True,
                      n_jobs=1)

# Train the RandomForestClassifier
gsRFC = gsRFC.fit(X_train, Y_train)

# Print the training score of the best model
print("Training score of the best model:" + str(gsRFC.best_score_))
print(gsRFC.best_score_)

# Print the model parameters of the best model
print(gsRFC.best_params_)

# Print the test score of the best model
clfRFC = gsRFC.best_estimator_
print('Test accuracy of the best model: %.3f' % clfRFC.score(X_test, Y_test))

# Print the MSE of the best model
pred_Y = gsRFC.predict(X_test)
print("MSE of the best model: " + str(mean_squared_error(Y_test, pred_Y, squared=False)))

print(pred_Y)

############################# Part 4D ################################

# In the report