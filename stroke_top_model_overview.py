import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from lazypredict import LazyClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE,SMOTEN,SMOTENC
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("full_data.csv")
target = "stroke"
x = data.drop([target,"gender"],axis=1)
y = data[target]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2024)

# Print the class distribution before balancing
print("Before balancing:")
print(y_train.value_counts(normalize=True))

categorical_features = ['ever_married', 'Residence_type', 'work_type', 'smoking_status']

categorical_features_indices = []

for feature_name in categorical_features:
    feature_index = x_train.columns.get_loc(feature_name)
    categorical_features_indices.append(feature_index)

ever_married_values = x_train["ever_married"].unique()
residence_type_values = x_train["Residence_type"].unique()
smoking_status_values = ["smokes","formerly smoked", "Unknown","never smoked"]


ord_transformer = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("ord",OrdinalEncoder(categories=[ever_married_values,residence_type_values,smoking_status_values]))
])

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())
])

nom_transformer = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("scaler",OneHotEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ("ord_feartures",ord_transformer,["ever_married","Residence_type","smoking_status"]),
    ("num_features",num_transformer,["age","hypertension","heart_disease","avg_glucose_level","bmi"]),
    ("nom_features",nom_transformer,["work_type"])
])

clf = Pipeline(steps=[
    ("preprocessor",preprocessor),
])

x_train = clf.fit_transform(x_train)
x_test = clf.transform(x_test)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)

























