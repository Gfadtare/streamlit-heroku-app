#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Load the dataset
df = pd.read_csv('gga.csv')
df.head()


# In[4]:


df.fillna(df.mean(), inplace=True)


# In[5]:


# Split data into features and target for both models
X = df.drop(columns=['Personal Loan', 'ID'])
y_loan = df['Personal Loan']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[6]:


# 1. Drop irrelevant columns
df_cleaned = df.drop(columns=['ID', 'ZIP Code'])


# In[7]:


# 3. Feature Scaling using Standardization (Z-score normalization)
scaler = StandardScaler()
scaled_columns = ['Income', 'CCAvg', 'Mortgage']
df_cleaned[scaled_columns] = scaler.fit_transform(df_cleaned[scaled_columns])


# In[8]:


# 4. Outlier Detection and Removal
def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


# In[9]:


df_cleaned = remove_outliers(df_cleaned, scaled_columns)


# In[10]:


# 5. Feature Engineering

# 1. Income per Family Member
df_cleaned['Income_Per_Family'] = df_cleaned['Income'] / df_cleaned['Family']

# 2. Experience Level (Example thresholds: Junior: 0-5 years, Mid: 6-15 years, Senior: 16+ years)
def experience_level(experience):
    if experience <= 5:
        return 'Junior'
    elif 6 <= experience <= 15:
        return 'Mid'
    else:
        return 'Senior'

df_cleaned['Experience_Level'] = df_cleaned['Experience'].apply(experience_level)

# Convert Experience_Level to dummy variables
df_cleaned = pd.get_dummies(df_cleaned, columns=['Experience_Level'], drop_first=True)

# 3. High Mortgage Indicator
# Assuming a threshold of $100,000 (normalized value will depend on your data)
df_cleaned['High_Mortgage'] = (df_cleaned['Mortgage'] > 100000).astype(int)

# Check the newly engineered features
print("Engineered Data Head:")
print(df_cleaned.head())


# In[11]:


# 6. Splitting the Data
X = df_cleaned.drop(columns=['Personal Loan'])  # Features
y = df_cleaned['Personal Loan']  # Target


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


print("Cleaned Data Head:")
print(df_cleaned.head())

print("\nTraining Set Shape:", X_train.shape, y_train.shape)
print("Testing Set Shape:", X_test.shape, y_test.shape)


# # Advanced Feature Engineering 

# In[15]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Convert the polynomial features back to a DataFrame to keep track of feature names
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

# 2. Principal Component Analysis (PCA)
pca = PCA(n_components=10)  # Reduce to 10 principal components
X_pca = pca.fit_transform(X_poly_df)

# Convert the PCA features back to a DataFrame
X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# View the updated dataset with PCA features
print("PCA-Transformed Dataset:")
print(X_pca_df.head())

# Save the PCA-transformed dataset to a CSV file
X_pca_df.to_csv('pca_transformed_data.csv', index=False)


# In[16]:


#3. Train-test split using PCA-transformed data
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca_df, y, test_size=0.2, random_state=42)

# 4. Train the model using the PCA-transformed data
model_pca = LogisticRegression(random_state=42)
model_pca.fit(X_train_pca, y_train)

# 5. Make Predictions
y_pred_pca = model_pca.predict(X_test_pca)

# 6. Evaluate the Model
print("Confusion Matrix (PCA):")
print(confusion_matrix(y_test, y_pred_pca))

print("\nClassification Report (PCA):")
print(classification_report(y_test, y_pred_pca))

print("Accuracy Score (PCA):", accuracy_score(y_test, y_pred_pca))




# In[17]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_pca, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test_pca)

# Evaluate the model
print("Confusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

print("Accuracy Score (Random Forest):", accuracy_score(y_test, y_pred_rf))


# In[18]:


from sklearn.ensemble import GradientBoostingClassifier

# Initialize and train the Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_pca, y_train)

# Make predictions
y_pred_gb = gb_model.predict(X_test_pca)

# Evaluate the model
print("Confusion Matrix (Gradient Boosting):")
print(confusion_matrix(y_test, y_pred_gb))

print("\nClassification Report (Gradient Boosting):")
print(classification_report(y_test, y_pred_gb))

print("Accuracy Score (Gradient Boosting):", accuracy_score(y_test, y_pred_gb))


# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Assuming X_pca_df and y are already defined
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca_df, y, test_size=0.2, random_state=42)

# Define base models
base_models = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42))
]

# Define the meta-model
meta_model = LogisticRegression(random_state=42)

# Define the stacking model
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Train the stacking model
stacking_model.fit(X_train_pca, y_train)

# Make predictions
y_pred_stack = stacking_model.predict(X_test_pca)

# Evaluate the model
print("Confusion Matrix (Stacking):")
print(confusion_matrix(y_test, y_pred_stack))

print("\nClassification Report (Stacking):")
print(classification_report(y_test, y_pred_stack))

print("Accuracy Score (Stacking):", accuracy_score(y_test, y_pred_stack))


# In[20]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# Define hyperparameter grids
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1]
}

# Initialize base models
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)

# Perform grid search for RandomForest
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1)
grid_search_rf.fit(X_train_pca, y_train)

# Perform grid search for GradientBoosting
grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=param_grid_gb, cv=5, n_jobs=-1)
grid_search_gb.fit(X_train_pca, y_train)

# Get the best models
best_rf = grid_search_rf.best_estimator_
best_gb = grid_search_gb.best_estimator_


# In[21]:


# Define the meta-model
meta_model = LogisticRegression(random_state=42)

# Define the stacking model with the best base models
stacking_model = StackingClassifier(
    estimators=[('rf', best_rf), ('gb', best_gb)],
    final_estimator=meta_model,
    cv=5
)

# Train the stacking model
stacking_model.fit(X_train_pca, y_train)


# In[22]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Make predictions
y_pred_stack = stacking_model.predict(X_test_pca)

# Evaluate
print("Confusion Matrix (Stacking):")
print(confusion_matrix(y_test, y_pred_stack))

print("\nClassification Report (Stacking):")
print(classification_report(y_test, y_pred_stack))

print("Accuracy Score (Stacking):", accuracy_score(y_test, y_pred_stack))


# In[23]:


import pickle

# Save the stacking model
with open('stacking_model.pkl', 'wb') as model_file:
    pickle.dump(stacking_model, model_file)

# Save the scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")


# In[49]:


get_ipython().run_line_magic('system', 'jupyter nbconvert --to script Untitled.ipynb')

|


# In[ ]:




