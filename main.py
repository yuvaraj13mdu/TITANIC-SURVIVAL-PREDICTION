# Import necessary libraries
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

# Load the Titanic training data
train = pd.read_csv('C:\\Users\\welcme\\Desktop\\CODSOFT\\tested.csv')

# statistical info
train.describe()

# datatype info
train.info()

# Missing Data
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
plt.title('Missing Data Visualization')
plt.show()

# Visualize survival based on sex
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')
plt.title('Survival Visualization by Sex')
plt.show()

# Visualize survival based on passenger class
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')
plt.title('Survival Visualization by Passenger Class')
plt.show()

# Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(data=train['Age'].dropna(), kde=False, color='darkred', bins=30)
plt.title('Distribution of Age')
plt.show()

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')

#Impute missing age based on passenger class

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

# Drop Cabin column and rows with missing Embarked data
train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)

# Convert categorical features to dummy variables
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train = pd.concat([train, sex, embark], axis=1)



# Building a Logistic Regression model
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1),train['Survived'], test_size=0.30,random_state=101)


# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Building a Logistic Regression model
logmodel = LogisticRegression(max_iter=1000)  # Increase max_iter to 1000
logmodel.fit(X_train_scaled, y_train)
predictions = logmodel.predict(X_test_scaled)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, predictions))

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Get the coefficients and corresponding feature names
coefficients = logmodel.coef_[0]
feature_names = X_train.columns

# Create a DataFrame to display the coefficients
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the coefficients by absolute value to see the most impactful features
coefficients_df['Absolute Coefficient'] = np.abs(coefficients_df['Coefficient'])
coefficients_df = coefficients_df.sort_values(by='Absolute Coefficient', ascending=False)

# Display the top coefficients
print(coefficients_df.head())

# Interpretation
print("Interpretation:")
for index, row in coefficients_df.iterrows():
    if row['Coefficient'] > 0:
        print(f"A unit increase in {row['Feature']} increases the odds of survival.")
    else:
        print(f"A unit increase in {row['Feature']} decreases the odds of survival.")

# Plot feature coefficients with confidence intervals
plt.figure(figsize=(10, 6))
plt.errorbar(coefficients_df['Coefficient'], coefficients_df['Feature'], xerr=coefficients_df['Absolute Coefficient'], fmt='o')
plt.title('Feature Coefficients and Confidence Intervals')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.grid(True)
plt.show()

# Calculate predicted probabilities for the positive class
y_pred_prob = logmodel.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()