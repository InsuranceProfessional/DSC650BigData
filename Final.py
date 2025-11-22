# %% Imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# %% Load the data
file_path = "C:\\Users\\champ\\OneDrive\\Documents\\MS Data Science\\DSC550 Data Mining\\Project\\train_clean.csv"
df = pd.read_csv(file_path)

# %% Filter out extreme values
df = df[df['Annual_Income'] <= 1_000_000_000]

# %% Drop unneeded columns
df = df.drop(columns=['ID', 'Customer_ID', 'Month'])

# %% Handle categorical variables
label_cols = [
    'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score',
    'Occ_Accountant', 'Occ_Architect', 'Occ_Developer', 'Occ_Doctor', 'Occ_Engineer',
    'Occ_Entrepreneur', 'Occ_Journalist', 'Occ_Lawyer', 'Occ_Manager', 'Occ_Mechanic',
    'Occ_Media_Manager', 'Occ_Musician', 'Occ_Scientist', 'Occ_Teacher', 'Occ_Writer', 'Occ________'
]

for col in label_cols:
    if col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Unknown")
        # Encode categorical labels
        df[col] = LabelEncoder().fit_transform(df[col])

# %% Drop remaining missing values
df = df.dropna()

# %% Define features and label
X = df.drop(columns=['Credit_Score'])
y = df['Credit_Score']

# %% Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% Train Random Forest with tuned hyperparameters
rf = RandomForestClassifier(
    n_estimators=310,
    max_depth=22,
    min_samples_split=4,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)

# %% Predictions
y_pred = rf.predict(X_test)

# %% Evaluation
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %% Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Poor', 'Standard', 'Good'],
            yticklabels=['Poor', 'Standard', 'Good'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest (Tuned)')
plt.tight_layout()
plt.show()
