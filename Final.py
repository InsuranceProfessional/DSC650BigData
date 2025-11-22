# %% Imports
import happybase
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# %% HBase Connection
connection = happybase.Connection('master')  # replace with your HBase master host
table = connection.table('final')

# %% Load data from HBase into DataFrame
rows = table.scan()  # get all rows
data = []
for key, value in rows:
    row = {k.decode().split(":")[1]: v.decode() for k, v in value.items()}
    row['HBASE_ROW_KEY'] = key.decode()
    data.append(row)

df = pd.DataFrame(data)

# %% Convert numeric columns to proper types
numeric_cols = [
    'Age', 'Annual_Income', 'Monthly_In_hand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
    'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
    'Median_Occupation_Income', 'Income_vs_Occupation_Median', 'Investment_to_Salary_Ratio'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# %% Handle categorical variables
label_cols = [
    'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score',
    'Occ_Accountant', 'Occ_Architect', 'Occ_Developer', 'Occ_Doctor', 'Occ_Engineer',
    'Occ_Entrepreneur', 'Occ_Journalist', 'Occ_Lawyer', 'Occ_Manager', 'Occ_Mechanic',
    'Occ_Media_Manager', 'Occ_Musician', 'Occ_Scientist', 'Occ_Teacher', 'Occ_Writer', 'Occ________'
]

for col in label_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")
        df[col] = LabelEncoder().fit_transform(df[col])

# %% Drop remaining missing values
df = df.dropna()

# %% Define features and label
X = df.drop(columns=['Credit_Score', 'HBASE_ROW_KEY'])
y = df['Credit_Score']

# %% Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% Train Random Forest
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


# %% Write predictions back to HBase
for idx, row_key in enumerate(X_test.index):
    table.put(
        df.loc[row_key, 'HBASE_ROW_KEY'],
        {b'cf:Predicted_Credit_Score': str(y_pred[idx]).encode()}
    )

print("Predictions written back to HBase table 'final'.")
