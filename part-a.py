import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# 1. Data Loading & Exploration
print("====== 1. Data Loading & Exploration ======")
df = pd.read_csv('suv_data.csv')
print("First 5 rows:\n", df.head())
print("\nShape of dataset:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# 2. Data Preprocessing
print("\n====== 2. Data Preprocessing ======")
# Optional: Handle missing values (if any)
df = df.dropna()

# Encode categorical variables (e.g., Gender)
if 'Gender' in df.columns:
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    print("Encoded 'Gender' successfully.")

# Select relevant features (Age, EstimatedSalary) and Separate target (y)
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
print("Features (X) and Target (y) separated.")

# 3. Train-Test Split (80/20)
print("\n====== 3. Train-Test Split ======")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Training data shape (X):", X_train.shape)
print("Testing data shape (X):", X_test.shape)

# 4. Feature Scaling (Apply Standard Scaling on features)
print("\n====== 4. Feature Scaling ======")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling applied. First scaled row:", X_train_scaled[0])

# 5. Model Training (Logistic Regression)
print("\n====== 5. Model Training ======")
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print("Logistic Regression model trained successfully!")
