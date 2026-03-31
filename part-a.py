import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('suv_data.csv')
print("First 5 rows:\n", df.head())
print("\nShape of dataset:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())




df = df.dropna()


if 'Gender' in df.columns:
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    print("Encoded 'Gender' successfully.")


X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
print("Features (X) and Target (y) separated.")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Training data shape (X):", X_train.shape)
print("Testing data shape (X):", X_test.shape)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling applied. First scaled row:", X_train_scaled[0])


model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print("Logistic Regression model trained successfully!")
