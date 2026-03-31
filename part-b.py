import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.colors as mcolors

# Load Data
df = pd.read_csv('suv_data.csv').dropna()
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

def evaluate_model(test_size):
    print(f"\n=========================================")
    print(f"Evaluating with test_size={test_size} ({(1-test_size)*100:.0f}/{test_size*100:.0f} Split)")
    print(f"=========================================")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Training
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # 1. Model Evaluation (Predict on test data)
    y_pred = model.predict(X_test_scaled)
    
    # Compute Accuracy & Confusion Matrix
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("Accuracy Score:", acc)
    print("Confusion Matrix:\n", cm)
    
    # 2. Visualization - Plot decision boundary
    plot_decision_boundary(X_test_scaled, y_test.values, model, f"Decision Boundary (Test Size {test_size})", f"decision_boundary_{test_size}.png")
    
    return acc

def plot_decision_boundary(X_set, y_set, model, title, filename):
    try:
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        
        plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.3, cmap=mcolors.ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=['red', 'green'][i], label=j, edgecolors='k')
        
        plt.title(title)
        plt.xlabel('Age (Scaled)')
        plt.ylabel('Estimated Salary (Scaled)')
        plt.legend()
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved successfully as '{filename}'")
    except Exception as e:
        print("Warning: Matplotlib is required to plot decision boundaries. Error:", e)

# Run standard evaluation (80/20)
acc_20 = evaluate_model(0.20)

# 3. Improvement - Try different test sizes (70/30, 75/25)
acc_30 = evaluate_model(0.30)
acc_25 = evaluate_model(0.25)

print("\n--- Accuracy Comparison ---")
print(f"Test Size 20% (80/20): {acc_20*100:.2f}%")
print(f"Test Size 25% (75/25): {acc_25*100:.2f}%")
print(f"Test Size 30% (70/30): {acc_30*100:.2f}%")

print("\n--- Interpretation ---")
print("The Decision Boundary Plot visualizes how the model separates the '1's (Purchased) from '0's (Not Purchased) using a linear boundary.")
