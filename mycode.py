import pandas as pd
from sklearn.datasets import load_iris
import os

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Add target column
df["target"] = iris.target

data_path = "data"
os.makedirs(data_path,exist_ok=True)

# Save to CSV
df.to_csv(f"{data_path}/iris_dataset.csv", index=False)

print("Iris dataset saved as iris_dataset.csv")
