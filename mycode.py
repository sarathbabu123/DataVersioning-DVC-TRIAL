import pandas as pd
from sklearn.datasets import load_iris
import os

# Load the Iris dataset
iris = load_iris()

# Version 1 data
# Create a DataFrame
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Version 2 Data
# sepal ratio
df["sepal ratio"] = df["sepal length (cm)"]/df["sepal width (cm)"]

cols  = list(df.columns)
cols.remove("sepal ratio")
width_index = cols.index("sepal width (cm)")
cols.insert(width_index+1,"sepal ratio")

df = df[cols]

print(df.head())

# # Version 3
# # Petal ratio
# df["petal ratio"] = df["petal length (cm)"]/df["petal width (cm)"]

# cols  = list(df.columns)
# cols.remove("petal ratio")
# width_index = cols.index("petal width (cm)")
# cols.insert(width_index+1,"petal ratio")

# Add target column
df["target"] = iris.target

data_path = "data"
os.makedirs(data_path,exist_ok=True)

# Save to CSV
df.to_csv(f"{data_path}/iris_dataset.csv", index=False)

print("Iris dataset saved as iris_dataset.csv")
