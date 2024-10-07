import pandas as pd

# Read the CSV file and select the "Make" and "baseModel" columns
df = pd.read_csv("all-vehicles-model.csv", sep=";")
df = df[["Make", "baseModel", "Model"]]

# convert to lower case
df["Make"] = df["Make"].str.lower()
df["baseModel"] = df["baseModel"].str.lower()
df["Model"] = df["Model"].str.lower()

# Remove duplicates based on "Make" and "baseModel", sort the DataFrame, and reset the index
df.drop_duplicates(["Make", "baseModel", "Model"], inplace=True)
df.sort_values(["Make", "baseModel", "Model"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(df[df["Make"] == "land rover"])

# Save the processed DataFrame to a new CSV file without the index column
df.to_csv("make_model.csv", index=False)