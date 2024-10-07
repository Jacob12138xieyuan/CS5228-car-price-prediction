import pandas as pd

# drop_duplicated_column = "baseModel"
drop_duplicated_column = "Model"

# Read the CSV file and select the "Make" and "baseModel" columns
df = pd.read_csv("all-vehicles-model.csv", sep=";")
df = df[["Make", drop_duplicated_column]]

# Remove duplicates based on "Make" and "baseModel", sort the DataFrame, and reset the index
df.drop_duplicates(["Make", drop_duplicated_column], inplace=True)
df.sort_values(["Make", drop_duplicated_column], inplace=True)
df.reset_index(drop=True, inplace=True)

print(df[df["Make"] == "Land Rover"])

# Save the processed DataFrame to a new CSV file without the index column
df.to_csv("make_model.csv", index=False)