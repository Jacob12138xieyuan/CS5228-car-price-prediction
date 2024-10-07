import pandas as pd

# Read the CSV file and select the "Make" and "baseModel" columns
df = pd.read_csv("make_baseModel_mapping/all-vehicles-model.csv", sep=";")
df = df[["Make", "baseModel"]]

# Remove duplicates based on "Make" and "baseModel", sort the DataFrame, and reset the index
df.drop_duplicates(["Make", "baseModel"], inplace=True)
df.sort_values(["Make", "baseModel"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(df[df["Make"] == "Land Rover"])

# Save the processed DataFrame to a new CSV file without the index column
df.to_csv("make_baseModel_mapping/make_baseModel.csv", index=False)