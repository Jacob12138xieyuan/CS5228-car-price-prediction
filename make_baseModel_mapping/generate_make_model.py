import pandas as pd

df = pd.read_csv("make_baseModel_mapping/all-vehicles-model.csv", sep=";")
df = df[["Make", "baseModel"]]
df.drop_duplicates(["Make", "baseModel"], inplace=True)
df.sort_values(["Make", "baseModel"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(df[df["Make"] == "Land Rover"])
df.to_csv("make_baseModel_mapping/make_baseModel.csv")