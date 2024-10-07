import pandas as pd

df = pd.read_csv("all-vehicles-model.csv", sep=";")
df = df[["Make", "baseModel"]]
df.drop_duplicates(["Make", "baseModel"], inplace=True)
df.sort_values(["Make", "baseModel"], inplace=True)

print(df[df["Make"] == "Land Rover"])
df.to_csv("make_baseModel.csv")