import pandas as pd

df = pd.read_csv("data/text/UNK.csv", parse_dates=True)
df.index = pd.to_datetime(df["Date"])
del df["Label"]
del df["Date"]
df = df.stack().to_frame().reset_index()
del df["level_1"]
df.columns = ["Date", "Text"]
df.index = df["Date"]
del df["Date"]
df.to_csv("data/text/UNK_stacked.csv", header=True)
