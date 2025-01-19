import pandas as pd

df = pd.read_json("retrieval/data.json")
df.shape
df.columns
df.head(3)
details_df = pd.json_normalize(df['fields'])
for col in details_df.columns:
    print(details_df[col].value_counts())
details_df.dropna(axis=0, subset=["description"])[['title', 'description']].to_csv("retrieval/title_desc.csv")

titles_df = details_df.dropna(axis=0, subset=["description"])

"""
You are a helpful career assistant.
You will be given a CV and a job offer.
Please tell us, on a scale 1-10, how good match it is
"""
