from dotenv import load_dotenv
import openai
import pandas as pd

load_dotenv(".envs/local/.openai")

openai.api_key = "YOUR_API_KEY"
df = pd.read_json("retrieval/data.json")
df.shape
df.columns
df.head(3)
details_df = pd.json_normalize(df['fields'])
for col in details_df.columns:
    print(details_df[col].value_counts())
details_df.dropna(axis=0, subset=["description"])[['title', 'description']].to_csv("retrieval/title_desc.csv")

titles_df = details_df.dropna(axis=0, subset=["description"])

system_prompt = "You are a career guidance assistant specializing in evaluating job offers based on the qualifications, skills, and experience listed in the given CV. Your goal is to assess how well the job offer aligns with the CV. When evaluating job offers, assign a numerical score that reflects the suitability of the CV for the position, considering skills, experience, and alignment with the candidate’s career focus."
user_query = "Here is the CV: {}. Here is the job offer:{}"
