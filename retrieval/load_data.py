import os
from dotenv import load_dotenv
import openai
import pandas as pd

load_dotenv(".envs/local/.openai")

openai.api_key = os.environ.get("OPENAI_KEY")
df = pd.read_json("retrieval/data.json")
print(df.shape)
print(df.columns)
df.head(3)
details_df = pd.json_normalize(df['fields'])
for col in details_df.columns:
    print(details_df[col].value_counts())
details_df.dropna(axis=0, subset=["description"])[['title', 'description']].to_csv("retrieval/title_desc.csv")

titles_df = details_df.dropna(axis=0, subset=["description"])

def evaluate_job_offer(cv_text, job_offer_text):
    system_prompt = "You are a career guidance assistant specializing in evaluating job offers based on the qualifications, skills, and experience listed in the given CV. Your goal is to assess how well the job offer aligns with the CV. When evaluating job offers, assign a numerical score that reflects the suitability of the CV for the position, considering skills, experience, and alignment with the candidate’s career focus."
    user_prompt = f"""
    Here is the CV: {cv_text}
    Here is the job offer: {job_offer_text}

    Evaluate the suitability of the job offer based on the given CV and assign a score out of 100.
    """

    # Sending the request to OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
    )

    return response['choices'][0]['message']['content']
