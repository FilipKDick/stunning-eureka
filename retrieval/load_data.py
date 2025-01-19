import os
import re

from dotenv import load_dotenv
from openai import OpenAI

import pandas as pd

load_dotenv(".envs/.local/.openai")
client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

df = pd.read_json("retrieval/pracuje.json")
with open("retrieval/cv.txt") as f:
    cv = f.read()

details_df = pd.json_normalize(df['fields'])

details_df["full_content"] = details_df["title"].fillna('') + details_df["description"].fillna('') + details_df["responsibilities"].fillna('') + details_df["requirements"].fillna('')
full_offers_only = details_df[details_df['full_content'] != '']
full_offers_only.to_csv("retrieval/title_desc.csv", index=False)

def evaluate_job_offer(cv_text, job_offer_text):
    system_prompt = "You are a career guidance assistant specializing in evaluating job offers based on the qualifications, skills, and experience listed in the given CV. Your goal is to assess how well the job offer aligns with the CV. When evaluating job offers, assign a numerical score that reflects the suitability of the CV for the position, considering skills, experience, and alignment with the candidate’s career focus."
    user_prompt = f"""
    Here is the CV: {cv_text}
    Here is the job offer: {job_offer_text}

    Evaluate the suitability of the job offer based on the given CV and assign a score out of 100.
    """

    # Sending the request to OpenAI
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.7)

    return response.choices[0].message.content

# answer = evaluate_job_offer(cv_text=cv, job_offer_text=full_offers_only["full_content"].iloc[0])
# print(answer)S

full_offers_only["fit"] = full_offers_only["full_content"].apply(lambda job_offer: evaluate_job_offer(cv, job_offer))
full_offers_only.to_csv("retrieval/final_evals.csv", index=False)

def extract_score(response_text):
    match = re.search(r'\b(\d{1,3})/100\b', response_text)  # Match any number followed by "/100"
    if match:
        return int(match.group(1))  # Return the score as an integer
    else:
        return None  # Return None if no match is found

full_offers_only["numbers"] = full_offers_only["fit"].apply(lambda fit_value: extract_score(fit_value))
full_offers_only.sort_values("numbers").dropna(subset="numbers")['url'][-5:]
