from src.helper.utils import load_data
from src.helper.logger import success
from openai import OpenAI

import os
import re
import pandas as pd


def extract_skills_gpt3(job_description):
  # Set your OpenAI API key from the environment
  API_KEY = os.getenv("OPENAI_API_KEY")
  client = OpenAI(api_key=API_KEY)

  # system = f"You are an expert in job analysis. Your task is to identify skills required for a job based on its description, selecting only from the following predefined skills list: {', '.join(skill_list)}. You are not allowed to use any other skills than those mentioned in the skills list. Do not infer or add skills not mentioned in the description and provide the skills in a simple, comma-separated format."
  # system = f"You are an expert in job analysis. Your task is to identify skills required for a job based on its description. You are only to identify soft skills. Do not infer or add skills not mentioned in the description and provide the skills in a simple, comma-separated format."
  system = f"You are an expert in job analysis. Your task is to extract at most 10 skills required for a job based on its description. Do not infer or add skills not mentioned in the description. You are required to present me the skills in a raw list format: [skill1, skill2, ... skill10]."

  # prompt = f"Extract and list the skills required for Present the skills in a simple, comma-separated list. No explanations or additional text. Job Description: '{job_description_str}' Skills:"
  prompt = f"Identify at most 10 skills required for this job based on the description. Present them to me in a raw list format [skill1, skill2, ..., skill10]. Description: '{job_description}'"

  # response = client.chat.completions.create(model="gpt-3.5-turbo",
  #                                          messages=[
  #                                              {"role": "system",
  #                                               "content": system},
  #                                              {"role": "user", "content": prompt},
  #                                          ])

  # response = client.chat.completions.create(model="gpt-3.5-turbo",
  #                                          messages=[
  #                                              {"role": "system", "content": "You are a professional job recruiter. Your task is to extract 7 most relevant skills required for a job position and present them in a raw list format: [skill1, skill2, ... skill7]."},
  #                                              {"role": "user", "content": "Extract 7 most relevant skills. Here is the job description: '''would like part ryanair group amazing cabin crew family k crew customer oriented love delivering great service want fast track career opportunity would delighted hear experience required bag enthusiasm team spirit europe largest airline carrying k guest daily flight looking next generation cabin crew join u brand new copenhagen base flying board ryanair group aircraft amazing perk including discounted staff travel destination across ryanair network fixed roster pattern free training industry leading pay journey becoming qualified cabin crew member start week training course learn fundamental skill require part day day role delivering top class safety customer service experience guest course required study exam taking place regular interval training culminates supernumerary flight followed cabin crew wing member ryanair group cabin crew family immersed culture day one career opportunity endless including becoming number base supervisor european base manager regional manager aspire becoming director inflight life cabin crew fun rewarding however demanding position safety number priority required operate early late shift report duty early morning early roster return home midnight afternoon roster morning person think twice applying requirement bag enthusiasm customer serviceoriented background ie previous experience working bar restaurant shop etc applicant must demonstrate legal entitlement work unrestricted basis across euyou must cm cm height must able swim meter unaided help hardworking flexible outgoing friendly personality adaptable happy work shift roster enjoy dealing public ability provide excellent customer service attitude comfortable speaking writing english ease passion travelling meeting new people benefit free cabin crew training course adventure experience lifetime within cabin crew network explore new culture city colleague day flexible day day staff roster unlimited highly discounted staff travel rate sale bonus free uniform year security working financially stable airline daily per diem provided whilst training direct employment contract highly competitive salary package click link start new exciting career sky'''"},
  #                                              {"role": "assistant",
  #                                                  "content": "[Customer Service Orientation, Flexibility and Adaptability, Communication Skills, Teamwork, Safety Awareness, Physical Fitness, Interpersonal Skills]"},
  #                                              {"role": "user", "content": f"Extract 7 most relevant skills. Here is the job description: '''{job_description}'''"},
  #                                          ])

  response = client.chat.completions.create(model="gpt-3.5-turbo",
                                            messages=[
                                                {"role": "system",
                                                  "content": system},
                                                {"role": "user", "content": "Identify at most 10 skills required for this job based on the description. Present them to me in a raw list format [skill1, skill2, ..., skill10]. Description: 'If you are customer oriented, love delivering a great service & want fast track career opportunities, we would be delighted to hear from you! No experience required, just bags of enthusiasm & team spirit! As Europe’s largest airline carrying over 550k guests on over 3,000 daily flights, we are looking for the next generation of cabin crew to join us at our brand new Copenhagen base. Flying on board Ryanair Group aircraft there are some amazing perks, including; discounted staff travel to over 230+ destinations across the Ryanair network, a fixed 5/3 roster pattern, free training & industry leading pay.Your journey to becoming a qualified cabin crew member will start on a 6 Week training course where you will learn all of the fundamental skills that you will require as part of your day to day role delivering a top class safety & customer service experience to our guests. '"},
                                                {"role": "assistant",
                                                    "content": "[Customer Service Orientation, Teamwork"},
                                                {"role": "user", "content": prompt},
                                            ])
  skills_response = response.choices[0].message.content
  return skills_response


def skill_extraction(save_skills=False):

  df_clean = load_data("processed")
  df_raw = load_data("raw")

  df_clean = df_clean[['id', 'description']]
  df_raw = df_raw[['id', 'description']]

  # Obtain the original unprocessed job descriptions from the jobs that appear in the clean dataset
  merged = pd.merge(df_clean, df_raw, on='id', how="left",
                    suffixes=('_clean', '_raw'))

  # Drop duplicates based on id
  merged = merged.drop_duplicates(subset=['id'])

  extracted_skills = {"id": [], "skills": [], "description_raw": []}

  N = len(merged)
  count = 0

  for _, row in merged.iterrows():
    job_description = row['description_raw']
    job_description = job_description.replace("\n", " ")
    pattern = r'(?<=[a-z])(?=[A-Z])'
    job_description = re.sub(pattern, ' ', job_description)
    # Remove the last 56 trash characters
    job_description = job_description[:-56]

    skills = extract_skills_gpt3(job_description)
    _id = row['id']

    extracted_skills["id"].append(_id)
    extracted_skills["skills"].append(skills)
    extracted_skills["description_raw"].append(job_description)

    count += 1

    # Print progress in place
    print(f"\r💬 Skills for {_id} extracted! Progress: {count}/{N}", end="")

  extracted_skills_df = pd.DataFrame(extracted_skills)
  success("Skills extracted")
  if save_skills:
    name = "skills_extracted_gpt3.csv"
    extracted_skills_df.to_csv(
        f"extracted_skills/{name}", index=False)
    success(f"Skills saved to extracted_skills/{name}")
  return extracted_skills_df


if __name__ == "__main__":
  skill_extraction(save_skills=False)
