import os
import asyncio
import time
import csv
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from collections import Counter
from supabase import create_client, Client
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
)
from langchain.prompts import load_prompt, ChatPromptTemplate


load_dotenv()

url: str = os.environ["SUPABASE_URL"]
key: str = os.environ["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

SYSTEM_EVAL = load_prompt('eval.yaml')
system_eval: SystemMessagePromptTemplate = SystemMessagePromptTemplate(prompt=SYSTEM_EVAL)
llm = AzureChatOpenAI(deployment_name = os.environ['OPENAI_API_DEPLOYMENT_NAME'], temperature=1)


async def get_eval(joint, session_id: str, voe: bool = True):
    """Given a list of joined thought-response pairs, get the eval"""
    data = []
    prompt = ChatPromptTemplate.from_messages([
        system_eval,
    ])
    chain = prompt | llm

    print(f"SESSION ID: {session_id}")

    for item in joint:
        result = await chain.ainvoke(
            {"ai_message": item[0], "user_prediction_thought": item[2], "actual": item[1]},
        )
        if result:
            data.append(
                {
                    "prediction_thought": item[2],
                    "actual": item[1],
                    "assessment": result.content,
                    "session_id": session_id,
                    "conversation_turn": item[3],
                    "voe": voe,
                }
            )
            print(f"ASSESSMENT: {result.content}")
        else:
            continue

    return data



async def main():
    """ Read in data, execute loop, collect results"""

    voe_session_ids = pd.read_csv(os.environ['CSV_TABLE']).values.ravel().tolist()
    before_count = len(voe_session_ids)

    # query to see what's already been done
    existing_session_ids = supabase.table(os.environ['SUPABASE_RESULTS_TABLE']).select('session_id').execute()
    # Convert existing_session_ids.data to a set for faster lookup
    existing_session_ids_set = set(d['session_id'] for d in existing_session_ids.data)
    # remove existing from unique
    unique_session_ids = [d for d in voe_session_ids if d not in existing_session_ids_set]
    after_count = len(unique_session_ids)

    print(f"Removed {before_count - after_count} session IDs")

    conversations = []
    # each response is for a different session id
    for session_id in unique_session_ids:
        response = (
            supabase.table(os.environ['SUPABASE_MESSAGE_TABLE'])
            .select('*')
            .eq('session_id', session_id)
            .execute()
        )
        conversations.append(response.data)




    # loop through the conversations (session_id responses)
    for conversation in tqdm(conversations):
    
        temp = pd.DataFrame(conversation)
        temp = temp.sort_values('timestamp')

        session_id = temp['session_id'][0]

        ai_responses = temp[(temp['message_type'] == 'response') & (temp['message'].apply(lambda x: x['type'] == 'ai'))]
        ai_responses_list = ai_responses['message'].apply(lambda x: x['data']['content']).tolist()

        human_thoughts = temp[(temp['message_type'] == 'thought') & (temp['message'].apply(lambda x: x['type'] == 'human'))]
        human_thought_list = human_thoughts['message'].apply(lambda x: x['data']['content']).tolist()

        ai_predictions = temp[(temp['message_type'] == 'user_prediction_thought_revision') & (temp['message'].apply(lambda x: x['type'] == 'ai'))]
        ai_prediction_list = ai_predictions['message'].apply(lambda x: x['data']['content']).tolist()

        # stagger the human thought list, we start making predictions after the first message
        joint = list(zip(ai_responses_list, human_thought_list[1:], ai_prediction_list, list(range(1, len(ai_responses_list)))))

        data = await get_eval(joint, session_id)

        # write to supabase
        if data:
            response = supabase.table(os.environ['SUPABASE_RESULTS_TABLE']).insert(data).execute()
        else:
            print(f"JOINT: {joint}")
            raise Exception("DATA RETURNED FROM get_eval WAS NONE")

        time.sleep(0.001)


if __name__ == "__main__":
    asyncio.run(main())
