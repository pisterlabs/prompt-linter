import sys, os
import pandas as pd
from tqdm import tqdm
import time
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain

sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

import utils.paper_utils as pu
import utils.prompts as ps
import utils.db as db

summaries_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "summaries")
meta_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "arxiv_meta")
review_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "weekly_reviews")

## LLM model.
llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.1)

## Initialize chain.
summarizer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ps.WEEKLY_SYSTEM_PROMPT),
        ("user", ps.WEEKLY_USER_PROMPT),
        (
            "user",
            "Tip: Remember to add plenty of citations! Use the format (arxiv:1234.5678)`.",
        ),
    ]
)

llm_chain = LLMChain(
    llm=llm,
    prompt=summarizer_prompt,
    verbose=False,
)


def main(date_str: str):
    """Generate weekly review of main highlights and takeaways from papers."""
    ## Check if we have the summary.
    if db.check_weekly_summary_exists(date_str):
        # print(f"Summary for {date_str} already exists.")
        return

    ## Get data to generate summary.
    date_st = pd.to_datetime(date_str)
    weekly_content_df = db.get_weekly_summary_inputs(date_str)

    ## Get weekly total counts for last 4 weeks.
    prev_mondays = pd.date_range(
        date_st - pd.Timedelta(days=7 * 4), date_st, freq="W-MON"
    )
    prev_mondays = [date.strftime("%Y-%m-%d") for date in prev_mondays]
    weekly_counts = {
        date_str: len(db.get_weekly_summary_inputs(date_str))
        for date_str in prev_mondays
    }

    date_end = date_st + pd.Timedelta(days=6)
    date_st_long = date_st.strftime("%B %d, %Y")
    date_end_long = date_end.strftime("%B %d, %Y")
    weekly_content_md = f"# Weekly Review ({date_st_long} to {date_end_long})\n\n"

    ## Add table of weekly paper counts.
    weekly_content_md += f"## Weekly Publication Trends\n"
    weekly_content_md += "| Week | Total Papers |\n"
    weekly_content_md += "| --- | --- |\n"
    for tmp_date_str, count in weekly_counts.items():
        date = pd.to_datetime(tmp_date_str)
        date_st_long = date.strftime("%B %d, %Y")
        date_end = date + pd.Timedelta(days=7)
        date_end_long = date_end.strftime("%B %d, %Y")
        if tmp_date_str == date_str:
            weekly_content_md += (
                f"| **{date_st_long} to {date_end_long}** | **{count}** |\n\n\n"
            )
        else:
            weekly_content_md += f"| {date_st_long} to {date_end_long} | {count} |\n"

    # weekly_content_md += f"*Total papers published this week: {len(weekly_content_df)}*\n"
    weekly_content_md += f"## Papers Published This Week\n\n"

    for idx, row in weekly_content_df.iterrows():
        paper_markdown = pu.format_paper_summary(row)
        weekly_content_md += paper_markdown
        if idx >= 30:
            weekly_content_md += f"\n\n*...and {len(weekly_content_df) - idx} more.*"
            break

    with get_openai_callback() as cb:
        ## Generate summary.
        weekly_summary_md = llm_chain.run(weekly_content=weekly_content_md)
        tstp_now = pd.Timestamp.now()
        date = pd.to_datetime(date_str)
        weekly_summary_df = pd.DataFrame(
            {"date": [date], "tstp": [tstp_now], "review": [weekly_summary_md]}
        )
        db.upload_df_to_db(weekly_summary_df, "weekly_reviews", pu.db_params)
        # print(cb)
        # print(f"Done with {date_str}!")


if __name__ == "__main__":
    start_dt = "2023-11-13"
    end_dt = "2023-11-15"
    date_range = pd.date_range(start_dt, end_dt, freq="W-MON")
    date_range = [date.strftime("%Y-%m-%d") for date in date_range]
    for date_str in tqdm(date_range):
        main(date_str)
        time.sleep(5)
