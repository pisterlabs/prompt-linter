from datetime import datetime, timedelta
import pandas as pd
from openai_client import categorize_text


def fetch_issues_and_prs(logger, today, days, repo):
    logger.info(f"Fetching all issues and pull requests in the past {days} days...")
    all_issues = repo.get_issues(state="all")

    days_ago = datetime.now() - timedelta(days=days)

    issues = [
        issue
        for issue in all_issues
        if issue.updated_at >= days_ago and not issue.pull_request
    ]
    prs = [
        issue
        for issue in all_issues
        if issue.updated_at >= days_ago and issue.pull_request
    ]

    issues_dict = [
        {
            "number": issue.number,
            "title": issue.title,
            "user": issue.user.login,
            "state": issue.state,
            "created_at": issue.created_at,
            "updated_at": issue.updated_at,
            "comments": issue.comments,
            "labels": ", ".join([label.name for label in issue.labels]),
            "body": issue.body,
            "is_pr": False,
        }
        for issue in issues
    ]
    prs_dict = [
        {
            "number": pr.number,
            "title": pr.title,
            "user": pr.user.login,
            "state": pr.state,
            "created_at": pr.created_at,
            "updated_at": pr.updated_at,
            "comments": pr.comments,
            "labels": ", ".join([label.name for label in pr.labels]),
            "body": pr.body,
            "is_pr": True,
        }
        for pr in prs
    ]

    logger.info(f"Fetched {len(issues)} issues and {len(prs)} pull requests.")
    all_data_dict = issues_dict + prs_dict
    all_data_df = pd.DataFrame(all_data_dict)
    # # Print column names
    # print(f"Columns: {all_data_df.columns}")
    # # Add a new column 'category' to the DataFrame
    # all_data_df["category"] = all_data_df["body"].apply(categorize_text)
    # all_data_df.to_csv(f"{today}/data.csv", index=False)
    return issues, prs
