import os
from jira import JIRA
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool


jira_server = os.getenv('JIRA_INSTANCE_URL', None) # e.g. https://jira.example.com
jira_username = os.getenv('JIRA_USERNAME', None) # e.g. jira_username
jira_password = os.getenv('JIRA_API_TOKEN', None) # e.g. jira_user_password.

class IssueDetailInput(BaseModel):
    """Search Jira issue input parameters."""
    issue_key: str = Field(..., description="The key of the jira issue")

class IssueDetailTool(BaseTool):
    name = "jira_detail_issue"
    description = "Find details of a Jira issue, try to explain the issue and its status like a human."

    def _run(self, issue_key:str):
        issue_results = get_jira_issue_details(issue_key)
        return issue_results

    def _arun(self, issue_key:str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = IssueDetailInput

def get_jira_issue_details(issue_key):
    # 建立 JIRA 連線
    try:
        jira = JIRA(server=jira_server, basic_auth=(jira_username, jira_password))
    except Exception as e:
        return {"status": "failure", "reason": str(e)}

    # 查詢 issue
    try:
        issue = jira.issue(issue_key)
        assignee = issue.fields.assignee.displayName if issue.fields.assignee else None
        status = issue.fields.status.name
        title = issue.fields.summary
        description = issue.fields.description

        # 取得最新的兩個 comments
        comments = issue.fields.comment.comments
        latest_comments = comments[-2:] if len(comments) > 2 else comments
        latest_comments = [{"author": comment.author.displayName, "body": comment.body} for comment in latest_comments]

        return {
            "status": "success",
            "issue_details": {
                "title": title,
                "issue_key": issue_key,
                "assignee": assignee,
                "description": description,
                "status": status,
                "latest_comments": latest_comments
            }
        }
    except Exception as e:
        return {"status": "failure", "reason": str(e)}
    
def main():
    # 使用範例
    response2 = get_jira_issue_details("JIRA-1234")
    print(response2)

if __name__ == "__main__":
    main()
