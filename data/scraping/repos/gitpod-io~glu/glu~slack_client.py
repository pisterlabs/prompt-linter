from glu.config_loader import config
from gidgethub.sansio import Event
from html import unescape as html_unescape
from slack_sdk.web.async_client import AsyncWebClient
from glu.openai_client import openai
from gidgethub.abc import GitHubAPI


class CustomAsyncWebClient(AsyncWebClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.username: str = config["slack"]["username"]
        self.icon_emoji: str = config["slack"]["icon_emoji"]


slack_client = CustomAsyncWebClient(token=config["slack"]["api_token"])


async def send_github_issue(
    event: Event, gh: GitHubAPI, channel: str, what: str
) -> None:
    sender = event.data["sender"]["login"]
    item_url: str | None = None
    if event.event == "issues":
        item_url = event.data["issue"]["html_url"]
    elif event.event == "pull_request":
        item_url = event.data["pull_request"]["html_url"]
    elif event.event == "issue_comment":
        item_url = event.data["comment"]["html_url"]

    item_title = html_unescape(
        (
            event.data["issue"]["title"]
            if event.event == "issues" or event.event == "issue_comment"
            else event.data["pull_request"]["title"]
        )
    )
    text = f"<https://github.com/{sender}|{sender}> {what} <{item_url}|{item_title}>"  # noqa: E501

    # Main message
    main_message = await slack_client.chat_postMessage(
        username=slack_client.username,
        icon_emoji=slack_client.icon_emoji,
        channel=channel,
        text=f"{sender} {what}: {item_title}",
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text,
                },
            }
        ],
    )
    # Also send item body inside thread
    item_body: str | None = None
    if event.event == "issues":
        item_body = event.data["issue"]["body"]
    elif event.event == "pull_request":
        item_body = event.data["pull_request"]["body"]
    elif event.event == "issue_comment":
        item_body = event.data["comment"]["body"]

    if item_body is not None:
        text = f"{item_body}\n\n_View it on <{item_url}|GitHub>_"
        # text = f'_View it on <{item_url}|GitHub>_'
        await slack_client.chat_postMessage(
            as_user=True,
            link_names=False,
            unfurl_links=False,
            unfurl_media=False,
            username=slack_client.username,
            icon_emoji=slack_client.icon_emoji,
            channel=str(main_message["channel"]),
            thread_ts=main_message["ts"],
            text=f"{item_url} body",
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": text,
                    },
                }
            ],
        )

        # Auto triage
        if (event.event == "issues" or event.event == "pull_request") and event.data[
            "action"
        ] == "opened":
            user_prompt = f"Title: {item_title}\n\nBody:\n{item_body}"
            ai_system_prompt = config["github"]["user_activity"]["auto_triage"][
                "system_prompt"
            ]
            max_tokens = config["github"]["user_activity"]["auto_triage"]["max_tokens"]

            ai_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": ai_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                n=1,
                max_tokens=max_tokens,
            )
            targets = ai_response["choices"][0]["message"]["content"]

            if "false" not in targets.lower():
                labels = []

                for team in config["github"]["user_activity"]["to_slack"]["teams"]:
                    team_label = str(team["label_id_or_name"])

                    # Post on slack
                    if team_label in targets:
                        labels.append(team_label)
                        await slack_client.chat_postMessage(
                            as_user=True,
                            link_names=False,
                            unfurl_links=False,
                            unfurl_media=False,
                            username=slack_client.username,
                            icon_emoji=slack_client.icon_emoji,
                            channel=str(main_message["channel"]),
                            thread_ts=main_message["ts"],
                            text=f"Triaged to {team_label}",
                        )

                if labels:
                    # Add labels to issue
                    api_url = (
                        event.data["issue"]["url"]
                        if event.event == "issues"
                        else event.data["pull_request"]["url"]
                    ) + "/labels"

                    await gh.post(api_url, data={"labels": labels})
