from datetime import datetime, timedelta

from langchain.tools import tool


@tool
def say(text: str) -> str:
    """Say what you want to say. This is a tool that can be used by the agent when it wants to say, write or speak
    something."""
    return text


@tool
def today(text: str) -> str:
    """Get today's date and time information.
    You MUST use this tool every time you want to get today's date or time information."""
    return (f"Today is {datetime.now().strftime('%A %d %B %Y')} and the time is {datetime.now().strftime('%H:%M')}\n"
            f"In ISO format this is {datetime.now().isoformat()}")


@tool
def time_since_today(text: str) -> str:
    """Get relative time and date since today.
    You MUST use this tool every time you want to get relative time and date since today, In example, when you want to
    know when was yesterday, or when was 2 days ago.
    The input for this model is the difference in seconds from now. For example, if you want to know what was the date
    and time 1 day ago, you would input 86400, which is the number of seconds in a day.
    """

    seconds = int(text.strip())
    ts = datetime.now() - timedelta(seconds=seconds)
    return (f"Today minus {seconds} seconds is {ts.strftime('%A %d %B %Y')} and the time is {ts.strftime('%H:%M')}\n"
            f"In ISO format this is {ts.isoformat()}")


@tool
def date_difference(text: str) -> str:
    """Return the difference between two dates in seconds.
    You MUST use this tool every time you want to get the difference between two dates or times.
    This is useful for when you want to know how long it has been since something happened, or find out how long it will
    be until something happens, or how long it has been since something happened.

    The input to this tool is a comma separated start and end time in ISO format. For example:
    `2021-01-01T00:00:00,2021-01-01T23:59:59` would mean you want to see the difference between the 1-1-2021 00:00 and 1-1-2021 23:59.
    """

    start, end = text.strip().split(",")
    start = datetime.fromisoformat(start.strip())
    end = datetime.fromisoformat(end.strip())
    return str((end - start).total_seconds())
