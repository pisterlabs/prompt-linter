from datetime import datetime

from langchain.tools import StructuredTool

from common.base_thread import get_logger


def watch() -> datetime:
    """useful for when you want to know what day or time it is."""

    dt = datetime.now()
    get_logger().info(f"the current dt is {dt}")
    return dt


watch = StructuredTool.from_function(watch)
