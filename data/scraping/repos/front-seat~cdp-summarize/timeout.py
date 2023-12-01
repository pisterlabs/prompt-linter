import logging

import requests

logger = logging.getLogger(__name__)


class TimeoutSession(requests.Session):
    """
    A requests session that allows you to ensure that the same timeouts
    are used for all requests.
    """

    def __init__(self, connect_timeout: float, read_timeout: float, force: bool):
        super().__init__()
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self._force = force

    def request(self, method, url, **kwargs):  # type: ignore
        """Construct a Request, prepare, and send it."""
        if self._force or ("timeout" not in kwargs):
            logger.debug(
                "Explicitly setting timeout to %s, %s",
                self.connect_timeout,
                self.read_timeout,
            )
            kwargs["timeout"] = (self.connect_timeout, self.read_timeout)

        return super().request(method, url, **kwargs)


def configure_openai_api_timeouts(connect_timeout: float, read_timeout: float) -> None:
    import openai

    openai.requestssession = lambda: TimeoutSession(
        connect_timeout=connect_timeout, read_timeout=read_timeout, force=True
    )
