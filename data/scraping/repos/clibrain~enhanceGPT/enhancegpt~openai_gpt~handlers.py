from typing import Any, Callable

import openai
from openai import error
from typing_extensions import override

from enhancegpt.base.config_manager import BaseConfigsManager
from enhancegpt.base.errors import NoConfigAvailable
from enhancegpt.base.loggers import logger as root_logger
from enhancegpt.base.retry_handler import BaseRetryHandler
from enhancegpt.openai_gpt.config import OpenAIConfig


class OpenAIApiRetryHandler(BaseRetryHandler):
    def __init__(self, func: Callable, manager: BaseConfigsManager,) -> None:
        super().__init__(func, manager)

    @override
    def handle_exception(self, exception: openai.OpenAIError) -> None:
        """Handle exception and update active config state."""
        if self.active_config is None:
            raise NoConfigAvailable("No usable config")

        log = f"{exception.__class__.__name__}: {exception}"
        root_logger.error(log)

        self.active_config.state.update_stats(success=False)
        match type(exception):
            case (error.Timeout | error.APIError | error.RateLimitError | error.ServiceUnavailableError):
                self.active_config.state.start_cooldown()
            case error.APIConnectionError:
                # retry = getattr(exception, "should_retry", False)
                # if not retry:
                #     self.active_config.state.disable(log)
                self.active_config.state.start_cooldown(3)
            case (error.AuthenticationError | error.PermissionError):
                self.active_config.state.disable(log)
            case error.InvalidRequestError:
                # TODO: We need to take care about this error.
                # TODO: This error is raised in multiple cases.
                # TODO: some of them are not fatel.
                # TODO: So we need to check the error message.
                pass

    @override
    def get_kwds(self) -> dict[str, Any]:
        """Get kwargs for the function."""
        if self.active_config is None:
            raise NoConfigAvailable("No usable config")

        # get active openai config
        _config: OpenAIConfig = self.active_config  # type: ignore

        kwds = _config.serialize()
        api_key = _config.api_key.get_secret_value()

        kwds["api_key"] = api_key

        return kwds
