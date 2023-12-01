"""Agent that can browse a webpage and communicate its state."""

from typing import Callable, Any, Protocol
from functools import cached_property
from pathlib import Path
from dataclasses import dataclass
import logging

import langchain
from autogen import UserProxyAgent, AssistantAgent
from langchain.schema import SystemMessage
from langchain.cache import SQLiteCache
from browserpilot.agents.gpt_selenium_agent import (
    GPTSeleniumAgent,
    logger as browserpilot_logger,
)

from hivemind.config import (
    BASE_WORK_DIR,
    LANGCHAIN_CACHE_DIR,
    BROWSERPILOT_DATA_DIR,
    CHROMEDRIVER_LOCATION,
)
from hivemind.toolkit.types import HivemindAgent
from hivemind.toolkit.text_formatting import dedent_and_strip
from hivemind.toolkit.autogen_support import get_last_user_reply
from hivemind.toolkit.browserpilot_support import run_browserpilot_with_instructions
from hivemind.toolkit.semantic_filtering import filter_semantic_html
from hivemind.toolkit.webpage_inspector import WebpageInspector
from hivemind.toolkit.text_validation import validate_text, find_llm_validation_error
from hivemind.toolkit.autogen_support import (
    is_termination_msg,
    ConfigDict,
    DEFAULT_CONFIG_LIST,
    continue_agent_conversation,
)

langchain.llm_cache = SQLiteCache(
    database_path=str(LANGCHAIN_CACHE_DIR / ".langchain.db")
)


def make_hivemind_user_proxy(
    agent: HivemindAgent,
    function_map: dict[str, Callable[[Any], Any]],
    llm_config: ConfigDict,
) -> UserProxyAgent:
    """Make a user proxy agent."""
    return UserProxyAgent(
        name=f"{agent.name}_user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=is_termination_msg,
        code_execution_config={"work_dir": agent.get_output_dir},
        llm_config=llm_config,
        system_message="Reply TERMINATE if the task has been solved or cannot be solved by the assistant.",
        function_map=function_map,
    )


def test_validate_text() -> None:
    """Test validate_text."""
    requirements = "Text must contain a number."
    number_text = "What's 3 plus 2?"
    validated, result = validate_text(number_text, requirements)
    print(result)
    assert validated
    non_number_text = "What's the capital of France?"
    validated, result = validate_text(non_number_text, requirements)
    print(result)
    assert not validated


@dataclass
class BrowserDaemon:
    """Agent for browsing and reading webpages instead of scraping them. Can perform basic actions like clicking on elements and typing in textareas from natural language."""

    @property
    def name(self) -> str:
        """Name of the agent."""
        return "browser_daemon"

    @property
    def page_id(self) -> str:
        """ID of the page."""
        raise NotImplementedError

    @cached_property
    def browserpilot_agent(self) -> GPTSeleniumAgent:
        """Browserpilot instance for interacting with the driver."""
        return GPTSeleniumAgent(
            chromedriver_path=str(CHROMEDRIVER_LOCATION),
            close_after_completion=False,
            model_for_instructions="gpt-4",
            model_for_responses="gpt-3.5-turbo",
            user_data_dir=str(BROWSERPILOT_DATA_DIR),
        )

    @property
    def page_source(self) -> str:
        """Return the page source."""
        return self.browserpilot_agent.driver.page_source

    @property
    def page_semantic_source(self) -> str:
        """Return the page semantic source."""
        return filter_semantic_html(self.page_source).prettify()

    _inspector: WebpageInspector | None = None

    @property
    def inspector(self) -> WebpageInspector:
        """Return the webpage inspector."""
        if self._inspector is None:
            self._inspector = WebpageInspector(self.page_semantic_source, [])
        if self._inspector.html != self.page_semantic_source:
            self._inspector.update_page(self.page_semantic_source)
        return self._inspector

    @property
    def work_dir(self) -> Path:
        """Return the working directory for the daemon."""
        return BASE_WORK_DIR / "browser_daemon"

    @property
    def llm_config(self) -> ConfigDict:
        """Return the config for the agent's interfacing LLM."""
        return {
            "functions": [
                {
                    "name": "go_to_url",
                    "description": "Go to a particular URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to go to.",
                            }
                        },
                        "required": ["url"],
                    },
                },
                {
                    "name": "skim",
                    "description": "Skim the contents of the currently zoomed in section.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
                {
                    "name": "zoom_into_subsection",
                    "description": "Zoom into a subsection of the currently zoomed in section.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subsection": {
                                "type": "string",
                                "description": "The subsection to zoom into. Must be part of the currently zoomed in section.",
                            },
                        },
                        "required": ["subsection"],
                    },
                },
                {
                    "name": "zoom_out",
                    "description": "Zoom out one level from the current zoom level.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
                {
                    "name": "click_element",
                    "description": "Click on a specific element on the page",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element_description": {
                                "type": "string",
                                "description": "A natural language description of the element that uniquely identifies it on the page. If using the element text, always use the exact text. Example: \"the 'Contact Us' link\". Do not use full sentences.",
                            }
                        },
                        "required": ["element_description"],
                    },
                },
                {
                    "name": "type_text",
                    "description": "Type text into a text field.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text_field_description": {
                                "type": "string",
                                "description": "A natural language description of the text field that uniquely identifies it.",
                            },
                            "text": {
                                "type": "string",
                                "description": "The text to type into the text field.",
                            },
                        },
                        "required": ["text_field_description"],
                    },
                },
            ],
            "raise_on_ratelimit_or_timeout": None,
            "request_timeout": 600,
            "seed": 42,
            "config_list": DEFAULT_CONFIG_LIST,
            "temperature": 0,
        }

    @property
    def function_map(self) -> dict[str, Callable[..., Any]]:
        """Map names defined in the LLM config to the actual functions."""
        return {
            "go_to_url": self.go_to_url,
            "skim": self.skim,
            "zoom_into_subsection": self.zoom_in,
            "zoom_out": self.zoom_out,
            "click_element": self.click_element,
            "type_text": self.type_text,
        }

    @property
    def allowed_actions(self) -> str:
        """Return a list of allowed actions."""
        return dedent_and_strip(
            """
            - go to a particular URL
            - skim the contents of the currently zoomed in section of the page
            - zoom in to a subsection of the currently zoomed in section
            - zoom out on the page
            - read the text in a subsection of the page
            - scroll up or down
            - click on an element
            - type into a text field
            """
        )

    def validate(self, message: str) -> tuple[bool, str]:
        """Validate a message, and return an error message if it is invalid."""
        instructions = """
        You are a message validation bot. Your purpose is to check for specific components that must be present in a message, and to return an error message if they are missing.

        The message you received is:
        ```text
        {message}
        ```

        The message is supposed to be a natural language command to a web browsing agent to do ONE of the following:
        {allowed_actions}

        If the message can't be classified as a command for one of these actions, or contains multiple actions, then it is invalid.

        Output your validation result as a markdown `text` block. If the message has no errors, output the following:
        ```text
        N/A
        ```

        If the message is invalid due not being one of these commands, output the following (fill in the error message):
        ```text
        ERROR: Your message was invalid due to not being one of the allowed actions.
        ```
        """
        instructions = dedent_and_strip(instructions).format(
            message=message, allowed_actions=self.allowed_actions
        )
        messages = [SystemMessage(content=instructions)]
        error = find_llm_validation_error(messages)
        error_info = dedent_and_strip(
            """
            Error: Your message was invalid due to not being one of the allowed actions.
            Original Message: "{message}"
            Allowed Actions: I am only able to do one and only one of the following actions:
            {allowed_actions}
            """
        ).format(message=message, allowed_actions=self.allowed_actions)
        return error is None, "" if error is None else error_info

    def go_to_url(self, url: str) -> str:
        """Go to a URL."""
        run_browserpilot_with_instructions(
            self.browserpilot_agent, instructions=f"Go to the URL `{url}`."
        )
        return f"Successfully navigated to `{url}`.\n\nThe current URL is: `{self.browserpilot_agent.driver.current_url}`.\n\nThe current page title is: `{self.browserpilot_agent.driver.title}`.\n\nYou are currently zoomed in on the following section of the page: `root` (the whole page)."

    def skim(self) -> str:
        """Skim the contents of the page."""
        return self.inspector.section_outline

    @property
    def page_title(self) -> str:
        """Return the page title."""
        return self.browserpilot_agent.driver.title

    @property
    def current_url(self) -> str:
        """Return the current URL."""
        return self.browserpilot_agent.driver.current_url

    @property
    def zoom_path(self) -> str:
        """Return the current zoom path."""
        return self.inspector.breadcrumb_display

    @property
    def current_subsection(self) -> str:
        """Return the current subsection."""
        return self.inspector.current_section_name

    @property
    def zoomed_in(self) -> bool:
        """Return whether the user is zoomed in."""
        return bool(self.inspector.breadcrumbs)

    def action_feedback(self, status_message: str) -> str:
        """Return feedback after some action."""
        feedback = """
        Current page title: `{title}`.
        Current URL: `{url}`.
        You are {view_status} the following section of the page: `{subsection}`.
        Full zoom path to this section: `{zoom_path}`.
        """
        feedback = status_message + "\n\n" + dedent_and_strip(feedback)
        return feedback.format(
            title=self.page_title,
            url=self.current_url,
            view_status="zoomed in on" if self.zoomed_in else "viewing",
            subsection=(
                self.current_subsection
                if self.zoomed_in
                else f"{self.current_subsection} (the whole page)"
            ),
            zoom_path=self.zoom_path,
        )

    def zoom_in(self, subsection: str) -> str:
        """Zoom in to a particular section of the page."""
        self.inspector.zoom_in(subsection)
        return self.action_feedback(status_message="Successfully zoomed in.")

    def zoom_out(self) -> str:
        """Zoom out of the current view."""
        self.inspector.zoom_out()

        return self.action_feedback(status_message="Successfully zoomed out.")

    def element_found(self, element_description: str) -> bool:
        """Check whether an element exists."""
        instructions = f"""Find the {element_description}". Do not click on or otherwise interact with it."""

        # instructions = f"""Find the element that matches the following identifying description: "{element_description}". Do not click on or otherwise interact with it."""
        try:
            run_browserpilot_with_instructions(
                self.browserpilot_agent, instructions=instructions
            )
            return True
        except Exception as error:  # pylint: disable=broad-except
            if "Failed to execute" in error.args[0]:
                return False
            raise error

    def element_not_found_message(self, element_description: str) -> str:
        """Return the message to display when an element isn't found."""
        return f'I was unable to find an element that matches the following identifying description: "{element_description}"\nPlease note that this does NOT mean the element doesn\'t exist. It may be possible to find the element by adjusting the identifying description, such as changing the text to match the exact text of the element, or using a different element tag, etc.'

    def interact_with_element(
        self, element_description: str, instructions: str, success_message: str
    ) -> dict[str, str | bool]:
        """Use the browserpilot agent to interact with an element."""
        if not self.element_found(element_description):
            return {
                "success": False,
                "message": self.action_feedback(
                    status_message=self.element_not_found_message(element_description)
                ),
            }
        error_message = run_browserpilot_with_instructions(
            self.browserpilot_agent, instructions=instructions
        )
        success = not error_message
        return {
            "success": success,
            "message": self.action_feedback(
                status_message=success_message if success else error_message
            ),
        }

    def click_element(self, element_description: str) -> dict[str, str | bool]:
        """Click an element with a particular description."""
        return self.interact_with_element(
            element_description=element_description,
            instructions=f"Click on the {element_description}.",
            success_message="Successfully clicked on element.",
        )

    def type_text(
        self, text_field_description: str, text: str
    ) -> dict[str, str | bool]:
        """Type text into a text field."""
        return self.interact_with_element(
            element_description=text_field_description,
            instructions=f"Type the following text into the {text_field_description}: `{text}`.",
            success_message="Successfully typed text into text field.",
        )

    def run(
        self,
        message: str,
    ) -> tuple[str, Callable[[str], str]]:
        """Run the agent."""
        user_proxy = make_hivemind_user_proxy(
            agent=self,
            function_map=self.function_map,
            llm_config=self.llm_config,
        )
        assistant = AssistantAgent(
            name=f"{self.name}_assistant",
            llm_config=self.llm_config,
            system_message="Use one of your functions to fulfill the user's request. Only report on whether you were successful or not. Do not summarize or repeat information that was output as a result of the function call. Do not offer to perform other tasks than what you've been asked.",
        )
        continue_conversation = continue_agent_conversation(user_proxy, assistant)
        validated, error = self.validate(message)
        if not validated:
            return error, continue_conversation
        user_proxy.initiate_chat(
            assistant,
            message=message,
        )
        return get_last_user_reply(user_proxy, assistant), continue_conversation


def test_validate() -> None:
    """Test validation."""
    validated, _ = BrowserDaemon().validate("Make a sandwich.")
    assert not validated
    validated, _ = BrowserDaemon().validate("Go to https://google.com")
    assert validated


def test_go_to_url() -> None:
    """Test go_to_url."""
    agent = BrowserDaemon()
    result, _ = agent.run("Go to https://google.com")
    agent.browserpilot_agent.driver.quit()
    assert "google" in result


def test_sequential_actions() -> None:
    """Test performing actions in sequence."""
    agent = BrowserDaemon()
    _, next_command = agent.run("Go to https://github.com/microsoft/autogen")
    result = next_command("Go to https://google.com")
    assert "google" in result


def test_root_breadcrumbs() -> None:
    """Test whether going to a URL comes back with the correct zoom breadcrumbs."""
    agent = BrowserDaemon()
    result, _ = agent.run("Go to https://github.com/microsoft/autogen")
    validated, error = validate_text(
        text=result,
        requirements="The text must mention that user is on the root zoom level of the page.",
    )
    assert validated, error


def test_browserpilot() -> None:
    """Test the browserpilot agent."""
    agent = GPTSeleniumAgent(
        chromedriver_path=str(CHROMEDRIVER_LOCATION),
        close_after_completion=False,
        model_for_instructions="gpt-4",
        model_for_responses="gpt-3.5-turbo",
        user_data_dir=str(BROWSERPILOT_DATA_DIR),
    )
    run_browserpilot_with_instructions(agent, "Go to https://google.com")
    agent.driver.quit()


def test_page_source() -> None:
    """Test prettifying the page source."""
    agent = BrowserDaemon()
    agent.run("Go to https://github.com/microsoft/autogen")
    print(agent.page_semantic_source)
    Path("page.html").write_text(agent.page_semantic_source, encoding="utf-8")


def test_skim_page() -> None:
    """Test skimming the contents of a page."""
    agent = BrowserDaemon()
    _, next_command = agent.run("Go to https://github.com/microsoft/autogen")
    result = next_command("Skim the contents.")
    validated, error = validate_text(
        text=result,
        requirements="The text must be a hierarchical outline.",
    )
    assert validated, error


def test_zoom_into_subsection() -> None:
    """Test zooming into a particular part of a page."""
    agent = BrowserDaemon()
    _, next_command = agent.run("Go to https://github.com/microsoft/autogen")
    result = next_command("Zoom into the 'Navigation' section.")
    # result should include 'Navigation' within zoom breadcrumbs
    assert "Navigation" in result


def test_zoom_out() -> None:
    """Test zooming out of a view."""
    agent = BrowserDaemon()
    _, next_command = agent.run("Go to https://github.com/microsoft/autogen")
    agent.zoom_in("Navigation")
    result = next_command("Zoom out.")
    assert "whole page" in result


def test_element_exception() -> None:
    """Test exception thrown for an element that isn't found."""
    agent = BrowserDaemon()
    instructions = """Go to Google.com
    Find the button with the text "blah".
    """
    try:
        run_browserpilot_with_instructions(
            agent.browserpilot_agent, instructions=instructions
        )
    except Exception as error:  # pylint: disable=broad-except
        if "Failed to execute" not in error.args[0]:
            raise error


def test_check_element() -> None:
    """Test checking whether an element exists."""
    agent = BrowserDaemon()
    logger_level = browserpilot_logger.getEffectiveLevel()
    # suppress logging since we expect exception messages
    browserpilot_logger.setLevel(logging.WARNING)
    agent.go_to_url("https://google.com")
    assert not agent.element_found("a button with the text 'blah'")
    assert agent.element_found("the 'About' link")
    browserpilot_logger.setLevel(logger_level)


def test_click_element() -> None:
    """Test clicking on an element."""
    agent = BrowserDaemon()
    agent.go_to_url("https://github.com/microsoft/autogen")
    result, _ = agent.run("Click on the 'Issues' link")
    assert "Successfully clicked on element" in result


def test_inspector_source_update() -> None:
    """Test that the inspector source updates after the page source updates."""
    agent = BrowserDaemon()
    agent.go_to_url("https://github.com/microsoft/autogen")
    agent.go_to_url("https://google.com")
    assert (
        agent.inspector.html == agent.page_semantic_source
    ), "Inspector source not updated after page source changed."


def test_type_text() -> None:
    """Test typing text into some text field."""
    agent = BrowserDaemon()
    agent.go_to_url("https://google.com")
    agent.click_element("textarea with the title of 'Search'")
    agent.run("Type 'Hello world!' into the textarea with the title of 'Search'")
    run_browserpilot_with_instructions(agent.browserpilot_agent, "Wait for 5 seconds.")
    # this is a manual inspection test


# ....
# > TODO: ask a question <- should be qna oracle
# > TODO: set memory folder
# > TODO: refactor: update functions to return success/failure
# > TODO: convert image of page to element list # needs gpt-4v


def test() -> None:
    """Test the agent."""
    # test_root_breadcrumbs()
    # test_validate()
    # test_go_to_url()
    # test_skim_page()
    # test_zoom_into_subsection()
    # test_zoom_out()
    # test_element_exception()
    # test_check_element()
    # test_click_element()
    # test_inspector_source_update()
    # test_type_text()


if __name__ == "__main__":
    test()
