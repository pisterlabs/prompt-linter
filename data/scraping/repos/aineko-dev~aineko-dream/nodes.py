"""Example file showing how to create nodes."""

import os
import re
import ast
import tempfile
import time
from typing import Optional, Union

import bandit
from openai import OpenAI
import uvicorn
from aineko.core.node import AbstractNode
from dotenv import load_dotenv
from github import Auth, Github

from aineko_dream.config import API


class GitHubDocFetcher(AbstractNode):
    """Node that fetches code documents from GitHub."""

    def _pre_loop_hook(self, params: Optional[dict] = None) -> None:
        """Initialize connection with GitHub and fetch latest document."""
        # Set environment variables
        load_dotenv()

        # Set parameters
        self.access_token = os.environ.get("GITHUB_ACCESS_TOKEN")
        self.organization = params.get("organization")
        self.repo = params.get("repo")
        self.branch = params.get("branch")
        self.file_path = params.get("file_path")
        self.retries = 0
        self.max_retries = 5
        self.retry_sleep = 5

        # Validate parameters
        if self.access_token is None:
            raise ValueError(
                "GITHUB_ACCESS_TOKEN is not set. Add to .env file or "
                "ensure it is set as an environment variable."
            )
        if self.organization is None:
            raise ValueError("organization is not set in GitHubDocFetcher params.")
        if self.repo is None:
            raise ValueError("repo is not set in GitHubDocFetcher params.")
        if self.branch is None:
            raise ValueError("branch is not set in GitHubDocFetcher params.")

        # Initialize github client
        auth = Auth.Token(token=self.access_token)
        self.github_client = Github(auth=auth)

        # Fetch current document
        self.emit_new_document()

    def _execute(self, params: Optional[dict] = None) -> Optional[bool]:
        """Update document in response to commit events."""
        # Check for new commit events from GitHub
        message = self.consumers["github_event"].consume()
        if message is None:
            return

        # Check if message is valid
        if (
            "repository" not in message["message"]
            or "organization" not in message["message"]["repository"]
            or "name" not in message["message"]["repository"]
            or "ref" not in message["message"]
        ):
            self.log(f"Received invalid event from GitHub: {message}", level="error")
            return

        # Skip if event is not from the repo or branch we are tracking
        if (
            message["message"]["repository"]["organization"] != self.organization
            or message["message"]["repository"]["name"] != self.repo
            or message["message"]["ref"] != f"refs/heads/{self.branch}"
        ):
            return

        # Fetch latest document and send update
        self.log("Received event from GitHub, fetching latest document.")
        self.emit_new_document()

    def fetch_github_contents(self) -> Union[dict, None]:
        """Download code from a github repo."""
        try:
            if isinstance(self.file_path, list):
                repo = self.github_client.get_repo(f"{self.organization}/{self.repo}")
                contents = []
                for file_path in self.file_path:
                    cur_contents = repo.get_contents(file_path, ref=self.branch)
                    if isinstance(cur_contents, list):
                        contents.extend(cur_contents)
                    else:
                        contents.append(cur_contents)
            else:
                repo = self.github_client.get_repo(f"{self.organization}/{self.repo}")
                contents = repo.get_contents(self.file_path, ref=self.branch)
            if isinstance(contents, list):
                return {f.path: f.decoded_content.decode("utf-8") for f in contents}
            return {contents.path: contents.decoded_content.decode("utf-8")}
        except Exception as err:  # pylint: disable=broad-except
            self.log(
                f"Error downloading {self.organization}/{self.repo} branch {self.branch}. Error: {str(err)}",
                level="critical",
            )

        # Retry if failed
        self.log(
            f"Unable to download {self.organization}/{self.repo} branch {self.branch}",
            level="error",
        )
        if self.retries + 1 < self.max_retries:
            self.log("Retrying to get latest document from github...")
            self.retries += 1
            time.sleep(self.retry_sleep)
            self.fetch_github_contents()
        else:
            self.log(
                "Exceeded maximum retries. Returning None.",
                level="critical",
            )
            return None

    def emit_new_document(self) -> None:
        """Emit new document."""
        github_contents = self.fetch_github_contents()
        if github_contents is not None:
            document = {
                "metadata": {
                    "source": "github",
                    "event": "commit",
                    "organization": self.organization,
                    "repo": self.repo,
                    "branch": self.branch,
                },
                "document": github_contents,
            }
            self.producers["document"].produce(document)
            self.log(
                f"Fetched documents for {self.organization}/{self.repo} branch {self.branch}"
            )


class PromptModel(AbstractNode):
    """Node that generates prompts for LLMs."""

    def _pre_loop_hook(self, params: Optional[dict]) -> None:
        self.document = None
        self.template = ""
        for prompt in ["guidelines", "documentation", "instructions"]:
            with open(f"aineko_dream/prompts/{prompt}", "r", encoding="utf-8") as f:
                self.template += f.read()
                self.template += "\n\n"

    def _execute(self, params: Optional[dict] = None) -> Optional[bool]:
        """Required; function repeatedly executes.

        Accesses inputs via `self.consumer`, and outputs via
        `self.producer`.
        Logs can be sent via the `self.log` method.
        """
        # Update documents
        document_message = self.consumers["document"].consume()
        if document_message is not None:
            self.log("Updating documents with latest update...")
            self.document = document_message["message"]["document"]

        # Fetch user prompt
        user_prompt = self.consumers["user_prompt"].consume()
        if user_prompt is None:
            return

        # Skip if documentation is not yet available
        if self.document is None:
            self.log(
                "Documents are not ready. Please wait.",
                level="error",
            )
            self.producers["prompt_error"].produce(
                {
                    "request_id": user_prompt["message"]["request_id"],
                    "message": "Error: Documents are not ready. Please wait.",
                }
            )
            return

        # Create prompt
        self.log("Creating prompt from user input...")
        prompt = self.template.format(
            instructions=user_prompt["message"]["prompt"],
            documentation=self.document,
        )
        message = {
            "chat_messages": [
                {
                    "role": "system",
                    "content": f"You are a helpful software engineering assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "request_id": user_prompt["message"]["request_id"],
        }
        self.producers["generated_prompt"].produce(message)


class OpenAIClient(AbstractNode):
    """Node that queries OpenAI LLMs."""

    def _pre_loop_hook(self, params: Optional[dict] = None) -> None:
        """Initialize connection with OpenAI."""
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = params.get("model")
        self.max_tokens = params.get("max_tokens")
        self.temperature = params.get("temperature")

    def _execute(self, params: Optional[dict] = None) -> Optional[bool]:
        """Query OpenAI LLM."""
        message = self.consumers["generated_prompt"].consume()
        if message is None:
            return
        messages = message["message"]["chat_messages"]
        # Query OpenAI LLM
        self.log("Querying OpenAI LLM...")
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                stream=False,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            self.log("OpenAI LLM response received")
            message["message"]["chat_messages"].append(
                {
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                }
            )
            response = {
                "chat_messages": messages,
                "request_id": message["message"]["request_id"],
            }
            self.producers["llm_response"].produce(response)
        except Exception as err:  # pylint: disable=broad-except
            self.log(
                f"Error querying OpenAI LLM. Prompt was {messages}. Error: {str(err)}",
                level="error",
            )
            self.producers["prompt_error"].produce(
                {
                    "request_id": message["message"]["request_id"],
                    "message": f"Error: Unable to query LLM.",
                }
            )
            return


class LLMResponseFormatter(AbstractNode):
    """Node that formats LLM responses."""

    def _pre_loop_hook(self, params: Optional[dict] = None) -> None:
        """Initialize node state."""
        pass

    def _execute(self, params: Optional[dict] = None) -> Optional[bool]:
        """Format response from LLM."""
        llm_response = self.consumers["llm_response"].consume(how="next")
        if llm_response is None:
            return

        # Format response
        try:
            chat_messages = llm_response["message"]["chat_messages"]
            llm_response["message"].update(self.format_response(chat_messages))
            self.producers["formatted_llm_response"].produce(llm_response["message"])
        except Exception as err:  # pylint: disable=broad-except
            self.log(
                f"Error formatting LLM response. LLM Response was {llm_response}. Error: {str(err)}",
                level="error",
            )
            self.producers["prompt_error"].produce(
                {
                    "request_id": llm_response["message"]["request_id"],
                    "message": f"Error: Unable to format LLM response.",
                }
            )

    def format_response(self, chat_messages):
        """Format response from LLM."""
        # Get last message from LLM
        llm_response = chat_messages[-1]["content"]

        # Extract Python code from response
        python_pattern = r"```python(.*?)```"
        python_match = re.search(python_pattern, llm_response, re.DOTALL)
        python_code = python_match.group(1) if python_match else None

        # Extract YAML from response
        yaml_pattern = r"```yaml(.*?)```"
        yaml_match = re.search(yaml_pattern, llm_response, re.DOTALL)
        yaml_code = yaml_match.group(1) if yaml_match else None

        return {
            "python_code": python_code,
            "yaml_code": yaml_code,
        }


class PythonEvaluation(AbstractNode):
    """Node that evaluates Python code."""

    def _pre_loop_hook(self, params: Optional[dict] = None) -> None:
        self.evaluation_name = "python"

    def _execute(self, params: Optional[dict] = None) -> None:
        """Evaluate Python code."""
        message = self.consumers["formatted_llm_response"].consume()
        if message is None:
            return

        # Evaluate Python code
        python_code = message["message"]["python_code"]
        try:
            ast.parse(python_code)
        except SyntaxError as err:
            self.producers["evaluation_result"].produce(
                {
                    "request_id": message["message"]["request_id"],
                    f"evaluation_result_{self.evaluation_name}": {
                        "result": "FAIL",
                        "message": {
                            "role": "user",
                            "content": f"I found a SyntaxError error in the Python code. Restate the previous response with fixes for the error. Here is the error: {str(err)}",
                        },
                    },
                }
            )
        except Exception as err:  # pylint: disable=broad-except
            self.producers["evaluation_result"].produce(
                {
                    "request_id": message["message"]["request_id"],
                    f"evaluation_result_{self.evaluation_name}": {
                        "result": "ERROR",
                        "message": str(err),
                    },
                }
            )
            return
        self.producers["evaluation_result"].produce(
            {
                "request_id": message["message"]["request_id"],
                f"evaluation_result_{self.evaluation_name}": {
                    "result": "PASS",
                    "message": None,
                },
            }
        )


class SecurityEvaluation(AbstractNode):
    """Node that evaluates security of code."""

    def _pre_loop_hook(self, params: Optional[dict] = None) -> None:
        self.evaluation_name = "security"

    def _execute(self, params: Optional[dict] = None) -> None:
        """Evaluate Python code."""
        message = self.consumers["formatted_llm_response"].consume()
        if message is None:
            return

        # Evaluate Python code
        python_code = message["message"]["python_code"]
        issues_list = []
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, mode="w"
            ) as tmpfile:
                tmpfile.write(python_code)

            # Setup Bandit and run tests on the temporary file
            b_mgr = bandit.manager.BanditManager(bandit.config.BanditConfig(), "file")
            b_mgr.discover_files([tmpfile.name], None)
            b_mgr.run_tests()

            # Store results
            results = b_mgr.get_issue_list(
                sev_level=bandit.constants.LOW,
                conf_level=bandit.constants.LOW,
            )

            # Cleanup (remove the temporary file)
            tmpfile.close()
            os.remove(tmpfile.name)

            issues_list = [
                {
                    "line": issue.lineno,
                    "test_id": issue.test_id,
                    "severity": issue.severity,
                    "confidence": issue.confidence,
                    "issue_text": issue.text,
                }
                for issue in results
            ]
        except Exception as err:  # pylint: disable=broad-except
            try:
                tmpfile.close()
                os.remove(tmpfile.name)
            except Exception as err:  # pylint: disable=broad-except
                pass
            self.producers["evaluation_result"].produce(
                {
                    "request_id": message["message"]["request_id"],
                    f"evaluation_result_{self.evaluation_name}": {
                        "result": "ERROR",
                        "message": str(err),
                    },
                }
            )

        if issues_list:
            self.producers["evaluation_result"].produce(
                {
                    "request_id": message["message"]["request_id"],
                    f"evaluation_result_{self.evaluation_name}": {
                        "result": "FAIL",
                        "message": {
                            "role": "user",
                            "content": f"I found Bandit issues in the Python code. Restate the previous response with fixes for the issues. Here are the issues: {issues_list}",
                        },
                    },
                }
            )
        else:
            self.producers["evaluation_result"].produce(
                {
                    "request_id": message["message"]["request_id"],
                    f"evaluation_result_{self.evaluation_name}": {
                        "result": "PASS",
                        "message": None,
                    },
                }
            )


class EvaluationModel(AbstractNode):
    """Node that evaluates LLM responses based on evaluations."""

    def _pre_loop_hook(self, params: Optional[dict] = None) -> None:
        self.state = {}
        self.second_requests = {}

    def _execute(self, params: Optional[dict] = None) -> None:
        """Evaluate LLM responses."""
        # Check for response errors and remove from state
        response_error = self.consumers["prompt_error"].consume()
        if response_error is not None:
            if response_error["message"]["request_id"] in self.state:
                del self.state[response_error["message"]["request_id"]]

        # Check for new prompts and add to state
        message = self.consumers["generated_prompt"].consume()
        if message is not None:
            self.update_state(message["message"])

        # Check for new LLM responses and add to state
        message = self.consumers["llm_response"].consume()
        if message is not None:
            self.update_state(message["message"])

        # Check for new evaluation results and add to state
        message = self.consumers["evaluation_result"].consume()
        if message is not None:
            self.update_state(message["message"])

        # Evaluate model and submit final response
        self.evaluate_model()

        # Cleanup state
        self.cleanup_state()

    def update_state(self, message: dict) -> None:
        """Update state with latest message."""
        if message["request_id"] not in self.state:
            self.state[message["request_id"]] = {}
        self.state[message["request_id"]].update(message)
        self.state[message["request_id"]]["timestamp"] = time.time()

    def cleanup_state(self) -> None:
        """Cleanup state."""
        self.state = {
            k: v for k, v in self.state.items() if time.time() - v["timestamp"] <= 100
        }

    def evaluate_model(self) -> None:
        """Evaluate model and submit final response."""
        ids_to_remove = []
        for request_id, message in self.state.items():
            # Skip if evaluation results are not ready
            if (
                "evaluation_result_python" not in message
                or "evaluation_result_security" not in message
            ):
                continue
            # If both evaluations pass, submit final response
            if (
                message["evaluation_result_python"]["result"] == "PASS"
                and message["evaluation_result_security"]["result"] == "PASS"
            ):
                self.producers["final_response"].produce(
                    {
                        "request_id": request_id,
                        "response": message["chat_messages"][-1]["content"],
                    }
                )
                self.log(f"Final response submitted for request {request_id}")
                if request_id in self.second_requests:
                    del self.second_requests[request_id]
                ids_to_remove.append(request_id)
                continue
            # If either evaluation fails, make new prompt to fix issues
            if message["evaluation_result_python"]["result"] == "FAIL":
                message["chat_messages"].append(
                    message["evaluation_result_python"]["message"]
                )
            if message["evaluation_result_security"]["result"] == "FAIL":
                message["chat_messages"].append(
                    message["evaluation_result_security"]["message"]
                )
            if request_id not in self.second_requests:
                self.log(f"Creating second prompt for request {request_id}")
                self.producers["generated_prompt"].produce(
                    {
                        "chat_messages": message["chat_messages"],
                        "request_id": request_id,
                    }
                )
                self.second_requests[request_id] = True
            else:
                self.log(f"Second prompt already created for request {request_id}")
                self.producers["evaluation_error"].produce(
                    {
                        "request_id": request_id,
                        "response": "Error: Couldn't generate a valid response. Evaluations did not pass. Sorry!",
                    }
                )
                del self.second_requests[request_id]
            ids_to_remove.append(request_id)

        # Remove requests from state
        for request_id in ids_to_remove:
            del self.state[request_id]


class ResponseCache(AbstractNode):
    """Node that emits the SMA of the yield data."""

    def _pre_loop_hook(self, params: Optional[dict] = None) -> None:
        """Optional; used to initialize node state."""
        self.state = {}
        self.cleanup_interval = params.get("cleanup_interval", 100)
        self.last_cleanup_time = time.time()

    def _execute(self, params: Optional[dict] = None) -> Optional[bool]:
        """Polls consumers, updates the state, and calculates the SMA."""
        # Update state df with new messages
        for dataset in ["final_response", "evaluation_error", "prompt_error"]:
            message = self.consumers[dataset].consume()
            if message is None:
                return
            self.update_state(message["message"])
            self.producers["response_cache"].produce(self.state)

    def update_state(self, message):
        """Updates the state with the latest APY values."""
        # Append to state dataframe
        self.state[message["request_id"]] = message
        self.state[message["request_id"]]["timestamp"] = time.time()

        # Cleanup state
        if time.time() - self.last_cleanup_time >= self.cleanup_interval:
            # Cleanup state dataframe
            self.state = {
                k: v
                for k, v in self.state.items()
                if time.time() - v["timestamp"] <= self.cleanup_interval
            }
            self.last_cleanup_time = time.time()


class APIServer(AbstractNode):
    """Node that runs the API server.

    Uvicorn is the HTTP server that runs the FastAPI app.
    The endpoints and logic for the app is contained in coframe/api.
    """

    def _pre_loop_hook(self, params: Optional[dict] = None):
        """Initialize node state."""
        pass

    def _execute(self, params: dict):
        """Start the API server."""
        # pylint: disable=too-many-function-args
        self.log("Starting API server...")
        config = uvicorn.Config(
            params.get("app"),
            port=API.get("PORT"),
            log_level="info",
            host="127.0.0.1",
        )
        server = uvicorn.Server(config)
        server.run()
