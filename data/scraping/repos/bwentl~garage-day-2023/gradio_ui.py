import sys
import gradio as gr
import random
import time
import hashlib
from datetime import datetime

from langchain.chat_models import AzureChatOpenAI


sys.path.append("./")
from src.models import LlamaModelHandler
from src.chain_sequence import ChainSequence
from src.agent_multi_step_critic import AgentMultiStepCritic
from src.prompts.customer_triage import (
    TRIAGE_PROCESS_A1,
    TRIAGE_PROCESS_A2,
    TRIAGE_PROCESS_A3,
)
from src.util import agent_logs, get_epoch_time, get_secrets


class WebUI:
    """a simple and awesome ui to display agent actions and thought processes"""

    gradio_app = None
    shared_chat_history = None

    def __init__(self, func, ui_type="garage_day_idea"):
        # clear old logs
        agent_logs.clear_log()
        # initialize app layouts
        if ui_type == "garage_day_idea":
            self.gradio_app = self._init_garage_day_idea(
                self._clear_log_before_func(func)
            )
        # initialize chat history
        self.customer_chat_history = []
        self.agent_chat_history = []
        # last message sent
        self.last_customer_query_msg = ""
        self.last_agent_response_msg = ""
        # initialize llm model
        testAgent = LlamaModelHandler()
        eb = testAgent.load_hf_embedding()
        model_name = "llama-7b"
        lora_name = "alpaca-lora-7b"
        pipeline, model, tokenizer = testAgent.load_llama_llm(
            model_name=model_name, lora_name=lora_name, max_new_tokens=200
        )
        self.pipeline = pipeline
        # self.pipeline = AzureChatOpenAI(
        #     openai_api_base=get_secrets("azure_openapi_url"),
        #     deployment_name=get_secrets("azure_openai_deployment"),
        #     openai_api_key=get_secrets("azure_openapi"),
        #     openai_api_type="azure",
        #     openai_api_version="2023-03-15-preview",
        #     model_name="gpt-35-turbo",
        #     temperature=0.1,
        #     max_tokens=1000,
        # )
        categorize_args = {
            "new_session": True,
            "use_cache_from_log": False,
        }
        urgency_args = {
            "new_session": False,
            "use_cache_from_log": False,
        }
        chain_helper_categorize_config = [
            {
                "name": "task1",
                "type": "simple",
                "chain_name": "categorize_helper",
                "input_template": TRIAGE_PROCESS_A1,
            }
        ]
        chain_helper_urgency_config = [
            {
                "name": "task2",
                "type": "simple",
                "chain_name": "urgency_helper",
                "input_template": TRIAGE_PROCESS_A2,
            }
        ]
        self.translink_helper_categorize_chains = ChainSequence(
            config=chain_helper_categorize_config,
            pipeline=self.pipeline,
            **categorize_args,
        )
        self.translink_helper_urgency_chains = ChainSequence(
            config=chain_helper_urgency_config, pipeline=self.pipeline, **urgency_args
        )
        # initialize multi step critic agent for qa
        # testAgent = LlamaModelHandler()
        # eb = testAgent.get_hf_embedding()
        # define tool list (excluding any documents)
        test_tool_list = ["wiki", "searx"]
        test_doc_info = {
            "translink": {
                "tool_name": "Translink Reports",
                "description": "published policy documents on transportation in Metro Vancouver by TransLink.",
                "files": [
                    "index-docs/translink/2020-11-12_capstan_open-house_boards.pdf",
                    "index-docs/translink/2020-11-30_capstan-station_engagement-summary-report-final.pdf",
                    "index-docs/translink/rail_to_ubc_rapid_transit_study_jan_2019.pdf",
                    "index-docs/translink/t2050_10yr-priorities.pdf",
                    "index-docs/translink/TransLink - Transport 2050 Regional Transportation Strategy.pdf",
                    "index-docs/translink/translink-ubcx-summary-report-oct-2021.pdf",
                    "index-docs/translink/ubc_line_rapid_transit_study_phase_2_alternatives_evaluation.pdf",
                    "index-docs/translink/ubc_rapid_transit_study_alternatives_analysis_findings.pdf",
                ],
            },
        }
        # initiate agent executor
        answer_kwarg = {
            "new_session": False,
            "use_cache_from_log": False,
            "log_tool_selector": False,
            "generate_search_term": True,
            "doc_use_type": "aggregate",
        }
        self.translink_helper_answer_chains = AgentMultiStepCritic(
            pipeline=self.pipeline,
            embedding=eb,
            tool_names=test_tool_list,
            doc_info=test_doc_info,
            verbose=True,
            **answer_kwarg,
        )

    @staticmethod
    def _clear_log_before_func(func):
        def inner1(prompt):
            # clear old logs
            agent_logs.clear_log()
            return func(prompt)

        return inner1

    def generate_response(self, input_text):
        annotated_text_1 = self.translink_helper_categorize_chains.run(input_text)
        annotated_text_2 = self.translink_helper_urgency_chains.run(input_text)
        annotated_text_3 = self.translink_helper_answer_chains.run(input_text)
        final_text = f"""Customer Inquiry: {input_text}\nCategory: {annotated_text_1}.\nPriority: {annotated_text_2}.\nAdditional Information: {annotated_text_3}"""
        return final_text

    def respond(self, message, chat_history):
        self.customer_chat_history = chat_history
        self.last_customer_query_msg = message
        # insert chatbot interruption
        annotated_message = self.generate_response(message)
        self.agent_chat_history.append(
            (self.last_agent_response_msg, annotated_message)
        )
        return "", self.agent_chat_history

    def agent_respond(self, message, chat_history):
        self.agent_chat_history = chat_history
        self.last_agent_response_msg = message
        self.customer_chat_history.append(
            (self.last_customer_query_msg, self.last_agent_response_msg)
        )
        # # revise agent chat
        # self.agent_chat_history = self.agent_chat_history[:-1]
        # self.agent_chat_history.append(
        #     (self.last_agent_response_msg, self.last_customer_query_msg)
        # )
        return "", self.customer_chat_history

    def _init_garage_day_idea(self, func):
        # resource:
        # - https://gradio.app/theming-guide/#discovering-themes
        # - https://gradio.app/quickstart/#more-complexity
        # - https://gradio.app/reactive-interfaces/
        # - https://gradio.app/blocks-and-event-listeners/
        # - https://gradio.app/docs/#highlightedtext

        with gr.Blocks(theme="sudeepshouche/minimalist") as demo:
            gr.Markdown("# TextTransit Instant Messenging System")
            thought_out = None
            # with gr.Tab("Demo"):
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    with gr.Tab("Customer"):
                        customer_chat = gr.Chatbot(
                            label="Customer Feedback",
                            interactive=True,
                            lines=6,
                            # value="Buses constantly late, missed important meetings.",
                        )
                        customer_msg = gr.Textbox()
                        customer_send = gr.Button("Send to TransLink")
                        clear = gr.Button("Clear")
                        # text_input = gr.Textbox(
                        #     info="Your thoughts are important to us, please tell us about your transit experience.",
                        #     placeholder="Enter here",
                        #     lines=3,
                        #     value="Buses constantly late, missed important meetings.",
                        # )
                        # text_button.click(func, inputs=text_input, outputs=text_output)
                        # text_button = gr.Button("Send Feedback")
                        # text_output = gr.Textbox(lines=5, label="Automated response")

                    with gr.Tab("TransLink Staff"):
                        server_status = gr.Radio(
                            ["online", "offline"],
                            label="Status",
                            value="online",
                            interactive=True,
                            info="Set offline to delegate automated response to Bot.",
                        )
                        agent_chat = gr.Chatbot(
                            label="Current Chat with Customer",
                            interactive=True,
                            lines=6,
                        )
                        agent_msg = gr.Textbox()
                        # revise_chat = gr.Button("Request Revision")
                        agent_send = gr.Button("Send to Customer")
                        # clear = gr.Button("Clear")
                        # revise_chat.click(respond, [msg, chatbot], [msg, chatbot])
                        # server_status.change(filter, server_status, [rowA, rowB, rowC])
                        # clear.click(lambda: None, None, chatbot, queue=False)

                    with gr.Tab("Virtual Assistant"):
                        with gr.Column(scale=1, min_width=600):
                            thought_out = gr.HTML(
                                label="Thought Process", scroll_to_output=True
                            )
                            customer_chat.change(
                                self.get_thought_process_log,
                                inputs=[],
                                outputs=thought_out,
                                queue=True,
                                every=1,
                            )

            #         with gr.Tab("Administrator"):
            #             shutdown_server = gr.Button("Shutdown Server")
            # shutdown_server.click(demo.close)

            customer_msg.submit(
                self.respond, [customer_msg, customer_chat], [agent_msg, agent_chat]
            )
            customer_send.click(
                self.respond, [customer_msg, customer_chat], [agent_msg, agent_chat]
            )
            agent_msg.submit(
                self.agent_respond,
                [agent_msg, agent_chat],
                [customer_msg, customer_chat],
            )
            agent_send.click(
                self.agent_respond,
                [agent_msg, agent_chat],
                [customer_msg, customer_chat],
            )
            clear.click(lambda: None, None, customer_chat, queue=False)
        return demo

    def get_thought_process_log(self):
        langchain_log = agent_logs.read_log()
        process_html = langchain_log
        # clean up new lines
        process_html = (
            process_html.replace(" \n", "\n")
            .replace("\n\n\n", "\n")
            .replace("\n\n", "\n")
            .replace(": \n", ": ")
            .replace(":\n", ": ")
        )
        # convert new lines to html
        process_html = process_html.replace("\n", "<br>")
        # add colors to different content
        # https://htmlcolors.com/color-names
        # color Tools Available Black
        process_html = process_html.replace(
            "Tools available:", """<p style="color:Black;">Tools available:"""
        )
        # color Question Black
        process_html = process_html.replace(
            "Question:", """</p><p style="color:Black;">Question:"""
        )
        process_html = process_html.replace(
            "Query:", """</p><p style="color:#348017;">Query:"""
        )
        # color Thought Medium Forest Green
        process_html = process_html.replace(
            "Thought:", """</p><p style="color:#348017;">Thought:"""
        )
        # color Action Bee Yellow
        process_html = process_html.replace(
            "Action:", """</p><p style="color:#E9AB17;">Action:"""
        )
        # color Action Bee Yellow
        process_html = process_html.replace(
            "Action Input:", """</p><p style="color:#E9AB17;">Action Input:"""
        )
        # color Observation Denim Dark Blue
        process_html = process_html.replace(
            "Observation:", """</p><p style="color:#151B8D;">Observation:"""
        )
        # color Observation Black
        process_html = process_html.replace(
            "Final Answer:", """</p><p style="color:Black;"><b>Final Answer</b>:"""
        )
        # color Answer Denim Dark Blue
        process_html = process_html.replace(
            "Answer:", """</p><p style="color:#151B8D;">Answer:"""
        )
        # add closing p
        process_html = f"""{process_html}</p>"""
        return process_html

    @staticmethod
    def generate_auth():
        temp_user = "test"
        temp_password = hashlib.md5(str(get_epoch_time()).strip().encode()).hexdigest()
        auth = (temp_user, temp_password)
        print(f"A temporary auth value as been generated: {auth}")
        return auth

    def launch(
        self,
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        prevent_thread_lock=False,
        auth=None,
        share=False,
    ):
        if share == True and auth is None:
            auth = self.generate_auth()
        self.gradio_app.queue().launch(
            server_name=server_name,
            server_port=server_port,
            inbrowser=inbrowser,
            prevent_thread_lock=prevent_thread_lock,
            auth=auth,
            share=share,
        )


if __name__ == "__main__":

    def test_func(prompt):
        answer = f"Question: {prompt}\nThis is a test output."
        import time

        time.sleep(5)
        return answer

    # test this class
    ui_test = WebUI(test_func)
    ui_test.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=False,
        prevent_thread_lock=False,
        auth=None,
        share=False,
    )
