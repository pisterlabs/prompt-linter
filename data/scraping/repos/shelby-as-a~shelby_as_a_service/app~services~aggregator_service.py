# region Imports
import os
import sys
from datetime import datetime, timedelta, timezone
import yaml
import re
import random
from importlib import import_module
import openai
from dotenv import load_dotenv
import time

from services.tiny_jmap_library.tiny_jmap_library import TinyJMAPClient
from services.data_processing_service import TextProcessing
from services.log_service import Logger
from bs4 import BeautifulSoup
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from models.models import IndexModel

# endregion


class Aggregator:
    def __init__(self, service_name):
        self.service_name = service_name
        self.log = Logger(
            self.service_name,
            f"{self.service_name}_aggregator_service",
            f"{self.service_name}_aggregator_service.md",
            level="INFO",
        )
        load_dotenv()

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.service_dir = "app/content_aggregator/"
        self.prompt_path = "app/prompt_templates/aggregator/"
        config_module_path = f"content_aggregator.config"
        self.config = import_module(config_module_path).MonikerAggregatorConfig
        self.vector_db = VectorIndex(self)

        self.start_time_unix_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.end_time_unix_ms = self.start_time_unix_ms - (
            self.config.story_time_range * 60 * 60 * 1000
        )  # subtract 24 hours in milliseconds

    def aggregate_email_newsletters(self):
        email_ag = AggregateEmailNewsletter(self)
        incoming_emails = email_ag.get_emails()
        relevant_emails = email_ag.pre_check_email(incoming_emails)
        stories = email_ag.split_email(relevant_emails)

        self.vector_db.upsert_email_text(stories)
        self.log.print_and_log(f"Total cost: ${self.calculate_cost()}")
        email_ag.archive_emails(relevant_emails)

    def create_newsletter(self):
        newsletter_cr = CreateNewsletter(self)
        top_vectorstore_content = newsletter_cr.find_top_stories()
        # Potentially perform operations on top_vectorstore_content here
        summarized_stories = newsletter_cr.summarize_merged_stories(
            top_vectorstore_content
        )
        summarized_stories = newsletter_cr.create_titles(summarized_stories)
        summarized_stories = newsletter_cr.create_emojis(summarized_stories)
        intro_text = newsletter_cr.create_intro(summarized_stories)
        hash_tags = newsletter_cr.create_hash_tags(summarized_stories)
        newsletter_cr.create_post(summarized_stories, intro_text, hash_tags)

        self.log.print_and_log(f"Total cost: ${self.calculate_cost()}")

    def check_response(self, response):
        # Check if keys exist in dictionary
        parsed_response = (
            response.get("choices", [{}])[0].get("message", {}).get("content")
        )

        self.total_prompt_tokens += int(response.get("usage").get("prompt_tokens", 0))
        self.total_completion_tokens += int(
            response.get("usage").get("completion_tokens", 0)
        )

        if not parsed_response:
            raise ValueError(f"Error in response: {response}")

        return parsed_response

    def calculate_cost(self):
        prompt_cost = 0.03 * (self.total_prompt_tokens / 1000)
        completion_cost = 0.06 * (self.total_completion_tokens / 1000)
        total_cost = prompt_cost + completion_cost
        # total_cost = math.ceil(total_cost * 100) / 100
        return total_cost


class AggregateEmailNewsletter:
    def __init__(self, main_ag: Aggregator):
        self.main_ag = main_ag
        self.run_output_dir = self.create_ag_folder()

    def create_ag_folder(self):
        aggregations_path = f"{self.main_ag.service_dir}/aggregations"
        today = datetime.now().strftime("%Y_%m_%d")
        # Initialize run number
        run_num = 1

        # Construct initial directory path
        run_output_dir = os.path.join(aggregations_path, f"{today}_run_{run_num}")

        # While a directory with the current run number exists, increment the run number
        while os.path.exists(run_output_dir):
            run_num += 1
            run_output_dir = os.path.join(aggregations_path, f"{today}_run_{run_num}")

        # Create the directory
        os.makedirs(run_output_dir, exist_ok=True)

        return run_output_dir

    def get_emails(self):
        # Get the current UTC date and time
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=self.main_ag.config.email_look_back_hours)
        # This date format required
        now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        client = TinyJMAPClient(
            hostname="api.fastmail.com",
            username=os.environ.get("JMAP_USERNAME"),
            token=os.environ.get("JMAP_TOKEN"),
        )
        account_id = client.get_account_id()

        # Query for the mailbox ID
        inbox_res = client.make_jmap_call(
            {
                "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
                "methodCalls": [
                    [
                        "Mailbox/query",
                        {
                            "accountId": account_id,
                            "filter": {"name": self.main_ag.config.newsletter_inbox},
                        },
                        "a",
                    ]
                ],
            }
        )

        inbox_id = inbox_res["methodResponses"][0][1]["ids"][0]
        assert len(inbox_id) > 0

        email_query_res = client.make_jmap_call(
            {
                "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
                "methodCalls": [
                    [
                        "Email/query",
                        {
                            "accountId": account_id,
                            "filter": {
                                "inMailbox": inbox_id,
                                "after": start_time_str,
                                "before": now_str,
                            },
                            "limit": 50,
                        },
                        "b",
                    ]
                ],
            }
        )

        # Extract the email IDs from the response
        email_ids = email_query_res["methodResponses"][0][1]["ids"]

        # Get the email objects
        email_get_res = client.make_jmap_call(
            {
                "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
                "methodCalls": [
                    [
                        "Email/get",
                        {
                            "accountId": account_id,
                            "ids": email_ids,
                            "properties": [
                                "id",
                                "blobId",
                                "threadId",
                                "mailboxIds",
                                "keywords",
                                "size",
                                "receivedAt",
                                "messageId",
                                "inReplyTo",
                                "references",
                                "sender",
                                "from",
                                "to",
                                "cc",
                                "bcc",
                                "replyTo",
                                "subject",
                                "sentAt",
                                "hasAttachment",
                                "preview",
                                "bodyValues",
                                "textBody",
                                "htmlBody",
                            ],
                            "bodyProperties": [
                                "partId",
                                "blobId",
                                "size",
                                "name",
                                "type",
                                "charset",
                                "disposition",
                                "cid",
                                "language",
                                "location",
                            ],
                            "fetchAllBodyValues": True,
                            "fetchHTMLBodyValues": True,
                            "fetchTextBodyValues": True,
                        },
                        "c",
                    ]
                ],
            }
        )

        emails = email_get_res["methodResponses"][0][1]["list"]
        sorted_emails = sorted(emails, key=lambda email: email["receivedAt"])
        incoming_emails = []
        email_count = 0
        for email in sorted_emails:
            if "htmlBody" in email and email["htmlBody"]:
                body_part_id = email["htmlBody"][0]["partId"]
            elif "textBody" in email and email["textBody"]:
                body_part_id = email["textBody"][0]["partId"]
            else:
                continue  # Skip this email if neither htmlBody nor textBody is present
            body_content = email["bodyValues"][body_part_id]["value"]

            email_info = {
                "email_id": email["id"],
                "subject": email["subject"],
                "incoming_email_number": email_count,
                "from": email["from"][0]["email"],
                "received_at": email["receivedAt"],
                "text": body_content,
                "links": [""],  # Placeholder for future implementation
            }
            incoming_emails.append(email_info)
            email_count += 1

        return incoming_emails

    def pre_check_email(self, incoming_emails=None):
        if not incoming_emails:
            self.main_ag.log.print_and_log("No emails in inbox.")
            sys.exit()

        self.main_ag.log.print_and_log(f"Got: {len(incoming_emails)} emails")

        with open(
            os.path.join(
                self.main_ag.prompt_path, "aggregator_pre_check_email_template.yaml"
            ),
            "r",
            encoding="utf-8",
        ) as stream:
            prompt_template = yaml.safe_load(stream)

        logit_bias_weight = 100
        logit_bias = {str(k): logit_bias_weight for k in range(15, 15 + 2)}
        relevant_emails = []
        incoming_emails_text = []
        email_count = 0

        for email in incoming_emails:
            soup = BeautifulSoup(email["text"], "html.parser")
            bs4_text_content = soup.get_text()
            # Removes excessive whitespace chars
            text_content = TextProcessing.strip_excess_whitespace(bs4_text_content)
            # Removes start/end of the string
            chars_to_remove = min(
                len(text_content), self.main_ag.config.email_footer_removed_chars
            )
            text_content = text_content[:-chars_to_remove]
            chars_to_remove = min(
                len(text_content), self.main_ag.config.email_footer_removed_chars
            )
            text_content = text_content[chars_to_remove:]

            tok = TextProcessing.tiktoken_len(text_content)

            self.main_ag.log.print_and_log(f"email token count: {tok}")

            if email_count >= self.main_ag.config.email_max_per_run:
                continue

            match tok:
                case _ if tok > self.main_ag.config.email_token_count_max:
                    self.main_ag.log.print_and_log(
                        f"{email['subject']}\n exceded email_token_count_max!"
                    )
                    self.archive_emails([email])
                    continue
                case _ if tok > 2500:
                    email_pre_check_window = 0.48
                case _ if tok > 2000:
                    email_pre_check_window = 0.45
                case _ if tok > 1500:
                    email_pre_check_window = 0.40
                case _ if tok > 1000:
                    email_pre_check_window = 0.35
                case _ if tok > 500:
                    email_pre_check_window = 0.30
                case _ if tok < 500:
                    email_pre_check_window = 0.25
                case _ if tok < 200:
                    self.main_ag.log.print_and_log(f"{email['subject']}\ntoo short!")
                    self.archive_emails([email])
                    continue

            length = len(text_content)
            start_index = int(length * email_pre_check_window)
            end_index = int(length * (1 - email_pre_check_window))

            prompt_template[1]["content"] = text_content[start_index:end_index]

            response = openai.ChatCompletion.create(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model=self.main_ag.config.LLM_decision_model,
                messages=prompt_template,
                max_tokens=1,
                logit_bias=logit_bias,
            )

            email["text"] = text_content
            email["relevant_email_number"] = email_count

            checked_response = self.main_ag.check_response(response)
            if checked_response == "1":
                self.main_ag.log.print_and_log(
                    f"🟢 {email['subject']}: LLM thinks it's newsworthy. Keeping it!"
                )

                relevant_emails.append(email)
                email_count += 1
            else:
                self.main_ag.log.print_and_log(
                    f"🔴 {email['subject']}: LLM thinks it's not newsworthy. Rejected!"
                )
                self.archive_emails([email])

            incoming_emails_text.append(email)

        # Writing the dictionary into a YAML file
        with open(
            f"{self.run_output_dir}/1_incoming_emails.yaml", "w", encoding="UTF-8"
        ) as yaml_file:
            yaml.dump(incoming_emails_text, yaml_file, default_flow_style=False)
        # Writing the dictionary into a YAML file
        with open(
            f"{self.run_output_dir}/2_relevant_emails.yaml", "w", encoding="UTF-8"
        ) as yaml_file:
            yaml.dump(relevant_emails, yaml_file, default_flow_style=False)

        self.main_ag.log.print_and_log(f"Found: {len(incoming_emails)} relevant emails")

        return relevant_emails

    def split_email(self, relevant_emails):
        with open(
            os.path.join(
                self.main_ag.prompt_path, "aggregator_split_email_template.yaml"
            ),
            "r",
            encoding="utf-8",
        ) as stream:
            prompt_template = yaml.safe_load(stream)

        pre_split_output = []
        stories = []
        stories_count = 0

        for email in relevant_emails:
            content = f"{email['text']}"
            self.main_ag.log.print_and_log(
                f"\nNow splitting and summarizing: {email['subject']}"
            )

            # Loop over the list of dictionaries in data['prompt_template']
            for role in prompt_template:
                if role["role"] == "user":  # If the 'role' is 'user'
                    role[
                        "content"
                    ] = content  # Replace the 'content' with 'prompt_message'

            response = openai.ChatCompletion.create(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model=self.main_ag.config.LLM_writing_model,
                messages=prompt_template,
                max_tokens=750,
            )

            checked_response = self.main_ag.check_response(response)

            self.main_ag.log.print_and_log(
                f"split_email response token count: {TextProcessing.tiktoken_len(checked_response)}"
            )

            email["summary"] = checked_response
            pre_split_output.append(email)

            # Split the text by patterns of numbered lists like n.
            list_pattern = r"\s+\d\.(?!\d)\s+"
            list_matches = re.findall(list_pattern, checked_response)

            # Split the text by patterns of [n], (n)
            brackets_pattern = r"\[\d+\]|\(\d+\)"
            brackets_matches = re.findall(brackets_pattern, checked_response)

            # Split the text by patterns of .n.
            dot_pattern = r"\.\d+\."
            dot_matches = re.findall(dot_pattern, checked_response)
            
            # Split the text by patterns of n)
            num_bracket_pattern = r"\d+\)"
            num_bracket_matches = re.findall(num_bracket_pattern, checked_response)

            # Check which pattern has the most matches
            pattern_counts = {
                'list': len(list_matches),
                'brackets': len(brackets_matches),
                'dot': len(dot_matches),
                'num_bracket': len(num_bracket_matches),
            }

            most_common_pattern = max(pattern_counts, key=pattern_counts.get)

            # Using match-case statement for better readability and structuref
            match most_common_pattern:
                case 'list':
                    # Removes the first item in the list if it starts with "n. "
                    checked_response = re.sub(r"^\d\.\s", "", checked_response)
                    splits = re.split(list_pattern, checked_response)
                case 'dot':
                    # Removes the first item if it starts with ".n. "
                    checked_response = re.sub(r"^\.\d+\.\s", "", checked_response)
                    splits = re.split(dot_pattern, checked_response)
                case 'brackets':
                    # Removes the first item if it starts with "[n] "
                    checked_response = re.sub(r"^\[\d+\]\s|^\(\d+\)\s", "", checked_response)
                    splits = re.split(brackets_pattern, checked_response)
                case 'num_bracket':  # Assuming you would like to handle this case similarly to 'brackets'
                    # Removes the first item if it starts with "n) "
                    checked_response = re.sub(r"^\d+\)\s", "", checked_response)
                    splits = re.split(num_bracket_pattern, checked_response)
                case _:
                    break

            # self.log.print_and_log each part
            for story in splits:
                story = TextProcessing.remove_all_white_space_except_space(story)
                story_token_count = TextProcessing.tiktoken_len(story)

                if story_token_count > self.main_ag.config.story_token_count_min:
                    story_info = {
                        "title": email["subject"],
                        "data_source_name": email["from"],
                        "date_indexed": email["received_at"],
                        "content": story,
                        "links": [""],  # Placeholder for future implementation
                        "url": "",  # Placeholder for future implementation of browser available newsletters
                        "target_type": "email_text",
                    }

                    stories.append(story_info)
                    stories_count += 1

        # Writing the dictionary into a YAML file
        with open(
            f"{self.run_output_dir}/3_all_stories.yaml", "w", encoding="UTF-8"
        ) as yaml_file:
            yaml.dump(stories, yaml_file, default_flow_style=False)
        # Writing the dictionary into a YAML file
        with open(
            f"{self.run_output_dir}/3b_pre_split_output.yaml", "w", encoding="UTF-8"
        ) as yaml_file:
            yaml.dump(pre_split_output, yaml_file, default_flow_style=False)

        return stories

    def archive_emails(self, relevant_emails):
        client = TinyJMAPClient(
            hostname="api.fastmail.com",
            username=os.environ.get("JMAP_USERNAME"),
            token=os.environ.get("JMAP_TOKEN"),
        )
        account_id = client.get_account_id()

        # Query for the mailbox ID
        inbox_res = client.make_jmap_call(
            {
                "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
                "methodCalls": [
                    [
                        "Mailbox/query",
                        {
                            "accountId": account_id,
                            "filter": {
                                "name": self.main_ag.config.email_ingested_folder
                            },
                        },
                        "a",
                    ]
                ],
            }
        )
        inbox_id = inbox_res["methodResponses"][0][1]["ids"][0]
        assert len(inbox_id) > 0

        email_ids = [email["email_id"] for email in relevant_emails]
        updates = {email_id: {"mailboxIds": {inbox_id: True}} for email_id in email_ids}
        email_query_res = client.make_jmap_call(
            {
                "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
                "methodCalls": [
                    ["Email/set", {"accountId": account_id, "update": updates}, "0"]
                ],
            }
        )
        response_data = email_query_res["methodResponses"][0][1]

        for email_id in email_ids:
            if "updated" in response_data and email_id in response_data["updated"]:
                self.main_ag.log.print_and_log(
                    f"Email with ID {email_id} moved to {self.main_ag.config.email_ingested_folder}!"
                )
            elif (
                "notUpdated" in response_data
                and email_id in response_data["notUpdated"]
            ):
                error = response_data["notUpdated"][email_id]["type"]
                self.main_ag.log.print_and_log(
                    f"Email with ID {email_id} update failed with error: {error}"
                )
            else:
                self.main_ag.log.print_and_log(
                    f"Unexpected response for email with ID {email_id}!"
                )


class VectorIndex:
    def __init__(self, main_ag: Aggregator):
        self.main_ag = main_ag
        self.index_config = IndexModel()

        pinecone.init(
            environment=self.index_config.index_env,
            api_key=os.environ.get("PINECONE_API_KEY"),
        )

        indexes = pinecone.list_indexes()
        if self.main_ag.config.index_name not in indexes:
            self.main_ag.log.print_and_log(
                f"{self.main_ag.config.index_name} not found in Pinecone"
            )

        self.vectorstore = pinecone.Index(self.main_ag.config.index_name)

        self.embedding_retriever = OpenAIEmbeddings(
            model=self.index_config.index_embedding_model,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            chunk_size=self.index_config.index_embedding_batch_size,
            request_timeout=self.index_config.index_openai_timeout_seconds,
        )

    def upsert_email_text(self, content):
        self.main_ag.log.print_and_log(
            f"Initial index stats: {self.vectorstore.describe_index_stats()}\n"
        )
        index_resource_stats = self.vectorstore.describe_index_stats(
            filter={"target_type": {"$eq": "email_text"}}
        )
        existing_resource_vector_count = (
            index_resource_stats.get("namespaces", {})
            .get(self.main_ag.config.index_namespace, {})
            .get("vector_count", 0)
        )
        self.main_ag.log.print_and_log(
            f"Existing vector count for {self.main_ag.config.index_namespace}: {existing_resource_vector_count}"
        )

        text_content = []

        for story in content:
            text_content.append(story["content"])
            # Convert to unix ms from str
            dt = datetime.strptime(story["date_indexed"], "%Y-%m-%dT%H:%M:%SZ")
            story["date_indexed"] = int(
                dt.replace(tzinfo=timezone.utc).timestamp() * 1000
            )

        # Get dense_embeddings
        dense_embeddings = self.embedding_retriever.embed_documents(text_content)

        vectors_to_upsert = []
        vector_counter = existing_resource_vector_count + 1
        for i, story in enumerate(content):
            prepared_vector = {
                "id": f"id-{self.main_ag.config.index_namespace}-{vector_counter}",
                "values": dense_embeddings[i],
                "metadata": story,
            }
            vector_counter += 1
            vectors_to_upsert.append(prepared_vector)

        self.main_ag.log.print_and_log(f"Upserting {len(vectors_to_upsert)} vectors")
        self.vectorstore.upsert(
            vectors=vectors_to_upsert,
            namespace=self.main_ag.config.index_namespace,
            batch_size=self.index_config.index_vectorstore_upsert_batch_size,
            show_progress=True,
        )

        index_resource_stats = self.vectorstore.describe_index_stats(
            filter={"target_type": {"$eq": "email_text"}}
        )
        new_resource_vector_count = (
            index_resource_stats.get("namespaces", {})
            .get(self.main_ag.config.index_namespace, {})
            .get("vector_count", 0)
        )
        self.main_ag.log.print_and_log(
            f"Indexing complete for: {self.main_ag.config.index_namespace}\nPrevious vector count: {existing_resource_vector_count}\nNew vector count: {new_resource_vector_count}\n"
        )

    def matching_in_period(self, topic_embeddings, target_type):
        initial_matching_vectors = []
        filter = {
            "target_type": {"$eq": target_type},
            "date_indexed": {
                "$gte": self.main_ag.end_time_unix_ms,
                "$lte": self.main_ag.start_time_unix_ms,
            },
        }

        query_response = self.vectorstore.query(
            namespace=self.main_ag.config.index_namespace,
            include_values=False,
            include_metadata=False,
            filter=filter,
            vector=topic_embeddings,
            top_k=20,
        )
        # Destructures the QueryResponse object the pinecone library generates.
        for v in query_response.matches:
            if v.score > self.main_ag.config.story_topic_score:
                initial_matching_vectors.append(v)

        return initial_matching_vectors

    def semantically_similar_sources(self, vectors, target_type):
        # Set filter for entire date range used
        filter = {
            "target_type": {"$eq": target_type},
            "date_indexed": {
                "$gte": self.main_ag.end_time_unix_ms,
                "$lte": self.main_ag.start_time_unix_ms,
            },
        }

        stories = []
        for vector in vectors:
            if any(vector.id in t for t in stories):
                continue
            semantically_similar_sources = set()
            query_response = self.vectorstore.query(
                namespace=self.main_ag.config.index_namespace,
                include_values=False,
                include_metadata=False,
                filter=filter,
                id=vector.id,
                top_k=20,
            )

            # Retain only the top N highest scored matches
            top_matches = sorted(
                query_response.matches, key=lambda x: x["score"], reverse=True
            )[:5]
            for v in top_matches:
                if v.id == vector.id:
                    semantically_similar_sources.add(v.id)
                    continue
                # Removing any below score threshold
                if v.score > self.main_ag.config.story_correlation_score:
                    semantically_similar_sources.add(v.id)

            stories.append(semantically_similar_sources)

        return stories

    def sort_and_merge_stories(self, vectors):
        if not vectors:
            return
        if isinstance(vectors[0], dict):
            vectors = [set(d.items()) for d in vectors]
        # If a set of sources contain the same source, we merge the sets
        i = 0
        while i < len(vectors):
            j = i + 1
            while j < len(vectors):
                if (
                    vectors[i] & vectors[j]
                ):  # Check for intersection using set intersection
                    vectors[i] = vectors[i].union(vectors[j])
                    vectors.pop(j)
                else:
                    j += 1
            i += 1

        # Sort by length of the sub-lists in descending order and get the top 10
        return sorted(vectors, key=len, reverse=True)[:10]

    def fetch_vectors_from_id(self, list_of_ids_lists):
        output = []
        for id_list in list_of_ids_lists:
            query_response = self.vectorstore.fetch(
                namespace=self.main_ag.config.index_namespace,
                ids=list(id_list),
            )
            output.append(query_response.vectors)

        return output


class CreateNewsletter:
    def __init__(self, main_ag: Aggregator):
        self.main_ag = main_ag
        self.target_type = "email_text"
        self.run_output_dir = self.create_newsletter_folder()

    def create_newsletter_folder(self):
        aggregations_path = (
            f"{self.main_ag.service_dir}/newsletter/{self.main_ag.config.moniker_name}/"
        )
        today = datetime.now().strftime("%Y_%m_%d")
        # Initialize run number
        run_num = 1

        # Construct initial directory path
        run_output_dir = os.path.join(aggregations_path, f"{today}_run_{run_num}")

        # While a directory with the current run number exists, increment the run number
        while os.path.exists(run_output_dir):
            run_num += 1
            run_output_dir = os.path.join(aggregations_path, f"{today}_run_{run_num}")

        # Create the directory
        os.makedirs(run_output_dir, exist_ok=True)

        return run_output_dir

    def find_top_stories(self):
        all_topics_similar_stories = []
        # For each topic
        for topic in self.main_ag.config.topic_keywords:
            # Get topic embeddings from open ai (maybe we should ask for more keywords here)
            topic_embeddings = self.main_ag.vector_db.embedding_retriever.embed_query(
                topic
            )
            # Find all stories in the defined time range that meet the story_topic_score threshold
            initial_matching_vectors = self.main_ag.vector_db.matching_in_period(
                topic_embeddings, self.target_type
            )
            if not initial_matching_vectors:
                continue
            # For each story, find sources for that story that meet story_correlation_score threshold
            similar_vectors_sets = self.main_ag.vector_db.semantically_similar_sources(
                initial_matching_vectors, self.target_type
            )
            if not similar_vectors_sets:
                continue
            # All sources within a topic are compared, and any two stories using the same source are merged.
            # Also only retains top post_max_stories based on count of sources
            all_topics_similar_stories.extend(
                self.main_ag.vector_db.sort_and_merge_stories(similar_vectors_sets)
            )

        # All sources across all topics are merged
        # Prunes stories below post_max_stories threshold
        prune_and_merged_stories = self.main_ag.vector_db.sort_and_merge_stories(
            all_topics_similar_stories
        )
        top_vectorstore_content = self.main_ag.vector_db.fetch_vectors_from_id(
            prune_and_merged_stories
        )

        return sorted(top_vectorstore_content, key=len, reverse=True)[
            : self.main_ag.config.post_max_stories
        ]

    def summarize_merged_stories(self, top_vectorstore_content):
        with open(
            os.path.join(
                self.main_ag.prompt_path,
                "aggregator_summarize_merged_stories_template.yaml",
            ),
            "r",
            encoding="utf-8",
        ) as stream:
            prompt_template = yaml.safe_load(stream)

        story_strings = []
        # Create strings to be fed to GPT for summarization.
        for story_sources in top_vectorstore_content:
            story_string = ""
            for source in story_sources.items():
                story_string += f"\nsource: {source[1].metadata['content']}\n"
            story_strings.append(story_string)

        summarized_stories = []
        # Writing the dictionary into a YAML file
        with open(
            f"{self.run_output_dir}/5a_top_story_strings.yaml", "w", encoding="UTF-8"
        ) as yaml_file:
            yaml.dump(story_strings, yaml_file, default_flow_style=False)

        for story in story_strings:
            prompt_template[1]["content"] = story
            
            retries = 3
            success = False
            
            while retries > 0 and not success:
                try:
                    response = openai.ChatCompletion.create(
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        model=self.main_ag.config.LLM_writing_model,
                        messages=prompt_template,
                        max_tokens=self.main_ag.config.story_length,
                    )
                    checked_response = self.main_ag.check_response(response)
                    summarized_story = {}
                    summarized_story["summary"] = checked_response
                    summarized_stories.append(summarized_story)
                    success = True
                except Exception as e:
                    retries -= 1
                    if retries == 0:  # If no more retries left, raise the exception
                        raise e
                    time.sleep(20)  # Timeout before retrying

        with open(
            f"{self.run_output_dir}/5_summarized_stories.yaml", "w", encoding="UTF-8"
        ) as yaml_file:
            yaml.dump(summarized_stories, yaml_file, default_flow_style=False)

        return summarized_stories

    def create_titles(self, summarized_stories):
        with open(
            os.path.join(
                self.main_ag.prompt_path, "aggregator_create_titles_template.yaml"
            ),
            "r",
            encoding="utf-8",
        ) as stream:
            prompt_template = yaml.safe_load(stream)
        
        for story in summarized_stories:
            prompt_template[1]["content"] = f"Story: {story['summary']}"

            retries = 3
            success = False
            
            while retries > 0 and not success:
                try:
                    response = openai.ChatCompletion.create(
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        model=self.main_ag.config.LLM_writing_model,
                        messages=prompt_template,
                        max_tokens=25,
                    )
                    checked_response = self.main_ag.check_response(response)
                    checked_response = checked_response.strip('"')
                    checked_response = checked_response.strip("'")
                    story["title"] = checked_response
                    success = True
                except Exception as e:
                    retries -= 1
                    if retries == 0:  # If no more retries left, raise the exception
                        raise e
                    time.sleep(20)  # Timeout before retrying

        return summarized_stories

    def create_emojis(self, summarized_stories):
        
        with open(
            os.path.join(
                self.main_ag.prompt_path, "aggregator_create_emojis_template.yaml"
            ),
            "r",
            encoding="utf-8",
        ) as stream:
            prompt_template = yaml.safe_load(stream)

        for summary in summarized_stories:
            prompt_template[1]["content"] = f"Story: {summary['title']}\n"
            
            retries = 3
            success = False
            
            while retries > 0 and not success:
                try:
                    response = openai.ChatCompletion.create(
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        model=self.main_ag.config.LLM_writing_model,
                        messages=prompt_template,
                        max_tokens=3,
                    )
                    checked_response = self.main_ag.check_response(response)

                    # If no exception was raised, the call was successful
                    summary["emoji"] = checked_response
                    success = True
                    
                except Exception as e:
                    retries -= 1
                    if retries == 0:  # If no more retries left, raise the exception
                        raise e
                    time.sleep(20)  # Timeout before retrying

        return summarized_stories

    def create_intro(self, summarized_stories):
        with open(
            os.path.join(
                self.main_ag.prompt_path, "aggregator_create_intro_template.yaml"
            ),
            "r",
            encoding="utf-8",
        ) as stream:
            prompt_template = yaml.safe_load(stream)

        content = f"Username: {self.main_ag.config.moniker_name}\n"
        for summary in summarized_stories:
            content += f"Story title: {summary['title']}\n"

        prompt_template[1]["content"] = content
        retries = 3
        success = False
        
        while retries > 0 and not success:
            try:
                response = openai.ChatCompletion.create(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    model=self.main_ag.config.LLM_writing_model,
                    messages=prompt_template,
                    max_tokens=50,
                )
                checked_response = self.main_ag.check_response(response)
                success = True
                    
            except Exception as e:
                retries -= 1
                if retries == 0:  
                    raise e
                time.sleep(20)  

        return checked_response

    def create_hash_tags(self, summarized_stories):
        logit_bias_weight = 100
        logit_bias = {str(k): logit_bias_weight for k in range(15, 64 + 26)}
        # #
        logit_bias["2"] = logit_bias_weight
        # ' '
        logit_bias["220"] = logit_bias_weight

        with open(
            os.path.join(
                self.main_ag.prompt_path, "aggregator_create_hash_tags_template.yaml"
            ),
            "r",
            encoding="utf-8",
        ) as stream:
            prompt_template = yaml.safe_load(stream)

        content = "Keywords: "
        for keyword in self.main_ag.config.topic_keywords:
            content += f"[{keyword}]"
        content += f"\nStories: "
        for summary in summarized_stories:
            content += f"[Story title: '{summary['title']}']"

        prompt_template[1]["content"] = content

        response = openai.ChatCompletion.create(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=self.main_ag.config.LLM_writing_model,
            messages=prompt_template,
            max_tokens=50,
        )
        checked_response = self.main_ag.check_response(response)

        filtered = re.sub(r"[^a-z#]", "", checked_response.lower())
        filtered = re.sub(r"(?<=\w)#", " #", filtered)
        return filtered

    def create_post(self, summarized_stories, intro_text, hash_tags):
        dots1 = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]
        dots2 = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"]
        dots3 = ["🔵", "🟢", "🟡", "🟠", "🟣", "🟤", "⚪️", "⚫️", "🔴"]
        dots4 = ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘"]
        all_dots = [dots1, dots2, dots3, dots4]
        selected_dots = random.choice(all_dots)

        content = intro_text
        content += "\n\n"

        for i, summary in enumerate(summarized_stories):
            if i <= len(selected_dots):
                content += f"{selected_dots[i]} {summary['title']} — {summary['summary']} {summary['emoji']}\n\n"
            else:
                content += (
                    f"{summary['title']} — {summary['summary']} {summary['emoji']}\n\n"
                )
        content += hash_tags
        with open(
            f"{self.run_output_dir}/6_output.md", "w", encoding="UTF-8"
        ) as text_file:
            text_file.write(content)
