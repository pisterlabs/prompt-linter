import base64
import os
from typing import Any, List
import gitlab
import googleapiclient.discovery
from kubernetes import client
from tempfile import NamedTemporaryFile

from slack_sdk import WebClient
from agent.toolkits.base import create_k8s_engineer_agent
from agent.toolkits.toolkit import K8sEngineerToolkit
from doc_indexes.k8s_index import KubernetesIndex
from tools.git_integrator.tool import GitModel
from tools.gitlab_integration.tool import GitlabModel
from tools.k8s_explorer.tool import KubernetesOpsModel
from tools.k8s_sme.tools import KubernetesSMEModel
from tools.slack_integration.tool import SlackModel
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent import AgentExecutor
from langchain.callbacks.base import CallbackManager, BaseCallbackHandler


class AgentFactory:
    def __init__(self, model_name: str = "gpt-4"):
        self.k8s_model = KubernetesOpsModel.from_k8s_client(
            k8s_client=kubernetes_api(get_cluster()))

        git_username = os.getenv("GIT_USERNAME", "k8s-engineer")
        git_password = os.getenv("GIT_PASSWORD", "dummy-password")
        self.git_model = GitModel(username=git_username, password=git_password)

        gitlab_private_token = os.getenv("GITLAB_PRIVATE_TOKEN", "dummy-token")
        gitlab_url = os.getenv("GITLAB_URL", "https://gitlab.com")

        gl = gitlab.Gitlab(url=gitlab_url, private_token=gitlab_private_token)
        self.gitlab_model = GitlabModel(gl=gl)

        slack_token = os.environ["SLACK_BOT_TOKEN"]
        slack_client = WebClient(token=slack_token)
        slack_channel = os.environ["SLACK_CHANNEL_ID"]
        self.slack_model = SlackModel(
            client=slack_client, channel=slack_channel)

        k8s_doc_url = os.getenv(
            "K8S_DOC_URL", "https://github.com/dohsimpson/kubernetes-doc-pdf/raw/master/PDFs/Reference.pdf")
        k8s_index = KubernetesIndex(doc_url=k8s_doc_url)
        self.k8s_sme_model = KubernetesSMEModel(index=k8s_index)

        self.llm = ChatOpenAI(
            temperature=0, model_name=model_name, max_tokens=1024)

    def new_k8s_engineer(self, handlers: List[BaseCallbackHandler], slack_channel: str = None, slack_thread_ts: str = None) -> AgentExecutor: 
        cm = CallbackManager(handlers=handlers) if handlers else None
        k8s_engineer_toolkit = K8sEngineerToolkit.from_llm(llm=self.llm, k8s_model=self.k8s_model, git_model=self.git_model, gitlab_model=self.gitlab_model,
                                                           slack_model=self.slack_model, k8s_sme_model=self.k8s_sme_model, slack_channel=slack_channel, slack_thread_ts=slack_thread_ts, callback_manager=cm,  verbose=True)
        return create_k8s_engineer_agent(llm=self.llm, toolkit=k8s_engineer_toolkit, callback_manager=cm, verbose=True)


def gcp_token(*scopes):
    credentials = googleapiclient._auth.default_credentials()
    scopes = [f'https://www.googleapis.com/auth/{s}' for s in scopes]
    scoped = googleapiclient._auth.with_scopes(credentials, scopes)
    googleapiclient._auth.refresh_credentials(scoped)
    return scoped.token


def kubernetes_api(cluster):
    config = client.Configuration()
    config.host = f'https://{cluster["endpoint"]}'

    config.api_key_prefix['authorization'] = 'Bearer'
    config.api_key['authorization'] = gcp_token('cloud-platform')

    with NamedTemporaryFile(delete=False) as cert:
        cert.write(base64.decodebytes(
            cluster['masterAuth']['clusterCaCertificate'].encode()))
        config.ssl_ca_cert = cert.name

    api = client.ApiClient(configuration=config)

    return api


def get_cluster():
    project_id = os.getenv("GOOGLE_PROJECT_ID")
    zone = os.getenv("GOOGLE_REGION")
    cluster_id = os.getenv("GOOGLE_CLUSTER_ID")

    credentials = googleapiclient._auth.default_credentials()
    service = googleapiclient.discovery.build(
        'container', 'v1', credentials=credentials)
    cluster = service.projects().locations().clusters().get(
        name=f'projects/{project_id}/locations/{zone}/clusters/{cluster_id}').execute()
    return cluster
