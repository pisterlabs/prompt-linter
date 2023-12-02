from datasette import hookimpl
from uppercase import Uppercase
from openai_embeddings import Embeddings
from jinja_sandbox import JinjaSandbox


@hookimpl
def register_enrichments():
    return [Uppercase(), Embeddings(), JinjaSandbox()]
