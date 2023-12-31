#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredURLLoader
from zenml.steps import BaseParameters, step


class WebUrlLoaderParameters(BaseParameters):
    """Params for WebUrlLoader.

    Attributes:
        urls: List of URLs to load documents from.
    """

    urls: List[str] = ["docs.zenml.io"]


@step
def web_url_loader_step(params: WebUrlLoaderParameters) -> List[Document]:
    """Loads documents from a list of URLs.

    Args:
        params: Parameters for the step.

    Returns:
        List of langchain documents.
    """
    loader = UnstructuredURLLoader(urls=params.urls)
    return loader.load()
