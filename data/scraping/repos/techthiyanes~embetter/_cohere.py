import os
import numpy as np

import cohere
from embetter.base import EmbetterBase


def _batch(iterable, n=1):
    len_iter = len(iterable)
    for ndx in range(0, len_iter, n):
        yield iterable[ndx : min(ndx + n, len_iter)]


class CohereEncoder(EmbetterBase):
    """
    Encoder that can numerically encode sentences.

    Note that this is an **external** embedding provider. If their API breaks, so will this component.

    This encoder will require the `COHERE_KEY` environment variable to be set.
    If you have it defined in your `.env` file, you can use python-dotenv to load it.

    You also need to install the `cohere` library beforehand.

    ```
    python -m pip install cohere
    ```

    Arguments:
        model: name of model, can be "small" or "large"

    **Usage**:

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    from embetter.grab import ColumnGrabber
    from embetter.external import CohereEncoder
    from dotenv import load_dotenv

    load_dotenv()  # take environment variables from .env.

    # Let's suppose this is the input dataframe
    dataf = pd.DataFrame({
        "text": ["positive sentiment", "super negative"],
        "label_col": ["pos", "neg"]
    })

    # This pipeline grabs the `text` column from a dataframe
    # which then get fed into Cohere's endpoint
    text_emb_pipeline = make_pipeline(
        ColumnGrabber("text"),
        CohereEncoder(model="large")
    )
    X = text_emb_pipeline.fit_transform(dataf, dataf['label_col'])

    # This pipeline can also be trained to make predictions, using
    # the embedded features.
    text_clf_pipeline = make_pipeline(
        text_emb_pipeline,
        LogisticRegression()
    )

    # Prediction example
    text_clf_pipeline.fit(dataf, dataf['label_col']).predict(dataf)
    ```
    """

    def __init__(self, model="large"):
        from cohere import Client

        self.client = Client(os.getenv("COHERE_KEY"))
        self.model = model

    def transform(self, X, y=None):
        """Transforms the text into a numeric representation."""
        result = []
        for b in _batch(X, 10):
            response = self.client.embed(b)
            result.extend(response.embeddings)
        return np.array(result)
