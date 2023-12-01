import json
import os
from typing import Any, List
from timeit import default_timer as timer

from app_modules.init import app_init
from app_modules.llm_inference import LLMInference
from app_modules.utils import print_llm_response

from datasets import load_from_disk
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.schema import BaseRetriever
from langchain.schema.document import Document

import evaluate
import json
from pathlib import Path


class DatasetRetriever(BaseRetriever):
    eval_ds: Any

    def __init__(self, eval_ds):
        super().__init__()
        self.eval_ds = eval_ds

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        docs = []
        eval_ds = self.eval_ds
        for i in range(eval_ds.num_rows):
            if str(eval_ds[i]["query"]) == query:
                passages = eval_ds[i]["passages"]
                passage_text = passages["passage_text"]
                for j in range(len(passage_text)):
                    page_content = passage_text[j]
                    source = passages["url"][j]
                    docs.append(
                        Document(page_content=page_content, metadata={"source": source})
                    )

                break

        return docs


class QAChainWithMsMacroDataset(LLMInference):
    retriever: Any

    def __init__(self, eval_ds, llm_loader):
        super().__init__(llm_loader)
        self.retriever = DatasetRetriever(eval_ds)

    def create_chain(self, inputs) -> Chain:
        qa = ConversationalRetrievalChain.from_llm(
            self.llm_loader.llm,
            self.retriever,
            max_tokens_limit=self.llm_loader.max_tokens_limit,
            return_source_documents=True,
        )

        return qa


bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


def calc_metrics(ds, predictions):
    references = [ds[i]["wellFormedAnswers"][0] for i in range(ds.num_rows)]

    assert len(references) == len(
        predictions
    ), f"lengths are difference: {len(references)} != {len(predictions)}"

    bleu_scores = bleu.compute(
        predictions=predictions, references=references, max_order=1
    )
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    return {"bleu_scores": bleu_scores, "rouge_scores": rouge_scores}


def calc_all_metrics(ds, answers):
    if os.environ.get("TEST_FIRST_5") == "true":
        query_types = ["NUMERIC", "DESCRIPTION"]
    else:
        query_types = ["NUMERIC", "DESCRIPTION", "ENTITY", "PERSON", "LOCATION"]

    result = {}
    result["OVERALL"] = calc_metrics(ds, answers)

    for query_type in query_types:
        predictions = [
            answers[i] for i in range(ds.num_rows) if ds[i]["query_type"] == query_type
        ]
        result[query_type] = calc_metrics(
            ds.filter(lambda example: example["query_type"] == query_type), predictions
        )

    return result


llm_loader = app_init()

if __name__ == "__main__":
    if os.environ.get("TEST_FIRST_5") == "true":
        eval_ds = load_from_disk("./data/datasets/ms_macro/").select(range(5))
        query_types = ["NUMERIC", "DESCRIPTION"]
    else:
        eval_ds = load_from_disk("./data/datasets/ms_macro/")
        query_types = ["NUMERIC", "DESCRIPTION", "ENTITY", "PERSON", "LOCATION"]

    qa_chain = QAChainWithMsMacroDataset(eval_ds, llm_loader)

    start = timer()
    answers = []
    for i in range(eval_ds.num_rows):
        inputs = {"question": str(eval_ds[i]["query"]), "chat_history": []}
        result = qa_chain.call_chain(
            inputs,
            None,
            None,
            True,
        )
        answers.append(result["answer"])
        print_llm_response(result)

    generation_end = timer()
    generation_time = generation_end - start

    result = calc_all_metrics(eval_ds, answers)

    evaluation_time = timer() - generation_end

    for i in range(eval_ds.num_rows):
        n = i + 1
        print(f"Q-{n:03d}: {eval_ds[i]['query']}")
        print(f"A-{n:03d}: {answers[i]}")
        print(f"G-{n:03d}: {eval_ds[i]['wellFormedAnswers'][0]}\n")

    print(f"\n\nscores: {json.dumps(result, indent=2)}\n")

    filename = os.environ.get("CSV_FILENAME")
    if filename is not None and len(filename) > 0:
        query_types.append("OVERALL")

        # create a Path object with the path to the file
        path = Path(filename)
        file_exists = path.is_file()

        file = open(filename, "a")
        if not file_exists:
            file.write(
                "model_name,repetition_penalty,generation_time,evaluation_time,total_tokens,total_words,tokens_per_second,total_tokens_over_total_words"
            )

            for query_type in query_types:
                file.write(f",{query_type.lower()}_bleu,{query_type.lower()}_rougeL")

            file.write(",total_words_over_total_tokens\n")

        llm_model_type = os.environ.get("LLM_MODEL_TYPE")
        if llm_model_type == "openai":
            model_name = os.environ.get("OPENAI_MODEL_NAME")
            repetition_penalty_str = ""
        elif llm_model_type == "mosaicml":
            model_name = os.environ.get("MOSAICML_MODEL_NAME_OR_PATH").split("/")[-1]
            repetition_penalty_str = os.environ.get("ML_RP")
        else:
            model_name = os.environ.get("HUGGINGFACE_MODEL_NAME_OR_PATH").split("/")[-1]
            repetition_penalty_str = os.environ.get("HF_RP")

        total_words = result["OVERALL"]["bleu_scores"]["translation_length"]

        file.write(
            f"{model_name},{repetition_penalty_str},{generation_time:.3f},{evaluation_time:.3f},{llm_loader.streamer.total_tokens},{total_words},{llm_loader.streamer.total_tokens/generation_time:.3f},{llm_loader.streamer.total_tokens/total_words:.3f}"
        )

        for query_type in query_types:
            scores = result[query_type]
            file.write(
                f",{scores['bleu_scores']['bleu']:.4f},{scores['rouge_scores']['rougeL']:.4f}"
            )

        file.write(f",{total_words/llm_loader.streamer.total_tokens:.3f}\n")
        file.close()
        print(f"All results saved to {filename}")
