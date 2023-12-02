import logging
from ...templates import Dataset
from ...file_utils import stream_jsonl, dump_jsonl
from ...utils import download, mark_done, done_path, sha256sum
from pathlib import Path
import re

logger = logging.getLogger(__name__)


def question_answer_to_pile_format(
    qa_dict,
    q_key="question",
    a_key="answer",
    question_prompt="Q: ",
    answer_prompt="A: ",
    separator=" ",
    include_original_as_meta=False,
):
    """
    Converts a question-answer dictionary to a pile formatted dictionary.
    """
    q = qa_dict[q_key]
    a = qa_dict[a_key]
    d = {"text": question_prompt + q + separator + answer_prompt + a, "meta": {}}
    if include_original_as_meta:
        d["meta"] = qa_dict
    return d


class _GradeSchoolMath(Dataset):
    name = None
    license = "MIT License"
    checksum = None
    urls = None

    original_set_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    original_set_checksum = (
        "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
    )

    remove_calculator_strings = False

    def replicate(self):
        # download original dataset from openai
        out_path = self.dataset_dir() / str(Path(self.original_set_url).name)
        logger.info(f"Downloading {self.original_set_url} to {out_path}")
        download(
            url=self.original_set_url,
            out_path=out_path,
            mirrors=None,
            checksum=self.original_set_checksum,
            force=False,
        )

        # reformat to pile format
        logger.info(f"Converting {self.name} to pile format")
        converted_path = self.dataset_dir() / str(next(self.paths()))
        converted = [
            question_answer_to_pile_format(qa) for qa in stream_jsonl(out_path)
        ]
        if self.remove_calculator_strings:
            # calculator strings in this dataset are always inbetween << and >>
            # they are there so that the model can indicate when to outsource to a calculator,
            # but we might not always want this behavior
            # below removes them
            converted = [
                {
                    "text": re.sub(r"\<{2,}.*?\>{2,}", "", q["text"]),
                    "meta": q["meta"],
                }
                for q in converted
            ]

        dump_jsonl(converted, converted_path)

        # remove original dataset and done path
        out_path.unlink()
        done_path(out_path).unlink()

        # compare checksum with expected checksum
        checksum = sha256sum(str(converted_path))
        if checksum != self.checksum:
            error_msg = f"{converted_path} checksum mismatch.\nExpected: {self.checksum}\nActual: {checksum}"
            error_msg += "\nHas the dataset changed since it was last downloaded?"
            logger.error(error_msg)
        else:
            logger.info(f"{self.name} checksum verified.")

        # mark done
        mark_done(converted_path)

    def documents(self):
        self.raise_if_not_exists()
        return stream_jsonl(list(self.paths())[0])

    def paths(self):
        fp = (
            "grade_school_math_no_calc.jsonl"
            if self.remove_calculator_strings
            else "grade_school_math.jsonl"
        )
        paths = [str(self.dataset_dir() / fp)]
        for path in paths:
            yield path

    def examples(self):
        example_fp = (
            "grade_school_math_no_calc.jsonl"
            if self.remove_calculator_strings
            else "grade_school_math.jsonl"
        )
        return list(stream_jsonl(Path(__file__).parent / example_fp))

    def size_on_disk(self):
        return None

    def size(self):
        return None

    def num_docs(self):
        return 7473


class GradeSchoolMath(_GradeSchoolMath):
    name = "Grade School Math"
    checksum = "14d2c64f3c53305dc8e3cea8d5023f93b3494dc631e875403e920d949054e49c"
    urls = ["http://eaidata.bmk.sh/data/grade_school_math.jsonl"]

    def size_on_disk(self):
        return 4169410

    def size(self):
        return 3955729


class GradeSchoolMathNoCalc(_GradeSchoolMath):
    name = "Grade School Math (no calculator strings)"
    remove_calculator_strings = True
    checksum = "56a3429dcf4dfaeb4ce04546ec0c9b0181271b8d9dc953a417ba0a99f852e331"
    urls = ["http://eaidata.bmk.sh/data/grade_school_math_no_calc.jsonl"]

    def size(self):
        return 3656968

    def size_on_disk(self):
        return 3870649
