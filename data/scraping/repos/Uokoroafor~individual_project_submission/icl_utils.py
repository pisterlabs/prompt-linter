# This files contains helper functions for in context learning (ICL) experiments.


import random
from typing import List, Tuple, Optional, Dict

import pandas as pd

from utils.data_utils import read_in_data
from utils.file_utils import create_training_folder
from utils.logging_utils import TrainingLogger
from utils.plot_utils import plot_predictions, plot_errors
from utils.train_utils import set_seed
from utils.time_utils import EpochTimer
import openai


class ContextGenerator:
    def __init__(
        self,
        data: str,
        line_delimiter: str = None,
        ans_delimiter: str = None,
        question: Optional[str] = None,
        answer_str: Optional[str] = None,
        train_indices: Optional[List[int]] = None,
        val_indices: Optional[List[int]] = None,
        test_indices: Optional[List[int]] = None,
    ):
        """This class is used to generate contexts for ICL experiments. It will be used to generate contexts of the form:
        context + question + answer

        Args:
            data (str): The data to be used.
            line_delimiter (str, optional): The text to split the lines on. Defaults to "\n".
            ans_delimiter (str, optional): The text to split the answers on. Defaults to " ans: ".
            question (Optional[str], optional): The question to be used. Defaults to None.
            answer_str (Optional[str], optional): The string to use for the answer.
            train_indices (Optional[List[int]], optional): The indices to be used for training. Defaults to None.
            val_indices (Optional[List[int]], optional): The indices to be used for validation. Defaults to None.
            test_indices (Optional[List[int]], optional): The indices to be used for testing. Defaults to None.
        """

        self.line_delimiter = line_delimiter
        self.ans_delimiter = ans_delimiter
        self.question = question
        self.answer_str = answer_str
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.prep_txt_data(
            data=data,
            line_delimiter=line_delimiter,
            ans_delimiter=ans_delimiter,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )

    def prep_line(
        self, line: str, ans_delimiter: Optional[str] = None
    ) -> Tuple[str, str]:
        """Prepares a line of data by splitting it into the context and answer.

        Args:
            line (str): The line to be prepared.
            ans_delimiter (str): The delimiter to split the line on.
        Returns:
            Tuple[str, str]: The context and answer.
        """

        if ans_delimiter is None:
            ans_delimiter = self.ans_delimiter

        # Split the line into the context and answer
        context, answer = line.split(ans_delimiter)

        return context, answer

    def prep_txt_data(
        self,
        data: str,
        line_delimiter: Optional[str] = None,
        ans_delimiter: Optional[str] = None,
        train_indices: Optional[List[int]] = None,
        val_indices: Optional[List[int]] = None,
        test_indices: Optional[List[int]] = None,
        only_test: Optional[bool] = False,
    ) -> Optional[List[List[str]]]:
        """Prepares the data for training and testing.

        Args:
            data (str): The data to be prepared.
            line_delimiter (str): The text to split the lines on.
            ans_delimiter (str): The text to split the answers on.
            train_indices (Optional[List[int]]): The indices to be used for training.
            val_indices (Optional[List[int]]): The indices to be used for validation.
            test_indices (Optional[List[int]]): The indices to be used for testing.
            only_test (Optional[bool]): Whether to data is only for testing.
        """
        if line_delimiter is None:
            line_delimiter = self.line_delimiter
        if ans_delimiter is None:
            ans_delimiter = self.ans_delimiter

        # Split into questions and answers
        # data = [line.split(ans_delimiter) for line in data.rsplit(line_delimiter)]
        data = [
            list(self.prep_line(line, ans_delimiter=ans_delimiter))
            for line in data.rsplit(line_delimiter)
        ]

        if only_test:
            return data

        elif train_indices is not None:
            # Get the data for the indices
            self.train_data = [data[i] for i in train_indices]
            if val_indices is not None:
                self.val_data = [data[i] for i in val_indices]
            if test_indices is not None:
                self.test_data = [data[i] for i in test_indices]
        else:
            self.train_data = data

        return None

    def prep_test_data(
        self,
        data: str,
        line_delimiter: Optional[str] = None,
        ans_delimiter: Optional[str] = None,
    ) -> None:
        """Get test data and prepare it for testing.

        Args:
            data (str): The data to be prepared.
            line_delimiter (str): The text to split the lines on.
            ans_delimiter (str): The text to split the answers on.
        """
        data = self.prep_txt_data(
            data=data,
            line_delimiter=line_delimiter,
            ans_delimiter=ans_delimiter,
            only_test=True,
        )

        return data

    def split_data(
        self,
        train_indices: List[int],
        val_indices: Optional[List[int]],
        test_indices: List[int],
    ):
        """Splits the data into train, validation and test sets.

        Args:
            train_indices (List[int]): The indices to be used for training.
            val_indices (Optional[List[int]]): The indices to be used for validation. If None, then no validation set is used.
            test_indices (List[int]): The indices to be used for testing.
        """
        # Get the data for the indices
        self.train_data = [self.train_data[i] for i in train_indices]
        if val_indices is not None:
            self.val_data = [self.train_data[i] for i in val_indices]
        self.test_data = [self.train_data[i] for i in test_indices]

    def generate_prompt(
        self,
        context: str,
        question: str = "",
        answer: str = "",
        answer_str: Optional[str] = None,
    ) -> str:
        """Generates a prompt for the question and answer. If an answer is provided, then a new line is appeded to the prompt.

        Args:
            context (str): The context to be used.
            question (str): The question to be used.
            answer (str): The answer to be used.
            answer_str (Optional[str]): The string to use for the answer. If None, then the default is " ans: ".
        Returns:
            str: The generated prompt.
        """
        if answer_str is None:
            answer_str = self.answer_str

        prompt = context + question + answer_str
        if answer:
            prompt += answer

        return prompt

    def generate_few_shot_prompt(
        self,
        context: str,
        question: Optional[str] = None,
        answer: str = "",
        num_shots: int = 5,
    ) -> str:
        """Generates a prompt for the question and answer. If an answer is provided, then a new line is appended to the prompt.
        It picks num_shots random contexts, questions and answers from self.train_data then appends the given context, question and answer at the end.

        Args:

            context (str): The context to be used.
            question (str): The question to be used.
            answer (str): The answer to be used.
            num_shots (int): The number of shots to use.
        Returns:
            str: The generated prompt.
        """

        if question is None:
            question = self.question

        # Get the indices of the data
        indices = self.get_random_indices(num_shots)

        # Get the contexts, questions and answers
        contexts = [self.train_data[i][0] for i in indices]
        answers = [self.train_data[i][1] for i in indices]

        # Generate the prompts
        prompts = [
            self.generate_prompt(context_, question, answer_)
            for context_, answer_ in zip(contexts, answers)
        ]

        # Append the given context, question and answer to the end
        prompts.append(self.generate_prompt(context, question, answer))

        return "\n".join(prompts)

    def get_random_indices(self, num_indices: int):
        """Gets a list of random indices from the data.

        Args:
            num_indices (int): The number of indices to get.
        Returns:
            List[int]: The list of random indices.
        """

        # Get the indices of the data
        indices = list(range(len(self.train_data)))
        # Shuffle the indices
        random.shuffle(indices)

        return indices[:num_indices]


class LLMTrainer:
    def __init__(self, cg: ContextGenerator):
        """This uses pretrained LLMs from OpenAI to perform in context learning on a dataset. Trainer is somewhat of a misnomer but it is in keeping with the nomenclature for the other trainers.

        Args:
            cg (ContextGenerator): The context generator to be used. It should already have train and test data.
        """
        self.cg = cg
        self.path = create_training_folder()
        self.logger = None

    def evaluate(
        self,
        num_shots: int = 5,
        model_name="Davinci",
        plotting: bool = True,
        save_preds: bool = True,
        verbose: bool = True,
        test_data: Optional[str] = None,
        batch_size: int = 5,
    ) -> Tuple[float, List[float], List[float]]:
        """Trains and evaluates the model on the test data. Again, train here is a misnomer, but it is in keeping with the nomenclature for the other trainers.

        Args:
            num_shots (int): The number of shots to use.
            model_name (str): The name of the model to use.
            plotting (bool): Whether to plot the predictions.
            save_preds (bool): Whether to save the predictions.
            verbose (bool): Whether to print out the MSE.
            test_data (Optional[str]): The test data to be used. If None, then the test data from the context generator is used.
            batch_size (int): The number of prompts to send in one batch.
        """
        timer = EpochTimer()
        timer.start()
        oos_str = "_oos" if test_data is not None else ""

        if test_data is not None:
            test_data = self.cg.prep_test_data(data=test_data)
        else:
            test_data = self.cg.test_data

        # Generate the prompts
        contexts = [test_data[i][0] for i in range(len(test_data))]
        answers = [test_data[i][1] for i in range(len(test_data))]
        prompts = [
            self.cg.generate_few_shot_prompt(context=context, num_shots=num_shots)
            for context in contexts
        ]

        if self.logger is None:
            logger = TrainingLogger(
                self.path + "/training_logs/training_log.txt",
                name="in_context_learning_log",
                verbose=verbose,
            )
            self.logger = logger
            self.logger.log_info(
                f"Logger created at {self.path}/training_logs/training_log.txt"
            )
            self.logger.log_info(
                f"Using {model_name} model. with batch size {batch_size}"
            )

        # Get the predictions
        predictions = self.get_predictions(
            prompts=prompts, model_name=model_name, batch_size=batch_size
        )

        # Convert the predictions and actuals to floats and calculate the mean squared error
        actuals, predictions, count = self.convert_strings_to_floats(
            actuals=answers, predictions=predictions
        )

        if count > 0:
            self.logger.log_info(f"Unable to convert {count} strings to floats.")

        mse = sum(
            [(pred - actual) ** 2 for pred, actual in zip(predictions, actuals)]
        ) / len(predictions)

        # Log the MSE
        self.logger.log_info(
            f"MSE for {num_shots}-shot for {len(prompts)} Test Examples: {mse:,.4f}"
        )

        if plotting:
            plot_saved_path = (
                (
                    f"{self.path}/training_logs/{model_name}_{num_shots}shot_predictions{oos_str}.png"
                )
                if save_preds
                else None
            )
            plot_predictions(
                predictions,
                actuals,
                model_name=f"{model_name}_{num_shots}shot",
                saved_path=plot_saved_path,
            )

        if save_preds:
            self.save_predictions(
                predictions,
                actuals,
                model_name=f"{model_name}_{num_shots}shot{oos_str}",
            )

        timer.lap()
        self.logger.log_info(timer.print_total_time(label="Total time taken: "))
        return mse, predictions, actuals

    def save_predictions(
        self, predictions: List[float], actuals: List[float], model_name: str
    ):
        """Saves the predictions and actuals to a csv file.

        Args:
            predictions (List[float]): The predictions.
            actuals (List[float]): The actuals.
            model_name (str): The name of the model.
        """
        df = pd.DataFrame({"predictions": predictions, "actuals": actuals})
        df.to_csv(
            f"{self.path}/training_logs/{model_name}_predictions.csv", index=False
        )

    def get_predictions(
        self, prompts: List[str], model_name: str = "Davinci", batch_size: int = 5
    ) -> List[str]:
        """Gets the predictions from the model.

        Args:
            prompts (List[str]): The prompts to be used.
            model_name (str): The name of the model to use.
            batch_size (int): The number of prompts to send in one batch.
        Returns:
            List[str]: The predictions.
        """

        # Get the predictions
        predictions = self.get_responses(
            prompts=prompts, model_name=model_name, batch_size=batch_size
        )

        return predictions

    def get_responses(
        self,
        prompts: List[str],
        model_name: str = "Davinci",
        max_tokens: int = 5,
        batch_size: int = 5,
    ) -> List[str]:
        """Gets the responses from the model in batches.

        Args:
            prompts (List[str]): The prompts to be used.
            model_name (str): The name of the model to use.
            max_tokens (int): The maximum number of tokens to generate.
            batch_size (int): The number of prompts to send in one batch.
        Returns:
            List[str]: The responses.
        """

        responses = []

        # Split prompts into batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]

            # Get the responses for the current batch
            batch_responses = openai.Completion.create(
                engine=model_name,
                prompt=batch,
                max_tokens=max_tokens,
            )

            # Append batch_responses to the master list
            responses.extend(self.extract_batched_responses(batch_responses))

        return responses

    def extract_batched_responses(self, api_response: Dict):
        """Extracts the completion text from batched API response.

        Args:
            api_response (dict): The response from the OpenAI API for batched prompts.

        Returns:
            List[str]: Extracted completion texts.
        """
        extracted_responses = []

        # Loop over each item in the batched response
        for item in api_response["choices"]:
            # Extract the text of the completion and append to the list
            extracted_responses.append(item["text"].strip())

        # Only return the first num_choices
        return extracted_responses

    def convert_strings_to_floats(
        self, actuals: List[str], predictions: List[str]
    ) -> Tuple[List[float], List[float], int]:
        """Converts the actuals and predictions to floats.

        Args:
            actuals (List[str]): The actuals.
            predictions (List[str]): The predictions.
        Returns:
            Tuple[List[float], List[float], int]: The actuals and predictions as floats, and the number of errors (i.e.
            the number of strings that could not be converted to floats if any).

        """
        # If it is unable to convert a string to a float, if we are unable to convert a string to a float, then we will
        # skip the entry for both the actuals and predictions and log it and continue
        actuals_ = []
        predictions_ = []
        count = 0
        message = ""
        for i in range(len(actuals)):
            try:
                pred = float(predictions[i])
                actual = float(actuals[i])
                predictions_.append(pred)
                actuals_.append(actual)
            except ValueError:
                count += 1
                message += (
                    f"Unable to convert {actuals[i]} or {predictions[i]} to a float.\n"
                )
                continue

        # Save the message to a text file if there are any errors
        if count > 0:
            with open(f"{self.path}/training_logs/conversion_errors.txt", "w") as f:
                f.write(message)
        return actuals_, predictions_, count

    def calculate_prompt_size(self):
        """Given there is a limit on the number of tokens that can be sent to the API, this method calculates the
        number of tokens in a prompt to help determine the batch size.
        """
        # First generate a random prompt
        context = self.cg.train_data[0][0]
        question = self.cg.question
        answer = self.cg.train_data[0][1]
        prompt = self.cg.generate_prompt(
            context=context, question=question, answer=answer
        )
        # Count the number of tokens in the prompt
        num_tokens = count_tokens_in_batch([prompt])
        return num_tokens


def count_tokens_in_batch(batch: List[str]) -> int:
    """Count the number of tokens in a batch of prompts.

    Args:
        batch (List[str]): The batch of prompts.
    Returns:
        int: The number of tokens in the batch.
    """
    # Use the rubric that 3 chars is 1 token
    total_tokens = 0

    for prompt in batch:
        total_tokens += len(prompt) / 3

    return int(total_tokens)


if __name__ == "__main__":
    set_seed(6_345_789)
    file_path = "../data/freefall/variable_height/descriptive_text.txt"
    test_file_path = "../data/freefall/variable_height/oos_descriptive_text.txt"
    train_data = read_in_data(file_path, make_dict=False)
    test_data = read_in_data(test_file_path, make_dict=False)
    # print(data)

    cg = ContextGenerator(
        data=train_data,
        line_delimiter="\n",
        ans_delimiter=" ans: Ball is at y=",
        question=", what is the value of y?",
        answer_str=" ans: ",
    )
    cg.prep_test_data(data=test_data)

    # Create the trainer
    trainer = LLMTrainer(cg=cg)

    # # read the 5th line from the file and test prep_line
    # with open(file_path, "r") as f:
    #     lines = f.readlines()
    #
    # context, answer = cg.prep_line(lines[4], ans_delimiter=" ans: Ball is at y=")

    import os
    from dotenv import load_dotenv

    # Load variables from .env
    load_dotenv()

    # Fetch the API key from the environment variables
    api_key = os.environ.get("OPENAI_API_KEY")

    # Use the API key with the OpenAI package
    openai.api_key = api_key

    shots = [1, 2, 3, 4]
    errors = []

    for num_shot in range(1, 5):
        mse, _, _ = trainer.evaluate(
            num_shots=num_shot,
            model_name="text-davinci-003",
            plotting=True,
            save_preds=True,
            verbose=True,
        )

        errors.append(mse)
    plot_errors(
        errors,
        shots,
        model_name="text-davinci-003",
        saved_path=f"{trainer.path}/training_logs/errors_vs_shots.png",
    )

    # for num_shot in range(5):
    #     predictions = []
    #     actuals = []
    #
    #     for i in range(len(cg.test_data)):
    #         # Generate the prompts
    #         context = cg.test_data[i][0]
    #         answer = cg.test_data[i][1]
    #         few_shot_prompt = cg.generate_few_shot_prompt(context=context,
    #                                                       num_shots=num_shot)
    #         response = openai.Completion.create(
    #             engine="text-davinci-003",
    #             prompt=few_shot_prompt,
    #             max_tokens=5,
    #         )
    #         answer_ = response.choices[0].text.strip()
    #         actuals.append(answer)
    #         predictions.append(answer_)
    #
    #     # convert the predictions and actuals to floats and calculate the mean squared error
    #     predictions = [float(pred) for pred in predictions]
    #     actuals = [float(actual) for actual in actuals]
    #
    #     mse = sum([(pred - actual) ** 2 for pred, actual in zip(predictions, actuals)]) / len(predictions)
    #     print(f'Number of shots: {num_shot}, MSE: {mse:,.2f}')
    #     print()
    #     # plot_predictions(predictions, actuals, model_name=f'text-davinci-003 {num_shot} shots')
    #     errors.append(mse)
    #
    # print(errors)
    # plot_errors(errors, shots, model_name='text-davinci-003')
