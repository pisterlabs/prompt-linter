import os

import openai
import pandas as pd
from dotenv import load_dotenv

from utils.data_utils import read_in_data
from utils.icl_utils import ContextGenerator, LLMTrainer
from utils.plot_utils import plot_errors
from utils.train_utils import set_seed
from utils.logging_utils import TrainingLogger

# Load variables from .env
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Set the seed
seed = 6_345_789
set_seed(seed)
# Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

# Create the logger
batch_logger = TrainingLogger("../../GPT_training_logs.txt", verbose=False)
file_paths = ["descriptive_text.txt"]

data_folder = "data/shelf_bounce/variable_ballheight/"

for file_path in file_paths:
    # file_path = "descriptive_text.txt"  # Update this to the file containing the data
    oos_test_data_path = f"oos_{file_path}"

    function_name = "Freefall Environment"  # Update this to the name of the dataset being trained (or the name of the function)

    train_indices_path = "train_indices.csv"
    val_indices_path = "val_indices.csv"
    test_indices_path = "test_indices.csv"

    # Get train and test indices and convert to lists
    train_indices = (
        pd.read_csv(data_folder + train_indices_path).values.flatten().tolist()
    )
    val_indices = pd.read_csv(data_folder + val_indices_path).values.flatten().tolist()
    test_indices = (
        pd.read_csv(data_folder + test_indices_path).values.flatten().tolist()
    )

    # truncate test indices to 1000
    test_indices = test_indices[:1000]

    # Load the datasets
    data = read_in_data(data_folder + file_path, make_dict=False)
    oos_data = read_in_data(data_folder + oos_test_data_path, make_dict=False)

    # Context Generator Args
    line_delimiter = "\n"
    ans_delimiter = " ans: y="
    question = ", what is y?"
    answer_str = " ans: "
    model_name = "text-davinci-003"

    # Create the context generator
    context_generator = ContextGenerator(
        data=data,
        line_delimiter=line_delimiter,
        ans_delimiter=ans_delimiter,
        question=question,
        answer_str=answer_str,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    # Create the trainer
    trainer = LLMTrainer(cg=context_generator)
    # token capacity for question and answer is 4096. Will use a conservative 3000 to allow for other tokens
    token_capacity = 3600
    batch_size = token_capacity // (trainer.calculate_prompt_size() + 5)
    # This will be divided by the number of shots in the evaluate function

    num_shot = 15
    errors = []
    oos_errors = []

    # for num_shot in num_shots:
    # Evaluate the model on test data
    mse, _, _ = trainer.evaluate(
        num_shots=num_shot,
        model_name="text-davinci-003",
        plotting=False,
        save_preds=True,
        verbose=True,
        batch_size=max(batch_size // num_shot, 5),
    )

    batch_logger.log_info(f"Training log is saved at {trainer.path} for")
    batch_logger.log_info(f"{function_name} on {data_folder} data with {file_path}")
    batch_logger.log_info(f"Test loss: {mse:.4f}")
