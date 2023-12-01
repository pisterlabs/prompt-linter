# Imports
from utils import (
    print_success,
    print_info,
    print_warning,
    print_error,
    print_final_results,
    save_results_to_json,
)
from config import load_configuration
from api_communication import AnthropicAPI
from prompt_processing import PromptProcessor
from user_input import prompt_user, get_test_cases_count


def main():
    config = load_configuration()
    api_key = config.get("anthropic_api_key")

    # Early exit if no API key found
    if not api_key:
        print_error("No API key found.")
        return

    api = AnthropicAPI(api_key)
    prompt_processor = PromptProcessor(api)
    goal = prompt_user()
    num_test_cases = get_test_cases_count()

    combined_results, test_results = [], {}
    test_cases, first_iteration = None, True

    while True:
        prompt_template, cleaned_prompt = prompt_processor.generate_and_clean_prompt(
            goal, test_results
        )
        if prompt_template is None:
            return  # Prompt generation failed
        if num_test_cases == 0:
            print_info("\n*** No test cases to evaluate. ***")
            break
        placeholders = prompt_processor.identify_placeholders(prompt_template)
        if placeholders is None:
            return  # Placeholder identification failed

        input_vars_detected = placeholders[0] != "None"
        if first_iteration and input_vars_detected:
            test_cases = prompt_processor.setup_test_cases(
                num_test_cases, prompt_template, placeholders
            )
            if test_cases is None:
                return  # Test case generation failed

        if input_vars_detected:
            # Skip processing if no test cases are defined
            if not test_cases:
                print_warning("No test cases available.")
                break

            (
                test_results,
                combined_results,
                failed_tests,
            ) = prompt_processor.process_test_cases(
                test_cases, prompt_template, combined_results, test_results
            )
            if (
                test_results is None
                and combined_results is None
                and failed_tests is None
            ):
                return  # Prompt Execution or Evaluation failed

            if not failed_tests:
                print_success("\n*** All test cases passed! ***")
                break
        else:
            (
                test_results,
                combined_results,
                failed_evaluation,
            ) = prompt_processor.process_no_input_var_case(
                prompt_template, combined_results, test_results
            )
            if (
                test_results is None
                and combined_results is None
                and failed_evaluation is None
            ):
                return  # Prompt Execution or Evaluation failed
            if not failed_evaluation:
                print_success(
                    "\n*** Evaluation passed! No input variables detected. ***"
                )
                break

        first_iteration = False

    save_results_to_json(combined_results)
    print_final_results(cleaned_prompt)


if __name__ == "__main__":
    main()
