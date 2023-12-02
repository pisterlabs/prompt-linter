import openai
from prog_params import ProgParams as prog_params
from src.chatgpt import ChatGPT
from src.framework import Framework
from src.bug import Bug
from src.prompts import Prompts as prompts
from src.proposed_patches import ProposedPatches
from src.utils import extract_patches_from_response

class RapidCapr(object):

    def __init__(self, chatgpt: ChatGPT, framework: Framework):
        self.name = "RapidCapr"
        self.chatgpt = chatgpt
        self.framework = framework
    
    def repair(self, bug: Bug, max_fpps_try_per_mode=1, max_mpps_try_per_mode=1, prompt_token_limit=1500, total_token_limit_target=3000, 
               max_sample_count=50, similarity_threshold=0.5, max_rounds=1, ask_for_bug_description=False):

        modes = [list(bug.bug_type.split())[0], "SF"]
        prefix = f"{self.framework.name}_{bug.project}_{bug.bug_id}"
        total_call_tries, total_cost = 0, 0
        first_plausible_patch_try = None
        plausible_patches, plausible_patch_diffs = [], []
        test_failure_count, test_error_count, total_length = 0, 0, 0
        max_mpps_try_per_mode = max_mpps_try_per_mode if not prog_params.stop_on_first_plausible_patch else 0

        for round in range(1, max_rounds + 1):
        
            proposed_patches = ProposedPatches()

            for mode in modes:

                n_shot_bugs=self.framework.get_n_shot_bugs(n=1, bug=bug, mode=mode)
                
                for run_condition, call_try_limit, construct_prompt_function in [(False, max_fpps_try_per_mode, prompts.construct_fpps_prompt), 
                                                                                (True, max_mpps_try_per_mode, prompts.construct_mpps_prompt)]:
                    call_tries = 0

                    while(call_tries < call_try_limit and proposed_patches.contains_plausible_patch(mode) == run_condition):
                        total_call_tries += 1
                        call_tries += 1

                        prompt, num_of_samples = construct_prompt_function(bug=bug, mode=mode, proposed_patches=proposed_patches, n_shot_bugs=n_shot_bugs,
                                                                        prompt_token_limit=prompt_token_limit, total_token_limit_target=total_token_limit_target,
                                                                        ask_for_bug_description=ask_for_bug_description)
                        try:
                            responses, cost = self.chatgpt.call(prompt, num_of_samples=max(1, min(max_sample_count,num_of_samples)), prefix=f"{prefix}_R{round}_{total_call_tries}")
                            total_cost += cost
                        except openai.error.InvalidRequestError as e:
                            total_cost += prog_params.gpt35_model_token_limit # Exceeded Token limit
                            continue

                        for response in responses:
                            patches = extract_patches_from_response(bug=bug, response=response, response_mode=mode, similarity_threshold=similarity_threshold)
                            for patch, patch_mode in patches:
                                test_result, result_reason, patch_diff = self.framework.validate_patch(bug=bug, proposed_patch=patch, mode=patch_mode)
                                proposed_patches.add(response=response, test_result=test_result, result_reason=result_reason, mode=patch_mode, 
                                                    patch=patch, patch_diff=patch_diff)
                                if first_plausible_patch_try is None and test_result == 'PASS':
                                    first_plausible_patch_try = total_call_tries
            
                if proposed_patches.contains_plausible_patch(mode=mode) == True:
                    break

            plausible_patches, plausible_patch_diffs = proposed_patches.get_plausible_patches(), proposed_patches.get_plausible_patch_diffs()
            test_failure_count += proposed_patches.get_test_failure_count()
            test_error_count += proposed_patches.get_test_error_count()
            total_length += proposed_patches.total_length()

            if proposed_patches.contains_plausible_patch() == True:
                break
        
        return (plausible_patches, plausible_patch_diffs, total_cost, first_plausible_patch_try, None, total_call_tries, test_failure_count, test_error_count, total_length)
