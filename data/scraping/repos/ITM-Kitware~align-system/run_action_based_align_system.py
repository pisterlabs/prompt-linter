import sys
import json

from rich.highlighter import JSONHighlighter

from align_system.utils import logging
from align_system.interfaces.cli_builder import build_interfaces
from align_system.algorithms.llm_baseline import LLMBaseline
from align_system.algorithms.llama_index import LlamaIndex
from align_system.similarity_measures import build_force_choice_func
from align_system.prompt_engineering.common import (
    prepare_action_based_prompt,
    prepare_treatment_selection_prompt,
    prepare_tagging_selection_prompt)


log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


def add_cli_args(parser):
    parser.add_argument('-m', '--model',
                        type=str,
                        default="falcon",
                        help="LLM Baseline model to use")
    parser.add_argument('-t', '--align-to-target',
                        action='store_true',
                        default=False,
                        help="Align algorithm to target KDMAs")
    parser.add_argument('-a', '--algorithm',
                        type=str,
                        default="llama_index",
                        help="Algorithm to use")
    parser.add_argument('-A', '--algorithm-kwargs',
                        type=str,
                        required=False,
                        help="JSON encoded dictionary of kwargs for algorithm "
                             "initialization")
    parser.add_argument('--similarity-measure',
                        type=str,
                        default="bert",
                        help="Similarity measure to use (default: 'bert')")
    parser.add_argument('-l', '--loglevel',
                        type=str,
                        default='INFO')


def main():
    log.debug(f"[bright_black]CMD: {' '.join(sys.argv)}[/bright_black]",
              extra={'markup': True, 'highlighter': None})
    run_action_based_align_system(
        **build_interfaces(add_cli_args, "ALIGN Action Based System CLI",
                           supported_interfaces={'TA3ActionBased'}))


# TODO: Find a better location for this (or pull in from TA3 client
# module)
TREATMENT_LOCATIONS = ['right forearm',
                       'left forearm',
                       'right calf',
                       'left calf',
                       'right thigh',
                       'left thigh',
                       'right stomach',
                       'left stomach',
                       'right bicep',
                       'left bicep',
                       'right shoulder',
                       'left shoulder',
                       'right side',
                       'left side',
                       'right chest',
                       'left chest',
                       'right wrist',
                       'left wrist',
                       'left face',
                       'right face',
                       'left neck',
                       'right neck',
                       'unspecified']

CASUALTY_TAGS = ['MINIMAL',
                 'DELAYED',
                 'IMMEDIATE',
                 'EXPECTANT']


def run_action_based_align_system(interface,
                                  model,
                                  align_to_target=False,
                                  algorithm="llm_baseline",
                                  algorithm_kwargs=None,
                                  similarity_measure="bert",
                                  loglevel="INFO"):
    log.setLevel(loglevel)

    scenario = interface.start_scenario()
    scenario_dict = scenario.to_dict()

    if align_to_target:
        alignment_target_dict = scenario.get_alignment_target()

    force_choice_func = build_force_choice_func(similarity_measure)

    # Load the system / model
    algorithm_kwargs_parsed = {}
    if algorithm_kwargs is not None:
        algorithm_kwargs_parsed = json.loads(algorithm_kwargs)

    if algorithm == "llm_baseline":
        algorithm = LLMBaseline(
            model_use=model, distributed=False,
            **algorithm_kwargs_parsed)
    elif algorithm == "llama_index":
        # TODO: This is a hacky way to have the "Knowledge" KDMA
        # determine whether or not domain documents should be loaded.
        # Should remove, or move to llama_index code
        if align_to_target:
            for kdma_dict in alignment_target_dict.get('kdma_values', ()):
                if kdma_dict['kdma'].lower() == 'knowledge':
                    if kdma_dict['value'] > 1:
                        log.info("** Setting 'retrieval_enabled' to True "
                                 "based on 'Knowledge' KDMA value ({})".format(
                                     kdma_dict['value']))
                        algorithm_kwargs_parsed['retrieval_enabled'] = True
                    else:
                        log.info("** Setting 'retrieval_enabled' to False "
                                 "based on 'Knowledge' KDMA value ({})".format(
                                  kdma_dict['value']))
                        algorithm_kwargs_parsed['retrieval_enabled'] = False

                    break

        algorithm = LlamaIndex(
            model_name=model,
            **algorithm_kwargs_parsed)

        algorithm.load_model()

    current_state = scenario.get_state()
    scenario_complete = current_state.get('scenario_complete', False)

    while not scenario_complete:
        available_actions = scenario.get_available_actions()

        untagged_casualties = [c for c in current_state['casualties']
                               if 'tag' not in c]

        # Don't let ADM choose to tag a casualty unless there are
        # still untagged casualties
        available_actions_unstructured =\
            [a['unstructured'] for a in available_actions
             if a['action_type'] != 'TAG_CASUALTY'
             or (a['action_type'] == 'TAG_CASUALTY'
                 and len(untagged_casualties) > 0)]

        prompt = prepare_action_based_prompt(
            scenario_dict['state']['unstructured'],
            current_state['mission'].get('unstructured'),
            current_state['unstructured'],
            current_state['casualties'],
            available_actions_unstructured,
            alignment_target=alignment_target_dict if align_to_target else None
        )
        log.info("[bold]* Action prompt for ADM *[/bold]",
                 extra={"markup": True})
        log.info(prompt)

        raw_response = str(algorithm.run_inference(prompt))
        log.info("[bold]* ADM raw response *[/bold]",
                 extra={"markup": True})
        log.info(raw_response)

        selected_action_idx, selected_action = force_choice_func(
            raw_response, available_actions_unstructured)

        log.info("[bold]* Mapped selection *[/bold]",
                 extra={"markup": True})
        log.info(selected_action)

        action_to_take = available_actions[selected_action_idx]

        if action_to_take['action_type'] == 'APPLY_TREATMENT':
            # Ask the system to specify the treatment to use and where

            # First casualty with the matching ID (should only be one)
            casualty_id = action_to_take['casualty_id']
            matching_casualties = [c for c in current_state['casualties']
                                   if c['id'] == casualty_id]

            assert len(matching_casualties) == 1
            casualty_to_treat = matching_casualties[0]

            treatment_prompt = prepare_treatment_selection_prompt(
                casualty_to_treat['unstructured'],
                casualty_to_treat['vitals'],
                current_state['supplies'])

            log.info("[bold]** Treatment prompt for ADM **[/bold]",
                     extra={"markup": True})
            log.info(treatment_prompt)

            raw_treatment_response =\
                str(algorithm.run_inference(treatment_prompt))

            log.info("[bold]** ADM raw treatment response **[/bold]",
                     extra={"markup": True})
            log.info(raw_treatment_response)

            # Map response to treatment and treatment location
            _, treatment = force_choice_func(
                raw_treatment_response,
                [s['type'] for s in current_state['supplies']])

            _, treatment_location = force_choice_func(
                raw_treatment_response,
                TREATMENT_LOCATIONS)

            log.info("[bold]** Mapped treatment selection **[/bold]",
                     extra={"markup": True})
            log.info("{}: {}".format(treatment, treatment_location))

            # Populate required parameters for treatment action
            action_to_take['parameters'] = {
                'treatment': treatment,
                'location': treatment_location}
        elif action_to_take['action_type'] == 'TAG_CASUALTY':
            # Ask the system to specify which triage tag to apply

            tagging_prompt = prepare_tagging_selection_prompt(
                untagged_casualties,
                CASUALTY_TAGS)

            log.info("[bold]** Tagging prompt for ADM **[/bold]",
                     extra={"markup": True})
            log.info(tagging_prompt)

            raw_tagging_response =\
                str(algorithm.run_inference(tagging_prompt))

            log.info("[bold]** ADM raw tagging response **[/bold]",
                     extra={"markup": True})
            log.info(raw_tagging_response)

            # Map response to casualty to tag
            casualty_to_tag_idx, _ = force_choice_func(
                raw_tagging_response,
                [c['unstructured'] for c in untagged_casualties])

            casualty_to_tag_id = untagged_casualties[casualty_to_tag_idx]['id']

            # Map response to tag
            _, tag = force_choice_func(
                raw_tagging_response,
                CASUALTY_TAGS)

            log.info("[bold]** Mapped tag selection **[/bold]",
                     extra={"markup": True})
            log.info("{}: {}".format(casualty_to_tag_id, tag))

            # Populate required parameters for treatment action
            action_to_take['casualty_id'] = casualty_to_tag_id
            action_to_take['parameters'] = {'category': tag}

        log.debug("[bold]*ACTION BEING TAKEN*[/bold]",
                  extra={"markup": True})
        log.debug(json.dumps(action_to_take, indent=4),
                  extra={"highlighter": JSON_HIGHLIGHTER})

        current_state = scenario.take_action(action_to_take)

        scenario_complete = current_state.get('scenario_complete', False)


if __name__ == "__main__":
    main()
