from collections import defaultdict
from enum import Enum
from random import choice
from tkinter import END, LEFT, Button, Frame, Label, OptionMenu, StringVar, Text, Tk
from typing import Any, Callable, Optional, Union

import fire

from cot_transparency.apis.openai import OpenAICompletionPrompt
from cot_transparency.data_models.io import ExpLoader
from cot_transparency.data_models.models import (
    StageTwoTaskOutput,
    StageTwoTaskSpec,
    TaskOutput,
    TaskSpec,
)


class ModificationType(str, Enum):
    mistake = "mistake"
    truncation = "truncation"
    no_modification = "no_modification"


StageTwoNestedDict = dict[tuple[str, str, str, str, ModificationType], dict[str, list[StageTwoTaskOutput]]]
StageOneNestedDict = dict[tuple[str, str, str], dict[str, list[TaskOutput]]]

# Dropdown names
dropdown_names = [
    "Task",
    "Model",
    "Formatter",
    "Modification Type",
    "Stage One Formatter",
]


class GUI:
    def __init__(
        self,
        frame: Frame,
        json_dict: StageTwoNestedDict | StageOneNestedDict,
        width: int = 150,
        update_callback: Optional[Callable[..., None]] = None,
        stage=1,
    ):
        self.is_stage_two = stage == 2

        config_width = width // 3
        self.frame = frame
        self.json_dict = json_dict
        self.keys = list(self.json_dict.keys())
        self.index = 0

        self.task_hashes: list = list(json_dict[self.keys[0]].keys())  # type: ignore
        # these are the task specs available for the selected json
        self.task_hash_idx = 0
        self.fontsize = 16
        self.alignment = "w"  # change to "center" to center align
        self.task_var = self.keys[0][0]
        self.update_callback = update_callback or (lambda: None)
        drop_down_width = 80

        # make all the dropdowns
        self.dropdowns = []
        self.dropdown_vars = []
        for i in range(1, len(self.keys[0])):
            var = StringVar(frame)
            var.set(self.keys[0][i])
            self.dropdown_vars.append(var)
            dropdown = OptionMenu(frame, var, *{key[i] for key in self.keys}, command=self.select_json)
            dropdown.config(font=("Arial", self.fontsize), width=drop_down_width)
            dropdown.pack(anchor="center")
            self.dropdowns.append((var, dropdown))

        self.label2 = Label(frame, text="Messages:", font=("Arial", self.fontsize))
        self.label2.pack(anchor=self.alignment)

        self.messages_text = Text(frame, width=width, height=20, font=("Arial", self.fontsize))
        self.messages_text.pack(anchor=self.alignment)

        self.config_and_output_frame = Frame(frame)
        self.config_and_output_frame.pack(anchor=self.alignment)

        self.config_frame = Frame(self.config_and_output_frame, width=config_width)
        self.label = Label(self.config_frame, text="Config:", font=("Arial", self.fontsize))
        self.label.pack(anchor=self.alignment)
        self.config_text = Text(
            self.config_frame,
            width=config_width,
            height=10,
            font=("Arial", self.fontsize),
        )
        self.config_text.pack(anchor=self.alignment)

        self.config_frame.pack(side=LEFT)

        self.output_frame = Frame(self.config_and_output_frame, width=width - config_width)
        self.label3 = Label(
            self.config_and_output_frame,
            text="Model Output:",
            font=("Arial", self.fontsize),
        )
        self.label3.pack(anchor=self.alignment)
        self.output_text = Text(
            self.output_frame,
            width=width - config_width,
            height=6,
            font=("Arial", self.fontsize),
        )
        self.output_text.pack(anchor=self.alignment)
        self.parsed_ans_text = Text(
            self.output_frame,
            width=width - config_width,
            height=4,
            font=("Arial", self.fontsize),
        )
        self.parsed_ans_text.pack(anchor=self.alignment)

        self.output_frame.pack(side=LEFT)

        if self.is_stage_two:
            self.cots_frame = Frame(frame)
            self.cot_texts = []
            self.cot_labels = []
            cots = ["Original COT:", "Modified COT:"]
            for cot in cots:
                cot_frame = Frame(self.cots_frame)
                cot_label = Label(cot_frame, text=cot, font=("Arial", self.fontsize))
                self.cot_labels.append(cot_label)
                cot_label.pack(anchor=self.alignment)
                cot_text = Text(cot_frame, width=width // 2, height=5, font=("Arial", self.fontsize))
                cot_text.pack(anchor=self.alignment)
                cot_text2 = Text(cot_frame, width=width // 2, height=5, font=("Arial", self.fontsize))
                cot_text2.pack(anchor=self.alignment)
                self.cot_texts.append((cot_text, cot_text2))
                cot_frame.pack(side=LEFT)
            self.cots_frame.pack(anchor=self.alignment)

        # Display the first JSON
        self.select_json()
        self.display_output()

    def select_task(self, task_name: str):
        self.task_var = task_name
        self.select_json()
        self.task_hashes = list(self.selected_exp.keys())
        self.task_hash_idx = 0
        self.display_output()

    def select_json(self, *args: Any, reset_index: bool = True):
        key = [self.task_var]
        for var in self.dropdown_vars:
            key.append(var.get())
        key = tuple(key)
        try:
            self.selected_exp = self.json_dict[key]  # type: ignore
            if reset_index:
                self.index = 0  # reset this as there may be fewer outputs
            self.display_output()

            # Do something with the data

        except KeyError:
            self.clear_fields()
            self.display_error()

    def prev_output(self):
        if self.index == 0:
            self.task_hash_idx = (self.task_hash_idx - 1) % len(self.task_hashes)
            new_len = len(self.selected_exp[self.task_hashes[self.task_hash_idx]])
            self.index = new_len - 1
        else:
            self.index = self.index - 1
        self.select_json(reset_index=False)

    def next_output(self):
        len_of_current = len(self.selected_exp[self.task_hashes[self.task_hash_idx]])
        if self.index == len_of_current - 1:
            self.task_hash_idx = (self.task_hash_idx + 1) % len(self.task_hashes)
            self.index = 0
        else:
            self.index = self.index + 1
        self.display_output()
        self.select_json(reset_index=False)

    def select_hash_index(self, idx: int):
        self.task_hash_idx = idx
        self.index = 0
        self.select_json(reset_index=False)

    def clear_fields(self):
        self.messages_text.delete("1.0", END)
        self.config_text.delete("1.0", END)
        self.output_text.delete("1.0", END)
        self.parsed_ans_text.delete("1.0", END)
        # clear other fields as necessary
        if self.is_stage_two:
            for cot_text in self.cot_texts:
                cot_text[0].delete("1.0", END)
                cot_text[1].delete("1.0", END)

    def display_error(self):
        self.messages_text.insert(END, "DATA NOT FOUND")
        self.config_text.insert(END, "DATA NOT FOUND")
        self.output_text.insert(END, "DATA NOT FOUND")
        self.parsed_ans_text.insert(END, "DATA NOT FOUND")
        # add to other fields as necessary

    def display_output(self):
        self.update_callback()
        if not self.update_callback():
            self.display_error()
        experiment = self.selected_exp

        # Clear previous text
        self.clear_fields()

        # Insert new text
        output = experiment[self.task_hashes[self.task_hash_idx]][self.index]

        formatted_output = OpenAICompletionPrompt(messages=output.task_spec.messages).format()
        self.config_text.insert(END, str(output.task_spec.inference_config.model_dump_json(indent=2)))
        self.messages_text.insert(END, formatted_output)
        self.output_text.insert(END, str(output.first_raw_response))
        self.parsed_ans_text.insert(END, str(output.first_parsed_response))

        # insert cot stuff
        task_spec = output.task_spec
        if isinstance(task_spec, StageTwoTaskSpec):
            if task_spec.trace_info:
                original_cot: list[str] = task_spec.trace_info.original_cot
                self.cot_texts[0][0].insert(END, "".join(original_cot))
                try:
                    self.cot_texts[0][1].insert(
                        END,
                        original_cot[task_spec.trace_info.get_mistake_inserted_idx()],
                    )
                    self.cot_texts[1][1].insert(END, task_spec.trace_info.get_sentence_with_mistake())
                except ValueError:
                    pass

                self.cot_labels[1].config(
                    text=(
                        f"Modified COT, mistake_idx: {task_spec.trace_info.mistake_inserted_idx}, "
                        f"CoT Length: {task_spec.n_steps_in_cot_trace}/{len(task_spec.trace_info.original_cot)}"
                    )
                )
                try:
                    complete_modified_cot = task_spec.trace_info.get_complete_modified_cot()
                except ValueError:
                    complete_modified_cot = ""
                self.cot_texts[1][0].insert(END, complete_modified_cot)


def convert_nested_dict_s2(
    outputs: list[StageTwoTaskOutput],
) -> StageTwoNestedDict:
    out: StageTwoNestedDict = defaultdict(dict)
    for output in outputs:
        # use keys
        task_name = output.task_spec.stage_one_output.task_spec.task_name
        model = output.task_spec.stage_one_output.task_spec.inference_config.model
        stage_one_formatter = output.task_spec.stage_one_output.task_spec.formatter_name
        stage_one_hash = output.task_spec.stage_one_output.task_spec.task_hash_with_repeat()
        formatter = output.task_spec.formatter_name
        assert output.task_spec.trace_info is not None
        if output.task_spec.trace_info.has_mistake:
            modification_type = ModificationType.mistake
        elif output.task_spec.trace_info.was_truncated:
            modification_type = ModificationType.truncation
        else:
            modification_type = ModificationType.no_modification

        exp = out[(task_name, model, stage_one_formatter, formatter, modification_type)]
        if stage_one_hash not in exp:
            exp[stage_one_hash] = [output]
        else:
            exp[stage_one_hash].append(output)

    # convert from default_dict to normal dict
    out = dict(out)
    return out


def convert_nested_dict_s1(outputs: list[TaskOutput]) -> StageOneNestedDict:
    out: StageOneNestedDict = defaultdict(dict)
    for output in outputs:
        # use keys
        task_name = output.task_spec.task_name
        model = output.task_spec.inference_config.model
        stage_one_formatter = output.task_spec.formatter_name + (output.task_spec.intervention_name or "")
        stage_one_hash = output.task_spec.task_hash_with_repeat()

        exp = out[(task_name, model, stage_one_formatter)]
        if stage_one_hash not in exp:
            exp[stage_one_hash] = [output]
        else:
            exp[stage_one_hash].append(output)

    # convert from default_dict to normal dict
    out = dict(out)
    return out


class CompareGUI:
    def __init__(
        self,
        master: Tk,
        task_list: list[StageTwoTaskOutput] | list[TaskOutput],
        width: int = 150,
        n_compare: int = 2,
        stage=1,
    ):
        width_of_each = width // n_compare
        self.fontsize = 16
        self.alignment = "center"  # change to "center" to center align

        nested_dict: Union[StageTwoNestedDict, StageOneNestedDict]
        if isinstance(task_list[0], StageTwoTaskOutput):
            nested_dict = convert_nested_dict_s2(task_list)  # type: ignore
        elif isinstance(task_list[0], TaskOutput):
            nested_dict = convert_nested_dict_s1(task_list)  # type: ignore
        else:
            raise ValueError(f"Unknown task type {type(task_list[0])}")

        self.keys = [i[0] for i in nested_dict.keys()]

        self.base_guis: list[GUI] = []
        for i in range(n_compare):
            frame = Frame(master)
            # add the callback to the last base_gui
            if i == n_compare - 1:
                callback = self.check_all_on_the_same_task
            else:
                callback = None

            self.base_guis.append(GUI(frame, nested_dict, width_of_each, callback, stage=stage))
            frame.grid(row=1, column=i)

        self.task_var = StringVar(master)
        self.task_var.set(self.keys[0])  # default value

        self.task_dropdown = OptionMenu(
            master,
            self.task_var,
            *{i for i in self.keys},
            command=self.select_task,  # type: ignore
        )
        self.task_dropdown.config(font=("Arial", self.fontsize))
        self.task_dropdown.grid(row=0, column=0, columnspan=n_compare)

        self.ground_truth_label = Label(master, text="Ground Truth:", font=("Arial", self.fontsize))
        self.ground_truth_label.grid(row=2, column=0, columnspan=n_compare)

        self.buttons_frame = Frame(master)
        self.buttons_frame.grid(row=3, column=0, columnspan=n_compare)

        self.prev_button = Button(
            self.buttons_frame,
            text="Prev",
            command=self.prev_output,
            font=("Arial", self.fontsize),
        )
        self.prev_button.pack(side=LEFT)

        self.next_button = Button(
            self.buttons_frame,
            text="Next",
            command=self.next_output,
            font=("Arial", self.fontsize),
        )
        self.next_button.pack(side=LEFT)

        self.random_button = Button(
            self.buttons_frame,
            text="Random",
            command=self.random_output,
            font=("Arial", self.fontsize),
        )
        self.random_button.pack(side=LEFT)

        self.display_output()

    def check_all_on_the_same_task(self):
        # check that all gui's are on the same index and have the same task_hash
        task_hashes = [gui.task_hashes[gui.task_hash_idx] for gui in self.base_guis]
        assert len(set(task_hashes)) <= 1, "Not all on the same task hash"

    def display_output(self):
        # Insert new text
        # exp: Union[ExperimentJsonFormat, StageTwoExperimentJsonFormat] = self.base_guis[0].selected_exp
        task_hash: str = self.base_guis[0].task_hashes[self.base_guis[0].task_hash_idx]
        is_stage_two = self.base_guis[0].is_stage_two
        output = self.base_guis[0].selected_exp[task_hash][0]

        if is_stage_two:
            task_spec_s2: StageTwoTaskSpec = output.task_spec  # type: ignore
            self.ground_truth_label.config(text=f"Ground Truth: {task_spec_s2.stage_one_output.task_spec.ground_truth}")
        else:
            task_spec: TaskSpec = output.task_spec  # type: ignore
            self.ground_truth_label.config(text=f"Ground Truth: {task_spec.ground_truth}")

    def prev_output(self):
        for gui in self.base_guis:
            gui.prev_output()
        self.display_output()

    def next_output(self):
        for gui in self.base_guis:
            gui.next_output()
        self.display_output()

    def random_output(self):
        idx = choice(range(len(self.base_guis[0].task_hashes)))
        for gui in self.base_guis:
            gui.select_hash_index(idx)
        self.display_output()

    def select_task(self, task_name: str):
        for gui in self.base_guis:
            gui.select_task(task_name)


def main(exp_dir: str, width: int = 175, n_compare: int = 1):
    stage = ExpLoader.get_stage(exp_dir)

    if stage == 1:
        loaded_jsons = ExpLoader.stage_one(exp_dir)
    else:
        loaded_jsons = ExpLoader.stage_two(exp_dir)

    list_of_all_outputs = []
    for exp in loaded_jsons.values():
        list_of_all_outputs.extend(exp.outputs)

    root = Tk()
    CompareGUI(root, list_of_all_outputs, width, n_compare, stage=stage)
    root.mainloop()


if __name__ == "__main__":
    fire.Fire(main)
