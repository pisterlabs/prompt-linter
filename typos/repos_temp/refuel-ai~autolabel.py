"""
    <s>[INST] <<SYS>>
    {task_guidelines}{output_guidelines}
    <</SYS>>
    {current_example}[/INST]\n""""""
    <s>[INST] <<SYS>>
    {task_guidelines}{output_guidelines}\n{seed_examples}
    <</SYS>>
    {current_example}[/INST]\n"""