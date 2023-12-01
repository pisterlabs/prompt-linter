import os
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from dotenv import load_dotenv

load_dotenv()

PRICE_PROMPT = 1.102e-5
PRICE_COMPLETION = 3.268e-5
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
# print only the first 4 characters of the API key and the last 4 characters of the API key
print(f"ANTHROPIC_API_KEY: {ANTHROPIC_API_KEY[:4]}...{ANTHROPIC_API_KEY[-4:]}")

claude_model = "claude-2.0"  # claude-instant-1, claude-2
input_file_name = "input.txt"
output_file_name = "NamuMatcha_Reuniao_ata.txt"

def count_used_tokens(prompt, completion, total_exec_time):
    input_token_count = anthropic.count_tokens(prompt)
    output_token_count = anthropic.count_tokens(completion)

    input_cost = input_token_count * PRICE_PROMPT
    output_cost = output_token_count * PRICE_COMPLETION

    total_cost = input_cost + output_cost

    return (
        "🟡 Used tokens this round: "
        + f"Input: {input_token_count} tokens, "
        + f"Output: {output_token_count} tokens - "
        + f"{format(total_cost, '.5f')} USD)"
        + f" - Total execution time: {format(total_exec_time, '.2f')} seconds"
    )


with open(input_file_name, "r", encoding="utf-8") as file:
    input_text = file.read()

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

TASK = """
Levando em consideração que para um bom resumo de atas devemos ter: 
1. Participantes da Reunião: Identificar quem estava presente é fundamental para entender o contexto da discussão.
2. Objetivo da Reunião: Uma declaração clara do propósito, ajudará a enquadrar o resumo e o foco da discussão.
3. Tópicos Discutidos: Um resumo dos principais pontos abordados durante a reunião, possivelmente divididos por tópico ou por quem levantou a questão.
4. Decisões Tomadas: Qualquer decisão tomada durante a reunião deve ser claramente indicada, incluindo quem a tomou e qualquer justificativa relevante.
5. Ações Futuras: Se houver tarefas ou ações a serem realizadas após a reunião, estas devem ser anotadas, incluindo quem é responsável pela tarefa e qualquer prazo relevante.
6. Questões Pendentes: Qualquer item ou questão que não tenha sido resolvido ou que precise ser discutido em uma reunião futura.
7. Observações Gerais: Qualquer outra informação que possa ser relevante para os participantes da reunião.

Abaixo teremos uma transcrição referente à Reunião NamuMatcha x Namastex Labs - 24/08/2023

"""

ACTION = "Agora, por favor, crie uma ata da reunião com os dados fornecidos."
PROMPT = f"""{HUMAN_PROMPT}
\n {TASK} Reunião:[{input_text}] 
\n {ACTION}
\n{AI_PROMPT}"""

# TASK_SPEAKERS = "Baseado nessa transcrição de reunião, identifique os SPEAKERS"

# PROMPT_SPEAKERS = f"""{HUMAN_PROMPT}
# \n Reunião:[{input_text}] 
# \n {TASK_SPEAKERS}
# \n{AI_PROMPT}"""

# start timer
time_start = time.time()

summary = anthropic.with_options(timeout=5 * 1000).completions.create(
    model=claude_model,
    max_tokens_to_sample=30000,
    prompt=PROMPT,
)

with open(output_file_name, "w", encoding="utf-8") as file:
    file.write(summary.completion)

total_exec_time: float = time.time() - time_start

print(count_used_tokens(PROMPT, summary.completion, total_exec_time))
