from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from starlette.responses import JSONResponse
from typing import Optional
import os
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from dotenv import load_dotenv

load_dotenv()

PRICE_PROMPT = 1.102e-5
PRICE_COMPLETION = 3.268e-5
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

app = FastAPI()

@app.post("/generate-summary")
async def generate_summary(MEETING_TITLE: str, MEETING_EXTRA_INFO: str, SPEAKERS_INFO: str, 
                           claude_model: str, input_text: str, output_file_name: str):
    print("MEETING_TITLE:", MEETING_TITLE)
    print("MEETING_EXTRA_INFO:", MEETING_EXTRA_INFO)
    print("SPEAKERS_INFO:", SPEAKERS_INFO)
    print("claude_model:", claude_model)
    print("input_text:", input_text)
    print("output_file_name:", output_file_name)
    
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
            + f" - Total execution time: {format(total_exec_time, '.2f')} seconds",
            total_cost  # Return the total cost as a float for further calculations
        )
        
    def identify_speakers(input_text):
        PROMPT_SPEAKERS = f"""{HUMAN_PROMPT}
            \n Reunião:[{input_text}] 
            \n Baseado nessa transcrição de reunião, identifique e liste os nomes dos SPEAKERS.
            \n {SPEAKERS_INFO}
            \n{AI_PROMPT}"""

        speakers_response = anthropic.with_options(timeout=5 * 1000).completions.create(
            model=claude_model, max_tokens_to_sample=30000, prompt=PROMPT_SPEAKERS
        )
        speakers = [speaker.strip() for speaker in speakers_response.completion.split(",")]
        return speakers, speakers_response.completion, PROMPT_SPEAKERS

    def update_input_file_with_speakers(input_text, speakers):
        for i, speaker in enumerate(speakers):
            placeholder = f"[SPEAKER_{i:02}]"
            input_text = input_text.replace(placeholder, speaker)
        return input_text

    def generate_meeting_ata(updated_input_text):
        TASK_ATA = f"""
        Levando em consideração que para um bom resumo de atas devemos ter: 
        1. Participantes da Reunião: Identificar quem estava presente é fundamental para entender o contexto da discussão.
        2. Objetivo da Reunião: Uma declaração clara do propósito, ajudará a enquadrar o resumo e o foco da discussão.
        3. Tópicos Discutidos: Um resumo dos principais pontos abordados durante a reunião, possivelmente divididos por tópico ou por quem levantou a questão.
        4. Decisões Tomadas: Qualquer decisão tomada durante a reunião deve ser claramente indicada, incluindo quem a tomou e qualquer justificativa relevante.
        5. Ações Futuras: Se houver tarefas ou ações a serem realizadas após a reunião, estas devem ser anotadas, incluindo quem é responsável pela tarefa e qualquer prazo relevante.
        6. Questões Pendentes: Qualquer item ou questão que não tenha sido resolvido ou que precise ser discutido em uma reunião futura.
        7. Observações Gerais: Qualquer outra informação que possa ser relevante para os participantes da reunião.

        Abaixo teremos uma transcrição referente à {MEETING_TITLE}
        """  
        ACTION_ATA = "Agora, por favor, crie uma ata da reunião com os dados fornecidos."
        PROMPT_ATA = f"""{HUMAN_PROMPT}
        \n {TASK_ATA} Reunião:[{updated_input_text}] 
        \n {ACTION_ATA}
        \n {MEETING_EXTRA_INFO}
        \n{AI_PROMPT}"""
        ata_response = anthropic.with_options(timeout=5 * 1000).completions.create(
            model=claude_model, max_tokens_to_sample=30000, prompt=PROMPT_ATA
        )
        return ata_response.completion, PROMPT_ATA

    anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Start timer
    time_start_first_run = time.time()
    total_costs = 0  # Initialize total costs

    # First run: Identify the speakers
    speakers, speakers_completion, PROMPT_SPEAKERS = identify_speakers(input_text)
    total_exec_time_first_run = time.time() - time_start_first_run
    message, cost = count_used_tokens(PROMPT_SPEAKERS, speakers_completion, total_exec_time_first_run)
    total_costs += cost  
    # Update the input text with the identified speakers
    updated_input_text = update_input_file_with_speakers(input_text, speakers)
    
    # Second run: Generate the meeting ata
    time_start_second_run = time.time()
    ata, PROMPT_ATA = generate_meeting_ata(updated_input_text)
    total_exec_time_second_run = time.time() - time_start_second_run
    message, cost = count_used_tokens(PROMPT_ATA, ata, total_exec_time_second_run)
    total_costs += cost  # Add the cost of the second run to the total
    
    final_message = f"🟢 Total costs for all rounds: {format(total_costs, '.5f')} USD"
    
    with open(output_file_name, "w", encoding="utf-8") as file:
        file.write(ata)
      
    return {
        "summary": ata,
        "message": final_message,
        "cost": format(total_costs, '.5f'),
        "file": output_file_name
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11301)
