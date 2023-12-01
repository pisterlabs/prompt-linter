import streamlit as st
import os
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from dotenv import load_dotenv

load_dotenv()

PRICE_PROMPT = 1.102e-5
PRICE_COMPLETION = 3.268e-5
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")


def main():
    st.title("Meeting Summary Generator")
    # Initialise input fields for the user to provide details using the st.session_state
    if "meeting_extra_info" not in st.session_state:
        st.session_state.meeting_extra_info = ""
    MEETING_EXTRA_INFO = st.text_area(
        "Extra Meetings Information", st.session_state.meeting_extra_info
    )

    if "speakers_info" not in st.session_state:
        st.session_state.speakers_info = ""
    SPEAKERS_INFO = st.text_area("Speakers Information", st.session_state.speakers_info)

    claude_model_options = ["claude-2.0", "claude-instant-1"]
    if "claude_model" not in st.session_state:
        st.session_state.claude_model = claude_model_options[0]
    claude_model = st.selectbox(
        "Select Model",
        claude_model_options,
        claude_model_options.index(st.session_state.claude_model),
    )

    if "output_file_name" not in st.session_state:
        st.session_state.output_file_name = "_ata.txt"
    output_file_name = st.text_input(
        "Output File Name", st.session_state.output_file_name
    )

    # File selector for the user to select the input file
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        input_text = uploaded_file.read().decode("utf-8")
    else:
        st.warning("Please upload a file.")
        return

    # Button to trigger the summary generation
    if st.button("Generate Summary"):
        progress = st.progress(0)
        progress_text = st.empty()

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
                total_cost,  # Return the total cost as a float for further calculations
            )

        def identify_speakers(input_text):
            PROMPT_SPEAKERS = f"""{HUMAN_PROMPT}
                \n {SPEAKERS_INFO}
                \n Reunião:[{input_text}] 
                \n Baseado nessa transcrição de reunião, identifique e liste os nomes dos SPEAKERS.
                \n Utilizando a seguinte formatação: "Nome, Nome, Nome"
                \n{AI_PROMPT}"""

            speakers_response = anthropic.with_options(
                timeout=5 * 1000
            ).completions.create(
                model=claude_model, max_tokens_to_sample=30000, prompt=PROMPT_SPEAKERS
            )
            speakers = [
                speaker.strip() for speaker in speakers_response.completion.split(",")
            ]
            return speakers, speakers_response.completion, PROMPT_SPEAKERS

        def update_input_file_with_speakers(input_text, speakers):
            for i, speaker in enumerate(speakers):
                placeholder = f"[SPEAKER_{i:02}]"
                input_text = input_text.replace(placeholder, speaker)
            return input_text

        def generate_meeting_ata(updated_input_text):
            with open("prompts/meeting-ata-br.txt", "r", encoding="utf-8") as file:
                TASK_ATA = file.read()

            ACTION_ATA = (
                "Agora, por favor, crie uma ata da reunião com os dados fornecidos."
            )
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

        # print only the first 4 characters of the API key and the last 4 characters of the API key
        print(f"ANTHROPIC_API_KEY: {ANTHROPIC_API_KEY[:4]}...{ANTHROPIC_API_KEY[-4:]}")

        # Start timer
        time_start_first_run = time.time()
        total_costs = 0  # Initialize total costs

        # First run: Identify the speakers
        progress_text.text("Identifying speakers...")
        speakers, speakers_completion, PROMPT_SPEAKERS = identify_speakers(input_text)
        st.write(speakers)
        total_exec_time_first_run = time.time() - time_start_first_run
        message, cost = count_used_tokens(
            PROMPT_SPEAKERS, speakers_completion, total_exec_time_first_run
        )
        print(message)
        total_costs += cost
        st.write(message)
        # Update the input text with the identified speakers
        updated_input_text = update_input_file_with_speakers(input_text, speakers)
        progress.progress(0.5)

        # Second run: Generate the meeting ata
        time_start_second_run = time.time()
        progress_text.text("Generating meeting summary...")
        ata, PROMPT_ATA = generate_meeting_ata(updated_input_text)
        total_exec_time_second_run = time.time() - time_start_second_run
        message, cost = count_used_tokens(PROMPT_ATA, ata, total_exec_time_second_run)
        print(message)
        st.write(message)
        total_costs += cost  # Add the cost of the second run to the total

        progress.progress(1)

        final_message = (
            f"🟢 Total costs for all rounds: {format(total_costs, '.5f')} USD"
        )
        print(final_message)
        st.write(final_message)

        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(ata)
        st.success(f"Summary saved to: {output_file_name}")

        # Display the result to the user
        st.subheader("Generated Summary:")
        st.write(ata)
        st.download_button(
            "Download Generated Summary",
            ata,
            file_name=f"{output_file_name}",
            mime="text/plain",
            key="download_summary",
        )

        st.download_button(
            "Download Modified Input File",
            updated_input_text,
            file_name=f"modified_{output_file_name}",
            mime="text/plain",
            key="download_modified_input",
        )
        # Display the cost details
        st.subheader("Cost Details:")
        st.write(f"Total costs for all rounds: {format(total_costs, '.5f')} USD")


if __name__ == "__main__":
    main()
