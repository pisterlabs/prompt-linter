
import os
import pdfplumber
import openai
import csv
from typing import List, Dict

# Diese Funktion sollte den Text aus einem PDF extrahieren
def textfrompdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = []
        for page in pdf.pages:
            text += [page.extract_text()]

    return text


def gpt_flashcards(pages):
    # Teilen Sie den Eingabetext in Segmente von x Zeilen auf
    

    # Liste für die Flashcards erstellen
    flashcard_list = []

    for segment in pages:
        # Führen Sie einen API-Request mit dem aktuellen Segment und dem Prompt durch
        response = make_api_request(segment)

        # Verarbeiten Sie die API-Antwort, um Flashcards zu extrahieren
        flashcard = extract_flashcard(response)
        # Fügen Sie die Flashcards der Liste hinzu
        flashcard_list.append(flashcard)

    return flashcard_list


def make_api_request(text):
    prompt = '''Erstelle eine Lernkarte für Anki mit einer Vorder- und Rückseite. Verwende HTML-Tags, um die Karten zu definieren: `<vorn></vorn>` für die Vorderseite und `<hinten></hinten>` für die Rückseite. Geben Sie nur den grundlegenden Karten-Strukturcode zurück.

Für die Frage auf der Vorderseite und die Musterantwort auf der Rückseite könnten folgende Beispiele dienen:
<vorn>Was ist die Hauptstadt von Frankreich?</vorn>
<hinten>Die Hauptstadt von Frankreich ist Paris.</hinten>

Frage auch nach definitionen mit: Erkläre den Begriff, was ist, was ist der Unterschied
Fügen Sie gerne zusätzliches Wissen zum Thema hinzu, aber halten Sie die Karten kurz und prägnant. bitte MAXIMAL eine Karte pro text erstellen! das ist enorm wichtig!
mach lieber eine karte mit zwei ähnlichen fragen ( zum beispiel ein A und B teil)
Solltest du denken dass der Text wenig sinn zu einem konkreten Thema ergibt, dann handelt es sich vermutlich um den text einer Folie mit Bildern oder einer Vorstelldung des Dozenten.
Lass diese Folien bitte aus und gibt -keine inhalte- zurück
die Frage sollte die Zentralen inhalte des textes bestmöglich abdecken.
die Rückseite sollte die Frage beantworten und zusätzliche Informationen enthalten, die Sie sich merken möchten.
solltest du denken, dass der text keine fachlichen bezug hat wie zb vorstellungsrunden oder nur ein name  bitte einfach überspringen und -keine inhalte- zurückgeben
hier ist der text:'''

    apikey = "ä"  # Replace with your OpenAI API key
    openai.api_key = apikey

    response = openai.Completion.create(
        engine="text-davinci-003",
        temperature = 0,
        max_tokens=1000,
        prompt = prompt+ text
        )
    api_response = response.choices[0].text.strip()

    # Extract the generated text from the OpenAI response
    print("---------------------------")
    print(api_response)
    print("---------------------------")
    return api_response


def extract_flashcard(api_response):

    flashcard = {}
    start_tag = "<vorn>"
    end_tag = "</vorn>"
    back_start_tag = "<hinten>"
    back_end_tag = "</hinten>"

    while start_tag in api_response and end_tag in api_response and back_start_tag in api_response and back_end_tag in api_response:
        start_index = api_response.index(start_tag)
        end_index = api_response.index(end_tag)
        back_start_index = api_response.index(back_start_tag)
        back_end_index = api_response.index(back_end_tag)

        front = api_response[start_index + len(start_tag):end_index]
        back = api_response[back_start_index + len(back_start_tag):back_end_index]

        flashcard={"front": front.strip(), "back": back.strip()}
        print(flashcard)

        # Remove the extracted flashcard from the API response
        api_response = api_response[end_index + len(end_tag):]

    return flashcard



def export_to_csv(flashcards: List[Dict[str, str]], output_folder: str) -> None:
    """Export flashcards to a CSV file."""
    csv_filename = "cards.csv"
    csv_path = os.path.join(output_folder, csv_filename)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["front", "back"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for flashcard in flashcards:
            writer.writerow(flashcard)

def main(input_folder: str, output_folder: str) -> None:
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all PDF files in the input folder
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        
        pages = textfrompdf(pdf_path)
        # Extract flashcards from the PDF
        flashcards = gpt_flashcards(pages)

        # Export flashcards to a CSV file
        export_to_csv(flashcards, output_folder)

if __name__ == "__main__":
    input_folder = r"/home/ian/Documents/Repository/MindWork/Anki"  # Set your input folder path
    output_folder = r"/home/ian/Documents/Repository/MindWork/Anki"  # Set your output folder path
    main(input_folder, output_folder)
