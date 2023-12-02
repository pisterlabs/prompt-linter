from django.core.management.base import BaseCommand
from tqdm import tqdm
from nucleus.ai_helpers import openai_titles_from_text_chat
from nucleus.models import GameLog
from nucleus.gdrive import fetch_airel_file_text
import json


class Command(BaseCommand):
    help = "Create up to 5 AI suggestion objects with just titles for each log"

    def handle(self, *args, **options):
        logs = GameLog.objects.all()
        for log in tqdm(logs):
            log_text = fetch_airel_file_text(log.google_id)
            try:
                response = openai_titles_from_text_chat(log_text)
                res_json = response["choices"][0]["message"]["content"]
                obj = json.loads(res_json)
                if len(obj["titles"]) > 5:
                    obj["titles"] = obj["titles"][:5]
                for title in obj["titles"]:
                    log.ailogsuggestion_set.create(title=title)
            except Exception as e:
                print(f"Error creating AI suggestion for {log}: {e}")
                continue
