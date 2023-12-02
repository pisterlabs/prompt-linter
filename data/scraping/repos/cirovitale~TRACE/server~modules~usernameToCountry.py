import threading
import openai
import json

def detectUsernameCountryOpenAI(username, results):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"FORNISCI RISPOSTA IN LINGUA INGLESE. Dato il seguente username [[[ {username} ]]], tenta di prevedere la nazionalità più probabile dell'utente in base al solo nome utente. Restituisci la previsione nel formato JSON indicato, usando 'NULL' se non puoi fare una previsione. Assicurati di utilizzare SOLO singole ('') o doppie (\") virgolette nel formato JSON. Non utilizzare virgolette doppie non escape all'interno delle stringhe JSON. Restituisci SOLO ed ESCLUSIVAMENTE un JSON con il formato seguente: {json.dumps({'isoPredicted': '[PREDIZIONE ISO 3166-1 alpha-2 o NULL]', 'reasons': '[MOTIVAZIONI DELLA SCELTA in LINGUA INGLESE]', 'completeAnswers': '[RISPOSTA DETTAGLIATA in LINGUA INGLESE]'})}"
                }
            ],
            temperature=1
        )
        results['data'] = response
    except Exception as e:
        results['error'] = str(e)

def predictFromUsername(username):
    print('Prediction from username of ', username + '...')
    results = {}
    # Setup thread call and start it
    timeout_seconds = 15
    thread = threading.Thread(target=detectUsernameCountryOpenAI, args=(username, results))
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        print("[OPENAI API] API call timed out")
        # Timeout handling
        return {
            "error": "API call timed out",
            "status": "408"
        }

    if 'error' in results:
        print(f"[OPENAI API] Error: {results['error']}")
        return {
            "error": results['error'],
            "status": "403"
        }

    return results['data']['choices'][0]['message']['content']