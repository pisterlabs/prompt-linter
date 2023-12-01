#!/usr/bin/python
import openai
from core.gpt import GPT
from colorama import Fore
def Sloka():
	openai.api_key = "<Your_OpenAI_GPT3_API_KEY>"
	gpt = GPT(engine="davinci",temperature=0.9)

	# To train GPT we are using some examples of sanskrit slokes so that it will automatically generate slokes . 
	prompt = """Sanskit Slokas:dehino’sminyathā dehe kaumāraṃ yauvanaṃ jarā।tathā dehāntaraprāptirdhīrastatra na muhyati,\n

	English Translation:
	Just as the boyhood, youth and old age come to the embodied Soul in this body, in the same manner, is the attaining of another body; the wise man is not deluded at that.

	Sanskit Slokas:ya enaṃ vetti hantāraṃ yaścainaṃ manyate hatamubhau tau na vijānīto nāyaṃ hanti na hanyate,
	English Translation:
	He who thinks that the soul kills, and he who thinks of it as killed, are both ignorant. The soul kills not, nor is it killed.

	Sanskit Slokas:na jāyate mriyate vā kadācin nāyaṃ bhūtvā bhavitā vā na bhūyaḥ ajo nityaḥ śāśvato’yaṃ purāṇo na hanyate hanyamāne śarīre,
	English Translation:
	The soul is never born, it never dies having come into being once, it never ceases to be.
	Unborn, eternal, abiding and primeval, it is not slain when the body is slain.

	Sanskit Slokas:nainaṃ chindanti śastrāṇi nainaṃ dahati pāvakaḥ na cainaṃ kledayantyāpo na śoṣayati mārutaḥ ,
	English Translation:
	Weapons do not cleave the soul, fire does not burn it, waters do not wet it, and wind does not dry it."""

	output = gpt.submit_request(prompt)
	print(Fore.BLUE + output.choices[0].text)

if __name__ == '__main__':
	print(Fore.RED + '''

                  _   _ _             
  /\ /\__ _ _   _| |_(_) |_   _  __ _ 
 / //_/ _` | | | | __| | | | | |/ _` |
/ __ \ (_| | |_| | |_| | | |_| | (_| |
\/  \/\__,_|\__,_|\__|_|_|\__, |\__,_|
                          |___/         > v1.0 by @powerexploit

''')
	print(Fore.WHITE + "[+] Work in progress....")
	print(Fore.WHITE + "[+] Kautilya generating Sanskrit Slokas....")
	Sloka()
	exit()
