from playwright.sync_api import sync_playwright
import json, time, pygame, utils, os
from itertools import product

username = utils.get_github_credentials()["username"]  # Replace with your own username
password = utils.get_github_credentials()["password"]  # Replace with your own password.
library = "cohere"  # Replace with the library you want to search for

CHARACTERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "_", "-", "a", "b", "c", "d", "e", "f", "g", "h", "i",
              "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

# Scraped Data will be stored here, along with the progress
if os.path.exists(f"results_{library}.json"):
    with open(f"results_{library}.json", "r") as f:
        charCombo_to_results = json.load(f)
else:
    charCombo_to_results = {}
    charCombo_to_results["~remaining_combinations~"] = CHARACTERS.copy()

def get_num_results(num_result):
    """
    Converts the number of results to an integer.
    :param num_result: The number of results as a string. e.g. "1.2k results"
    :return: The number of results as an integer. e.g. 1200
    """
    num_result = num_result.split()[0].replace(',', '')
    num_result = int(float(num_result[:-1]) * 1000) if "k" in num_result else int(num_result)
    return num_result

if __name__ == "__main__":
    with sync_playwright() as p:
        # You can pick your browser of choice: chromium, firefox, webkit
        browser = p.chromium.launch(headless=True)  # Set headless=False to watch the magic happen!
        page = browser.new_page()

        # Navigate to our starting point
        page.goto('https://www.github.com/login', wait_until='networkidle')
        page.fill('#login_field', username)  # Typing into the username field
        page.fill('#password', password)  # Typing into the password field
        page.press('#password', 'Enter')  # Submitting the form by pressing "Enter"

        # LOGIN Manually  -  Waiting for authentication (2 seconds)
        page.wait_for_timeout(2000)

        # Loop through all the three character combinations
        while len(charCombo_to_results["~remaining_combinations~"]) > 0:
            charCombo = charCombo_to_results["~remaining_combinations~"].pop(-1)
            for i in range(1, 6):
                # Go to Code Search
                page.goto(f'https://github.com/search?q=%22from+{library}%22+OR+%22import+{library}%22+language%3Apython+path%3A{charCombo}*+-repo%3Apisterlabs%2Fprompt-linter&type=code&ref=advsearch&p={i}')
                page.set_default_timeout(600000)  # Wait at most 10 minutes for a page to load
                # Keep waiting while we are still rate limited
                wait_count = 90
                while page.is_visible("#suggestions"):
                    print("##################################################")
                    text = page.eval_on_selector("body", "elem => elem.innerText")
                    print(f"{text}\n")
                    print(f"Still rate limited. Waiting {wait_count} seconds.")
                    print("##################################################")
                    page.wait_for_timeout(wait_count * 1000)  # Wait for wait_count seconds
                    # Reload
                    page.reload()
                    wait_count *= 2
                    wait_count = min(wait_count, 60)

                # Search is failing, play alarm!!!
                while page.is_visible(".cKVTEn"):
                        print("##################################################")
                        print("Search is failing.")
                        print("##################################################")
                        # for playing audio.wav file
                        pygame.mixer.init()
                        pygame.mixer.music.load("../scraping/alarm.wav")
                        # Keep playing the sound in a loop
                        while True:
                            time.sleep(1)
                            pygame.mixer.music.play()

                # Wait for the page to load
                page.wait_for_selector(".cgQapc")

                # Get number of results
                num_results = page.eval_on_selector(".cgQapc", "elem => elem.innerText")
                while "More" in num_results:
                    page.wait_for_timeout(1000)
                    num_results = page.eval_on_selector(".cgQapc", "elem => elem.innerText")
                num_results = get_num_results(num_results)  # Convert to integer

                # If the number of results for this charCombo exceeds the pagination limit (5 page, 20 per page),
                # then we skip this charCombo, and add the cartesian product of this charCombo with all characters.
                if num_results > 100 and len(charCombo) < 5:
                    print(f"Skipping {charCombo} as it has {num_results} results.")
                    charCombo_to_results["~remaining_combinations~"] += ["".join(pair) for pair in product([charCombo], CHARACTERS)]
                    break

                # Now, let's find those sneaky little hrefs
                hrefs = page.eval_on_selector_all('.SAskR', 'elements => elements.map(e => e.href)')
                if len(hrefs) == 0:
                    print(f"No results found for {charCombo}. Exiting.")
                    break

                # Storing the results
                charCombo_to_results[charCombo] = charCombo_to_results.get(charCombo, {})
                charCombo_to_results[charCombo]["num_results"] = num_results
                charCombo_to_results[charCombo]["hrefs"] = charCombo_to_results[charCombo].get("hrefs", []) + hrefs

                
                # Print out the results
                print(f"charCombo: {charCombo}; Total: {num_results}; Extracted: {len(charCombo_to_results[charCombo]['hrefs'])} files; Page: {i};")
            
            print(len(charCombo_to_results["~remaining_combinations~"]), "charCombos remaining.")
            # Save the results
            with open(f"results_{library}.json", "w") as f:
                json.dump(charCombo_to_results, f)
            print("##################################################")

        # Close the browser
        browser.close()

    # Save the results
    with open(f"results_{library}.json", "w") as f:
        json.dump(charCombo_to_results, f)

    print("Done!")

    # for playing audio.wav file
    pygame.mixer.init()
    pygame.mixer.music.load("../scraping/alarm.wav")
    # Keep playing the sound in a loop
    while True:
        time.sleep(1)
        pygame.mixer.music.play()
