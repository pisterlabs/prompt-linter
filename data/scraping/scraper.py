from playwright.sync_api import sync_playwright
import json, time, pygame

username = "DJPAUL2001"  # Replace with your own username
password = "Pl3453D0N07H4ckM3!"  # Replace with your own password.
library = "openai"  # Replace with the library you want to search for

characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "_", "a", "b", "c", "d", "e", "f", "g", "h", "i",
              "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "v", "u", "w", "x", "y", "z"]
two_char_combinations = [f"{char1}{char2}" for char1 in characters for char2 in characters]



if __name__ == "__main__":
    charCombo_to_results = {}

    with sync_playwright() as p:
        # You can pick your browser of choice: chromium, firefox, webkit
        browser = p.chromium.launch(headless=False)  # Set headless=False to watch the magic happen!
        page = browser.new_page()

        # Navigate to our starting point
        page.goto('https://www.github.com/login', wait_until='networkidle')
        page.fill('#login_field', username)  # Typing into the username field
        page.fill('#password', password)  # Typing into the password field
        page.press('#password', 'Enter')  # Submitting the form by pressing "Enter"

        # LOGIN Manually  -  Waiting for authentication (10 seconds)
        page.wait_for_timeout(10000)

        # Loop through all the two character combinations
        for charCombo in two_char_combinations:
            for i in range(1, 6):
                # Go to Code Search
                page.goto(f'https://github.com/search?q=%22from+{library}%22+OR+%22import+{library}%22+language%3Apython+path%3A{charCombo}*&type=code&ref=advsearch&p={i}')

                # Keep waiting while we are still rate limited
                wait_count = 1
                while page.is_visible("#suggestions"):
                    print("##################################################")
                    text = page.eval_on_selector("body", "elem => elem.innerText")
                    print(f"{text}\n")
                    print(f"Still rate limited. Waiting {wait_count} second.")
                    print("##################################################")
                    page.wait_for_timeout(wait_count* 1000)  # Wait for wait_count seconds
                    # Reload
                    page.reload()
                    wait_count *= 2
                    wait_count = min(wait_count, 60)
                
                # Wait for the page to load
                page.wait_for_selector(".cgQapc")

                # Get number of results
                num_results = page.eval_on_selector(".cgQapc", "elem => elem.innerText")
                while "More" in num_results:
                    page.wait_for_timeout(1000)
                    num_results = page.eval_on_selector(".cgQapc", "elem => elem.innerText")

                # Now, let's find those sneaky little hrefs
                hrefs = page.eval_on_selector_all('.vyRTf', 'elements => elements.map(e => e.href)')

                # Storing the results
                charCombo_to_results[charCombo] = charCombo_to_results.get(charCombo, {})
                charCombo_to_results[charCombo]["num_results"] = num_results
                charCombo_to_results[charCombo]["hrefs"] = charCombo_to_results[charCombo].get("hrefs", []) + hrefs

                if len(hrefs) == 0:
                    print("No results found. Exiting.")
                    break
                
                # Print out the results
                print(f"charCombo: {charCombo}; Total Files: {num_results}; Extracted: {len(charCombo_to_results[charCombo]['hrefs'])} files; Page: {i};")

        # Close the browser
        browser.close()

    # Save the results
    with open(f"results_{library}.json", "w") as f:
        json.dump(charCombo_to_results, f)

    print("Done!")

    # for playing audio.wav file
    pygame.mixer.init()
    pygame.mixer.music.load("alarm.wav")
    # Keep playing the sound in a loop
    while True:
        time.sleep(1)
        pygame.mixer.music.play()
