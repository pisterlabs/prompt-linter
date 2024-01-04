from playwright.sync_api import sync_playwright
import json, time, pygame, utils

username = utils.get_github_credentials()["username"]  # Replace with your own username
password = utils.get_github_credentials()["password"]  # Replace with your own password.
library = "anthropic"  # Replace with the library you want to search for

def get_repo_name(url):
    url_split = url.split("/")
    # Getting repo name
    repo_name = f"/".join(url_split[3:5])
    return repo_name

if __name__ == "__main__":
    # First pass, get list of repository names
    repo_to_results = {}

    with sync_playwright() as p:
        # You can pick your browser of choice: chromium, firefox, webkit
        browser = p.chromium.launch(headless=False)  # Set headless=False to watch the magic happen!
        page = browser.new_page()

        # Navigate to our starting point
        page.goto('https://www.github.com/login', wait_until='networkidle')
        page.fill('#login_field', username)  # Typing into the username field
        page.fill('#password', password)  # Typing into the password field
        page.press('#password', 'Enter')  # Submitting the form by pressing "Enter"

        # LOGIN Manually  -  Waiting for authentication (2 seconds)
        page.wait_for_timeout(2000)

        stil_searching = True
        URL = f"-repo%3Apisterlabs%2Fprompt-linter"
        while stil_searching:
            # Loop through all the three character combinations
            for i in range(1, 6):
                # Go to Code Search
                page.goto(f'https://github.com/search?q=%22from+{library}%22+OR+%22import+{library}%22+language%3Apython+{URL}&type=code&ref=advsearch&p={i}')

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
                        # Wait for 2 minutes
                        time.sleep(120)
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

                # Now, let's find those sneaky little hrefs
                hrefs = page.eval_on_selector_all('.SAskR', 'elements => elements.map(e => e.href)')

                # Storing results
                for href in hrefs:
                    repo_name = get_repo_name(href)
                    if href not in repo_to_results:
                        repo_to_results[repo_name] = []
                        URL += f"+-repo%3A{repo_name}"
                    repo_to_results[repo_name].append(href)

                if len(hrefs) == 0:
                    if i == 1:
                        stil_searching = False
                    print("Results remaining:", num_results)
                    print(repo_to_results)
                    break
            
        # Close the browser
        browser.close()

    # Save the results
    with open(f"results_{library}.json", "w") as f:
        json.dump(repo_to_results, f)

    print("Done!")

    # for playing audio.wav file
    pygame.mixer.init()
    pygame.mixer.music.load("../scraping/alarm.wav")
    # Keep playing the sound in a loop
    while True:
        time.sleep(1)
        pygame.mixer.music.play()
