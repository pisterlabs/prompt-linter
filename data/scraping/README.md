# Scraping using Playwright

usage: `python3 scraper.py`

WARNING: The library, and the GitHub username and password are hardcoded in the script. Please change them before running the script.

### Target libraries
libraries = ["openai", "llamaindex", "cohere", "guidance", "anthropic", "langchain"]

### Results for 11/30/2023

Result Counts from Github (Collected Manually):

    openai: 71.7k
    langchain: 50.2k
    cohere: 5.1k
    guidance: 1.6k
    anthropic: 1.5k
    llamaindex: 91

Library: openai
    
        Total number of results: 33952
        Total number of hrefs: 18206

Library: langchain

        Total number of results: 28803
        Total number of hrefs: 16775

Library: cohere

        Total number of results: 5380
        Total number of hrefs: 4121

Library: llamaindex

        Total number of results: 91
        Total number of hrefs: 91

Library: guidance

        Total number of results: 1303
        Total number of hrefs: 1195

- Total number of hrefs indicate the number of files downloaded. 

- Total number of results detected during scraping for openai and langchain are about half the results detected manually due to various reasons. 