import openai

class ArticleGenerator:

    INTENT_DESCRIPTION = "Schreibe einen neuen Artikel für die deutschen Nachrichten, passend zu diesem Thema:"
    INTENT_TITLE = "Schreibe einen Titel für diesen Artikel:"
    INTENT_TAGS = "Gib mir eine Komma-getrennte Liste Schlagworte für diesen Artikel:"
    INTENT_CATEGORY = "Welche Kategorie passt am besten für den Text: "

    def __init__(self, openai_apikey):
        self.openai_apikey = openai_apikey
        openai.api_key = self.openai_apikey

    def generateImage(self, title):
        response = openai.Image.create(
            prompt=title,
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']
        return image_url

    def generateTitle(self, description):
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=self.INTENT_TITLE + '"' + description + '"',
            temperature=0.9,
            max_tokens=3000,
            top_p=1,
            frequency_penalty=0.13,
            presence_penalty=0.3
        )
        return response.choices[0].text.strip()

    def generateDescription(self, title):
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=self.INTENT_DESCRIPTION + title + "\n\n",
            temperature=0.9,
            max_tokens=3500,
            top_p=1,
            frequency_penalty=0.13,
            presence_penalty=0.3
        )
        return response.choices[0].text.strip()

    def generateTags(self, description):
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=self.INTENT_TAGS + description,
            temperature=0.9,
            max_tokens=3500,
            top_p=1,
            frequency_penalty=0.13,
            presence_penalty=0.3
        )

        raw_tags = response.choices[0].text.strip()
        if "," in raw_tags:
            tags = raw_tags.split(",")
        elif "\n" in raw_tags:
            tags = raw_tags.split("\n")
        else:
            tags = raw_tags.split(" ")

        tags = [i.strip('#').strip("-").strip() for i in tags]
        return tags

    def getCategory(self, categories, description):
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=self.INTENT_CATEGORY + ' ,'.join(categories.values()) + "\n\n" + description,
            temperature=0.9,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0.13,
            presence_penalty=0.3
        )
        return response.choices[0].text.strip().rsplit(':', 1)[-1]

    def generateArticle(self, title, description, categories):

        article = {}
        try:
            article["description"] = self.generateDescription(title)
            article["title"] = self.generateTitle(article["description"])
            article["image"] = self.generateImage(article["title"])
            article["tags"] = self.generateTags(article["description"])
            article["category"] = self.getCategory(categories, article["description"])
        except Exception as e:
            print(e)

        return article
