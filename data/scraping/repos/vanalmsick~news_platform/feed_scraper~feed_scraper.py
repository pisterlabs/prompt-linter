"""This file is doing the article scraping"""

import datetime
import hashlib
import html
import math
import random
import threading
import time
import traceback
import urllib
from urllib.parse import urlparse

import feedparser
import langid
import ratelimit
import requests
from bs4 import BeautifulSoup
from django.conf import settings
from django.core.cache import cache
from django.core.validators import URLValidator
from django.db.models import Max, Q
from linkpreview import Link, LinkPreview
from linkpreview.exceptions import (
    InvalidContentError,
    InvalidMimeTypeError,
    MaximumContentSizeError,
)
from openai import OpenAI

from articles.models import Article, FeedPosition
from feeds.models import Feed


def postpone(function):
    """
    Reusable decorator function to run any function async to Django User request - i.e. that the user does not
    have to wait for completion of function to get the requested view
    """

    def decorator(*args, **kwargs):
        t = threading.Thread(target=function, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()

    return decorator


def update_feeds():
    """Main function that refreshes/scrapes articles from article feed sources."""
    start_time = time.time()

    # delete feed positions of inactive feeds
    inactive_feeds = Feed.objects.filter(~Q(active=True))
    for feed in inactive_feeds:
        delete_feed_positions(feed=feed)

    all_articles = Article.objects.exclude(content_type="video")
    all_articles.update(min_feed_position=None)
    all_articles.update(max_importance=None)
    all_articles.update(min_article_relevance=None)

    # get acctive feeds
    feeds = Feed.objects.filter(active=True, feed_type="rss")

    added_articles = 0
    for feed in feeds:
        added_articles += fetch_feed(feed)

    # calculate next refesh time
    end_time = time.time()

    now = datetime.datetime.now()
    if now.hour >= 18 or now.hour < 6 or now.weekday() in [5, 6]:
        print(
            "No AI summaries are generated during non-business hours (i.e. between"
            " 18:00-6:00 and on Saturdays and Sundays)"
        )
    else:
        min_article_relevance = (
            Article.objects.filter(
                has_full_text=True, categories__icontains="FRONTPAGE"
            )
            .exclude(publisher__name__in=["Risk.net", "The Economist"])
            .exclude(min_article_relevance__isnull=True)
            .order_by("min_article_relevance")[:20]
            .aggregate(Max("min_article_relevance"))["min_article_relevance__max"]
        )
        articles_add_ai_summary = (
            Article.objects.filter(
                has_full_text=True,
                ai_summary__isnull=True,
                categories__icontains="FRONTPAGE",
                min_article_relevance__lte=min_article_relevance,
            )
            .exclude(publisher__name__in=["Risk.net", "The Economist"])
            .exclude(min_article_relevance__isnull=True)
            .order_by("min_article_relevance")
        )
        add_ai_summary(article_obj_lst=articles_add_ai_summary)

    old_articles = Article.objects.filter(
        min_article_relevance__isnull=True,
        feed_position=None,
        added_date__lte=settings.TIME_ZONE_OBJ.localize(
            datetime.datetime.now() - datetime.timedelta(days=7)
        ),
    )
    if len(old_articles) > 0:
        print(f"Delete {len(old_articles)} old articles")
        old_articles.delete()
    else:
        print("No old articles to delete")

    print(
        f"Refreshed articles and added {added_articles} articles in"
        f" {int(end_time - start_time)} seconds"
    )

    # connection.close()


def calcualte_relevance(publisher, feed, feed_position, hash, pub_date):
    """
    This function calsucates the relvanec score for all artciles and videos depensing on user
    settings and article positions
    """
    random.seed(hash)

    random_int = random.randrange(0, 9) / 10000
    feed__importance = feed.importance  # 0-4
    feed__ordering = feed.feed_ordering  # r or d
    publisher__renowned = publisher.renowned  # -3-3
    content_type = "art" if feed.feed_type == "rss" else "vid"
    publisher_article_count = cache.get(f"publisher_{content_type}_cnt_{publisher.pk}")
    if pub_date is None:
        article_age = 3
    else:
        article_age = (
            settings.TIME_ZONE_OBJ.localize(datetime.datetime.now()) - pub_date
        ).total_seconds() / 3600

    factor_publisher__renowned = {
        3: 2 / 6,  # Top Publisher = 3x
        2: 4 / 6,  # Higly Renowned Publisher = 1.5x
        1: 5 / 6,  # Renowned Publisher = 1.2x
        0: 6 / 6,  # Regular Publisher = 1x
        -1: 8 / 6,  # Lesser-known Publisher = 0.75x
        -2: 10 / 6,  # Unknown Publisher = 0.6x
        -3: 12 / 6,  # Inaccurate Publisher = 0.5x
    }[publisher__renowned]

    # Publisher artcile ccount normalization
    if publisher_article_count is None or publisher_article_count < 25:
        factor_article_normalization = 25 / 100
    else:
        factor_article_normalization = min(publisher_article_count / 100, 6)

    factor_feed__importance = {
        4: 1 / 4,  # Lead Articles News: 4x
        3: 2 / 4,  # Breaking & Top News: 2x
        2: 3 / 4,  # Frontpage News: 1.3x
        1: 4 / 4,  # Latest News: 1x
        0: 5 / 4,  # Normal: 0.8x
    }[feed__importance]

    # age factor
    if feed.feed_type != "rss":  # videos
        factor_age = 10 / (1 + math.exp(-0.01 * article_age + 4)) + 1
    elif feed__ordering == "r":
        factor_age = 3 / (1 + math.exp(-0.25 * article_age + 4)) + 1
    else:  # d
        factor_age = 4 / (1 + math.exp(-0.25 * article_age + 4)) + 1

    article_relevance = round(
        feed_position
        * factor_publisher__renowned
        * factor_article_normalization
        * factor_feed__importance
        * factor_age
        + random_int,
        6,
    )
    article_relevance = min(float(article_relevance), 999999.0)

    return 9, float(article_relevance)


def delete_feed_positions(feed):
    """Deletes all feed positions of a respective feed"""
    all_feedpositions = feed.feedposition_set.all()
    all_feedpositions.delete()


@ratelimit.sleep_and_retry
@ratelimit.limits(calls=30, period=60)
def check_limit():
    """Empty function just to check for calls to API"""
    # limit of 90k tokens per minute = 90k / 3k per request = 30 requets
    # limit of 3500 requests per minute
    # min(3.5k, 30) = 30 requests per minute
    return


def add_ai_summary(article_obj_lst):
    """Use OpenAI's ChatGPT API to get artcile summaries"""
    print(f"Requesting AI article summaries for {len(article_obj_lst)} articles.")

    # openai.api_key = settings.OPENAI_API_KEY
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    TOTAL_API_COST = (
        0
        if cache.get("OPENAI_API_COST_LAUNCH") is None
        else cache.get("OPENAI_API_COST_LAUNCH")
    )
    COST_TOKEN_INPUT = 0.0015
    COST_TOKEN_OUTPUT = 0.002
    NET_USD_TO_GROSS_GBP = 1.2 * 0.785
    token_cost = 0
    articles_summarized = 0

    for article_obj in article_obj_lst:
        logging = [
            str(datetime.datetime.now().isoformat()),
            str(article_obj.publisher.name),
            str(article_obj.pk),
            str(article_obj.min_article_relevance),
            str(article_obj.title),
        ]
        try:
            soup = BeautifulSoup(article_obj.full_text, "html5lib")
            article_text = " ".join(html.unescape(soup.text).split())
            # article_text = re.sub(r'\n+', '\n', article_text).strip()
            if len(article_text) > 3000 * 5:
                article_text = article_text[: 3000 * 5]
            if len(article_text) / 5 < 500:
                continue
            elif len(article_text) / 5 < 1000:
                bullets = 2
            elif len(article_text) / 5 < 2000:
                bullets = 3
            else:
                bullets = 4
            check_limit()
            completion = client.chat.completions.create(
                # completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Summarize this article in {bullets} bullet"
                            f' points:\n"{article_text}"'
                        ),
                    }
                ],
            )
            article_summary = completion["choices"][0]["message"]["content"]
            article_summary = article_summary.replace("- ", "<li>").replace(
                "\n", "</li>\n"
            )
            article_summary = "<ul>\n" + article_summary + "</li>\n</ul>"
            token_cost += round(
                (completion["usage"]["prompt_tokens"] * COST_TOKEN_INPUT)
                + (completion["usage"]["completion_tokens"] * COST_TOKEN_OUTPUT),
                4,
            )
            setattr(article_obj, "ai_summary", article_summary)
            article_obj.save()
            articles_summarized += 1
            logging.extend(["SUCCESS", str(token_cost)])
        except Exception as e:
            print(f"Error getting AI article summary for {article_obj}:", e)
            logging.extend(["ERROR", str(0)])

        with open(str(settings.BASE_DIR) + "/data/ai_summaries.csv", "a+") as myfile:
            myfile.write(";".join(logging) + "\n")

    THIS_RUN_API_COST = round(float(token_cost / 1000 * NET_USD_TO_GROSS_GBP), 4)
    TOTAL_API_COST += THIS_RUN_API_COST
    cache.set("OPENAI_API_COST_LAUNCH", TOTAL_API_COST, 3600 * 1000)
    print(
        f"Summarized {articles_summarized} articles costing {THIS_RUN_API_COST} GBP."
        f" Total API cost since container launch {TOTAL_API_COST} GBP."
    )


def scarpe_img(url):
    """Manual image scraping from homepage - bascially searches for any image on homepage"""
    img_url = None
    try:
        _ = URLValidator(url)
        resp = requests.get(url)
        soup = BeautifulSoup(resp.content, "html5lib")
        body = soup.find("body")
        images = body.find_all("img")
        new_images = []
        for i in images:
            i_alt = ""
            i_class = ""
            i_src = ""
            try:
                i_src = str(i["src"]).lower()
                i_alt = str(i["alt"]).lower()
                i_class = str(i["class"]).lower()
            except Exception:
                pass
            if (
                len(i_src) > 3
                and any(
                    [
                        j in i_src
                        for j in [
                            ".avif",
                            ".gif",
                            ".jpg",
                            ".jpeg",
                            ".jfif",
                            ".pjpeg",
                            ".pjp",
                            ".png",
                            ".svg",
                            ".webp",
                        ]
                    ]
                )
                and "logo" not in i_class
                and "logo" not in i_alt
                and "author" not in i_class
                and "author" not in i_alt
            ):
                new_images.append(i)
        if len(new_images) > 0:
            images = new_images
            image = images[0]["src"]
            if "www." not in image and "http" not in image:
                url_parts = urlparse(url)
                image = url_parts.scheme + "://" + url_parts.hostname + image
            img_url = image

    except Exception as e:
        print(f'Error sccraping image for acticle from "{url}"')
        print(e)

    return img_url


class LinkGrabber:
    """linkpreview's LinkGrabber had bug with html content-type attribute - this custom class fixes the bug"""

    headers = {
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0)"
            " Gecko/20100101"
            " Firefox/95.0"
        ),
        "accept-language": "en-US,en;q=0.5",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def __init__(
        self,
        initial_timeout: int = 20,
        maxsize: int = 1048576,
        receive_timeout: int = 10,
        chunk_size: int = 1024,
    ):
        """
        :param initial_timeout in seconds
        :param maxsize in bytes (default 1048576 = 1 MB)
        :param receive_timeout in seconds
        :param chunk_size in bytes
        """
        self.initial_timeout = initial_timeout
        self.maxsize = maxsize
        self.receive_timeout = receive_timeout
        self.chunk_size = chunk_size

    def get_content(self, url: str, headers: dict = {}):
        r = requests.get(
            url,
            stream=True,
            timeout=self.initial_timeout,
            headers={**self.headers, **headers} if headers else self.headers,
        )
        r.raise_for_status()

        content_type = r.headers.get("content-type")
        if not content_type:
            raise InvalidContentError("Invalid content type")

        mime_type = content_type.split(";")[0].lower()
        if "text/html" not in mime_type:
            raise InvalidMimeTypeError("Invalid mime type")

        length = r.headers.get("Content-Length")
        if length and int(length) > self.maxsize:
            raise MaximumContentSizeError("response too large")

        size = 0
        start = time.time()
        content = b""
        for chunk in r.iter_content(self.chunk_size):
            if time.time() - start > self.receive_timeout:
                raise TimeoutError("timeout reached")

            size += len(chunk)
            if size > self.maxsize:
                raise MaximumContentSizeError("response too large")

            content += chunk

        return content, r.url


def scarpe_meta(url):
    """Scrape <meta> data from website"""
    try:
        while cache.get("metaScrapeWait") == "wait":
            print("meta scraping wait")
            time.sleep(2)
        grabber = LinkGrabber(
            initial_timeout=20,
            maxsize=1048576,
            receive_timeout=10,
            chunk_size=1024,
        )
        content, url = grabber.get_content(url)
        link = Link(url, content)
        preview = LinkPreview(link, parser="lxml")
        if hasattr(preview, "image") and preview.image is not None:
            if type(preview.image) is dict:
                img_url = preview.image["url"]
            else:
                img_url = preview.image
            if "www." not in img_url and "http" not in img_url:
                url_parts = urlparse(url)
                print(url_parts.scheme, "://", url_parts.hostname, img_url)
                preview.cust_image = (
                    url_parts.scheme + "://" + url_parts.hostname + img_url
                )
            else:
                preview.cust_image = img_url
            print(preview.cust_image)
        else:
            print("no image for", url)
        return preview
    except Exception as e:
        print("Error getting meta data:", e)
        traceback.print_exc()
        return None
    finally:
        cache.set("metaScrapeWait", "wait", 5)


def fetch_feed(feed):
    """Fetch/update/scrape all articles for a specific source feed"""
    added_articles = 0

    feed_url = feed.url
    if "http://FEED-CREATOR.local" in feed_url:
        feed_url = feed_url.replace(
            "http://FEED-CREATOR.local", settings.FEED_CREATOR_URL
        )
    if "http://FULL-TEXT.local" in feed_url:
        feed_url = feed_url.replace("http://FULL-TEXT.local", settings.FULL_TEXT)

    fetched_feed = feedparser.parse(feed_url)

    if len(fetched_feed.entries) > 0:
        delete_feed_positions(feed)

    for i, scraped_article in enumerate(fetched_feed.entries):
        hash_obj = hashlib.new("sha256")
        article__feed_position = i + 1
        article_kwargs = dict(
            min_feed_position=article__feed_position,
            publisher=feed.publisher,
            categories=(
                ""
                if feed.source_categories is None
                or len(feed.source_categories.split(";")) == 0
                else ";".join(
                    [str(i).upper() for i in feed.source_categories.split(";") + [""]]
                )
            ),
        )
        for kwarg_X, kwarg_Y in {
            "title": "title",
            "summary": "summary",
            "link": "link",
            "guid": "id",
            "pub_date": "published_parsed",
            "categories": "tags",
        }.items():
            if (
                hasattr(scraped_article, kwarg_Y)
                and scraped_article[kwarg_Y] is not None
                and scraped_article[kwarg_Y] != ""
            ):
                if (
                    kwarg_X in ["title", "summary"]
                    and scraped_article[kwarg_Y] is not None
                ):
                    bs_html = BeautifulSoup(scraped_article[kwarg_Y], "html.parser")
                    if bool(bs_html.find()):
                        article_kwargs[kwarg_X] = html.unescape(bs_html.get_text())
                    else:
                        article_kwargs[kwarg_X] = html.unescape(
                            scraped_article[kwarg_Y]
                        )
                elif (
                    kwarg_X in ["categories"]
                    and scraped_article[kwarg_Y] is not None
                    and len(scraped_article[kwarg_Y]) > 0
                ):
                    article_kwargs[kwarg_X] += (
                        ";".join(
                            [str(i["term"]).upper() for i in scraped_article[kwarg_Y]]
                        )
                        + ";"
                    )
                else:
                    article_kwargs[kwarg_X] = scraped_article[kwarg_Y]

        if (
            hasattr(scraped_article, "content")
            and len(scraped_article.content) > 0
            and hasattr(scraped_article.content[0], "value")
        ):
            article_kwargs["full_text"] = scraped_article.content[0].value

        # add unique id/hash
        if feed.publisher.unique_article_id == "guid" and "guid" in article_kwargs:
            article_kwargs["hash"] = f'{feed.publisher.pk}_{article_kwargs["guid"]}'
        elif feed.publisher.unique_article_id == "title":
            hash_obj.update(str(article_kwargs["title"]).encode())
            hash_str = hash_obj.hexdigest()
            article_kwargs["hash"] = f"{feed.publisher.pk}_{hash_str}"
        else:
            hash_obj.update(str(article_kwargs["link"]).encode())
            hash_str = hash_obj.hexdigest()
            article_kwargs["hash"] = f"{feed.publisher.pk}_{hash_str}"

        # make sure pub_date exists and is in the right format
        if (
            "pub_date" in article_kwargs
            and type(article_kwargs["pub_date"]) is time.struct_time
        ):
            article_kwargs["pub_date"] = datetime.datetime.fromtimestamp(
                time.mktime(article_kwargs["pub_date"])
            )
        elif "pub_date" not in article_kwargs and hasattr(
            fetched_feed, "published_parsed"
        ):
            article_kwargs["pub_date"] = datetime.datetime.fromtimestamp(
                fetched_feed["published_parsed"]
            )
        else:
            article_kwargs["pub_date"] = datetime.datetime.now()
        article_kwargs["pub_date"] = settings.TIME_ZONE_OBJ.localize(
            article_kwargs["pub_date"]
        )

        # check if artcile already exists
        search_article = Article.objects.filter(hash=article_kwargs["hash"])
        prev_article = None

        # if article exists but changed
        if (
            len(search_article) > 0
            and search_article[0].title is not None
            and "title" in article_kwargs
            and search_article[0].title != article_kwargs["title"]
        ):
            old_artcile = search_article[0]
            new_article = True
            hash_obj.update(str(old_artcile.title).encode())
            hash_str = hash_obj.hexdigest()
            setattr(old_artcile, "hash", f"{feed.publisher.pk}_{hash_str}")
            old_artcile.save()
            prev_article = old_artcile.pk

        # if article exists
        elif len(search_article) > 0:
            article_obj = search_article[0]
            new_article = False

        # article does not exist
        else:
            new_article = True

        # article does not exist yet
        if new_article:
            # get full text if settings say yes
            if feed.full_text_fetch == "Y":
                request_url = (
                    f'{settings.FULL_TEXT_URL}extract.php?url={urllib.parse.quote(article_kwargs["link"], safe="")}'
                )
                response = requests.get(request_url)
                if response.status_code == 200:
                    full_text_data = response.json()
                    if (
                        "news.google.com" in article_kwargs["link"]
                        and "effective_url" in full_text_data
                        and full_text_data["effective_url"] is not None
                    ):
                        request_url = (
                            f"{settings.FULL_TEXT_URL}extract.php?"
                            f"url={urllib.parse.quote(full_text_data['effective_url'], safe='')}"
                        )
                        response = requests.get(request_url)
                        if response.status_code == 200:
                            full_text_data = {**full_text_data, **response.json()}
                    for kwarg_X, kwarg_Y in {
                        "summary": "excerpt",
                        "author": "author",
                        "image_url": "og_image",
                        "full_text": "content",
                        "language": "language",
                    }.items():
                        if (
                            kwarg_Y in full_text_data
                            and full_text_data[kwarg_Y] is not None
                            and (
                                kwarg_X not in article_kwargs
                                or (
                                    kwarg_X == "language"
                                    and len(article_kwargs[kwarg_X]) < 2
                                    and len(full_text_data[kwarg_Y]) <= 5
                                )
                                or (
                                    kwarg_X == "author"
                                    and len(article_kwargs[kwarg_X]) < 5
                                    and len(full_text_data[kwarg_Y]) > 5
                                )
                                or (
                                    kwarg_X == "summary"
                                    and len(article_kwargs[kwarg_X]) < 70
                                    and len(full_text_data[kwarg_Y]) > 100
                                )
                                or (
                                    kwarg_X == "image_url"
                                    and len(article_kwargs[kwarg_X]) < 35
                                    and len(full_text_data[kwarg_Y]) > 35
                                )
                                or (
                                    kwarg_X == "full_text"
                                    and len(article_kwargs[kwarg_X]) * 1.75
                                    < len(full_text_data[kwarg_Y])
                                    and len(
                                        BeautifulSoup(
                                            article_kwargs[kwarg_X], "html5lib"
                                        ).text
                                    )
                                    * 1.75
                                    < len(
                                        BeautifulSoup(
                                            full_text_data[kwarg_Y], "html5lib"
                                        ).text
                                    )
                                )
                            )
                        ):
                            if kwarg_X in ["title", "summary", "author", "language"]:
                                bs_html = BeautifulSoup(
                                    full_text_data[kwarg_Y], "html.parser"
                                )
                                if bool(bs_html.find()):
                                    article_kwargs[kwarg_X] = html.unescape(
                                        bs_html.get_text()
                                    )
                                else:
                                    article_kwargs[kwarg_X] = html.unescape(
                                        full_text_data[kwarg_Y]
                                    )
                            else:
                                article_kwargs[kwarg_X] = full_text_data[kwarg_Y]

                    # if no image try scraping it differently
                    if (
                        "image_url" not in article_kwargs
                        or article_kwargs["image_url"] is None
                        or len(article_kwargs["image_url"]) < 10
                    ):
                        image_url = scarpe_img(url=article_kwargs["link"])
                        if image_url is not None:
                            article_kwargs["image_url"] = image_url
                            print(
                                "Successfully scrape image for article"
                                f" {feed.publisher.name}: {article_kwargs['title']}"
                            )
                        else:
                            print(
                                "Couldn't scrape image for article"
                                f" {feed.publisher.name}: {article_kwargs['title']}"
                            )

                else:
                    print(f"Full-Text fetch error response {response.status_code}")

            # clean up data
            if "full_text" in article_kwargs:
                soup = BeautifulSoup(article_kwargs["full_text"], "html.parser")
                for img in soup.find_all("img"):
                    img["style"] = (
                        "max-width: 100%; max-height: 80vh; width: auto; height: auto;"
                    )
                    if img["src"] == "src":
                        if "data-url" in img:
                            img["src"] = img["data-url"].replace("${formatId}", "906")
                        elif "data-src" in img:
                            img["src"] = img["data-src"]
                    img["referrerpolicy"] = "no-referrer"
                for a in soup.find_all("a"):
                    a["target"] = "_blank"
                    a["referrerpolicy"] = "no-referrer"
                for link in soup.find_all("link"):
                    if link is not None:
                        link.decompose()
                for meta in soup.find_all("meta"):
                    if meta is not None:
                        meta.decompose()
                for noscript in soup.find_all("noscript"):
                    if noscript is not None:
                        noscript.name = "div"
                for div_type, id in [
                    ("div", "barrierContent"),
                    ("div", "nousermsg"),
                    ("div", "trial_print_message"),
                    ("div", "print_blocked_message"),
                    ("div", "copy_blocked_message"),
                    ("button", "toolbar-item-parent-share-2909"),
                    ("ul", "toolbar-item-dropdown-share-2909"),
                ]:
                    div = soup.find(div_type, id=id)
                    if div is not None:
                        div.decompose()
                article_kwargs["full_text"] = soup.prettify()

                if prev_article is not None:
                    article_kwargs["full_text"] = (
                        '<a class="btn btn-outline-secondary my-2 ms-2" style="float:'
                        f' right;" href="/?article={prev_article}">Go to previous'
                        " article version</a>\n"
                        + article_kwargs["full_text"]
                    )

            # add additional properties
            if (
                "full_text" not in article_kwargs
                or len(article_kwargs["full_text"]) < 750
                or len(BeautifulSoup(article_kwargs["full_text"], "html5lib").text)
                < 750
                or (
                    "During your trial you will have complete digital access to FT.com"
                    " with everything in both of our Standard Digital and Premium"
                    " Digital packages."
                )
                in article_kwargs["full_text"]
            ):
                article_kwargs["has_full_text"] = False
            else:
                article_kwargs["has_full_text"] = True

            print(scraped_article.link)
            print(article_kwargs["link"])
            if "image_url" not in article_kwargs or article_kwargs["image_url"] is None:
                meta_data = scarpe_meta(url=article_kwargs["link"])
                if meta_data is not None:
                    for kwarg_X, kwarg_Y in {
                        "title": "title",
                        "summary": "description",
                        "image_url": "cust_image",
                    }.items():
                        if (
                            hasattr(meta_data, kwarg_Y)
                            and getattr(meta_data, kwarg_Y) is not None
                        ):
                            if (
                                kwarg_X not in article_kwargs
                                or article_kwargs[kwarg_X] is None
                                or article_kwargs[kwarg_X] == ""
                            ):
                                article_kwargs[kwarg_X] = getattr(meta_data, kwarg_Y)

            # check if breaking news
            if (
                "title" in article_kwargs
                and (
                    "liveblog" in article_kwargs["title"].lower()
                    or "breaking news" in article_kwargs["title"].lower()
                    or "live news" in article_kwargs["title"].lower()
                )
            ) or (
                "full_text" in article_kwargs
                and article_kwargs["full_text"] is not None
                and (
                    "livestream" in article_kwargs["full_text"].lower()
                    or "developing story" in article_kwargs["full_text"].lower()
                )
            ):
                article_kwargs["type"] = "breaking"
            else:
                article_kwargs["type"] = "normal"

            # check article language
            if "title" in article_kwargs and "summary" in article_kwargs:
                lang = langid.classify(
                    f'{article_kwargs["title"]}\n{article_kwargs["summary"]}'
                )
                article_kwargs["language"] = lang[0]

            # add article
            article_obj = Article(**article_kwargs)
            article_obj.save()
            added_articles += 1

            if prev_article is not None:
                full_text = (
                    "" if old_artcile.full_text is None else old_artcile.full_text
                )
                setattr(
                    old_artcile,
                    "full_text",
                    '<a class="btn btn-outline-danger cust-text-danger my-2 ms-2"'
                    f' style="float: right;" href="/?article={article_obj.pk}">Go to'
                    " updated article version</a>\n"
                    + full_text,
                )
                old_artcile.save()

        # Update article metrics
        (
            article_kwargs["max_importance"],
            article_kwargs["min_article_relevance"],
        ) = calcualte_relevance(
            publisher=feed.publisher,
            feed=feed,
            feed_position=article__feed_position,
            hash=article_kwargs["hash"],
            pub_date=article_kwargs["pub_date"],
        )
        for k, v in article_kwargs.items():
            value = getattr(article_obj, k)
            if value is None and v is not None:
                setattr(article_obj, k, v)
            elif "min" in k and v < value:
                setattr(article_obj, k, v)
            elif "max" in k and v > value:
                setattr(article_obj, k, v)
            elif (
                k == "categories"
                and article_kwargs["categories"] is not None
                and len(article_kwargs["categories"]) > 0
            ):
                for category in article_kwargs["categories"].split(";"):
                    if category.upper() not in v:
                        v += category.upper() + ";"
                setattr(article_obj, k, v)
        article_obj.save()

        # Add feed position linking
        feed_position = FeedPosition(
            feed=feed,
            position=article__feed_position,
            importance=article_kwargs["max_importance"],
            relevance=article_kwargs["min_article_relevance"],
        )
        feed_position.save()

        article_obj.feed_position.add(feed_position)

    print(
        f"Refreshed {feed} with {added_articles} new articles out of"
        f" {len(fetched_feed.entries)}"
    )
    return added_articles
