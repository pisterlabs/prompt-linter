from django.test import TestCase

from scrapy.http import HtmlResponse

from openaiapp.spiders import NewsSpider


class CrawlerTestCase(TestCase):
    def test_should_raise_unsupported_error_for_parsing_not_html(self):
        """
        Test that the spider raises an UnsupportedError when attempting to parse a non-HTML response.
        """
        raise NotImplementedError

    def test_should_spider_extract_from_html_response_all_text(self):
        """
        Test whether the spider correctly extracts all text from an HTML response.
        """
        domain = "example.com"
        url = f"http://www.{domain}"
        body = "<html><body><p>This is some sample text.</p></body></html>"
        headers = {"Content-Type": "text/html"}

        response = HtmlResponse(
            url=url, body=body.encode("utf-8"), headers=headers, encoding="utf-8"
        )
        spider = NewsSpider(domain=domain, start_urls=[url])

        spider.parse(response)

        expected_articles = [{"url": url, "text": "This is some sample text."}]
        self.assertEqual(expected_articles, spider.articles)

    def test_should_crawl_website_and_return_all_text(self):
        """
        Test that the spider crawls the website and linked HTML documents within the same domain, returning all text.
        """
        domain = "example.com"
        url = f"http://{domain}"
        link1 = f"http://{domain}/page1.html"
        link2 = f"http://{domain}/page2.html"
        link3 = "/page3.html"
        sharp_link = "#"
        mailto_link = "mailto:johnbrawo1231@gg123mail.com"
        js_link = "/file.js"
        other_link = "http://www.other.com"

        spider = NewsSpider(domain=domain, start_urls=[url])

        url_bodies = [
            (
                url,
                "<html><body><p>This is some sample text.</p></body></html>",
                {"Content-Type": "text/html"},
            ),
            (
                link1,
                "<html><body><p>This is some sample text.</p></body></html>",
                {"Content-Type": "text/html"},
            ),
            (
                link2,
                "<html><body><p>This is some sample text.</p></body></html>",
                {"Content-Type": "text/html"},
            ),
            (
                link3,
                "<html><body><p>This is some sample text.</p></body></html>",
                {"Content-Type": "text/html"},
            ),
            (
                sharp_link,
                "<html><body><p>[BAD-#]This is some sample text.</p></body></html>",
                {"Content-Type": "text/html"},
            ),
            (
                mailto_link,
                "<html><body><p>[BAD-mailto]This is some sample text.</p></body></html>",
                {"Content-Type": "text/html"},
            ),
            (js_link, "/*[BAD-js]*/", {"Content-Type": "file/js"}),
            (
                other_link,
                "<html><body><p>[BAD-other]This is some sample text.</p></body></html>",
                {"Content-Type": "text/html"},
            ),
        ]

        responses = [
            HtmlResponse(
                url=url, body=body.encode("utf-8"), headers=headers, encoding="utf-8"
            )
            for url, body, headers in url_bodies
        ]

        for response in responses:
            spider.parse(response)

        expected_articles = [
            {"url": url, "text": "This is some sample text."},
            {"url": link1, "text": "This is some sample text."},
            {"url": link2, "text": "This is some sample text."},
            {"url": link3, "text": "This is some sample text."},
        ]

        self.assertEqual(expected_articles, spider.articles)
