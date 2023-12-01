import asyncio
import random
from playwright.async_api import Playwright, async_playwright
from lxml import etree
from urllib.parse import unquote
import re
import os
import openai


class Google(object):
    def __init__(self):
        self.search_key = ''
        self.urls = []
        self.yzm = False
        self.mp3 = 0
        self.openai = openai
        self.openai.api_key = "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"

    def yzms(self):
        audio_file = open("12.mp3", "rb")
        transcript = self.openai.Audio.transcribe("whisper-1", audio_file)
        print(transcript.text)
        return transcript.text

    async def parser(self, response):
        html = etree.HTML(response)
        urls = html.xpath('//cite/text()')
        if urls:
            for url in urls:
                if url not in self.urls:
                    self.urls.append(url)
                    print(url)

    async def on_response(self, response):
        url = unquote(response.url, 'utf-8')
        if url.startswith(f'https://www.google.com/search?q={self.search_key}') and response.status == 200:
            try:
                html = await response.text()
                await self.parser(html)
            except Exception as e:
                pass
        elif "https://www.google.com/sorry/index?continue" in url:
            self.yzm = True
        elif 'https://www.google.com/recaptcha/api2/payload?p=' in url:
            self.mp3 += 1
            if self.mp3 > 1:
                with  open('12.mp3', 'wb') as f:
                    f.write(await response.body())

    async def search(self, page, key):
        self.search_key = key
        page.on('response', self.on_response)
        try:
            await page.goto(f'https://www.google.com/search?q={key}')
            await page.wait_for_load_state('networkidle')
        except Exception as e:
            print(f"Failed to load: {e}")
        while self.yzm:
            await page.wait_for_load_state('networkidle')
            for url in page.frames:
                box = re.findall("url='(https://www.google.com/recaptcha/api2/anchor?.*)'", str(url))
                if len(box) >= 1:
                    await page.frame(url=box[0]).get_by_role("checkbox", name="进行人机身份验证").click()
                    await page.wait_for_load_state('networkidle')
                    await page.wait_for_timeout(1000)
                buttons = re.findall("url='(https://www.google.com/recaptcha/api2/bframe?.*)'", str(url))
                if len(buttons) >= 1:
                    await page.frame(url=buttons[0]).click('//*[@id="recaptcha-audio-button"]')
                    await page.wait_for_load_state('networkidle')
                    await page.wait_for_timeout(1000)
                    await page.frame(url=buttons[0]).fill(selector='//*[@id="audio-response"]', value=self.yzms())
                    await page.frame(url=buttons[0]).click('//*[@id="recaptcha-verify-button"]')
                    await page.wait_for_load_state('networkidle')
                    await page.wait_for_timeout(5000)
            self.yzm = False
        print("开始翻页")
        delay = random.uniform(1, 1)
        await asyncio.sleep(delay)
        while self.yzm != True:
            try:
                await page.click('#pnnext', timeout=5000)
                await page.wait_for_load_state('networkidle')
            except Exception as e:
                break

    async def run(self, playwright: Playwright, key) -> None:

        browser = await playwright.chromium.launch(headless=False, args=["--no-sandbox"])
        context_options = {
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36",
            "viewport": {"width": 1440, "height": 900},
            "ignore_https_errors": True,
            "permissions": ["geolocation", "midi", "microphone", "camera"],
        }
        context = await browser.new_context(**context_options)
        page = await context.new_page()
        await context.add_init_script(path='stealth.min.js')

        await self.search(page, key)

        print(f"{len(self.urls)} URLs found:\n{sorted(self.urls)}")
        await browser.close()

    async def main(self, key) -> None:
        async with async_playwright() as playwright:
            await self.run(playwright, key)


if __name__ == '__main__':
    data = Google()
    asyncio.run(data.main('我好菜哦'))
