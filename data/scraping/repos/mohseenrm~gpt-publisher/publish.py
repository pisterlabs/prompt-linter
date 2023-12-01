import pendulum
import datetime
import random
import re
import requests

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.operators.bash import BashOperator

from sendgrid.helpers.mail import Mail
from sendgrid import SendGridAPIClient

import openai

from gpt_publisher.constants import (
    GPT_TOPICS,
    DATE_TIME_REGEX,
    TAGS_REGEX,
    TITLE_REGEX,
    WORD_REGEX,
    PREVIEW_REGEX,
    DESKTOP_REGEX,
    TABLET_REGEX,
    MOBILE_REGEX,
    FALLBACK_REGEX,
    UNSPLASH_BASE_URL,
    gpt_prompt,
)

with DAG(
    dag_id="publish_blog_post",
    # Once every week on Friday at 10:00 AM UTC
    schedule="0 10 * * 5",
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=60),
    tags=["gpt-publisher"],
    render_template_as_native_obj=True,
) as dag:

    def get_clone_link() -> str:
        token = Variable.get("GITHUB_TOKEN")
        username = Variable.get("GITHUB_USER")
        repo = Variable.get("GITHUB_REPOSITORY")
        clone_link = f"https://{username}:{token}@github.com/{repo}"
        return clone_link

    def get_website_repo_link() -> str:
        token = Variable.get("GITHUB_TOKEN")
        username = Variable.get("GITHUB_USER")
        repo = Variable.get("GITHUB_WEBSITE_REPOSITORY")
        clone_link = f"https://{username}:{token}@github.com/{repo}"
        return clone_link

    @task(task_id="pick_topic")
    def pick_topic():
        length = len(GPT_TOPICS)
        index = random.randint(0, length - 1)
        return GPT_TOPICS[index]

    @task(task_id="call_gpt")
    def call_gpt(ti=None):
        """
        Calls ChatGPT 4 to generate a blog post
        """
        theme = ti.xcom_pull(task_ids="pick_topic")
        assert theme is not None or theme is not ""
        print(f"Theme: {theme}")
        prompt = gpt_prompt(theme)
        openai_api_key = Variable.get("OPENAI_API_KEY")
        assert openai_api_key is not None or openai_api_key is not ""

        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}]
        )
        print(f"response: {response}")
        raw_text = response["choices"][0]["message"]["content"]
        # raw_text = "---\ntitle: \"Beyond Basics: Delving into the Depths of Modern Frontend Development\"\ndate: 2022-11-30\nhero:\n  preview: images/blog-preview.jpg\n  desktop: images/blog-desktop.jpg\n  tablet: images/blog-tablet.jpg\n  mobile: images/blog-mobile.jpg\n  fallback: images/blog-fallback.jpg\ntags: frontend_development, JavaScript, VueJs, ReactJs, AngularJs, NuxtJs, NextJs, PWA, WebPack, Babel\nexcerpt: \"In this blog, we will navigate the uncharted seas of advanced frontend development techniques, working with popular libraries and frameworks and working around common pitfalls and bottlenecks. Brace yourself for a journey to the \u2018beyond\u2019 of frontend development.\"\ntimeToRead: 10\nauthors:\n  - Mohseen Mukaddam\n---\n\nThe world of frontend development has taken enormous strides  in the recent years, from baking simple web pages with HTML and CSS to constructing complex single-page applications (SPAs) with modern libraries and frameworks.\n\nUnderstanding these modern technologies not only segregates ordinary frontend developers from the truly proficient ones, but it also enables one\u2019s ability to build more efficient, future-proof and maintainable applications.\n\nLet\u2019s dive in.\n\n## React, Angular, Vue - Choose Your Shield\n\nIn the battlefront of frontend development, you need a trustworthy shield. The most popular choices are React, Angular, and Vue. All three provide advanced features out of the box. However, choosing the one right for you depends on your use case. Understanding their strengths, weaknesses, and ideal use-cases is crucial.\n\n> \u201cAny application that can be written in JavaScript, will eventually be written in JavaScript.\u201d \u2013 Jeff Atwood.\n\nReact\u2019s strength lies in its flexibility and immense community support. It is backed by Facebook and its one-direction data flow is renowned. Here is a small sample of a React component:\n\n```javascript\nimport React from 'react';\n\nconst HelloWorld = () => {\n  return <p>Hello, world!</p>\n}\n\nexport default HelloWorld;\n```\nThis is arguably the most simple a React component can get. But the true power of React is realised when you need to manage state, handle user inputs and compose components. You can find many examples and resources [here](https://github.com/facebook/react).\n\nAngular, maintained by Google, is a full-fledged MVC framework rather than a lib like React or Vue, and provides much more straight out of the box. It automates two-way data binding, dependency injection and more, making it the right tool for larger scale applications.\n\nVue, on the other hand, provides the best of both Angular and React, with a lighter footprint. Here is a simple Vue component:\n\n```vue\n<template>\n  <p>{{ greeting }}</p>\n</template>\n\n<script>\nexport default {\n  data() {\n    return {\n      greeting: 'Hello, world!'\n    }\n  }\n}\n</script>\n```\nCompared to our previous React example, you can see that Vue separates the template (HTML) from the logic (JavaScript).\n\n## Building Performance-First Applications with NuxtJS and NextJS\n\nNext.js and Nuxt.js are powerful frameworks based on React and Vue respectively. These frameworks ensure better performance with features like auto code-splitting, pre-fetching, automatic routing and static site generation.\n\nIt\u2019s notable to mention the pitfalls when constructing SPAs - SEO. Often overlooked, SEO can hit you hard in the long-run as it scarifies the discoverability of your website. However, NuxtJS and NextJS are very SEO friendly.\n\nIt's vital to \"reduce, reduce, reduce. Then add lightness.\" as said by Colin Chapman. Your website needs to lose the extra pounds and pick up some speed.\n\n## WebPack and Babel\n\nJavaScript has come a long way since its inception. But with each version, the gap between modern JS (ES6/7/8) and browser-supported JS increases. Babel helps bridge this gap, converting modern JS into browser compatible JS. It's essential to any advanced frontend developer\u2019s toolkit. \n\nWebpack, on the other hand, is a module bundler. Simply put, it takes a bunch of modules, applies a series of transformations, and gives out a single (or multiple - code splitting) bundled file, optimised and ready for the browser.\n\n## Progressive Web Apps (PWAs)\n\nPWAs are the talk of the web-development town, combining the best of web and native apps. They can load when offline, receive push notifications, and even be added to a home screen. \n\nBeware, though. Although promising, PWAs are not perfect. They come with their own bag of issues including compatibility issues with iOS and the dread of managing a service worker.\n\n## Conclusion\n\nThe journey of traversing the 'beyond' of frontend development is bumpy. The terrain is hostile and pitfalls are common. However, the treasure that awaits is worth it and the view is amazing. Happy developing!\n\nRemember these wise words from Paul Graham, \"The web is the medium that rewards giving.\"\n\n---\n\nThis post barely scratches the surface of modern frontend development techniques. However, don't let that daint you. Keep exploring, keep learning; The journey is what matters after all."
        assert raw_text is not None or raw_text is not ""
        return raw_text

    @task(task_id="process_blog_post")
    def process_blog_post(ti=None):
        """
        Processes the blog post, sanitizes date and tags
        """
        post = ti.xcom_pull(task_ids="call_gpt")
        assert post is not None or post is not ""

        date = datetime.datetime.now().isoformat().split("T")[0]
        post = re.sub(DATE_TIME_REGEX, f"date: {date}", post)

        tags = re.findall(TAGS_REGEX, post)
        tags = tags[0].split(",")
        tags = [tag.strip() for tag in tags]
        tags = [
            tag.replace("_", " ").replace("-", " ").replace('"', "").replace("'", "")
            for tag in tags
        ]

        raw_title = re.findall(TITLE_REGEX, post)
        assert raw_title and len(raw_title) > 0

        raw_title = raw_title[0].strip()
        title_words = re.findall(WORD_REGEX, raw_title)

        assert title_words and len(title_words) > 0
        title_words = [word.lower() for word in title_words]
        title_words = "-".join(title_words)
        file_name = f"{date}-{title_words}.md"

        return {
            "post": post,
            "tags": tags,
            "date": date,
            "file_name": file_name,
            "title": title_words,
        }

    @task(task_id="fetch_images")
    def fetch_images(ti=None):
        context = ti.xcom_pull(task_ids="process_blog_post")
        tags = context["tags"][:2]
        query = ", ".join(tags)

        token = Variable.get("UNSPLASH_TOKEN")
        assert token is not None or token is not ""

        headers = {
            "Authorization": f"Client-ID {token}",
        }
        url = f"{UNSPLASH_BASE_URL}/search/photos"
        params = {
            "query": query,
        }
        response = requests.get(url, headers=headers, params=params)
        # print(f"response: {response}")
        # print(f"response.json(): {response.json()}")
        json = response.json()
        results = json["results"]

        assert results and len(results) > 0

        idx = random.randint(0, len(results) - 1)
        image_urls = results[idx]["urls"]

        preview = image_urls["small"]
        desktop = image_urls["full"]
        tablet = image_urls["regular"]
        mobile = image_urls["small"]
        fallback = image_urls["thumb"]

        return {
            **context,
            "preview": preview,
            "desktop": desktop,
            "tablet": tablet,
            "mobile": mobile,
            "fallback": fallback,
        }

    @task(task_id="process_images")
    def process_images(ti=None):
        context = ti.xcom_pull(task_ids="fetch_images")

        post = context["post"]
        title = context["title"]

        post = re.sub(PREVIEW_REGEX, f" /images/hero/{title}.preview.jpg", post)
        post = re.sub(DESKTOP_REGEX, f" /images/hero/{title}.desktop.jpg", post)
        post = re.sub(TABLET_REGEX, f" /images/hero/{title}.tablet.jpg", post)
        post = re.sub(MOBILE_REGEX, f" /images/hero/{title}.mobile.jpg", post)
        post = re.sub(FALLBACK_REGEX, f" /images/hero/{title}.fallback.jpg", post)

        # Weird hack for jinja2 template rendering, that strips out newlines
        post = post.replace("\n", "<NEW_LINE_TOKEN>")

        return {
            **context,
            "post": post,
        }

    @task(task_id="send_success_email")
    def send_success_email(ti=None):
        context = ti.xcom_pull(task_ids="publish_blog_post")
        content = context.split("@")
        assert len(content) == 2
        url, title = content[0], content[1]

        def capitalize_words(s):
            return re.sub(r"\w+", lambda m: m.group(0).capitalize(), s)

        title = capitalize_words(title).replace("-", " ")

        message = Mail(
            from_email="gpt-publisher@mohseen.dev",
            to_emails=["mohseenmukaddam6@gmail.com"],
        )
        message.dynamic_template_data = {
            "title": title,
            "blog_post_url": url,
        }
        message.template_id = Variable.get("SENDGRID_SUCCESS_TEMPLATE")

        try:
            sg = SendGridAPIClient(Variable.get("SENDGRID_API_KEY"))
            response = sg.send(message)
            code, body, headers = response.status_code, response.body, response.headers
            print(f"Response code: {code}")
            print(f"Response headers: {headers}")
            print(f"Response body: {body}")
            print("Success Message Sent!")
            return str(response.status_code)
        except Exception as e:
            print("Error: {0}".format(e))
            raise e

    @task(task_id="send_failure_email", trigger_rule="one_failed")
    def send_failure_email(ti=None):
        message = Mail(
            from_email="gpt-publisher@mohseen.dev",
            to_emails=["mohseenmukaddam6@gmail.com"],
        )
        message.template_id = Variable.get("SENDGRID_FAILURE_TEMPLATE")

        try:
            sg = SendGridAPIClient(Variable.get("SENDGRID_API_KEY"))
            response = sg.send(message)
            code, body, headers = response.status_code, response.body, response.headers
            print(f"Response code: {code}")
            print(f"Response headers: {headers}")
            print(f"Response body: {body}")
            print("Failure Message Sent!")
            return str(response.status_code)
        except Exception as e:
            print("Error: {0}".format(e))
            raise e

    clone_url = get_clone_link()
    website_repo_url = get_website_repo_link()
    pick_topic = pick_topic()
    call_gpt = call_gpt()
    process_blog_post = process_blog_post()
    fetch_images = fetch_images()
    process_images = process_images()
    send_success_email = send_success_email()
    send_failure_email = send_failure_email()

    run = BashOperator(
        task_id="publish_blog_post",
        bash_command="/opt/airflow/dags/scripts/publish.sh ",
        env={
            "GITHUB_WEBSITE_REPOSITORY": Variable.get("GITHUB_WEBSITE_REPOSITORY"),
            "GITHUB_REPOSITORY": Variable.get("GITHUB_REPOSITORY"),
            "CLONE_URL": clone_url,
            "CLONE_WEBSITE_REPO_URL": website_repo_url,
            "BLOG_FILENAME": "{{ ti.xcom_pull(task_ids='process_images')['file_name'] }}",
            "BLOG_CONTENT": "{{ ti.xcom_pull(task_ids='process_images')['post'] }}",
            "PREVIEW_URL": "{{ ti.xcom_pull(task_ids='process_images')['preview'] }}",
            "DESKTOP_URL": "{{ ti.xcom_pull(task_ids='process_images')['desktop'] }}",
            "TABLET_URL": "{{ ti.xcom_pull(task_ids='process_images')['tablet'] }}",
            "MOBILE_URL": "{{ ti.xcom_pull(task_ids='process_images')['mobile'] }}",
            "FALLBACK_URL": "{{ ti.xcom_pull(task_ids='process_images')['fallback'] }}",
            "BLOG_TITLE": "{{ ti.xcom_pull(task_ids='process_images')['title'] }}",
        },
        do_xcom_push=True,
    )

    end = BashOperator(
        task_id="end",
        bash_command='echo "Shutting down!"',
    )

    send_failure_email >> end

    # Setup fallback for failure
    pick_topic >> [call_gpt, send_failure_email]
    call_gpt >> [process_blog_post, send_failure_email]
    process_blog_post >> [fetch_images, send_failure_email]
    fetch_images >> [process_images, send_failure_email]
    process_images >> [run, send_failure_email]
    run >> [send_success_email, send_failure_email]
    send_success_email >> end


if __name__ == "__main__":
    dag.test()
