import datetime
import json
import os
import time

import httpx
import openai
import requests

import utilities as utils

config = utils.read_config()


class CustomHTTPTransport(httpx.HTTPTransport):
    def handle_request(
            self,
            request: httpx.Request,
    ) -> httpx.Response:
        if "images/generations" in request.url.path and request.url.params[
            "api-version"
        ] in [
            "2023-06-01-preview",
            "2023-07-01-preview",
            "2023-08-01-preview",
            "2023-09-01-preview",
            "2023-10-01-preview",
        ]:
            request.url = request.url.copy_with(path="/openai/images/generations:submit")
            response = super().handle_request(request)
            operation_location_url = response.headers["operation-location"]
            request.url = httpx.URL(operation_location_url)
            request.method = "GET"
            response = super().handle_request(request)
            response.read()

            timeout_secs: int = 120
            start_time = time.time()
            while response.json()["status"] not in ["succeeded", "failed"]:
                if time.time() - start_time > timeout_secs:
                    timeout = {
                        "error": {"code": "Timeout", "message": "Operation polling timed out."}}
                    return httpx.Response(
                        status_code=400,
                        headers=response.headers,
                        content=json.dumps(timeout).encode("utf-8"),
                        request=request,
                    )

                time.sleep(int(response.headers.get("retry-after")) or 10)
                response = super().handle_request(request)
                response.read()

            if response.json()["status"] == "failed":
                error_data = response.json()
                return httpx.Response(
                    status_code=400,
                    headers=response.headers,
                    content=json.dumps(error_data).encode("utf-8"),
                    request=request,
                )

            result = response.json()["result"]
            return httpx.Response(
                status_code=200,
                headers=response.headers,
                content=json.dumps(result).encode("utf-8"),
                request=request,
            )
        return super().handle_request(request)


client = openai.AzureOpenAI(
    azure_endpoint=config['aoai_endpoint'],
    api_key=config['aoai_key'],
    api_version='2023-10-01-preview',
    http_client=httpx.Client(
        transport=CustomHTTPTransport(),
    ),
)


def generate_image_with_text(text):
    generated_image = client.images.generate(prompt=text)
    image_url = generated_image.data[0].url  # extract image URL from response

    path = f'./downloads'
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = \
        f"{path}/{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
    results = requests.get(image_url).content  # download the image
    with open(file_path, "wb") as image_file:
        image_file.write(results)
    response = {'image_url': image_url, 'file_path': file_path}
    return response
