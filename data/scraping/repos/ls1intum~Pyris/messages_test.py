import pytest
from freezegun import freeze_time
from app.services.guidance_wrapper import GuidanceWrapper
import app.config as config


@pytest.fixture(scope="function")
def model_configs():
    llm_model_config = config.OpenAIConfig(
        type="openai",
        name="test",
        description="test",
        spec={"context_length": 100},
        llm_credentials={},
    )
    config.settings.pyris.llms = {"GPT35_TURBO": llm_model_config}
    api_key_config = config.APIKeyConfig(
        type="openai",
        token="secret",
        comment="test",
        spec={"context_length": 100},
        llm_access=["GPT35_TURBO"],
    )
    config.settings.pyris.api_keys = [api_key_config]


@freeze_time("2023-06-16 03:21:34 +02:00")
@pytest.mark.usefixtures("model_configs")
def test_send_message(test_client, headers, mocker):
    mocker.patch.object(
        GuidanceWrapper,
        "query",
        return_value={
            "response": "some content",
        },
        autospec=True,
    )

    body = {
        "template": {
            "id": 123,
            "content": "{{#user~}}I want a response to the following query:\
            {{query}}{{~/user}}{{#assistant~}}\
            {{gen 'response' temperature=0.0 max_tokens=500}}{{~/assistant}}",
        },
        "preferredModel": "GPT35_TURBO",
        "parameters": {
            "course": "Intro to Java",
            "exercise": "Fun With Sets",
            "query": "Some query",
        },
    }
    response_v1 = test_client.post(
        "/api/v1/messages", headers=headers, json=body
    )
    assert response_v1.status_code == 200
    assert response_v1.json() == {
        "usedModel": "GPT35_TURBO",
        "message": {
            "sentAt": "2023-06-16T01:21:34+00:00",
            "content": [{"textContent": "some content", "type": "text"}],
        },
    }


@freeze_time("2023-06-16 03:21:34 +02:00")
@pytest.mark.usefixtures("model_configs")
def test_send_message_v2(test_client, headers, mocker):
    mocker.patch.object(
        GuidanceWrapper,
        "query",
        return_value={
            "response": "some content",
        },
        autospec=True,
    )

    body = {
        "template": "{{#user~}}I want a response to the following query:\
            {{query}}{{~/user}}{{#assistant~}}\
            {{gen 'response' temperature=0.0 max_tokens=500}}{{~/assistant}}",
        "preferredModel": "GPT35_TURBO",
        "parameters": {
            "course": "Intro to Java",
            "exercise": "Fun With Sets",
            "query": "Some query",
        },
    }

    response_v2 = test_client.post(
        "/api/v2/messages", headers=headers, json=body
    )
    assert response_v2.status_code == 200
    assert response_v2.json() == {
        "usedModel": "GPT35_TURBO",
        "sentAt": "2023-06-16T01:21:34+00:00",
        "content": {
            "response": "some content",
        },
    }


def test_send_message_missing_model(test_client, headers):
    response = test_client.post("/api/v1/messages", headers=headers, json={})
    assert response.status_code == 404


def test_send_message_missing_params(test_client, headers):
    response = test_client.post(
        "/api/v1/messages",
        headers=headers,
        json={"preferredModel": "GPT35_TURBO"},
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "loc": ["body", "template"],
                "msg": "field required",
                "type": "value_error.missing",
            },
            {
                "loc": ["body", "parameters"],
                "msg": "field required",
                "type": "value_error.missing",
            },
        ]
    }


@pytest.mark.usefixtures("model_configs")
def test_send_message_raise_value_error(test_client, headers, mocker):
    mocker.patch.object(
        GuidanceWrapper, "query", side_effect=ValueError("value error message")
    )
    body = {
        "template": {
            "id": 123,
            "content": "some template",
        },
        "preferredModel": "GPT35_TURBO",
        "parameters": {"query": "Some query"},
    }
    response = test_client.post("/api/v1/messages", headers=headers, json=body)
    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "type": "other",
            "errorMessage": "value error message",
        }
    }


@pytest.mark.usefixtures("model_configs")
def test_send_message_raise_key_error(test_client, headers, mocker):
    mocker.patch.object(
        GuidanceWrapper, "query", side_effect=KeyError("key error message")
    )
    body = {
        "template": {
            "id": 123,
            "content": "some template",
        },
        "preferredModel": "GPT35_TURBO",
        "parameters": {"query": "Some query"},
    }
    response = test_client.post("/api/v1/messages", headers=headers, json=body)
    assert response.status_code == 400
    assert response.json() == {
        "detail": {
            "type": "missing_parameter",
            "errorMessage": "'key error message'",
        }
    }


def test_send_message_with_wrong_api_key(test_client):
    headers = {"Authorization": "wrong api key"}
    response = test_client.post("/api/v1/messages", headers=headers, json={})
    assert response.status_code == 403
    assert response.json()["detail"] == {
        "type": "not_authorized",
        "errorMessage": "Permission denied",
    }


def test_send_message_without_authorization_header(test_client):
    response = test_client.post("/api/v1/messages", json={})
    assert response.status_code == 401
    assert response.json()["detail"] == {
        "type": "not_authenticated",
        "errorMessage": "Requires authentication",
    }


@pytest.mark.usefixtures("model_configs")
def test_send_message_fail_three_times(
    test_client, mocker, headers, test_cache_store
):
    mocker.patch.object(
        GuidanceWrapper, "query", side_effect=Exception("some error")
    )
    body = {
        "template": {
            "id": 123,
            "content": "some template",
        },
        "preferredModel": "GPT35_TURBO",
        "parameters": {"query": "Some query"},
    }

    for _ in range(3):
        try:
            test_client.post("/api/v1/messages", headers=headers, json=body)
        except Exception:
            ...

    # Restrict access
    response = test_client.post("/api/v1/messages", headers=headers, json=body)
    assert test_cache_store.get("GPT35_TURBO:status") == "OPEN"
    assert test_cache_store.get("GPT35_TURBO:num_failures") == 3
    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "errorMessage": "Too many failures! Please try again later.",
            "type": "other",
        }
    }

    # Can access after TTL
    test_cache_store.delete("GPT35_TURBO:status")
    test_cache_store.delete("GPT35_TURBO:num_failures")
    response = test_client.post("/api/v1/messages", headers=headers, json=body)
    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "errorMessage": "some error",
            "type": "other",
        }
    }
