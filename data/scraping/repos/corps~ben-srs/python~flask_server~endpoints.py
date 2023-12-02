from __future__ import annotations

import abc
import json
import os.path
import uuid
import warnings
from typing import (
    TypeVar,
    Generic,
    Type,
    Any,
    Callable,
    Optional,
    Tuple,
    cast,
)

import openai as openai
import requests
import vcr  # type: ignore
from dropbox import Dropbox  # type: ignore
from dropbox.exceptions import HttpError  # type: ignore
from dropbox.files import (  # type: ignore
    ListFolderResult,
    FileMetadata as DbFileMetadata,
    DeletedMetadata,
    WriteMode,
)
from dropbox.oauth import OAuth2FlowResult  # type: ignore
from dropbox.users import FullAccount  # type: ignore
from flask import Response, request, make_response, redirect, session, send_file
from flask.testing import FlaskClient
from openai.openai_object import OpenAIObject
from pydantic import ValidationError, BaseModel
from werkzeug.exceptions import Unauthorized, TooManyRequests

from .app import app
from .conftest import (
    write_to_cache,
    get_value_from_cache_or_browser,
)
from .datasource import User

T = TypeVar("T")


class JsonEndpoint(BaseModel, Generic[T], abc.ABC):
    @abc.abstractmethod
    def handle(self, res: T) -> Response | None:
        pass


def user_dropbox() -> Tuple[User, Dropbox]:
    assert app.store
    if user_id := session.get("user_id"):
        if user := app.store.get(User, user_id):
            return user, Dropbox(
                app_secret=app.app_secret,
                app_key=app.app_key,
                oauth2_refresh_token=user.refresh_token,
            )
        warnings.warn(f"User id {user_id} in session had no user associated with it.")
        session.pop("user_id")

    raise Unauthorized()


JET = TypeVar("JET", bound=JsonEndpoint[Any])


_bad_json_response = json.dumps(dict(error="Json body needs to be a dictionary!"))


def as_json_handler(
    route: str,
    resp_factory: Callable[[], T],
    **options: Any,
) -> Callable[[Type[JET]], Type[JET]]:
    def wrapper(endpoint_cls: Type[JET]) -> Type[JET]:
        def handler():
            raw: Any = request.get_json(force=True)
            if not isinstance(raw, dict):
                return make_response((_bad_json_response, 400))

            try:
                json_endpoint = endpoint_cls.parse_obj(raw)
            except ValidationError as e:
                return make_response((e.json(), 400))

            json_resp = resp_factory()
            try:
                flask_resp = json_endpoint.handle(json_resp)
            except HttpError as e:
                if e.status_code == 429:
                    raise TooManyRequests()
                if e.status_code == 401:
                    session.pop("user_id")
                    raise Unauthorized()
                raise e

            if flask_resp is not None:
                return flask_resp
            if isinstance(json_resp, BaseModel):
                return make_response(json_resp.dict())
            return make_response(json_resp)

        handler.__name__ = endpoint_cls.__name__

        app.route(route, **options)(handler)
        return endpoint_cls

    return wrapper


def test_start():
    get_value_from_cache_or_browser("authorization-code", app.start_url)


@app.route("/")
def index_endpoint():
    return send_file(os.path.join(app.root_path, app.static_folder or "", "index.html"))


@app.route("/start")
def start_endpoint():
    session["csrf"] = uuid.uuid4().hex
    oauth_flow = app.oauth_flow(request, cast(dict, session), session["csrf"])
    return redirect(oauth_flow.start())


@app.route("/start_ext")
def start_ext_endpoint():
    return redirect(app.start_url)


class OauthTokenResponse(BaseModel):
    access_token: str
    account_id: str
    expires_in: int
    refresh_token: str
    token_type: str
    uid: str


@vcr.use_cassette(
    os.path.join(app.root_path, ".test_cache/login.yaml"),
    record_mode="once",
)
def test_login_endpoint(client: FlaskClient, authorization_code) -> None:
    assert app.store
    response = client.post("/login", json=Login())
    assert response.status_code == 401

    assert authorization_code
    response = client.post("/login", json=Login(authorization_code=authorization_code))

    json_response = response.get_json(force=True)
    assert json_response is not None
    assert json_response["success"]
    assert json_response["email"]
    with client.session_transaction() as session:
        assert session["user_id"] == 1
    assert (user := app.store.get(User, 1))
    assert json_response["access_token"]
    assert json_response["app_key"] == app.app_key

    response = client.post("/login", json=Login())
    json_response = response.get_json(force=True)
    assert json_response["success"]
    assert json_response["email"]
    assert json_response["access_token"]


class LoginResponse(BaseModel):
    success: bool = True
    email: str = ""
    access_token: str = ""
    app_key: str = ""


@as_json_handler("/login", lambda: LoginResponse(), methods=["POST"])
class Login(JsonEndpoint[LoginResponse]):
    authorization_code: Optional[str] = None

    def handle(self, resp: LoginResponse) -> Response | None:
        user: User | None = None
        try:
            user, _ = user_dropbox()
        except Unauthorized:
            pass

        db: Dropbox
        if user is None or self.authorization_code:
            if not self.authorization_code:
                raise Unauthorized()
            refresh_token = self.get_refresh_token()
            account, db = self.login(refresh_token)
        else:
            account, db = self.login(user.refresh_token)

        resp.email = account.email
        resp.access_token = db._oauth2_access_token
        resp.app_key = app.app_key
        return None

    def get_refresh_token(self) -> str:
        response = requests.post(
            "https://api.dropboxapi.com/oauth2/token",
            data=dict(
                code=self.authorization_code,
                grant_type="authorization_code",
            ),
            auth=(app.app_key, app.app_secret),
        )

        try:
            refresh_token = OauthTokenResponse.parse_obj(response.json()).refresh_token
        except ValidationError:
            print(response.json())
            raise Unauthorized()
        write_to_cache("refresh_token", refresh_token)
        return refresh_token

    def login(self, refresh_token: str) -> Tuple[FullAccount, Dropbox]:
        db = Dropbox(
            oauth2_refresh_token=refresh_token,
            app_key=app.app_key,
            app_secret=app.app_secret,
        )

        account: FullAccount = db.users_get_current_account()

        assert app.store
        with app.store.connection:
            user_id = app.store.upsert(
                User(
                    account_id=account.account_id,
                    email=account.email,
                    refresh_token=refresh_token,
                )
            )

        session["user_id"] = user_id
        session.permanent = True
        return account, db


class CompletionResponse(BaseModel):
    response: str = ""


@as_json_handler("/completion", lambda: CompletionResponse(), methods=["POST"])
class Completion(JsonEndpoint[CompletionResponse]):
    prompt: str = ""

    def handle(self, res: CompletionResponse) -> Response | None:
        user, _ = user_dropbox()
        openai.api_key = app.openapi_key
        response: OpenAIObject = openai.Completion.create(
            model="text-davinci-003",
            prompt=self.prompt,
            temperature=0.6,
            max_tokens=350,
            n=1,
        )

        res.response = response.choices[0].text
        return None


@vcr.use_cassette(
    os.path.join(app.root_path, ".test_cache/completion.yaml"),
    record_mode="once",
)
def test_completion_endpoint(client: FlaskClient, logged_in_user: User) -> None:
    response = client.post(
        "/completion", json=Completion(prompt="Just say 'moo'\nback to me.")
    )
    json_response = response.get_json(force=True)
    completion = CompletionResponse.parse_obj(json_response)
    assert "moo" in completion.response.lower()


@app.route("/authorize")
def authorize():
    if "csrf" in session:
        oauth_flow = app.oauth_flow(request, cast(dict, session), session["csrf"])
        result: OAuth2FlowResult = oauth_flow.finish(request.args)
        Login().login(result.refresh_token)
    return redirect("/")
