import json
from typing import Any, Dict

from autonomous import log
from autonomous.ai import OpenAI
from autonomous.model.automodel import AutoModel
from autonomous.storage.cloudinarystorage import CloudinaryStorage
from autonomous.storage.markdown import Page
from slugify import slugify


class TTRPGObject(AutoModel):
    _storage = CloudinaryStorage()
    _wiki_api = Page

    name: str = ""
    image_data: dict = None
    backstory: str = ""
    bs_summary: str = ""
    desc: str = ""
    dod: str = ""
    dob: str = ""
    traits: list = []
    world: Any = None
    wiki_id: str = ""
    wiki_path: str = ""
    notes: list[str] = ["TBD"]

    def __getattr__(self, key):
        if key == "genre" and self.world:
            return self.world.genre
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{key}'"
        )

    @property
    def story(self):
        return self.backstory

    @story.setter
    def story(self, value):
        self.backstory = value

    @property
    def slug(self):
        return slugify(self.name)

    @property
    def backstory_summary(self):
        if not self.bs_summary:
            primer = "As an expert AI in fictional Worldbuilding fof TTRPGs, summarize the following backstory into a concise paragraph, creating a readable summary that could help a person understand the main points of the backstory. Avoid unnecessary details."

            self.bs_summary = OpenAI().summarize_text(self.backstory, primer=primer)
            self.save()
        return self.bs_summary

    def image(self, url=None, raw=None, asset_id=0):
        if self.image_data is None:
            self.image_data = {}
        if url:
            self.image_data["url"] = url
        if raw:
            self.image_data["raw"] = raw
        if asset_id:
            self.image_data["asset_id"] = asset_id
        return self.image_data

    def save(self):
        self.get_image()
        return super().save()

    def get_image(self, url=None, update=False):
        log(self.image_data)
        self.image_data["url"] = url or self.image_data.get("url")
        if update or not self.image_data["url"]:
            if not self.image_data.get("raw"):
                resp = OpenAI().generate_image(
                    self.get_image_prompt(),
                    n=1,
                )
                # log(resp)
                self.image_data["raw"] = resp[0]

            if self.world:
                img_path = f"ttrpg/{self.world.slug}/{self.__class__.__name__.lower()}"
            else:
                img_path = f"ttrpg/{self.slug}/{self.__class__.__name__.lower()}"

            self.image_data = self._storage.save(
                self.image_data["raw"],
                folder=img_path,
                context={"caption": self.get_image_prompt()[:500]},
            )
        if self.image_data["url"]:
            self.image_data["raw"] = None
        return self.image_data["url"]

    def get_image_prompt(self):
        raise NotImplementedError

    @classmethod
    def generate(cls, prompt, primer):
        if hasattr(cls, "funcobj"):
            cls.funcobj["parameters"]["required"] = list(
                cls.funcobj["parameters"]["properties"].keys()
            )
            response = OpenAI().generate_text(prompt, primer, functions=cls.funcobj)
        else:
            response = OpenAI().generate_text(prompt, primer)

        json_invalid_max = 10
        json_retries = 1
        obj_data = None
        while json_invalid_max > 0:
            try:
                obj_data = json.loads(response, strict=False)
            except json.JSONDecodeError as e:
                response = response[: e.pos] + response[e.pos + 1 :]
                log(e, response)
                if json_retries:
                    if hasattr(cls, "funcobj"):
                        cls._funcobj["parameters"]["required"] = list(
                            cls._funcobj["parameters"]["properties"].keys()
                        )
                        response = OpenAI().generate_text(
                            prompt, primer, functions=cls._funcobj
                        )
                    else:
                        response = OpenAI().generate_text(prompt, primer)
                        json_invalid_max = 10
                    json_retries -= 1
                else:
                    json_invalid_max -= 1
        return obj_data

    def page_data(self):
        return {}

    def page_url(self, path="ttrpg"):
        # TODO: base_url = self._wiki_api.wiki_api.endpoint[:-1] if endswith
        return f"{self.wiki_path}"

    def canonize(self, api=None, root_path="ttrpg"):
        if not self.wiki_path:
            model = self.__class__.__name__.lower()
            w = f"{self.world.slug}/" if self.world else ""
            self.wiki_path = f"/{root_path}/{w}{model}/{self.slug}"

        config = {
            "Notes": self.notes,
            "Image": f"![{self.name}]({self.get_image()} =x350) \n\n {self.desc}",
            "Meta": [
                f"Genre: {self.genre}",
                f"World: {self.world.name if self.world else self.name}",
                f"pk: {self.pk}",
            ],
        }

        if self.traits:
            config["Meta"] += [f"Traits: {', '.join(self.traits)}"]

        if hasattr(self, "history") and self.history:
            config |= {"History": self.history}
        elif self.backstory:
            config |= {"Backstory": self.backstory}

        config |= self.page_data(root_path=root_path)

        if not api:
            api = self._wiki_api

        if self.wiki_id:
            res = api.push(
                config,
                title=self.name,
                id=self.wiki_id,
            )
        else:
            res = api.push(
                config,
                title=self.name,
                path=f"{self.wiki_path}",
                description=self.desc[: self.desc.find(".") + 1],
                tags=[
                    self.__class__.__name__,
                    self.world.slug if self.world else self.slug,
                    "ttrpg",
                ],
            )
            print(res)
            self.wiki_id = res.id
            self.save()
        return res
