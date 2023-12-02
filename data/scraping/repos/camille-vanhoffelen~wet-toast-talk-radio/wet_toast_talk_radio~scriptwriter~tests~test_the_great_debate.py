import asyncio

import pytest
from guidance.llms import Mock

from wet_toast_talk_radio.media_store import MediaStore, VirtualMediaStore
from wet_toast_talk_radio.media_store.media_store import ShowId
from wet_toast_talk_radio.scriptwriter.the_great_debate.show import (
    Guest,
    Polarity,
    TheGreatDebate,
)


def test_the_great_debate(  # noqa: PLR0913
    fake_llm, virtual_media_store, show_id, guest_in_favor, guest_against, topic
):
    show = TheGreatDebate(
        topic=topic,
        guest_in_favor=guest_in_favor,
        guest_against=guest_against,
        llm=fake_llm,
        media_store=virtual_media_store,
    )
    asyncio.run(show.arun(show_id=show_id))
    script_shows = virtual_media_store.list_script_shows()
    assert script_shows == [show_id]


def test_broken_the_great_debate(  # noqa: PLR0913
    broken_fake_llm, virtual_media_store, show_id, guest_in_favor, guest_against, topic
):
    show = TheGreatDebate(
        topic=topic,
        guest_in_favor=guest_in_favor,
        guest_against=guest_against,
        llm=broken_fake_llm,
        media_store=virtual_media_store,
    )
    with pytest.raises(AssertionError):
        asyncio.run(show.arun(show_id=show_id))


@pytest.fixture()
def topic() -> str:
    return "toilet paper"


@pytest.fixture()
def script(topic: str, guest_in_favor: Guest, guest_against: Guest) -> str:
    return f"""Julie: Welcome to THE GREAT DEBATE!
    {guest_in_favor.placeholder_name}: I love {topic}.
    {guest_against.placeholder_name}: I hate {topic}.
    {guest_in_favor.placeholder_name}: Let's agree to disagree."""


@pytest.fixture()
def bad_script() -> str:
    return "This isn't a script at all!"


@pytest.fixture()
def show_id() -> ShowId:
    return ShowId(show_i=0, date="2021-01-01")


@pytest.fixture()
def fake_llm(
    topic: str, script: str, guest_in_favor: Guest, guest_against: Guest
) -> Mock:
    in_favor_guest = f"Meet {guest_in_favor.placeholder_name}. {guest_in_favor.placeholder_name} loves {topic}."
    against_guest = f"Meet {guest_against.placeholder_name}. {guest_against.placeholder_name} hates {topic}."
    return Mock(output=[in_favor_guest, against_guest, script])


@pytest.fixture()
def broken_fake_llm(
    topic: str, bad_script: str, guest_against: Guest, guest_in_favor: Guest
) -> Mock:
    in_favor_guest = f"Meet {guest_in_favor.placeholder_name}. {guest_in_favor.placeholder_name} loves {topic}."
    against_guest = f"Meet {guest_against.placeholder_name}. {guest_against.placeholder_name} hates {topic}."
    return Mock(output=[in_favor_guest, against_guest, bad_script])


@pytest.fixture()
def virtual_media_store() -> MediaStore:
    return VirtualMediaStore(load_test_data=False)


@pytest.fixture()
def guest_in_favor() -> Guest:
    return Guest(
        name="Tamara",
        gender="female",
        trait="Cool",
        polarity=Polarity.IN_FAVOR,
        placeholder_name="Alice",
    )


@pytest.fixture()
def guest_against() -> Guest:
    return Guest(
        name="George",
        gender="male",
        trait="Grumpy",
        polarity=Polarity.AGAINST,
        placeholder_name="Bob",
    )
