import asyncio

import pytest
from guidance.llms import Mock

from wet_toast_talk_radio.media_store import MediaStore, VirtualMediaStore
from wet_toast_talk_radio.media_store.media_store import ShowId
from wet_toast_talk_radio.scriptwriter.modern_mindfulness import (
    ModernMindfulness,
)


def test_modern_mindfulness(
    fake_llm, virtual_media_store, show_id, situation, circumstance
):
    show = ModernMindfulness(
        situation=situation,
        circumstance=circumstance,
        llm=fake_llm,
        media_store=virtual_media_store,
    )
    asyncio.run(show.arun(show_id=show_id))
    script_shows = virtual_media_store.list_script_shows()
    assert script_shows == [show_id]


@pytest.fixture()
def situation() -> str:
    return "Taking the metro to go to work"


@pytest.fixture()
def circumstance() -> str:
    return "Having to pee really bad"


@pytest.fixture()
def show_id() -> ShowId:
    return ShowId(show_i=0, date="2021-01-01")


@pytest.fixture()
def fake_llm() -> Mock:
    events = """
        Missing your stop.
        Getting caught in the door.
        Sitting next to a loud chewer.
        Accidentally stepping on someone's foot.
        Making eye contact with your ex.
    """
    meditation = "Welcome to Modern Mindfulness! Now breathe. [breathes]"
    return Mock(output=[events, meditation])


@pytest.fixture()
def virtual_media_store() -> MediaStore:
    return VirtualMediaStore(load_test_data=False)
