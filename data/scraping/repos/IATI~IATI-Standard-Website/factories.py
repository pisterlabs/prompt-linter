import factory
from wagtail_factories import ImageFactory
from guidance_and_support.models import (
    GuidanceAndSupportPage,
    GuidanceGroupPage,
    GuidancePage,
)
from home.factories import BasePageFactory


class GuidanceAndSupportPageFactory(BasePageFactory):
    """Factory generating data for GuidanceAndSupportPage."""

    class Meta:
        model = GuidanceAndSupportPage


class GuidanceGroupPageFactory(BasePageFactory):
    """Factory generating data for GuidanceGroupPage."""

    section_image = factory.SubFactory(ImageFactory)
    button_link_text = factory.Faker(
        provider='sentence',
        nb_words=3
    )
    button_link_text_fr = factory.Faker(
        provider='sentence',
        nb_words=3,
        locale='fr_FR',
    )

    class Meta:
        model = GuidanceGroupPage


class GuidancePageFactory(BasePageFactory):
    """Factory generating data for GuidancePage."""

    class Meta:
        model = GuidancePage
