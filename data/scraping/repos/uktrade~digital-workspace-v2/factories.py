from content.test.factories import ContentOwnerFactoryMixin, ContentPageFactory
from working_at_dit.models import Guidance, HowDoI, PageWithTopics, Policy


class PageWithTopicsFactory(ContentPageFactory):
    class Meta:
        model = PageWithTopics


class GuidanceFactory(ContentOwnerFactoryMixin, PageWithTopicsFactory):
    class Meta:
        model = Guidance


class PolicyFactory(ContentOwnerFactoryMixin, PageWithTopicsFactory):
    class Meta:
        model = Policy


class HowDoIFactory(ContentOwnerFactoryMixin, PageWithTopicsFactory):
    class Meta:
        model = HowDoI
