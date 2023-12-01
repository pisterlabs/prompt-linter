import pytest
from guidance_and_support.factories import (
    GuidanceAndSupportPageFactory,
    GuidanceGroupPageFactory,
    GuidancePageFactory
)
from home.models import HomePage


@pytest.mark.django_db
class TestGuidancePage():
    """Tests EventPage."""

    @property
    def home_page(self):
        """Return HomePage created in migrations."""
        return HomePage.objects.first()

    def test_guidance_tree(self, client):
        """Test that event with random date is created."""
        guidance_and_support_index = GuidanceAndSupportPageFactory.create()
        guidance_group_page = GuidanceGroupPageFactory.create(parent=guidance_and_support_index)
        guidance_page = GuidancePageFactory.create_batch(20, parent=guidance_group_page)
        assert guidance_page is not None
