# Integration / functional output tests
import pytest
from django.core.management import call_command
from django.test import TestCase

from content.models import ContentPage
from extended_search.query import Filtered
from extended_search.query_builder import CustomQueryBuilder
from news.factories import NewsPageFactory
from peoplefinder.services.person import PersonService
from peoplefinder.test.factories import PersonFactory, TeamFactory
from tools.tests.factories import ToolFactory
from user.models import User
from user.test.factories import UserFactory
from working_at_dit.models import PoliciesAndGuidanceHome
from working_at_dit.tests.factories import GuidanceFactory, HowDoIFactory, PolicyFactory


class TestGeneratedQuery:
    def find_filter(self, query):
        if isinstance(query, Filtered):
            self.filter_parts.append(query)
            return query
        if hasattr(query, "subqueries"):
            subs = []
            for q in query.subqueries:
                subs.append(self.find_filter(q))
            return subs
        return query.subquery

    @pytest.mark.django_db
    def test_generated_search_applies_baseclass_and_subclass_filters(self):
        query = CustomQueryBuilder.get_search_query(ContentPage, "foo")

        extended_models = (
            CustomQueryBuilder.get_extended_models_with_unique_indexed_fields(
                ContentPage
            )
        )

        self.filter_parts = []
        self.find_filter(query)

        num_excludes = 0
        assert len(self.filter_parts) == len(extended_models) + 1
        for r in self.filter_parts:
            field, lookup, value = r.filters[0]
            assert field == "content_type"
            assert lookup in ("contains", "excludes")
            if lookup == "excludes":
                num_excludes += 1
                assert (
                    f"{ContentPage._meta.app_label}.{ContentPage.__name__}" not in value
                )
                for model in extended_models:
                    assert f"{model._meta.app_label}.{model.__name__}" in value
            else:
                assert value in [
                    f"{model._meta.app_label}.{model.__name__}"
                    for model in extended_models
                ]
        assert num_excludes == 1


def create_test_user():
    # Jane Smith - another normal user
    user, created = User.objects.get_or_create(
        username="johnsmith",
        first_name="John",
        last_name="Smith",
        email="john.smith@example.com",
        legacy_sso_user_id="john-smith-sso-user-id",
        is_staff=False,
        is_superuser=False,
    )
    if created:
        PersonService().create_user_profile(user)


class TestExpectedSearchResults(TestCase):
    @pytest.mark.django_db(transaction=True, reset_sequences=True)
    def setUp(self):
        """
        Create the following data for search testing:
        - 1 guidance page with "fruit" in the title
        - 1 policy page with "fruit" in the title
        - 1 howdoi page with "fruit" in the title
        - 1 tool page with "fruit" in the title
        - 1 news page with "fruit" in the title
        - 1 person with "fruit" in their name
        - 1 team with "fruit" in the name
        - 1 of each of the above without any "fruit" content
        """

        create_test_user()

        policies_and_guidance_home = PoliciesAndGuidanceHome.objects.first()
        self.content_owner = User.objects.get(username="johnsmith")
        self.content_owner_pages = [
            GuidanceFactory.create(
                parent=policies_and_guidance_home,
                title="How to eat Fruit",
                content_owner=self.content_owner.profile,
                content_contact_email=self.content_owner.email,
            ),
            GuidanceFactory.create(
                parent=policies_and_guidance_home,
                title="An irrelevant page",
                content_owner=self.content_owner.profile,
                content_contact_email=self.content_owner.email,
            ),
            PolicyFactory.create(
                parent=policies_and_guidance_home,
                title="What fruit can I eat?",
                content_owner=self.content_owner.profile,
                content_contact_email=self.content_owner.email,
            ),
            PolicyFactory.create(
                parent=policies_and_guidance_home,
                title="Policies not relating to food",
                content_owner=self.content_owner.profile,
                content_contact_email=self.content_owner.email,
            ),
            HowDoIFactory.create(
                title="How do I find fruit?",
                content_owner=self.content_owner.profile,
                content_contact_email=self.content_owner.email,
            ),
            HowDoIFactory.create(
                title="How to write test content",
                content_owner=self.content_owner.profile,
                content_contact_email=self.content_owner.email,
            ),
        ]
        self.pages = [
            ToolFactory.create(
                title="Fruit finder",
            ),
            ToolFactory.create(
                title="Internal tooling tool",
            ),
            NewsPageFactory.create(
                title="Big fruit news",
            ),
            NewsPageFactory.create(
                title="New news",
            ),
        ]
        self.people = [
            PersonFactory.create(
                first_name="Martin",
                last_name="Fruit",
                email="martin.fruit@example.com",
            ),
            PersonFactory.create(
                first_name="Adam", last_name="Ant", email="adam.ant@example.com"
            ),
        ]
        self.teams = [
            TeamFactory.create(
                name="Fruit pickers", slug="fruit-pickers", abbreviation="FRUIT"
            ),
            TeamFactory.create(name="Middle managers", slug="managers"),
        ]

        call_command("update_index")

        self.user = UserFactory()
        PersonService().create_user_profile(self.user)
        self.client.force_login(self.user)

    def tearDown(self):
        for p in self.content_owner_pages:
            p.delete()
        for p in self.pages:
            p.delete()
        for p in self.people:
            p.delete()
        for t in self.teams:
            t.delete()

    def test_workspace(self):
        response = self.client.get("/search/", {"query": ""}, follow=True)
        self.assertEqual(response.status_code, 200)
        for page in self.content_owner_pages:
            self.assertNotContains(response, page.title)
        for page in self.pages:
            self.assertNotContains(response, page.title)
        for person in self.people:
            self.assertNotContains(response, person.email)
        for team in self.teams:
            self.assertNotContains(response, team.name)

        self.assertContains(response, "All&nbsp;(0)")
        self.assertContains(response, "People&nbsp;(0)")
        self.assertContains(response, "Teams&nbsp;(0)")
        self.assertContains(response, "Guidance&nbsp;(0)")
        self.assertContains(response, "Tools&nbsp;(0)")
        self.assertContains(response, "News&nbsp;(0)")
        self.assertContains(response, "0 pages")
        self.assertContains(response, "0 people")
        self.assertContains(response, "0 teams")
        self.assertContains(response, "There are no matching pages.")
        self.assertContains(response, "There are no matching people.")
        self.assertContains(response, "There are no matching teams.")

    def test_fruits_all(self):
        response = self.client.get("/search/", {"query": "fruit"}, follow=True)
        self.assertEqual(response.status_code, 200)

        self.assertContains(response, "All&nbsp;(7)")
        self.assertContains(response, "People&nbsp;(1)")
        self.assertContains(response, "Teams&nbsp;(1)")
        self.assertContains(response, "Guidance&nbsp;(2)")
        self.assertContains(response, "Tools&nbsp;(1)")
        self.assertContains(response, "News&nbsp;(1)")
        self.assertContains(response, "5 pages")
        self.assertContains(response, "1 person")
        self.assertContains(response, "1 team")
        self.assertContains(response, "How to eat Fruit")
        self.assertContains(response, "What fruit can I eat?")
        self.assertContains(response, "How do I find fruit?")
        self.assertContains(response, "Fruit finder")
        self.assertContains(response, "Big fruit news")
        self.assertNotContains(response, "An irrelevant page")
        self.assertNotContains(response, "Policies not relating to food")
        self.assertNotContains(response, "How to write test parent")
        self.assertNotContains(response, "Internal tooling tool")
        self.assertNotContains(response, "New news")
        self.assertContains(response, "Martin Fruit")
        self.assertNotContains(response, "Adam Ant")
        self.assertContains(response, "Fruit pickers")
        self.assertNotContains(response, "Middle managers")

    def test_fruits_news(self):
        response = self.client.get("/search/news/", {"query": "fruit"}, follow=True)
        self.assertEqual(response.status_code, 200)

        self.assertContains(response, "All&nbsp;(7)")
        self.assertContains(response, "People&nbsp;(1)")
        self.assertContains(response, "Teams&nbsp;(1)")
        self.assertContains(response, "Guidance&nbsp;(2)")
        self.assertContains(response, "Tools&nbsp;(1)")
        self.assertContains(response, "News&nbsp;(1)")
        self.assertNotContains(response, "How to eat Fruit")
        self.assertNotContains(response, "What fruit can I eat?")
        self.assertNotContains(response, "How do I find fruit?")
        self.assertNotContains(response, "Fruit finder")
        self.assertContains(response, "Big fruit news")
        self.assertNotContains(response, "An irrelevant page")
        self.assertNotContains(response, "Policies not relating to food")
        self.assertNotContains(response, "How to write test parent")
        self.assertNotContains(response, "Internal tooling tool")
        self.assertNotContains(response, "New news")
        self.assertNotContains(response, "Martin Fruit")
        self.assertNotContains(response, "Adam Ant")
        self.assertNotContains(response, "Fruit pickers")
        self.assertNotContains(response, "Middle managers")

    def test_fruits_tools(self):
        response = self.client.get("/search/tools/", {"query": "fruit"}, follow=True)
        self.assertEqual(response.status_code, 200)

        self.assertContains(response, "All&nbsp;(7)")
        self.assertContains(response, "People&nbsp;(1)")
        self.assertContains(response, "Teams&nbsp;(1)")
        self.assertContains(response, "Guidance&nbsp;(2)")
        self.assertContains(response, "Tools&nbsp;(1)")
        self.assertContains(response, "News&nbsp;(1)")
        self.assertNotContains(response, "How to eat Fruit")
        self.assertNotContains(response, "What fruit can I eat?")
        self.assertNotContains(response, "How do I find fruit?")
        self.assertContains(response, "Fruit finder")
        self.assertNotContains(response, "Big fruit news")
        self.assertNotContains(response, "An irrelevant page")
        self.assertNotContains(response, "Policies not relating to food")
        self.assertNotContains(response, "How to write test parent")
        self.assertNotContains(response, "Internal tooling tool")
        self.assertNotContains(response, "New news")
        self.assertNotContains(response, "Martin Fruit")
        self.assertNotContains(response, "Adam Ant")
        self.assertNotContains(response, "Fruit pickers")
        self.assertNotContains(response, "Middle managers")

    def test_fruits_guidance(self):
        response = self.client.get("/search/guidance/", {"query": "fruit"}, follow=True)
        self.assertEqual(response.status_code, 200)

        self.assertContains(response, "All&nbsp;(7)")
        self.assertContains(response, "People&nbsp;(1)")
        self.assertContains(response, "Teams&nbsp;(1)")
        self.assertContains(response, "Guidance&nbsp;(2)")
        self.assertContains(response, "Tools&nbsp;(1)")
        self.assertContains(response, "News&nbsp;(1)")
        self.assertContains(response, "How to eat Fruit")
        self.assertContains(response, "What fruit can I eat?")
        self.assertNotContains(response, "How do I find fruit?")
        self.assertNotContains(response, "Fruit finder")
        self.assertNotContains(response, "Big fruit news")
        self.assertNotContains(response, "An irrelevant page")
        self.assertNotContains(response, "Policies not relating to food")
        self.assertNotContains(response, "How to write test parent")
        self.assertNotContains(response, "Internal tooling tool")
        self.assertNotContains(response, "New news")
        self.assertNotContains(response, "Martin Fruit")
        self.assertNotContains(response, "Adam Ant")
        self.assertNotContains(response, "Fruit pickers")
        self.assertNotContains(response, "Middle managers")

    def test_fruits_teams(self):
        response = self.client.get("/search/teams/", {"query": "fruit"}, follow=True)
        self.assertEqual(response.status_code, 200)

        self.assertContains(response, "All&nbsp;(7)")
        self.assertContains(response, "People&nbsp;(1)")
        self.assertContains(response, "Teams&nbsp;(1)")
        self.assertContains(response, "Guidance&nbsp;(2)")
        self.assertContains(response, "Tools&nbsp;(1)")
        self.assertContains(response, "News&nbsp;(1)")
        self.assertNotContains(response, "How to eat Fruit")
        self.assertNotContains(response, "What fruit can I eat?")
        self.assertNotContains(response, "How do I find fruit?")
        self.assertNotContains(response, "Fruit finder")
        self.assertNotContains(response, "Big fruit news")
        self.assertNotContains(response, "An irrelevant page")
        self.assertNotContains(response, "Policies not relating to food")
        self.assertNotContains(response, "How to write test parent")
        self.assertNotContains(response, "Internal tooling tool")
        self.assertNotContains(response, "New news")
        self.assertNotContains(response, "Martin Fruit")
        self.assertNotContains(response, "Adam Ant")
        self.assertContains(response, "Fruit pickers")
        self.assertNotContains(response, "Middle managers")

    def test_fruits_people(self):
        response = self.client.get("/search/people/", {"query": "fruit"}, follow=True)
        self.assertEqual(response.status_code, 200)

        self.assertContains(response, "All&nbsp;(7)")
        self.assertContains(response, "People&nbsp;(1)")
        self.assertContains(response, "Teams&nbsp;(1)")
        self.assertContains(response, "Guidance&nbsp;(2)")
        self.assertContains(response, "Tools&nbsp;(1)")
        self.assertContains(response, "News&nbsp;(1)")
        self.assertNotContains(response, "How to eat Fruit")
        self.assertNotContains(response, "What fruit can I eat?")
        self.assertNotContains(response, "How do I find fruit?")
        self.assertNotContains(response, "Fruit finder")
        self.assertNotContains(response, "Big fruit news")
        self.assertNotContains(response, "An irrelevant page")
        self.assertNotContains(response, "Policies not relating to food")
        self.assertNotContains(response, "How to write test parent")
        self.assertNotContains(response, "Internal tooling tool")
        self.assertNotContains(response, "New news")
        self.assertContains(response, "Martin Fruit")
        self.assertNotContains(response, "Adam Ant")
        self.assertNotContains(response, "Fruit pickers")
        self.assertNotContains(response, "Middle managers")
