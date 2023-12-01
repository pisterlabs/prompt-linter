"""Management command that creates the top-level default pages from the infrastructure architecture."""

from django.core.management.base import BaseCommand
from home.models import HomePage
from about.models import AboutPage
from contact.models import ContactPage
from events.models import EventIndexPage
from guidance_and_support.models import GuidanceAndSupportPage, CommunityPage
from news.models import NewsIndexPage
from iati_standard.models import IATIStandardPage
from using_data.models import UsingDataPage


DEFAULT_PAGES = [
    {"model": AboutPage, "title": "About", "slug": "about"},
    {"model": ContactPage, "title": "Contact", "slug": "contact"},
    {"model": EventIndexPage, "title": "Events", "slug": "events"},
    {"model": GuidanceAndSupportPage, "title": "Guidance and support", "slug": "guidance"},
    {"model": NewsIndexPage, "title": "News", "slug": "news"},
    {'model': IATIStandardPage, 'title': "IATI Standard", 'slug': 'iati-standard'},
    {'model': UsingDataPage, 'title': "Using IATI data", 'slug': 'using-data'},
    {'model': CommunityPage, 'title': "Community", 'slug': 'community'}
]


class Command(BaseCommand):
    """Management command that first rectifies some database problems with the HomePage model created by wagtail-modeltranslation and then creates the top-level default pages from the infrastructure architecture.

    The home_page needed a queryset update before the HomePage model is allowed to save in the CMS.

    The update method bypasses the validation of the save method and writes directly to the database, but the child pages need their URLs updated with save.
    If no missing pages are detected for the children, the home page will not be changed.

    TODO:
    1. If wagtail-modeltranslation or django-modeltranslation update, this command may no longer need to edit the home page.
    """

    help = 'Create the default pages that constitute the skeleton of the website information architecture.'

    def handle(self, *args, **options):
        """Implement the command handler."""
        missing_pages_detected = False

        home_page = HomePage.objects.live().first()
        if home_page is not None:

            for default_page in DEFAULT_PAGES:
                default_page_instance = default_page["model"].objects.live().first()
                if default_page_instance is None:
                    missing_pages_detected = True

            if missing_pages_detected:
                self.stdout.write(self.style.SUCCESS('Missing pages detected. Fixing and saving home page...'))
                home_page_queryset = HomePage.objects.live()
                home_page_queryset.update(
                    slug_en="home",
                    slug="home",
                    url_path_en="/home/",
                    url_path="/home/",
                    title_en="Home",
                    title="Home"
                )
                home_page = home_page_queryset.first()
                home_page.save()
                for default_page in DEFAULT_PAGES:
                    default_page_instance = default_page["model"].objects.live().first()
                    if default_page_instance is None:
                        msg = 'No {} page! Creating about page...'.format(default_page["title"])
                        self.stdout.write(self.style.WARNING(msg))
                        default_page_instance = default_page["model"](
                            title_en=default_page["title"],
                            slug_en=default_page["slug"],
                            title=default_page["title"],
                            slug=default_page["slug"]
                        )
                        home_page.add_child(instance=default_page_instance)
                        default_page_instance.save_revision().publish()
            else:
                self.stdout.write(self.style.SUCCESS('No missing pages detected. Skipping home page fixes...'))

            self.stdout.write(self.style.SUCCESS('Success.'))
