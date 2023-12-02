import logging

from django.core.management.base import BaseCommand

from apps.parkeergarages.scraper import GuidanceSignScraper

log = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Scrape GuidanceSigns"

    def handle(self, *args, **options):
        log.info("Starting Scraper")
        GuidanceSignScraper().start()
        log.info("Scraping Done")
