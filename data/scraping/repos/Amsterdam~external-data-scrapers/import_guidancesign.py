import logging

from django.core.management.base import BaseCommand

from apps.parkeergarages.importer import GuidanceSignImporter
from apps.parkeergarages.models import GuidanceSignSnapshot

log = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Import GuidanceSignSnapshot"

    def handle(self, *args, **options):
        log.info("Starting Importing")
        for snapshot in GuidanceSignSnapshot.objects.limit_offset_iterator(10):
            GuidanceSignImporter(snapshot).start_import()
        log.info("Import Done")
