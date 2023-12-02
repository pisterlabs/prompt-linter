# coding=utf-8

from django.core.management.base import BaseCommand, CommandError
from openair.records.models import Zone


class Command(BaseCommand):
    help = 'Run it to fill the DB with hardcoded zones'

    def handle(self, *args, **options):

        # hardcoded zones
        zones = [
            dict(name='צפון', url_id=7),
            dict(name='חיפה וקריות', url_id=8),
            dict(name='עמק יזרעאל', url_id=9),
            dict(name='שרון - כרמל', url_id=10),
            dict(name='שומרון', url_id=11),
            dict(name='שפלה פנימית', url_id=12),
            dict(name='גוש דן', url_id=13),
            dict(name='ירושלים', url_id=14),
            dict(name='איזור יהודה', url_id=15),
            dict(name='מישור החוף הדרומי', url_id=16),
            dict(name='צפון הנגב', url_id=18),
            dict(name='אילת', url_id=19),
        ]

        # delete old zones
        for zone in Zone.objects.all():
            zone.delete()

        # fill DB
        for z in zones:
            zone = Zone(name=z['name'], url_id=z['url_id'])
            zone.save()
