"""
Produce un piu csv uno per code list
dove ogni riga presenta
Project.crs:Project.recipient_id
e il valore anno dopo anno di quella code list.

CRSID | RECIPIENT | 2004 | ... | 2012

"""
import csvkit
from django.db.models import Count
import sys

from openaid.projects.models import Project

YEARS = [str(x) for x in range(2004, 2013)]
DEFAULT_YEARS_VALUES = dict([(y, u'') for y in YEARS])
FIELDS = ['crsid', 'recipient', 'agency'] + [str(x) for x in range(2004, 2013)]

def create_writer(name):
    f = open('%s.csv' % name, 'w')
    return f, csvkit.DictWriter(f, FIELDS)


def run():

    aid_file, aid_writer = create_writer('aid')
    aid_writer.writeheader()
    sector_file, sector_writer = create_writer('sector')
    sector_writer.writeheader()
    channel_file, channel_writer = create_writer('channel')
    channel_writer.writeheader()

    for i, project in enumerate(Project.objects.annotate(activity_count=Count('activity')).filter(activity_count__gt=1)):

        aids = {}
        sectors = {}
        channels = {}
        agency = ''

        for activity in project.activity_set.all():

            year = str(activity.year)

            aid = activity.aid_type.code if activity.aid_type else 'XXX'
            if year in aids:
                if aid not in aids[year]:
                    aids[year] += '/%s' % aid
            else:
                aids[year] = aid
            sector = activity.sector.code if activity.sector else 'XXX'
            if year in sectors:
                if sector not in sectors[year]:
                    sectors[year] += '/%s' % sector
            else:
                sectors[year] = sector
            channel = activity.channel.code if activity.channel else 'XXX'
            if year in channels:
                if channel not in channels[year]:
                    channels[year] += '/%s' % channel
            else:
                channels[year] = channel

            agency = activity.agency.name if activity.agency else ''

        line = {
            'crsid': project.crsid  ,
            'recipient': project.recipient.code,
            'agency': agency,
        }
        line.update(DEFAULT_YEARS_VALUES)

        for codelist, writer in [(aids, aid_writer), (sectors, sector_writer), (channels, channel_writer)]:

            codelist_line = line.copy()
            codelist_line.update(codelist)
            writer.writerow(codelist_line)

        sys.stdout.write("\rMulti activity project: %d" % (i), ending='')
        sys.stdout.flush()

    aid_file.close()
    sector_file.close()
    channel_file.close()