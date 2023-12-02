# coding=utf-8
__author__ = 'stefano'
import logging
from openpyxl import Workbook
from django.core.management.base import BaseCommand
from openaid.projects.models import Initiative


class Command(BaseCommand):

    help = 'export file with initiatives for check'
    logger = logging.getLogger('openaid')

    def handle(self, *args, **options):
        verbosity = options['verbosity']
        if verbosity == '0':
            self.logger.setLevel(logging.ERROR)
        elif verbosity == '1':
            self.logger.setLevel(logging.WARNING)
        elif verbosity == '2':
            self.logger.setLevel(logging.INFO)
        elif verbosity == '3':
            self.logger.setLevel(logging.DEBUG)

        output_filename= 'initiatives.xlsx'
        initiative_fields = ['code', 'title_it', 'title_en', 'loan_amount_approved','grant_amount_approved','total_project_costs','recipient_temp','status_temp']
        excel_header = ['initiative_code', 'title_it', 'title_en', 'loan_amount_approved','grant_amount_approved','total_project_costs','recipient','status']
        workbook = Workbook()
        ws_output = workbook.create_sheet(index=0, title='sheet')
        self.logger.info(u"start")

        # append headers to output file
        ws_output.append(excel_header)

        for init in Initiative.objects.all().order_by('recipient_temp','status_temp'):

            row = []

            # gets data from initiative
            for f in initiative_fields:
                value = getattr(init, f)
                if f == 'recipient_temp' and value is not None:
                    row.append(value.name)
                elif f == 'status_temp':
                    if value == '-':
                        row.append("NOT AVAILABLE")
                    else:
                        row.append(value+"%")
                else:
                    row.append(value)

            ws_output.append(row)

        # save output file
        workbook.save(output_filename)
        self.logger.info(u"Written {} initiatives to file:{}".format(Initiative.objects.all().count(),output_filename))