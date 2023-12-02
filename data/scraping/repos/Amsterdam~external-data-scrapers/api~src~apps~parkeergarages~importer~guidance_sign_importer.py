import logging

from apps.base.areas_importer import AreasImportFactory
from apps.base.base_importer import BaseImportFactory, BaseSnapshotImporter
from apps.base.geojson_importer import GeoJsonImportFactory
from apps.parkeergarages import constants
from apps.parkeergarages.forms import (GuidanceSignImporterForm,
                                       ParkingGuidanceDisplayImporterForm)
from apps.parkeergarages.models import GuidanceSign, ParkingGuidanceDisplay

log = logging.getLogger(__name__)


class GuidanceSignImportFactory(GeoJsonImportFactory, AreasImportFactory):
    model_form = GuidanceSignImporterForm
    raw_to_model_fields = constants.GUIDANCESIGN_RAW_TO_MODEL_FIELDS


class ParkingGuidanceDisplayImportFactory(BaseImportFactory):
    model_form = ParkingGuidanceDisplayImporterForm
    raw_to_model_fields = constants.PARKINGGUIDANCEDISPLAY_RAW_TO_MODEL_FIELDS

    def __init__(self, *args, **kwargs):
        self.guidance_sign = kwargs.pop('guidance_sign')
        super().__init__(*args, **kwargs)

    def finalize_model_instance(self, model_instance):
        super().finalize_model_instance(model_instance)
        model_instance.pub_date = self.guidance_sign.pub_date
        model_instance.guidance_sign_id = self.guidance_sign.api_id


class GuidanceSignImporter(BaseSnapshotImporter):
    '''
    Heavily overrides the BaseSnapshotImporter to allow importing the snapshot into
    two models.

    Each GuidanceSign contains several ParingGuidanceSignDisplay(s).
    '''

    def fetch_raw_data_from_snapshot(self):
        return self.snapshot.data['features']

    def store_guidance_sign(self, guidance_sign_list):
        '''
        GuidanceSigns are updated on every import not endlessly appended to the table.
        We use the model's update_or_create method for that.
        '''
        log.info(f'''
        Bulk updating or creating {len(guidance_sign_list)} GuidanceSign
        scraped at {self.snapshot.scraped_at}
        ''')
        for guidance_sign in guidance_sign_list:
            # Generate dict with the fields that need to be updated
            defaults = {x: guidance_sign.__dict__[x] for x in constants.GUIDANCESIGN_UPDATE_FIELDS}
            GuidanceSign.objects.update_or_create(api_id=guidance_sign.api_id, defaults=defaults)

    def start_import(self):
        '''
        Generates guidance_sign_list and parking_guidance_sign_list from one snapshot
        then calls different store methods to save each list in the db
        '''
        guidance_sign_list = []
        parking_guidance_display_list = []

        for raw_data in self.fetch_raw_data_from_snapshot():
            guidance_sign = GuidanceSignImportFactory(snapshot=self.snapshot).build_model_instance(raw_data)
            guidance_sign_list.append(guidance_sign)

            for parking_guidance_raw in raw_data['ParkingguidanceDisplay']:
                parking_guidance_display = ParkingGuidanceDisplayImportFactory(
                    snapshot=self.snapshot,
                    guidance_sign=guidance_sign
                ).build_model_instance(parking_guidance_raw)
                parking_guidance_display_list.append(parking_guidance_display)

        self.store_guidance_sign(guidance_sign_list)

        self.import_model = ParkingGuidanceDisplay
        self.store(parking_guidance_display_list)
