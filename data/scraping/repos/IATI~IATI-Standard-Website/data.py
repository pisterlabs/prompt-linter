"""Module for handling IATI standard reference data."""
import requests
import io
import os
from zipfile import ZipFile
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.management import call_command
from django.conf import settings
from django.utils.text import slugify
from iati_standard.models import ReferenceData, ActivityStandardPage, IATIStandardPage, ReferenceMenu, StandardGuidanceIndexPage, StandardGuidancePage
from iati_standard.inlines import StandardGuidanceTypes
from iati_standard.edit_handlers import GithubAPI
from guidance_and_support.models import GuidanceAndSupportPage


ACTIVITY_SORT_ORDER = {
    "iati-activities/iati-activity": 0,
    "iati-activity/iati-identifier": 1,
    "iati-activity/reporting-org": 2,
    "reporting-org/narrative": 3,
    "iati-activity/title": 4,
    "title/narrative": 5,
    "iati-activity/description": 6,
    "description/narrative": 7,
    "iati-activity/participating-org": 8,
    "participating-org/narrative": 9,
    "iati-activity/other-identifier": 10,
    "other-identifier/owner-org": 11,
    "owner-org/narrative": 12,
    "iati-activity/activity-status": 13,
    "iati-activity/activity-date": 14,
    "activity-date/narrative": 15,
    "iati-activity/contact-info": 16,
    "contact-info/organisation": 17,
    "organisation/narrative": 18,
    "contact-info/department": 19,
    "department/narrative": 20,
    "contact-info/person-name": 21,
    "person-name/narrative": 22,
    "contact-info/job-title": 23,
    "job-title/narrative": 24,
    "contact-info/telephone": 25,
    "contact-info/email": 26,
    "contact-info/website": 27,
    "contact-info/mailing-address": 28,
    "mailing-address/narrative": 29,
    "iati-activity/activity-scope": 30,
    "iati-activity/recipient-country": 31,
    "recipient-country/narrative": 32,
    "iati-activity/recipient-region": 33,
    "recipient-region/narrative": 34,
    "iati-activity/location": 35,
    "location/location-reach": 36,
    "location/location-id": 37,
    "location/name": 38,
    "name/narrative": 39,
    "location/description": 40,
    "location/activity-description": 41,
    "activity-description/narrative": 42,
    "location/administrative": 43,
    "location/point": 44,
    "point/pos": 45,
    "location/exactness": 46,
    "location/location-class": 47,
    "location/feature-designation": 48,
    "iati-activity/sector": 49,
    "sector/narrative": 50,
    "iati-activity/tag": 51,
    "tag/narrative": 52,
    "iati-activity/country-budget-items": 53,
    "country-budget-items/budget-item": 54,
    "budget-item/description": 55,
    "iati-activity/humanitarian-scope": 56,
    "humanitarian-scope/narrative": 57,
    "iati-activity/policy-marker": 58,
    "policy-marker/narrative": 59,
    "iati-activity/collaboration-type": 60,
    "iati-activity/default-flow-type": 61,
    "iati-activity/default-finance-type": 62,
    "iati-activity/default-aid-type": 63,
    "iati-activity/default-tied-status": 64,
    "iati-activity/budget": 65,
    "budget/period-start": 66,
    "budget/period-end": 67,
    "budget/value": 68,
    "iati-activity/planned-disbursement": 69,
    "planned-disbursement/period-start": 70,
    "planned-disbursement/period-end": 71,
    "planned-disbursement/value": 72,
    "iati-activity/capital-spend": 73,
    "iati-activity/transaction": 74,
    "transaction/transaction-type": 75,
    "transaction/transaction-date": 76,
    "transaction/value": 77,
    "transaction/description": 78,
    "transaction/provider-org": 79,
    "provider-org/narrative": 80,
    "transaction/receiver-org": 81,
    "receiver-org/narrative": 82,
    "transaction/disbursement-channel": 83,
    "transaction/sector": 84,
    "transaction/recipient-country": 85,
    "transaction/recipient-region": 86,
    "transaction/flow-type": 87,
    "transaction/finance-type": 88,
    "transaction/aid-type": 89,
    "transaction/tied-status": 90,
    "iati-activity/document-link": 91,
    "document-link/title": 92,
    "document-link/description": 93,
    "document-link/category": 94,
    "document-link/language": 95,
    "document-link/document-date": 96,
    "iati-activity/related-activity": 97,
    "iati-activity/legacy-data": 98,
    "iati-activity/conditions": 99,
    "conditions/condition": 100,
    "condition/narrative": 101,
    "iati-activity/result": 102,
    "result/title": 103,
    "result/description": 104,
    "result/document-link": 105,
    "result/reference": 106,
    "result/indicator": 107,
    "indicator/title": 108,
    "indicator/description": 109,
    "indicator/document-link": 110,
    "indicator/reference": 111,
    "indicator/baseline": 112,
    "baseline/comment": 113,
    "comment/narrative": 114,
    "indicator/period": 115,
    "period/period-start": 116,
    "period/period-end": 117,
    "period/target": 118,
    "target/comment": 119,
    "period/actual": 120,
    "actual/comment": 121,
    "iati-activity/crs-add": 122,
    "crs-add/other-flags": 123,
    "crs-add/loan-terms": 124,
    "loan-terms/repayment-type": 125,
    "loan-terms/repayment-plan": 126,
    "loan-terms/commitment-date": 127,
    "loan-terms/repayment-first-date": 128,
    "loan-terms/repayment-final-date": 129,
    "crs-add/loan-status": 130,
    "loan-status/interest-received": 131,
    "loan-status/principal-outstanding": 132,
    "loan-status/principal-arrears": 133,
    "loan-status/interest-arrears": 134,
    "iati-activity/fss": 135,
    "fss/forecast": 136,
}


ORGANISATION_SORT_ORDER = {
    "iati-organisations/iati-organisation": 0,
    "iati-organisation/organisation-identifier": 1,
    "iati-organisation/name": 2,
    "iati-organisation/reporting-org": 3,
    "iati-organisation/total-budget": 4,
    "total-budget/period-start": 5,
    "total-budget/period-end": 6,
    "total-budget/value": 7,
    "total-budget/budget-line": 8,
    "budget-line/value": 9,
    "budget-line/narrative": 10,
    "iati-organisation/recipient-org-budget": 11,
    "recipient-org-budget/recipient-org": 12,
    "recipient-org-budget/period-start": 13,
    "recipient-org-budget/period-end": 14,
    "recipient-org-budget/value": 15,
    "recipient-org-budget/budget-line": 16,
    "iati-organisation/recipient-region-budget": 17,
    "recipient-region-budget/recipient-region": 18,
    "recipient-region-budget/period-start": 19,
    "recipient-region-budget/period-end": 20,
    "recipient-region-budget/value": 21,
    "recipient-region-budget/budget-line": 22,
    "iati-organisation/recipient-country-budget": 23,
    "recipient-country-budget/recipient-country": 24,
    "recipient-country-budget/period-start": 25,
    "recipient-country-budget/period-end": 26,
    "recipient-country-budget/value": 27,
    "recipient-country-budget/budget-line": 28,
    "iati-organisation/total-expenditure": 29,
    "total-expenditure/period-start": 30,
    "total-expenditure/period-end": 31,
    "total-expenditure/value": 32,
    "total-expenditure/expense-line": 33,
    "expense-line/value": 34,
    "expense-line/narrative": 35,
    "iati-organisation/document-link": 36,
    "document-link/recipient-country": 97,
}


SORT_ORDER = {**ACTIVITY_SORT_ORDER, **ORGANISATION_SORT_ORDER}


def iati_order(json_page_obj):
    """Return IATI Standard element order."""
    family_tag = "/".join(json_page_obj["ssot_path"].split("/")[-2:])
    try:
        return SORT_ORDER[family_tag]
    except KeyError:
        return 999


def default_order(json_page_obj):
    """Return ordering based on metadata."""
    return json_page_obj["meta_order"], json_page_obj["title"]


def download_zip(url):
    """Download a ZIP file."""
    headers = {
        'Authorization': 'token %s' % settings.GITHUB_TOKEN,
        'Accept': 'application/octet-stream',
    }
    response = requests.get(url, headers=headers)
    return ZipFile(io.BytesIO(response.content))


def extract_zip(zipfile):
    """Extract zip in memory and yields (filename, file-like object) pairs."""
    with zipfile as thezip:
        for zipinfo in thezip.infolist():
            with thezip.open(zipinfo) as thefile:
                if not zipinfo.filename.startswith('.') and not zipinfo.filename.startswith('__MACOSX') and not zipinfo.is_dir():
                    yield thefile


def update_or_create_tags(observer, repo, tag=None, type_to_update=None):
    """Create or update tags."""
    observer.update_state(
        state='PROGRESS',
        meta='Retrieving data and media from Github'
    )
    git = GithubAPI(repo)

    if tag:
        data, media = git.get_data(tag)

        if type_to_update == "ssot":
            populate_media(observer, media, tag)
        populate_data(observer, data, tag)
        populate_index(observer, tag, type_to_update)

        observer.update_state(
            state='SUCCESS',
            meta='All tasks complete'
        )

    return True


def populate_media(observer, media, tag):
    """Use ZIP data to create reference download objects."""
    observer.update_state(
        state='PROGRESS',
        meta='Creating reference downloads'
    )

    if media:
        for item in extract_zip(download_zip(media.url)):
            output_path = os.path.join(settings.REFERENCE_DOWNLOAD_ROOT, item.name)
            if not settings.AZURE_ACCOUNT_NAME:
                output_dir = os.path.dirname(output_path)
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
            if default_storage.exists(output_path):
                default_storage.delete(output_path)
            default_storage.save(output_path, ContentFile(item.read()))

    else:
        raise ValueError('No data available for tag: %s' % tag)


def populate_data(observer, data, tag):
    """Use ZIP data to create reference data objects."""
    observer.update_state(
        state='PROGRESS',
        meta='Data retrieved, updating database'
    )

    if data:
        for item in extract_zip(download_zip(data.url)):
            if os.path.splitext(item.name)[1] == ".html":
                ssot_path = os.path.dirname(item.name)
                ReferenceData.objects.update_or_create(
                    ssot_path=ssot_path,
                    tag=tag,
                    defaults={'data': item.read().decode('utf-8')},
                )

    else:
        raise ValueError('No data available for tag: %s' % tag)


def child_equals_object(child_page, object):
    """Check whether a page has changed by comparing it to a Reference Data object."""
    if child_page.tag != object.tag:
        return False
    if getattr(child_page, "data_{}".format(object.language)) != object.data:
        return False
    return True


def create_or_update_from_object(parent_page, page_model, object):
    """Create ActivityStandardPage from ReferenceData object."""
    try:
        child_page = page_model.objects.get(
            ssot_path=object.ssot_path
        )
        if not child_equals_object(child_page, object):
            setattr(child_page, "data_{}".format(object.language), object.data)
            child_page.tag = object.tag
            child_page.locked = True
            child_page.locked_by = None
            child_page.save_revision().publish()
    except page_model.DoesNotExist:
        child_page = page_model(
            ssot_path=object.ssot_path,
            title=object.name,
            heading=object.name,
            slug=slugify(object.name),
            tag=object.tag
        )
        setattr(child_page, "data_{}".format(object.language), object.data)
        child_page.locked = True
        child_page.locked_by = None
        parent_page.add_child(instance=child_page)
        child_page.save_revision().publish()
    return child_page


def recursive_create(page_model, object_pool, parent_page, parent_path, recursed_page_paths):
    """Recursively create ActivityStandardPage objects."""
    objects = object_pool.filter(parent_path=parent_path)
    for object in objects:
        child_page = create_or_update_from_object(parent_page, page_model, object)
        if child_page.ssot_path not in recursed_page_paths:
            recursed_page_paths.append(child_page.ssot_path)
            recursive_create(page_model, object_pool, child_page, child_page.ssot_path, recursed_page_paths)
    return True


def recursive_create_menu(parent_page):
    """Recursively create reference menu."""
    page_obj = {
        "depth": parent_page.depth,
        "title": parent_page.title,
        "pk": parent_page.pk,
        "ssot_path": parent_page.specific.ssot_path,
        "meta_order": parent_page.specific.meta_order,
        "children": list()
    }
    page_children = parent_page.get_children()
    if len(page_children) == 0:
        return page_obj
    for page_child in page_children:
        page_obj["children"].append(
            recursive_create_menu(page_child)
        )
    if list(filter(page_obj["ssot_path"].endswith, SORT_ORDER.keys())):
        page_obj["children"].sort(key=iati_order)
    else:
        page_obj["children"].sort(key=default_order)
    return page_obj


def populate_index(observer, tag, type_to_update):
    """Use ReferenceData objects to populate page index."""
    observer.update_state(
        state='PROGRESS',
        meta='Populating index'
    )

    recursed_page_paths = []

    new_object_paths = set(ReferenceData.objects.filter(tag=tag).order_by().values_list('ssot_path'))
    old_object_paths = set(ReferenceData.objects.exclude(tag=tag).order_by().values_list('ssot_path'))
    to_delete = [item[0] for item in (old_object_paths - new_object_paths)]

    if type_to_update == "guidance":
        StandardGuidanceTypes.objects.all().delete()
        StandardGuidancePage.objects.filter(ssot_path__in=list(to_delete)).delete()

        standard_page = GuidanceAndSupportPage.objects.live().first()
        ssot_root_page = StandardGuidanceIndexPage.objects.live().descendant_of(standard_page).first()
        if not ssot_root_page:
            ssot_root_page = StandardGuidanceIndexPage(
                title="Standard Guidance",
                heading="Standard Guidance",
                slug="standard-guidance",
                section_summary="Use this section to find guidance on how to publish and interpret data on specific topics according to the IATI Standard.",
                button_link_text="Search the Standard guidance pages"
            )
            standard_page.add_child(instance=ssot_root_page)
            ssot_root_page.save_revision().publish()
        recursive_create(StandardGuidancePage, ReferenceData.objects.filter(tag=tag), ssot_root_page, "guidance", recursed_page_paths)

    elif type_to_update == "ssot":
        menu_json = []

        ActivityStandardPage.objects.filter(ssot_path__in=list(to_delete)).exclude(ssot_root_slug="developer").delete()

        standard_page = IATIStandardPage.objects.live().first()
        ssot_roots = [roots[0] for roots in ReferenceData.objects.filter(tag=tag).order_by().values_list('ssot_root_slug').distinct() if roots[0] not in ["guidance", "developer"]]

        for ssot_root in ssot_roots:
            objects = ReferenceData.objects.filter(tag=tag, ssot_path=ssot_root)
            for object in objects:
                ssot_root_page = create_or_update_from_object(standard_page, ActivityStandardPage, object)
            ssot_root_page.title = ssot_root
            ssot_root_page.slug = slugify(ssot_root)
            ssot_root_page.save_revision().publish()
            recursive_create(ActivityStandardPage, ReferenceData.objects.filter(tag=tag), ssot_root_page, ssot_root_page.ssot_path, recursed_page_paths)
            menu_json.append(recursive_create_menu(ssot_root_page))

            ReferenceMenu.objects.update_or_create(
                tag=tag,
                menu_type="ssot",
                defaults={'menu_json': menu_json},
            )
    else:
        menu_json = []

        ActivityStandardPage.objects.filter(ssot_path__in=list(to_delete), ssot_root_slug="developer").delete()

        standard_page = GuidanceAndSupportPage.objects.live().first()
        ssot_root = "developer"

        objects = ReferenceData.objects.filter(tag=tag, ssot_path=ssot_root)
        for object in objects:
            ssot_root_page = create_or_update_from_object(standard_page, ActivityStandardPage, object)
        ssot_root_page.title = ssot_root
        ssot_root_page.slug = slugify(ssot_root)
        ssot_root_page.save_revision().publish()
        recursive_create(ActivityStandardPage, ReferenceData.objects.filter(tag=tag), ssot_root_page, ssot_root_page.ssot_path, recursed_page_paths)
        menu_json.append(recursive_create_menu(ssot_root_page))

        ReferenceMenu.objects.update_or_create(
            tag=tag,
            menu_type="developer",
            defaults={'menu_json': menu_json},
        )

    call_command('update_index')
