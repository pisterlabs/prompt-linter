"""Module of context processors for the IATI Standard Website."""

import itertools
from django.conf import settings
from wagtail.models import Page, Site
from navigation.models import (
    PrimaryMenu,
    UtilityMenu,
    UsefulLinks,
)
from notices.models import GlobalNotice, PageNotice
from search.models import SearchPage
from guidance_and_support.models import SupportPage
from iati_standard.models import IATIStandardPage


def get_current_page(request):
    """Try and get a page safely."""
    try:
        # this try is here to protect against 500 errors when there is a 404 error
        # taken from https://github.com/torchbox/wagtail/blob/master/wagtail/wagtailcore/views.py#L17
        path_components = [component for component in request.path.split('/') if component]
        current_page, args, kwargs = Site.find_for_request(request).root_page.specific.route(request, path_components[1:])
        return current_page
    except Exception:
        return None


def captchakey(request):
    """Return the public captcha key."""
    return {'RECAPTCHA_KEY': settings.RECAPTCHA_PUBLIC_KEY}


def globals(request):
    """Return a global context dictionary for use by templates."""
    current_page = get_current_page(request)
    if current_page is not None:
        current_site = current_page.get_site()
        hostname = current_site.hostname
    else:
        hostname = None
    search_page = SearchPage.objects.all().live().first()
    support_page = SupportPage.objects.all().live().first()
    standard_page = IATIStandardPage.objects.live().first()

    return {
        'global': {
            'primary_menu': construct_nav(PrimaryMenu.for_request(request).primary_menu_links.all(), current_page),
            'utility_menu': construct_nav(UtilityMenu.for_request(request).utility_menu_links.all(), current_page),
            'useful_links': UsefulLinks.for_request(request).useful_links.all(),
            'twitter_handle': settings.TWITTER_HANDLE,
            'standard_page': standard_page,
            'search_page_url': search_page.url if search_page else '',
            'support_page_url': support_page.url if support_page else '',
            'global_notice': GlobalNotice.get_notice(request),
            'page_notice': PageNotice.get_notice(current_page, request),
            'hero_feed_srcs': 'fill-300x620 1w, fill-499x450 300w, fill-780x400 500w, fill-1000x650 780w, fill-1200x650 1000w, width-1400 1200w',
            'hero_srcs': 'width-300 300w, width-400 400w, width-500 500w',
            'hero_sizes': '(max-width: 499px) 0vw, (min-width: 780px) 50vw, (min-width: 1000px) 40vw',
            'tool_logo_srcs': 'width-150 150w',
            'tool_logo_sizes': '(max-width: 499px) 30vw, (max-width: 779px) 20vw, (max-width: 999px) 15vw, (min-width: 1000px) 13vw',
            'tool_listing_logo_srcs': 'width-80 80w, width-150 150w, width-40 40w, width-60 60w',
            'tool_listing_logo_sizes': '(max-width: 499px) 16vw, (max-width: 779px) 20vw, (min-width: 780px) 5vw',
            'case_study_srcs': 'fill-240x110 1w, fill-440x200 300w, fill-720x315 500w, fill-550x270 780w, fill-675x325 1000w',
            'people_srcs': 'max-240x240 240w, max-440x440 440w, max-190x190 190w, max-150x150 150w',
            'people_sizes': '(max-width: 499px) 100vw, (max-width: 779px) 25vw, (max-width: 1199px) 15vw, (min-width: 1200px) 10vw',
            'org_logo_srcs': 'width-32 32w, width-40 40w, width-60 60w, width-68 68w',
            'org_logo_sizes': '(max-width: 499px) 12vw, (max-width: 779px) 8vw, (min-width: 780px) 6vw',
            'event_srcs': 'width-440 440w, width-720 720w, width-600 600w, width-500 500w',
            'event_sizes': '(max-width: 779px) 90vw, (max-width: 999px) 60vw, (min-width: 1000px) 40vw',
            'get_involved_srcs': 'max-240x200 1w, max-440x200 300w, max-720x200 500w, max-300x200 780w, max-360x200 1000w',
            'chair_srcs': 'fill-440x440 440w, fill-180x180 180w, fill-70x70 70w, fill-90x90 90w',
            'chair_sizes': '(max-width: 499px) 90vw, (max-width: 779px) 23vw, (min-width: 780px) 7vw',
            'member_srcs': 'width-240 240w, width-170 170w, width-130 130w, width-160 160w',
            'member_sizes': '(max-width: 499px) 48vw, (max-width: 779px) 21vw, (min-width: 780px) 13vw',
            'guidance_srcs': 'width-440 440w, width-720 720w, width-800 800w, width-320 320w',
            'guidance_sizes': '(max-width: 779px) 90vw, (max-width: 949px) 85vw, (max-width: 999px) 40vw, (max-width: 1199px) 32vw, (min-width: 1200px) 26vw',
            'media_floated_srcs': 'width-200 200w, width-315 315w',
            'media_floated_sizes': '(max-width: 779px) 40vw, (max-width: 1199px) 32vw, (min-width: 1200px) 26vw',
            'media_centred_srcs': 'width-440 440w, width-700 700w, width-420 420w',
            'media_centred_sizes': '(max-width: 779px) 90vw, (max-width: 999px) 42vw, (min-width: 1000px) 35vw',
            'media_full_srcs': 'width-440 440w, width-700 700w',
            'media_full_sizes': '(max-width: 779px) 90vw, (max-width: 999px) 70vw, (max-width: 1199px) 58vw, (min-width: 1200px) 55vw',
            'getting_started_srcs': 'fill-240x200 1w, fill-440x200 300w, fill-720x200 500w, fill-300x200 780w, fill-360x200 1000w',
            'iati_tools_srcs': 'width-80 80w, width-150 150w, width-110 110w',
            'iati_tools_sizes': '(max-width: 499px) 16vw, (max-width: 779px) 20vw, (min-width: 780px) 10vw',
            'iati_news_srcs': 'width-440 440w, width-720 720w, width-420 420w, width-130 130w, width-155 155w',
            'iati_news_sizes': '(max-width: 779px) 90vw, (max-width: 949px) 44vw, (min-width: 950px) 10vw',
            'testimonial_srcs': 'fill-55x60 55w, fill-80x90 80w, fill-50x60 70w, fill-60x70 60w',
            'testimonial_sizes': '(max-width: 779px) 11vw, (min-width: 780px) 5vw',
            'featured_srcs': 'fill-120x250 950w',
            'news_list_srcs': 'width-440 440w, width-720 720w, width-570 570w, width-210 210w, width-260 260w',
            'news_list_sizes': '(max-width: 779px) 90vw, (max-width: 949px) 60vw, (min-width: 950px) 21vw',
            'news_featured_srcs': 'width-440 440w, width-720 720w, width-600 600w, width-500 500w',
            'news_featured_sizes': '(max-width: 779px) 90vw, (max-width: 999px) 60vw, (min-width: 950px) 41vw',
            'related_srcs': 'width-440 440w, width-720 720w, width-420 420w, width-155 155w, width-190 190w',
            'related_sizes': '(max-width: 779px) 90vw, (max-width: 949px) 44vw, (min-width: 950px) 15vw',
            'hostname': hostname,
        },
    }


def construct_nav(qs, current_page):
    """Construct a navigation menu."""
    nav = list(qs)
    for item in nav:
        item.active = False
        try:
            if Page.objects.filter(id=item.page.id).ancestor_of(current_page, inclusive=True).first():
                item.active = True
        except AttributeError:
            pass

    for a, b in itertools.combinations(nav, 2):
        if a.active and b.active:
            if a.page.depth > b.page.depth:
                b.active = False
            else:
                a.active = False

    return nav
