"""View defintions for the search app."""

from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.shortcuts import render
from wagtail.search.models import Query
from wagtail.core.models import Page

from about.models import AboutPage, AboutSubPage, CaseStudyPage, HistoryPage, PeoplePage
from contact.models import ContactPage
from events.models import EventPage
from guidance_and_support.models import GuidanceAndSupportPage, GuidanceGroupPage, \
    GuidancePage, KnowledgebasePage
from news.models import NewsPage
from home.models import StandardPage
from iati_standard.models import IATIStandardPage


def search(request):
    """Process a user input for a search query and return a page containing results."""
    per_page = 10
    searchable_models = [
        AboutPage, AboutSubPage, CaseStudyPage, HistoryPage,
        PeoplePage, ContactPage, EventPage, GuidanceAndSupportPage,
        GuidanceGroupPage, GuidancePage, KnowledgebasePage, NewsPage,
        StandardPage, IATIStandardPage,
    ]
    search_query = request.GET.get('query', None)
    page = request.GET.get('page', 1)

    # Search
    if search_query:
        search_results = [r for m in searchable_models
                          for r in m.objects.live().search(search_query)]
        query = Query.get(search_query)

        # Record hit
        query.add_hit()
    else:
        search_results = Page.objects.none()

    # Pagination
    paginator = Paginator(search_results, per_page)
    try:
        search_results = paginator.page(page)
    except PageNotAnInteger:
        search_results = paginator.page(1)
    except EmptyPage:
        search_results = paginator.page(paginator.num_pages)

    # Pagination range
    # TODO: currently the logic for this lives in home.models.AbstractIndexPage
    # this is a duplication that might be worth looking into and DRYing
    total_pages = search_results.paginator.num_pages

    range_start = search_results.number - 5 if search_results.number > 5 else 1
    if search_results.number < (total_pages - 4):
        range_end = search_results.number + 4
    else:
        range_end = total_pages

    return render(request, 'search/search.html', {
        'search_query': search_query,
        'search_results': search_results,
        'paginator_range': [i for i in range(range_start, range_end + 1)],
        'paginator': paginator,
    })
