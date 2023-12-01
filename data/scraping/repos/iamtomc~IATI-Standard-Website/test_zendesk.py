"""A module of unit tests for guidance and support."""
import pytest
from django.http import HttpRequest

from guidance_and_support.zendeskhelper import generate_ticket


LEGITIMATE_USER = HttpRequest()
LEGITIMATE_USER.path = "/en/a-test-path"
LEGITIMATE_USER.POST = {
    'phone': '',
    'email': 'test@user.com',
    'textarea': 'A very serious matter.',
    'name': 'A legitimate user'
}
LEGITIMATE_USER.expected_output = {
    'request': {
        'requester': {
            'name': 'A legitimate user',
            'email': 'test@user.com'
        },
        'comment': {
            'body': 'A request was sent from /en/a-test-path.\nA very serious matter.'
        },
        'subject': 'Automated request from A legitimate user'
    }
}


SPAM_BOT = HttpRequest()
SPAM_BOT.path = "/en/a-test-path"
SPAM_BOT.POST = {
    'phone': '555-555-5555',
    'email': 'test@user.com',
    'textarea': 'A very serious matter.',
    'name': 'A legitimate user'
}
SPAM_BOT.expected_output = False


@pytest.mark.parametrize("user", [LEGITIMATE_USER, SPAM_BOT])
def test_generate_ticket(user):
    """Test a ticket from a valid user and a spam bot."""
    ticket = generate_ticket(user)
    assert ticket == user.expected_output
