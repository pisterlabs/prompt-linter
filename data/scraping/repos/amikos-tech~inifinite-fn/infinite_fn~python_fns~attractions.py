import uuid

from func_ai.utils.llm_tools import OpenAIInterface


def get_attractions_for_location(location: str):
    """
    Returns a list of attractions for a given location. The function will return at most 10 attrachions.
    Each attraction will have a short description.

    :param location: The location for which to return attractions. E.g. "London"
    :return: Returns a list of attractions
    """
    llm_interface = OpenAIInterface()
    _resp = llm_interface.send(
        f"What are the best attractions in {location}? Give me at most 10 attractions. Provide a short description for each attraction.")
    return _resp['content']


attraction_bookings = {}


def book_attraction(location: str, name_for_booking: str, attraction_name: str, date_and_time: str,
                    persons: int) -> str:
    """
    Books an attraction for a given date and time.

    :param location: The location of the attraction
    :param name_for_booking: The name of the person making the booking.
    :param attraction_name: The name of the attraction to book
    :param date_and_time: The date and time to book the attraction for
    :param persons: The number of persons
    :return: Returns the booking number
    """
    booking_uuid = str(uuid.uuid4())
    attraction_bookings[booking_uuid] = {
        "location": location,
        "name_for_booking": name_for_booking,
        "attraction_name": attraction_name,
        "date_and_time": date_and_time,
        "persons": persons}
    return booking_uuid


def get_attraction_booking_by_id(booking_uuid: str) -> dict[str, any]:
    """
    Returns the booking for a given booking uuid.

    :param booking_uuid: The booking uuid
    :return: Returns the booking
    """
    return attraction_bookings[booking_uuid]


def get_attraction_booking_by_name(name_for_booking: str) -> dict[str, any]:
    """
    Returns the booking for a given booking uuid.

    :param booking_uuid: The booking uuid
    :return: Returns the booking
    """
    return next(iter(
        [booking for booking in attraction_bookings.values() if booking['name_for_booking'] == name_for_booking]))
