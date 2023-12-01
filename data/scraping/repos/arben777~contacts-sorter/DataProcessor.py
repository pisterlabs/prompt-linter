from Contact import Contact , CategorizedContact
from typing import List
from OpenAICaller import OpenAICaller

class DataProcessor:

    @staticmethod
    def categorize_contacts(contacts: List[Contact]) -> List[CategorizedContact]:
        return [OpenAICaller.categorize_contact(contact) for contact in contacts]
    
    @staticmethod
    def sort_contacts(categorized_contacts: List[CategorizedContact]) -> List[CategorizedContact]:
        return sorted(categorized_contacts, key=lambda x: (x.industry, x.organization, -x.seniority))
