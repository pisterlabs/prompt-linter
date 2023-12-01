from typing import List
from objects.issue import Issue
from colorama import init, Fore, Back, Style
from datetime import datetime
from openpyxl import Workbook
from openpyxl.worksheet.hyperlink import Hyperlink
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Font, Alignment, numbers
from openpyxl.utils import get_column_letter, quote_sheetname
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

class Epic:
    def __init__(self, id, key, summary, create_date):
        self.id = id
        self.key = key
        self.summary = summary
        self.create_date = datetime.strptime(create_date, "%Y-%m-%dT%H:%M:%S.%f%z")
        self.issues: List[Issue] = []
        self.sub_labels = []
        self.team = ""
        self.estimate = 0
        self.issues_with_points = 0
        self.issues_points = 0
        self.issues_with_no_points = 0
        self.description = ""
        self.comments = []

    def add_issue(self, issue):
        self.issues.append(issue)   

    def add_sub_label(self, sub_label):
        self.sub_labels.append(sub_label)
    
    def add_comment(self, comment):
        self.comments.append(comment)

    def set_team(self, team):
        self.team = team
    
    def set_estimate(self, estimate):
        self.estimate = estimate

    def set_issues_with_points(self, issues_with_points):
        self.issues_with_points = issues_with_points
    
    def set_issues_points(self, issues_points):
        self.issues_points = issues_points
    
    def set_issues_with_no_points(self, issues_with_no_points):
        self.issues_with_no_points = issues_with_no_points

    def set_description(self, description):
        self.description = description

    def get_sublevels(self):
        sub_label_print = ""
        count_label = 0
        for sub_label in self.sub_labels:
            if (count_label == 0):
                sub_label_print += sub_label
                count_label += 1
            else:
                sub_label_print += sub_label + ", "
        return sub_label_print

    def print_epic(self):
        print(Fore.YELLOW + Style.BRIGHT + 
              "Epic-" + str(self.key) + ": " + Fore.LIGHTYELLOW_EX + Style.NORMAL + self.summary + Fore.WHITE + 
              " Created: " + str(self.create_date.month) + "/" + str(self.create_date.day) + "/" + str(self.create_date.year) +
              "\n    " + Fore.BLUE + Style.BRIGHT + self.team + Fore.RED + Style.NORMAL + " Estimate: " + str(self.estimate) +
              "\n    " + Fore.YELLOW + Style.NORMAL + f"{self.issues_with_points} issues have points and {self.issues_with_no_points} don't. {self.issues_points} points total." +
              "\n    " + Fore.MAGENTA + " Sub Labels: " + Fore.WHITE + str(self.sub_labels) + Style.RESET_ALL)

