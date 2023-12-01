import random

import openai


class Individual:
    def __init__(self, categories):
        self.categories = categories

    def get_notes(self):
        return self.categories[0].notes

    def is_scored(self):
        return all(category.is_scored() for category in self.categories)

    def score(self, verbose=False, n=3):
        for category in self.categories:
            print(f"====== Scoring {category.category_name()} ======")
            category.score(verbose=verbose, n=n)
            print(
                f"Average score for {category.category_name()}: {category.average_score()}/{category.best_possible_score()}"
            )

    def total_score(self):
        return sum(category.average_score() for category in self.categories)

    def normalized_score(self):
        return sum(category.normalized_score() for category in self.categories) / len(
            self.categories
        )

    def best_possible_score(self):
        return sum(category.best_possible_score() for category in self.categories)

    def make_recommendation(self, verbose=False):
        num_missing_points = self.best_possible_score() - self.total_score()

        # Pick a category to improve on, weighted by how many points they're missing
        # Categories with worse scores (more missing points) are probably easier to improve
        missing_point = random.uniform(0, num_missing_points)
        current = 0
        for category in self.categories:
            current += category.best_possible_score() - category.average_score()
            if current > missing_point:
                return category, category.make_recommendation(verbose=verbose)

    def apply_recommendation(self, category, recommendation, verbose=False):
        # Given the premise, conditioning info, and notes, apply the recommendation to make new notes
        prompt = category.apply_recommendation_prompt(recommendation)
        if verbose:
            print(prompt)
        application = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            n=1,
            temperature=1,
        )
        if verbose:
            print(application.choices[0].message.content)

        revised_notes_begin = "====== REVISED NOTES ======"
        revised_notes_end = "====== END REVISED NOTES ======"

        # Extract the revised notes from the response
        revised_notes = (
            application.choices[0]
            .message.content.split(revised_notes_begin)[1]
            .split(revised_notes_end)[0]
        )

        # Sanity check: make sure the revised notes are different from the original notes, and long enough to maybe be notes
        if (
            revised_notes == category.notes
            or len(revised_notes) < len(category.notes) / 2
        ):
            return self.apply_recommendation(category, recommendation, verbose=verbose)

        return revised_notes
