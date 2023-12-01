"""
You are given the task of generating insights for Fiscal Year Highlights from the annual report of the company. 

Given below is the output format, which has the subsections.
Write atleast 50 words for each subsection.
Incase you don't have enough info you can just write: No information available
---
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"performance_highlights": {"title": "Performance Highlights", "description": "Key performance metrics and achievements over the fiscal year.", "type": "string"}, "major_events": {"title": "Major Events", "description": "Highlight of significant events, acquisitions, or strategic shifts that occurred during the year.", "type": "string"}, "challenges_encountered": {"title": "Challenges Encountered", "description": "Challenges the company faced during the year and how they managed or overcame them.", "type": "string"}, "milestone_achievements": {"title": "Milestone Achievements", "description": "Milestones achieved in terms of projects, expansions, or any other notable accomplishments.", "type": "string"}}, "required": ["performance_highlights", "major_events", "challenges_encountered", "milestone_achievements"]}
```
---
"""