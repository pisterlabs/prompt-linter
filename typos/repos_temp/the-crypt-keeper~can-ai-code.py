"""
You are going to evaluate the results of language models on a {{language}} programming challenge: {{task}}
Automated tests have been used to verify corectness each solution produced, a detailed description of the results of each test will be provided.
For each model, you will be provided the code produced by the model and the result of all tests.
Compare and contrast the solutions each model produced.  Do not repeat any of the generated code back to me.  Highlight differences in solution approaches, test results, and provide a final summary of cohort performance on this challenge.

""""""
---
Model: {{id}}
Test Result: {{check_summary}}
Test Details:
{{passing_tests}}{{failing_tests}}
Code:
```{{language}}
{{code}}
```
""""""
---
Analysis:"""