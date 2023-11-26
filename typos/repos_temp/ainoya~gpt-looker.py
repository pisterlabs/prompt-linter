"""
        Given an input question, first create a syntactically correct JSON. The JSON is Looker SDK's run_inline_query function's models.WriteQuery argument. Do not use "fields": ["*"] in the JSON. Field names must include the view name. For example, fields: ["pet.id"]. The JSON must include the view name. For example, "view": "pet".

        # LookML Reference

        ```
        {context}
        ```

        # Question
        {question}"""