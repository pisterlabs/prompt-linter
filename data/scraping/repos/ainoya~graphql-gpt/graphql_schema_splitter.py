from typing import List
from langchain.text_splitter import TextSplitter
import yaml


class HasuraGraphQLSchemaSplitter(TextSplitter):
    """Attempts to split the text along Hasura GraphQL Schema syntax."""

    def __init__(self, **kwargs):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)

    # Extract data from schema whose kind is OBJECT
    def split_text(self, graphql_schema) -> List[str]:
        objects = [x for x in graphql_schema['__schema']
                   ['types'] if x['kind'] == 'OBJECT']

        # Limit the keys in the type of the args array of the fields to kind and name only
        for obj in objects:
            for field in obj['fields']:
                # Delete the type key
                if 'type' in field:
                    del field['type']
                if 'isDeprecated' in field:
                    del field['isDeprecated']
                # If the description is `An array relationship`, delete the args
                if field['description'] == 'An array relationship':
                    del field['args']
                    continue
                for arg in field['args']:
                    arg['type'] = {k: arg['type'][k] for k in ['kind', 'name']}
                    # Delete the defaultvalue key
                    if 'defaultValue' in arg:
                        del arg['defaultValue']
            # Delete the isDeprecated key
            if 'isDeprecated' in field:
                del field['isDeprecated']

        # Delete the object with the name of query_root
        objects = [x for x in objects if x['name'] != 'query_root']

        # Remove the key recursively whose value is null or an empty array in the content of objects
        def remove_null(obj):
            if isinstance(obj, dict):
                for key in list(obj.keys()):
                    if obj[key] is None or obj[key] == []:
                        del obj[key]
                    else:
                        remove_null(obj[key])
            elif isinstance(obj, list):
                for item in obj:
                    remove_null(item)
            return obj

        objects = remove_null(objects)
        # Convert the object to a string array of yamlized
        return [yaml.dump(obj) for obj in objects]
