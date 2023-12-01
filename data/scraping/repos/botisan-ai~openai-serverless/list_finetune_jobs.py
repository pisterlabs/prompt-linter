import os
import json
import traceback
import openai
from typing import Any, Dict
from dotenv import dotenv_values

config = {
    **dotenv_values(os.path.join(os.path.dirname(__file__), '..', '.env')),
    **os.environ,
}

openai.api_key = config.get('OPENAI_API_KEY')


def handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    try:
        uid = event.get('requestContext', {}).get(
            'authorizer', {}).get('principalId')

        finetunes_response = openai.FineTune.list()

        return {
            'statusCode': 200,
            'body': json.dumps(finetunes_response),
            'headers': {
                'Content-Type': 'application/json',
            },
        }
    except Exception:
        traceback.print_exc()
        return {
            'statusCode': 400,
            'body': json.dumps({'message': 'error occurred, please check logs'}),
            'headers': {
                'Content-Type': 'application/json',
            },
        }


if __name__ == '__main__':
    print(config)
