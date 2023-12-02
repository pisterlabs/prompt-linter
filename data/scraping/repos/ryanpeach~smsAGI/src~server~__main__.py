import os

from flask import Flask, request
from langchain.schema import HumanMessage
from loguru import logger
from sqlalchemy.orm import sessionmaker
from twilio.rest import Client

from lib.sql import ENGINE, Goal, SuperAgent, ThreadItem, User
from server.agents.user_agent import UserAgent

# On startup, we need to create a couple objects
_Session = sessionmaker(bind=ENGINE)

app = Flask(__name__)
CONFIG = None


@app.route("/health")
async def health():
    return {
        "statusCode": 200,
        "body": "Healthy!",
    }


@app.route("/sms", methods=["POST"])
async def sms_reply():
    # Get the message from the request
    logger.info(request.values)
    incoming_msg = request.values.get("Body", "").lower()
    phone_number = incoming_msg["From"]
    user_msg = HumanMessage(content=incoming_msg)
    user = User.get_from_phone_number(session=session, phone_number=phone_number)
    if user is None:
        return {
            "statusCode": 404,
            "body": "User not found!",
        }
    with _Session() as session:
        primary_agent = User.get_primary_agent(session=session)
        ThreadItem.create(session=session, super_agent=primary_agent, msg=user_msg)
        user_agent = UserAgent(
            super_agent=primary_agent, session=session, config=CONFIG
        )
        await user_agent.arun(incoming_msg)

        # Reset wait_for_response upon receiving a message
        primary_agent.wait_for_response = False

        session.commit()

    return {
        "statusCode": 200,
        "body": "Message received!",
    }


if __name__ == "__main__":
    # On startup, we need to create a couple objects if they don't exist
    user_phone_number = os.environ["TWILIO_TO_PHONE_NUMBER"]
    with _Session() as session:
        user = User.get_from_phone_number(
            session=session, phone_number=user_phone_number
        )
        if user is not None:
            logger.info(f"Found admin user")
        else:
            logger.info("Creating admin user and agent.")
            user = User(name="admin", phone_number=user_phone_number)
            super_agent = SuperAgent(
                name="admin", user=user, is_active=True, is_primary=True
            )
            goal = Goal(
                super_agent=super_agent, objective="Ask the user what they want of you."
            )
            session.add(user)
            session.add(super_agent)
            session.commit()
    app.run(debug=True)
