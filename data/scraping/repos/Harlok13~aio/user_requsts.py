from collections import namedtuple
from typing import Union

from sqlalchemy import text, update, select
from sqlalchemy.orm import sessionmaker

from bot_ai.data.schemas.user_model import User
# from bot_ai.lexicon.tokens_pay_lexicon import TOKENS

ProfileInfo = namedtuple('ProfileInfo', ('nickname', 'question_count', 'token_count', 'days_left', 'status'))


class UserRequest:
    def __init__(self, session: sessionmaker):
        self._session: sessionmaker = session

    @staticmethod
    def _set_days_left(
            current_status: str,
            purchased_status: str,
            days_left: int,
            days_sub: int
    ) -> int:
        """
        If the current status is the same as the one being purchased, then the number
        of subscription days increases. Otherwise, the counter is reset to zero and takes
        the standard value for the subscription
        :param current_status: current status of the user subscription
        :param purchased_status: purchased status of the user subscription
        :param days_left: days left in the subscription
        :param days_sub: number of days in the purchased subscription
        :return: days left in the subscription
        """
        if current_status == purchased_status:
            return days_left + days_sub
        return days_sub

    @staticmethod
    def _set_tokens(
            current_status: str,
            purchased_status: str,
            current_tokens: int,
            purchased_tokens
    ) -> Union[str, int]:
        """
        If the current status is unlimited, then the number of tokens is unlimited.
        if the current status matches the one being acquired, then the number of tokens
        does not change. If the status is different, the number of tokens takes the
        value that is in the subscription
        :param current_status: current status of the user subscription
        :param purchased_status: purchased status of the user subscription
        :param current_tokens: current number of tokens of the user
        :param purchased_tokens: purchased number of tokens of the user
        :return: number of tokens
        """
        if current_status == purchased_status:
            return current_tokens
        return purchased_tokens

    @staticmethod
    def _spent_tokens_counter(current_tokens: int, openai_spent_tokens: int) -> int:
        """
        Multiplies the number of tokens spent on the answer from openai
        :param current_tokens: current number of tokens of the user
        :param openai_spent_tokens: number of tokens spent by openai
        :return: number of tokens spent on the answer, it is total result
        """
        token_multiplier = 5
        total_spent_tokens: int = current_tokens - openai_spent_tokens * token_multiplier
        return total_spent_tokens

    @staticmethod
    def _check_user_tokens(user_id: int, current_user_tokens: int) -> bool:
        """
        Check if the user has enough tokens to purchase the subscription
        :param user_id: user id
        :param current_user_tokens: current number of tokens of the user
        :return: True if the user has enough tokens to purchase the subscription
        """
        if current_user_tokens > 0:
            return True
        return False

    async def set_subscription(
            self,
            purchased_status: str,
            user_id: int,
            days_sub: int,
            purchased_tokens: Union[int, str],
    ) -> None:
        """
        Set paid subscription status for user in database
        :param purchased_tokens: number of tokens purchased
        :param days_sub: number of days of subscription
        :param purchased_status: subscription type to be set
        :param user_id: just a user id in telegram
        :return:
        """
        print('set_default_sub', purchased_status, user_id)
        async with self._session.begin():
            stmt_get: ChunckedIteratorResult = await self._session.execute(  # type: ignore
                select(User.token_count, User.days_left, User.status).where(User.user_id == user_id)
            )
            token_count, days_left, current_status = stmt_get.fetchall()[0]

            stmt_set = update(User).where(User.user_id == user_id).values(
                status=purchased_status,
                days_left=self._set_days_left(
                    current_status,
                    purchased_status,
                    days_left,
                    days_sub
                ),
                token_count=self._set_tokens(
                    current_status,
                    purchased_status,
                    token_count,
                    purchased_tokens
                )
            )

        await self._session.execute(stmt_set)  # type: ignore
        await self._session.commit()  # type: ignore

    async def get_profile_info(self, user_id: int) -> ProfileInfo:
        """Get profile info from database"""
        print('get_profile_info', user_id)
        stmt_get: ChunckedIteratorResult = await self._session.execute(  # type: ignore
            select(
                User.nickname, User.question_count, User.token_count, User.days_left, User.status
            ).where(User.user_id == user_id)
        )

        profile: ProfileInfo = ProfileInfo(*stmt_get.fetchall()[0])
        return profile

    async def increase_question_count(self, user_id: int) -> None:
        """
        Increase question count for user in database
        :param user_id: just a user id in telegram
        :return:
        """
        async with self._session.begin():
            stmt_get: ChunckedIteratorResult = await self._session.execute(  # type: ignore
                select(User.question_count).where(User.user_id == user_id)
            )
            question_count: int = stmt_get.fetchone()[0] + 1
            stmt_set = update(User).where(User.user_id == user_id).values(
                question_count=question_count
            )
        await self._session.execute(stmt_set)  # type: ignore
        await self._session.commit()  # type: ignore

    async def decrease_user_tokens(self, user_id: int, openai_spent_tokens: int) -> None:
        """
        Decrease number of tokens for user in database. If number of tokens less than 0, then
        number of tokens is set to 0
        :param user_id: just a user id in telegram
        :param openai_spent_tokens: number of tokens spent by openai
        :return:
        """
        async with self._session.begin():
            stmt_get: ChunckedIteratorResult = await self._session.execute(  # type: ignore
                select(User.token_count).where(User.user_id == user_id)
            )
            current_tokens: int = stmt_get.fetchall()[0][0] - 1
            result_tokens: int = self._spent_tokens_counter(current_tokens, openai_spent_tokens)
            if result_tokens < 0:
                result_tokens = 0
            stmt_set = update(User).where(User.user_id == user_id).values(token_count=result_tokens)

        await self._session.execute(stmt_set)  # type: ignore
        await self._session.commit()  # type: ignore

    async def check_user_tokens(self, user_id: int) -> bool:
        """
        Check if the user can send a question to the bot
        :param user_id: just a user id in telegram
        :return: True if the user can send a question, otherwise False
        """
        async with self._session.begin():
            stmt_get: ChunckedIteratorResult = await self._session.execute(  # type: ignore
                select(User.token_count).where(User.user_id == user_id)
            )
            token_count: int = stmt_get.fetchone()[0]
        return token_count > 0

    async def set_tokens(self, user_id: int, purchased_tokens: int, ) -> None:
        """
        Set tokens for user in database
        :param user_id: just a user id in telegram
        :param purchased_tokens: quantity of purchased tokens
        :return:
        """
        async with self._session.begin():
            stmt_get: ChunckedIteratorResult = await self._session.execute(  # type: ignore
                select(User.token_count).where(User.user_id == user_id)
            )
            await self._session.execute(  # type: ignore
                update(User).where(User.user_id == user_id).values(
                    token_count=stmt_get.fetchone()[0] + purchased_tokens
                )
            )
        await self._session.commit()  # type: ignore

    async def get_user_status(self, user_id: int) -> str:
        """
        Get user status from database
        :param user_id: just a user id in telegram
        :return: current user status
        """
        async with self._session.begin():
            stmt_get: ChunckedIteratorResult = await self._session.execute(  # type: ignore
                select(User.status).where(User.user_id == user_id)
            )
            current_status = stmt_get.fetchone()[0]
        return current_status
