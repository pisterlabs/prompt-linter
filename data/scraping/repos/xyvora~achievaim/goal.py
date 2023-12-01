from bson import ObjectId
from fastapi import HTTPException
from starlette.status import (
    HTTP_204_NO_CONTENT,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from app.api.deps import CurrentUser, logger
from app.core.config import config
from app.core.utils import APIRouter, process_openai_to_smart_goal
from app.exceptions import (
    DuplicateGoalError,
    InvalidApiKeyError,
    InvalidTemperatureError,
    NoGoalsFoundError,
    NoRecordsDeletedError,
    NoRecordsUpdatedError,
    QuotaExceededError,
)
from app.models.smart_goal import GoalSuggestionCreate, SmartGoal
from app.models.user import Goal, GoalCreate
from app.services.goal_service import create_goal as create_goal_service
from app.services.goal_service import delete_goal_by_id as delete_goal_by_id_service
from app.services.goal_service import delete_goal_by_name as delete_goal_by_name_service
from app.services.goal_service import get_goal_by_id as get_goal_by_id_service
from app.services.goal_service import get_goal_by_name as get_goal_by_name_service
from app.services.goal_service import get_goals_by_user_id
from app.services.goal_service import update_goal as update_goal_service
from app.services.openai import generate_smart_goal

router = APIRouter(tags=["Goal"], prefix=f"{config.V1_API_PREFIX}/goal")


@router.post("/")
async def create_goal(goal: GoalCreate, current_user: CurrentUser) -> list[Goal]:
    """Add a new goal."""
    logger.info("Creating goal for user %s", current_user.id)
    try:
        goals = await create_goal_service(ObjectId(current_user.id), goal)
    except DuplicateGoalError:
        logger.error("Goal with the name %s already exists for user %s", goal.goal, current_user.id)
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Goal with the name {goal.goal} already exists",
        )
    except ValueError as e:  # pragma: no cover
        if "Goal is required" in str(e):
            logger.info("No goal provided: %s", e)
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Goal is required")
        if "Goal IDs must be unique" in str(e):
            logger.info("Goal IDs must be unique: %s", e)
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Goal IDs must be unique")
        if "Goal names must be unique" in str(e):
            logger.info("Goal names must be unique: %s", e)
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST, detail="Goal names must be unique"
            )
        logger.info("An error occurred %s", e)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while adding the goal",
        )
    except Exception as e:  # pragma: no cover
        logger.error("An error occurred while adding the goal: %s", e)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while adding the goal",
        )

    return goals


@router.delete("/{goal_id}", response_model=None, status_code=HTTP_204_NO_CONTENT)
async def delete_goal(goal_id: str, current_user: CurrentUser) -> None:
    """Delete a user's goal by ID."""
    try:
        await delete_goal_by_id_service(ObjectId(current_user.id), goal_id)
    except NoGoalsFoundError:
        logger.info("User %s has no goals", current_user.id)
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="No goals found for user")
    except NoRecordsDeletedError:
        logger.info("No goal with the id %s found for user %s", goal_id, current_user.id)
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail=f"No goal with the id {goal_id} found"
        )


@router.delete("/goal-name/{goal_name}", response_model=None, status_code=HTTP_204_NO_CONTENT)
async def delete_goal_by_name(goal_name: str, current_user: CurrentUser) -> None:
    """Delete a user's goal by name."""
    try:
        await delete_goal_by_name_service(ObjectId(current_user.id), goal_name)
    except NoGoalsFoundError:
        logger.info("User %s has no goals", current_user.id)
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="No goals found for user")
    except NoRecordsDeletedError:
        logger.info("No goal with the name %s found for user %s", goal_name, current_user.id)
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail=f"No goal with the name {goal_name} found"
        )


@router.get("/")
async def get_user_goals(current_user: CurrentUser) -> list[Goal] | None:
    """Get goals for a user."""
    logger.info("Getting goals for user %s", current_user.id)
    goals = await get_goals_by_user_id(ObjectId(current_user.id))

    return goals


@router.get("/{goal_id}")
async def get_goal_by_id(goal_id: str, current_user: CurrentUser) -> Goal:
    """Get a specifiic goal by goal ID."""
    goal = await get_goal_by_id_service(ObjectId(current_user.id), goal_id)

    if not goal:
        logger.info("No goal named %s found for ID %s", goal_id, current_user.id)
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=f"No goal ID {goal_id} found")

    return goal


@router.get("/goal-name/{goal_name}")
async def get_goal_by_name(goal_name: str, current_user: CurrentUser) -> Goal:
    """Get a specifiic goal."""
    goal = await get_goal_by_name_service(ObjectId(current_user.id), goal_name)

    if not goal:
        logger.info("No goal named %s found for user %s", goal_name, current_user.id)
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail=f"No goal named {goal_name} found"
        )

    return goal


@router.put("/")
async def update_goal(goal: Goal, current_user: CurrentUser) -> list[Goal]:
    """Update a goal's information."""
    try:
        return await update_goal_service(ObjectId(current_user.id), goal)
    except NoGoalsFoundError:
        logger.info("User %s has no goals", current_user.id)
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="No goals found for user")
    except NoRecordsUpdatedError:  # pragma: no cover
        logger.info("No goal with the id %s found for user %s", goal.id, current_user.id)
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail=f"No goal with the id {goal.id} found"
        )
    except DuplicateGoalError:  # pragma: no cover
        logger.error("Goal with the name %s already exists for user %s", goal.goal, current_user.id)
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Goal with the name {goal.goal} already exists",
        )


@router.post("/openai-goal")
async def openai_goal(goal: GoalSuggestionCreate, current_user: CurrentUser) -> SmartGoal:
    """Get goal suggestions from OpenAI."""
    logger.info("Getting goal suggestions from OpenAI for user %s", current_user.id)
    try:
        response = await generate_smart_goal(
            goal.goal, model=goal.model, temperature=goal.temperature
        )
    except InvalidTemperatureError as e:
        logger.info("An error occurred %s", e)
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Temperature must be between 0 and 2",
        )
    except InvalidApiKeyError as e:
        logger.info("An error occurred %s", e)
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    except QuotaExceededError as e:
        logger.info("An error occurred %s", e)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred generating the goal",
        )
    except Exception as e:  # pragma: no cover
        logger.error("An error occurred while generating the goal suggestions: %s", e)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the goal suggestions",
        )

    return process_openai_to_smart_goal(response, goal)
