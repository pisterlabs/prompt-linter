from enum import Enum
from langchain.pydantic_v1 import BaseModel, Field

with open("./utils/exercises_uppercase.txt") as f:
    ALL_EXERCISES: list[str] = [e.replace("\n", "") for e in f.readlines()]

class ExerciseBodyArea(Enum):
    CHEST = 0
    SHOULDERS = 1
    BACK = 2
    ARMS = 3
    CORE = 4
    LEGS = 5

class ExerciseDifficulty(Enum):
    BEGINNER = 0
    INTERMEDIATE = 1
    EXPERT = 2

class Set(BaseModel):
    num_reps: int = Field("The number of reps to be done, if applicable. If it's a time-based exercise, \
                          set this field to 0")
    weight: int = Field("The weight (in lbs) to be used for the exercise, if applicable. If it's a \
                        time-based or body weight exercise, set this field to 0")
    time: float = Field("How long (in minutes) to perform this exercise for, if applicable. If it's a rep-based exercise, \
                        set this field to 0. Please try to make core exercises time-based instead of rep-based.")
    rest: int = Field("How long to rest after this set, in seconds")

class Exercise(BaseModel):
    name: str = Field(description="The name of the exercise")
    bodyareas: list[ExerciseBodyArea] = Field(description="The body areas that the exercise targets. \
                                Possible values are chest, shoulders, back, arms, core, and legs")
    difficulty: ExerciseDifficulty = Field(description="The difficulty level of the exercise. \
                                           Possible values are beginner, intermediate, and expert")
    sets: list[Set] = Field(description="The list of sets to be done for this exercise")


class Workout(BaseModel):
    exercises: list[Exercise] = Field(description="The list of exercises to be done for the workout")

class WorkoutSpecification(BaseModel):
    duration: int = Field(description="The duration of the workout, in minutes.")
    intensity_level: int = Field(description="The intensity level of the workout. Can be the following values: \
                                 0 for low, 1 for medium, or 2 for high.")
    bodyarea: int = Field(description="The part of the body that the user wants to exercise. Can be one of \
                          6 numerical values: 0 for chest, 1 for shoulders, 2 for back, 3 for arms, 4 for core, or 5 for legs.")
    hours_slept: float = Field(description="The number of hours that the user slept the previous night. If not provided,\
        set this field to 8.0")