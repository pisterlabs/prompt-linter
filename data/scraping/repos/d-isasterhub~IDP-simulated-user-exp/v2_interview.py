import openai
import pandas as pd

from v2_profiling import (
    UserProfile
)

from v2_interview_util import (
    get_msg,
    get_msg_with_image
)

from v2_interview_util_prompts import (
    SYSTEM,
    USER
)

def create_user_profiles(path_to_csv, n=5, selection='first') -> [UserProfile]:
    """Constructs n UserProfiles based on user data given in a .csv file
        
    Args:
        path_to_csv (str) : path to a csv file with user data
        n (int) : the number of users to create profiles for
        selection () : how to select the n users from the dataset
    
    Returns:
        [UserProfile] : a list of n UserProfile objects
    """
    selection_methods = ['random', 'first', 'last']

    if selection not in selection_methods:
        raise ValueError("Invalid selection method. Expected one of: %s" % selection_methods)
    
    safe_n = n

    if selection == 'first':
        # if we only want the first n rows, we don't need to read the whole file
        df = pd.read_csv(path_to_csv, nrows=n)
        
    elif selection == 'last':
        df = pd.read_csv(path_to_csv)
        df = df.tail(n)
        if df.size < n:
            safe_n = df.size
    else: 
        df = pd.read_csv(path_to_csv)
        df = df.sample(n)
        if df.size < n:
            safe_n = df.size
    
    df = df.rename(columns={"Q_8" : "Q_4"})

    userprofiles = []

    for i in range(safe_n):
        userprofiles.append(UserProfile(df.iloc[i].squeeze()))

    return userprofiles   

def personalize_profiling_prompt(profiles: [UserProfile], prompt: str) -> [str]:
    """Individualizes the given prompt based on info stored in a UserProfile
    
    Args:
        profiles ([UserProfiles]) : list of UserProfile objects holding relevant profiling info
        prompt (str) : generic prompt with placeholders for profiling info
    
    Returns:
        [str] : a list of individualized profiling prompts
    """

    PROFILINGS = []
    for p in profiles:
        PROFILINGS.append[p.profiling_prompt(SYSTEM)]
    return PROFILINGS


def conduct_interview():
    PROFILING = ""
    # https://platform.openai.com/docs/api-reference/chat/create?lang=python
    response = openai.ChatCompletion.create(
        model = "gpt-4-vision-preview",
        max_tokens = 300,
        messages = 
            get_msg(role="system", prompt=PROFILING) +\
            get_msg_with_image(role="user", prompt=USER, image="3-Crested.png")
    )
    actual_response = response["choices"][0]["message"]["content"] # have a string

# Question-Image mapping:
# 1. 0-Crested.png
# 2. 1-Crested.png
# 3. 2-Crested.png
# 4. 3-Crested.png
# 5. 4-Crested.png
# 6. 15-Least.png
# 7. 16-Least.png
# 8. 17-Least.png
# 9. 18-Least.png
# 10. 19-Least.png
# 11. 31-Parakeet.png
# 12. 32-Parakeet.png
# 13. 37-Parakeet.png
# 14. 38-Parakeet.png
# 15. 47-Parakeet.png
# 16. 50-Rhinoceros.png
# 17. 51-Rhinoceros.png
# 18. 53-Rhinoceros.png
# 19. 54-Rhinoceros.png
# 20. 55-Rhinoceros.png


profiles = create_user_profiles("../../data-exploration-cleanup/cleaned_simulatedusers.csv")

for p in profiles:
    # print(p)
    print(p.profiling_prompt(SYSTEM))