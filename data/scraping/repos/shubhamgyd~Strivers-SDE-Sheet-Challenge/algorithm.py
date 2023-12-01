import json
import os

# Get the directory path of the current script
current_dir = os.path.dirname(_file_)

# Construct the path to the careers.json file
careers_json_path = os.path.join(current_dir, '..', 'data', 'careers.json')

# Load the career dataset from JSON
with open(careers_json_path, 'r') as file:
    careers_data = json.load(file)
    
# print(careers_data)

# # Single user input scenario
# # Modified user input scenario
user_response = {
    "interests": {
        "mathematics": 1,  # Not interested at all
        "teamwork": 4,  # Thrive in team settings
        "creative_activities": 5,  # Extremely interested
        "understanding_human_behavior": 3,  # Moderately interested
    },
    "strengths": {
        "rate_strengths": 3,  # Average
        "handle_challenges": 4,  # Effectively
        "rate_accomplishment": 4,  # Very effective
    },
    "values": {
        "motivation": 4,  # Very motivated
        "work_life_balance": 5,  # Extremely important
        "work_values": 5,  # Extremely important
        "financial_stability": 4,  # Very important
    }
}

test_q=('''
Section 1: Interests

On a scale of 1 to 5, how interested are you in solving complex mathematical problems?

(1) Not interested at all
(2) Slightly interested
(3) Moderately interested
(4) Very interested
(5) Extremely interested
How do you feel about working in a team and collaborating with others?

(1) I prefer working alone.
(2) I'm somewhat comfortable working in a team.
(3) I'm moderately comfortable working in a team.
(4) I'm comfortable working in a team.
(5) I thrive in team settings.
Do you enjoy creative activities like writing, painting, or playing a musical instrument?

(1) Not at all
(2) Slightly
(3) Moderately
(4) Very much
(5) Extremely
Are you interested in understanding human behavior, motivations, and emotions?

(1) Not interested
(2) Slightly interested
(3) Moderately interested
(4) Very interested
(5) Extremely interested
Section 2: Strengths

On a scale of 1 to 5, how would you rate your strengths as described by others?

(1) Very weak
(2) Weak
(3) Average
(4) Strong
(5) Very strong
On a scale of 1 to 5, how do you typically handle challenges or setbacks in your life or work?

(1) Poorly
(2) Somewhat poorly
(3) Moderately
(4) Effectively
(5) Very effectively
On a scale of 1 to 5, how would you rate the effectiveness of a specific accomplishment or project where you felt your skills and strengths were utilized?

(1) Not effective at all
(2) Slightly effective
(3) Moderately effective
(4) Very effective
(5) Extremely effective
Section 4: Values

On a scale of 1 to 5, what motivates you the most in your career choice?

(1) Not motivated
(2) Slightly motivated
(3) Moderately motivated
(4) Very motivated
(5) Extremely motivated
On a scale of 1 to 5, how important is work-life balance to you?

(1) Not important at all
(2) Slightly important
(3) Moderately important
(4) Very important
(5) Extremely important
On a scale of 1 to 5, what values do you prioritize in your work environment?

(1) Not important
(2) Slightly important
(3) Moderately important
(4) Very important
(5) Extremely important
On a scale of 1 to 5, how important is financial stability and compensation in your career choices?

(1) Not important at all
(2) Slightly important
(3) Moderately important
(4) Very important
(5) Extremely important
''')
# # Calculate scores and recommend a career based on the user's input
career_scores = {}
for career in careers_data:
    score = 0
    for category, responses in user_response.items():
        # print(category, responses)
        for key, user_value in responses.items():
            print(key, user_value)

            
            career_scores[x] += user_value
    #         career_value = career.get(category, {}).get(key, 3)  # Default to 3 (Moderate) if not found
    #         score += abs(career_value - user_value)
    # career_scores[career["name"]] = score

# Find the career with the lowest score (closest match to user responses)
'''
recommended_career = min(career_scores, key=career_scores.get)


print(f"Recommended Career: {recommended_career}")

from langchain.llms import Cohere
llm = Cohere(cohere_api_key="Pt5Aqeee3CA0uMkXnYNG16Q8yfwQ3rIZpV86MwB0", temperature=0.7)
'''
# print(llm.predict('''There are three variables provided to you which are {careers_data}{user_response}{test_q}. careers_data is a json file which contains a set of career options and their detail. test_q contains the assesment questions. user_response contains the answers given by the user to the test_q questions. on the  basis of the user_response, suggest the ideal career name from careers_data'''))
'''
'''
'''
PS C:\Users\Dell\Desktop\storm\Codestorm_priv\backend\algo> py algo.py
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}
interests {'mathematics': 1, 'teamwork': 4, 'creative_activities': 5, 'understanding_human_behavior': 3}
strengths {'rate_strengths': 3, 'handle_challenges': 4, 'rate_accomplishment': 4}
values {'motivation': 4, 'work_life_balance': 5, 'work_values': 5, 'financial_stability': 4}

'''