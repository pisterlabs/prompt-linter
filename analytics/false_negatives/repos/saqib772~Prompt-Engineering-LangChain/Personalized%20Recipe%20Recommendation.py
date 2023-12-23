from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for recipe recommendation
recipe_template = '''I need a personalized recipe recommendation based on the following preferences:
Cuisine: {cuisine}
Ingredients: {ingredients}
Dietary Restriction: {dietary_restriction}'''

recipe_prompt = PromptTemplate(
    input_variables=["cuisine", "ingredients", "dietary_restriction"],
    template=recipe_template
)

# Format the recipe recommendation prompt
recipe_prompt.format(
    cuisine="Italian",
    ingredients="tomatoes, basil, mozzarella",
    dietary_restriction="vegetarian"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
recipe_chain = LLMChain(llm=llm, prompt=recipe_prompt)

# Run the recipe recommendation chain
recipe_chain.run({
    "cuisine": "Italian",
    "ingredients": "tomatoes, basil, mozzarella",
    "dietary_restriction": "vegetarian"
})


#output of above Prompt


Vegetarian Caprese Pasta
Ingredients:
- 8 ounces of your favorite pasta
- 2 tablespoons of olive oil
- 2 cloves of garlic, minced
- 2 cups of cherry tomatoes, halved
- 2 tablespoons of balsamic vinegar
- Salt and pepper, to taste
- 2 tablespoons of fresh basil, chopped
- 1/2 cup of mozzarella cheese, cubed

Instructions:
'''
1. Bring a pot of salted water to a boil and cook the pasta according to the package instructions. Drain and set aside.
2. Heat the olive oil in a large skillet over medium-high heat. Add the garlic and cook for 1 minute.
3. Add the tomatoes and cook for another 2 minutes.
4. Add the balsamic vinegar, salt, and pepper to the skillet and stir to combine.
5. Add the cooked pasta to the skillet and stir to combine.
6. Add the basil and mozzarella and stir to combine.
7. Serve immediately. Enjoy! '''
