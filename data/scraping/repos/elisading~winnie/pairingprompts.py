import os
import openai
import json

prompts = [
    """It’s important to understand the basic relationships 
and taste interactions between wine and food. It’s 
equally important to be aware that different people 
have different  reactions and taste sensitivities to 
various flavor and aroma components, meaning that 
the same level of acidity or bitterness for instance 
will likely affect one person differently than another. 
Keep in mind this is not the same as personal 
preferences.  Pairing should therefore take into 
account the preferences of the individual, as well as 
the interactions between food and wine. """,
"""When you place food in your mouth, your taste 
buds will adapt to alter the per ceptions of the levels 
of salt, sugar, acid, etc. of the next items you taste. 
An example would be  the unpleasant reaction when  
you taste highly acidic foods after using toothpaste. Also, some foods such as chocolate or rich, creamy 
dishes have a mouth- coating effect that inhibits the 
sense of taste. """,
"""Basically, there are two components in food (sweetness and umami) that can make wines taste 
“harder” (more astringent and bitter, more acidic, 
less sweet and less fruity), and two components (salt 
and acid) that tend to make wines taste “softer” 
(less astringent and bitter, less acidic, sweeter and 
more fruity). In general, food has a greater impact 
on the taste of wine than the other way around, and 
in particular is more likely to affect a wine 
negatively.  """,
"""Sweet Food  
 Increases the sense of bitterness  and acidity 
in the wines   Will enhance the burning effect of alcohol  
 May make a dry wine seem to lose its sense 
of fruitiness  
 A general rule of thumb with sweet foods is to select a wine with a higher level of sweetness than the food  
""",
"""Umami in Food  
   Umami can be difficult to identify, compared to the other primary tastes. In general it is the savory taste 
you get from herbs and vegetables like mushrooms 
and asparagus; MSG; ripe cheeses; meats, seafood  
and hard cheeses. Although the negative impact of 
umami on wines in many of these foods can be 
overcome in meats, smoked seafood and cheeses, 
due to the significant levels of salt in these foods.  
  When you encounter foods that are difficult to pair 
due to high Umami, you can counter this negative 
effect by adding salt and/or acid to the dish and then 
pairing the wine based on those additions (quite 
often in the case of a sauce created for that dish , or 
by adding lemon to fish, for instance) . 
 Umami increases the perception of 
bitterness, acidity, alcohol b urn 
 Lessens the perception of body, sweetness, 
fruitiness in wine """,
""" Acid in Food  
   Acid is the most important element in the pulp, 
aside from the water and sugar. In general some 
acidity in foods or sauces can be a good thing when 
pairing and can quite often bring a very high acid 
wine into balance and highlight its fruitiness . Acid in 
wine gives the wine vivacity and makes it thirst 
quenching, therefor keeping acid in balance is critical 
when creating the wine. When pairing wine with 
foods with hi gher levels of acid it’s important to  make sure the wine has a high enough level of acid 
to avoid making the wine taste languid, flat or 
flabby.  
 Acid increases the perception of body, sweetness and fruitiness  
 Decreases the perception of acidity in wines  
 Helps make the wine thirst quenching  """,
"""Saltiness in Food 
   Salt is a very wine -friendly component which can 
mellow many of the more difficult and harder 
elements of a wine. Salt is a great contrast to acidity, 
which is why, for instance, Asian dishes with a  lot of 
soy sauce are a great match with a German Riesling. And of course, many of us love the contrast between 
sweet and salty, which is why a great Stilton or Blue 
Cheese matches so well with Port.  
 A great contrast to acidity  
 Decreases the perception of bitterness and 
acidity in wine  
 Increases the perception of body  
 Works well with sweet wines  """,
"""Bitterness in Food  
   Different persons will be affected differently by 
bitter flavors in both wine and food. In most cases, 
however, bitter flavors do not pair well with each 
other.  The level of bitterness in both the food and 
the wine may be at favorable levels separately, but 
when consumed together they may combine for an 
unpleasant taste profile.  
 Bitterness in food increases bitterness in wine.  
 Avoid wines with a high level of hard  
tannins  """,
"""Chili  Heat in Food  
   As with bitterness, chili  heat in food can affect 
different people in very different ways. In general, 
chili heat will increase, in an unpleasant way, the 
sensations of bitterness, acidity, and alcohol burn. 
The intensity of this reaction increases as the alcohol 
level of the wine increases. The alcohol can also 
increase the sensation of heat in the food.  
 Chili  increases the perception of acid, 
bitterness, and alcohol burn  
 Decreases the perception of body, sweetness and fruitiness in the wine """,
""" Flavor Profile  
   In general, it’s usually a safe bet to match the 
flavor intensities of the food and wine to be p aired, 
so that one doesn’t overpower the other. However, in some cases, more intensely flavored foods, such as a curry, can work well with a ligh ter style wine, 
like a slightly fizzy Lambrusco. Many lightly flavored desserts can also pair well with intense ly sweet 
dessert wines.""",
"""Acidity and Fat  
Most people, especially in America, enjoy the pairing 
of acidic wines with fatty or oily foods. The pairing 
gives the subjective sensation of the acid cutting through the richness of the food and thereby 
cleansing the pa late. Fat also has a way of softening 
highly tannic wines.  
This is why so many people enjoy a bold Cabernet or Syrah with a grilled rib eye, or a high acid and citrusy 
Sauvignon Blanc with many types of white fish. """,
"""Sweet and Salty  
   Although it doesn’t seem to work technically in 
wine pairing, who can argue with the wonderful (if 
somewhat subjective) combinations of sweet and 
salty? Who doesn’t like a chocolate dipped salty 
pretzel? A classic example of this would be the 
European tradition of pairing Stilton cheese with 
Port.  """,
"""Geography  
   All things being equal –  if you find a suitable pairing 
for your dish and you can source it from the same 
geographical area, you can discover some very 
wonderful and fun combinations, that also quit e 
often are accompanied by some interesting stories 
and history.  """,
"""Difficult Foods to Pair  
 Sweet – foods or dishes high in sugar 
should be paired with wines that have sweetness levels as high or higher than the 
food  
 Umami – fruity wines or wines with lower,  
softer tannin levels pair more favorably with dishes higher in umami as the umami 
brings out the bitterness of the tannins in the wine. High levels of umami can be 
balanced out by the addition of salt or acid  to a dish, but the addition shouldn’t change 
the basic character of the foods  
 Bitterness – in food will emphasize 
bitterness in the wine. Match bitter foods 
with white or neutral wines, or reds with 
lower levels of tannin  
 Chili Heat  - pair with white wines, slightly 
sweet whites and reds, or low- tanni n reds. 
Low alcohol wines are better, as the chili  
heat accentuates the alcohol burn and 
bitterness for many people. Chili  also 
reduces fruitiness or sweetness, so look for 
wines that are higher in these properties. """,
"""Wine Friendly Foods  
 Foods higher in acid and/or salt tend to be 
easier to match with most wines; keeping in 
mind that foods high in acid should be 
matched with a wine with a higher acid 
level, or the wine may taste too soft or 
flabby.  """,
"""Challenging Wines  
The more structural components in the wine and/or 
food to be paired, the more challenging, but also 
rewarding, the pairing can become. The most 
difficult wines are those with higher levels of 
bitterness from oak or tannins and/or high levels of 
alcohol and/or acid. The good news is that if you  find 
a suitable pairing the wine can reveal complexities and flavors that might not be detected if the wine 
were to be drunk without food. """,
"""Lower Risk Wines  
Neutral, unoaked wines with a small amount of residual sugar can be pretty safe with most foods, 
but also aren’t likely to produce very interesting 
experiences. """,
"""Pair great with great and simple with simple. A basic meatloaf sandwich doesn’t need an expensive merlot to make a nice 
combination. On the other hand, an 
expen sive crown roast of lamb or prime rib  
may be the perfect oc casion for breaking out that big and opulent Napa Cabernet or Bordea ux you’ve been saving. """,
"""Match delicate to delicate and bold to bold. 
A delicate red Burgundy or most any 
delicate white will be completely 
overshadowed by robust or spicy dishes. 
Likewise, bold, spicy and hot flavors work 
well with spicy, big flavored , lower tannin  
wines such as Zinfandel. """,
"""You can choose to mirror the wine to the dish, or set up a contrast, either can work. 
For instance: a California Chardonnay with 
lobster or pasta in cream  sauce would be an 
example of mirroring. Or, you can serve 
that same dish with a crisp , sharp, bubbling 
Champagne.  """,
"""Think flexibility: Oaky Chardonnay is a very 
popular varietal, but it is one of the least 
flexible white wines to pair with most foods. Sauvi gnon Blanc, dry Riesling from 
Germany or Alsace, or a more neutral wine like Pinot Grigio offer much more flexibility.  
More flexible red wines either have nice 
acidity, such as Chianti, red Burgundy and 
American Pinot Noir; or they are fruity with 
moderate or low tannins, such as Zinfandel, 
simpler Italian reds and southern Rhone 
wines.  """,
"""A couple of good ways to produce interesting and 
satisfying combinations are to use these principals to 
practice your own pairings, and make notes of what 
works and what doesn’t. Another way is to take 
notice of long established successful pairings and try 
to identify which of these principals are at work 
there, and use those same ideas when creating your 
own pairings. Also, consider what exactly you’re 
trying to pair the wine with when you have a dish 
with many different flavor profiles going on at the 
same time. Are you trying to match the protein, the 
sauce, the vegetable, etc. In general, the simpler the 
better.  
At the end of the day, the wines that appeal m ost to 
you at any given times are always a good choice; however, we feel that by practicing these principals 
over a period of time you will begin to discover new 
wine and food experiences that are even more 
appealing to you and your guests.  """,
"""
Cabernet is a complex and full -bodied wine characterized by aromas of black currants, blackberries, black cherries, 
plums, cedar, mint, clove, and strong hints of vanilla oak. “Cab” is intensely flavorful, and is more acidic and has 
harder tannins than Merlot. """,
"""Merlot is fruit -intensive and has softer and more velvety tannins than Cabernet. Merlot is characterized by black 
cherries, red cherries, an d oak flavors. In Bordeaux Merlot is almost always blended with Cabernet Sauvignon or 
Cabernet Franc. """,
"""Pinot Noir is complex, having a velvety texture, majestic flavor, and is extraordinary in bouquet. Pinot Noir is sophisticated  and has rich, fruit flavors of black cherries, blackbe rries, and plums as well as dried roses, tar, bark, 
earthy mushrooms, cola, and spicy black pepper.  Pinot has a higher level of acid, which makes it a more flexible 
food pairing wine.  """,
"""Zinfandel is a full -bodied, spicy wine with strong raspberry, blackberry, boysenberry, cranberry, and black cherry 
flavors, as well as hints of licorice, black pepper, plum, tobacco, cedar, vanilla, and light oak.  
 """,
"""Syrah is deeply colorful, powerful, peppery, spicy, and is characterized by black cherry, blackberry, tar, clove, 
thyme, leather, and roasted nut flavors. This wine exhibits a rich, smooth, supple texture, and even tannins.  
 """,
 """Sangiovese is a Medium -to-full-bodied wine with supple “warm” texture. Sangiovese boasts the flavors of raspberries, cherries, 
anise, and various spices.  It is the varietal used to make Chianti.    
 """,
 """At its best, Chardonnay features bold, rich fruit flavors of apple, fig, melon, pear, peach, pineapple, and citrus 
fruits. It also may possess hints of honey, butter, vanilla, butterscotch, and hazelnut. Most significant ly, as a result 
of oak- barrel aging, Chardonnay is well known for its flavors of apple, vanilla, and toasty oak.  
 """,
 """Sauvignon Blanc is brisk, strong, dry, and light -bodied, and features citrus, grassy, and floral scents . Most 
Sauvignon Blanc wines are dry and “un- oaked” (fermented in steel vats instead of oak barrels), with a bracing, 
lively acidity that balances the wine’s natural fruitiness.  
 """,
 """Chenin Blanc’s signature is its acidity combin ed with high alcohol content and a full -bodied, almost oily texture. 
Though Chenin Blanc is most often found in simple, dry wines, it can -  especially in the Loire Valley -  create great 
wines in a variety of styles from dry to sweet. It has a subtle fruiti ness (melon, peach, quince, apricot, and 
sometimes even a citrus quality.) It can also have spicy overtones together with a smooth hint of honey.  
 """,
 """Full-bodied and flamboyant, Gewurztraminer wines are invariably dee ply flavored, aromatic wines. Their flavor is 
somewhat exotic with a bit of apricots and grapefruits, floral roses, and herbal spices such as tarragon. Still, they are dry, refreshing wines that couple extremely well with food.  
 """,
 """These  are light -bodied, medium -dry, low -alcohol wines, often with lively acidity and crispness. A good Riesling has 
a distinctive floral or honeysuckle aroma with citrus fruits, peaches, and apples. For the most part, Rieslings should 
not be aged. They should be consumed fresh and young.  
 """,
 """ Pinot Grigio wines are medium- to-full-bodied, lightly fruity, somewhat peachy, with slight hints of grass and spice.  
 """

]

text = """
Some Basic Wine and Food Pairings
Asian Cuisine (umami) – Grüner Veltiner, Riesling, Sauvignon Blanc, Gamay, Pinot Noir
Caviar, Oysters, Smoked Salmon – Champagne or Dry Sparkling Wine
Olives, Almonds, Canapes, etc. – Sauvignon Blanc, Cava, Albarino, Chablis, Dry Sherry, Blush Wine
Cream Soups – White wine
Heavy Vegetable Soups – Lighter Red Wine
Barbeque Ribs & Chicken - Pinot Noir, Zinfandel, Rhone Wines, Carmenère, Lower Price Red Blends in General
Fried Foods – Sparkling Wines such as: Cava, Champagne, Crémant, Moscato d’ Asti, Prosecco
Fish: Poached, Grilled or Sautéed; Crab or Lobster – Chardonnay (Chablis), Sauvignon Blanc, Riesling, Soave, Verdejo from Spain
More complex Fish or Shellfish dishes – Champagne, Riesling, Soave, Chardonnay (Chablis), Gewurztraminer, Vermentino from Italy, Fino Sherry, Pinot Noir
Chicken or Turkey – Barbera, Dolcetto, Chardonnay, Torrontes from Argentina, Dry Rose
Pheasant – Champagne, Viognier, Mature Pinot Noir
Roast Ham or Pork – Gewurztraminer, Viognier, Blush wine, Beaujolais, Dolcetto from Italy
Lamb – Fine Red Bordeaux, Cabernet Sauvignon
Beef – Merlot, Beaujolais, Cabernet Sauvignon, Dry Italian Red Wines (Nebbiolo), Syrah, Petite Syrah, Tempranillo
Mediterranean (salty, oily dishes) – Assyrtiko, Carricante, Pinot Grigio, Vermentino, Cotes de Provence
Tomato Based Pizza & Pasta – Barbera, Corvina, Lambrusco, Nebbiolo, Sangiovese
Stews or Pot Roast – Beaujolais, Cabernet Sauvignon, Zinfandel, Pinot Noir, Basic Sangiovese
Salads – Chenin Blanc, French Colombard, Sauvignon Blanc, Assyrtiko from Greece
Cheese – Full-bodied Red Wine, Big Bordeaux, Cabernet Sauvignon, Pinot Noir, Chardonnay, Sauvignon Blanc, Fume Blanc, Port
Desserts – Any Sweet or Sparkling Wine
"""

lines = text.strip().split('\n')
#print(lines)
for line in lines:
    prompts.append(line)

print(len(prompts))

schema = {
    "type": "object",
    "properties": {
        "background": {
            "type": "string",
            "description": "Background information or context for the conversation"
        },
        "dialogue": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "human": {
                        "type": "string",
                    },
                    "bot": {
                        "type": "string"
                    }
                },
                "required": [
                    "human",
                    "bot"
                ]
            }
        }
    },
    "required": [
        "background",
        "dialogue"
    ]
}
openai.api_key = "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"

def get_response(schema, chunk):
	
	completion = openai.ChatCompletion.create(
	model="gpt-3.5-turbo-0613",
	messages=[
		{"role": "system", "content": "You are a sample text generator, who can synthetically generate background, human and bot dialogue."},
		{"role": "user", "content": f"Provide a background, human and bot dialogue based on the following context `{chunk}`"}
	],
	functions=[{"name": "generate_sample", "parameters": schema}],
	function_call={"name": "generate_sample"},
	temperature=0,
	)

	return completion.choices[0].message.function_call.arguments


output_file = 'food_pair_train.jsonl'

# Loop through the prompts and call the completion function for each

with open(output_file, 'w') as f:
	for i, chunk in enumerate(prompts):
		response = json.loads(get_response(schema, chunk))
		text = f"Background: {response["Background']} "
		for dialogue in response["dialogue"]:
			text += f"<human>: {dialogue['human']} "
			text += f"<bot>: {dialogue['bot']} "
		f.writelines([str({"text": text}) + "\n"])
		print(response)
"""                
with open(output_file, 'a', encoding='utf-8') as f_out:
    for prompt in prompts:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "system", "content": "You are a sample text generator, who can synthetically generate background, human and bot dialogue."},
                {"role": "user", "content": f"Background: {prompt}"}
            ],
            functions=[{"name": "generate_sample", "parameters": schema}],
            function_call={"name": "generate_sample"},
            temperature=0,
        )

        generated_text = completion.choices[0].message.function_call.arguments
        jsonl_object = {"text": generated_text}
        f_out.write(json.dumps(jsonl_object) + '\n')

        print(f"Generated JSONL object for prompt: {prompt}")"""

print("All prompts processed and saved to", output_file)


