from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
import os


animals_template =""" 
You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, wildlife photography, photograph, high quality, wildlife, f 1.8, soft focus, 8k, national geographic, award - winning photograph by nick nichols

here is the input: {input}
"""

archviz_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, by James McDonald and Joarc Architects, home, interior, octane render, deviantart, cinematic, key art, hyperrealism, \
                        sun light, sunrays, canon eos c 300, ƒ 1.8, 35 mm, 8k, medium - format print

here is the input: {input}
"""

building_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, shot 35 mm, realism, octane render, 8k, trending on artstation, 35 mm camera, unreal engine, hyper detailed, \
    photo - realistic maximum detail, volumetric light, realistic matte painting, hyper photorealistic, trending on artstation, ultra - detailed, realistic

here is the input: {input}
    """

cartoon_char_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, anthro, very cute kid's film character, disney pixar zootopia character concept artwork, 3d concept, detailed fur, \
    high detail iconic character for upcoming film, trending on artstation, character design, 3d artistic render, highly detailed, octane, blender, cartoon, shadows, lighting

here is the input: {input}
    """

concept_art_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, character sheet, concept design, contrast, style by kim jung gi, zabrocki, karlkka, \
          jayison devadas, trending on artstation, 8k, ultra wide angle, pincushion lens effect

here is the input: {input}
          """

cyberpunk_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, cyberpunk, in heavy raining futuristic tokyo rooftop cyberpunk night, sci-fi, fantasy, intricate, very very beautiful, \
    elegant, neon light, highly detailed, digital painting, artstation, concept art, soft light, hdri, smooth, sharp focus, illustration, \
    art by tian zi and craig mullins and wlop and alphonse mucha

here is the input: {input}
    """

digital_art_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, ultra realistic, concept art, intricate details, highly detailed, photorealistic, octane render, \
    8k, unreal engine, sharp focus, volumetric lighting unreal engine. art by artgerm and alphonse mucha

here is the input: {input}
    """

digital_art_landscape_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, epic concept art by barlowe wayne, ruan jia, light effect, volumetric light, \
    3d, ultra clear detailed, octane render, 8k, dark green

here is the input: {input}
    """

drawing_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, cute, funny, centered, award winning watercolor pen illustration, detailed, \
    disney, isometric illustration, drawing, by Stephen Hillenburg, Matt Groening, Albert Uderzo

here is the input: {input}
    """

fashion_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	photograph of a Fashion model, `input goes here`, full body, highly detailed and intricate, golden ratio, vibrant colors, hyper maximalist, \
    futuristic, city background, luxury, elite, cinematic, fashion, depth of field, colorful, glow, trending on artstation, ultra high detail, \
    ultra realistic, cinematic lighting, focused, 8k

here is the input: {input}
    """

landscape_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, birds in the sky, waterfall close shot 35 mm, realism, octane render, 8 k, exploration, \
    cinematic, trending on artstation, 35 mm camera, unreal engine, hyper detailed, photo - realistic maximum detail, volumetric light, \
    moody cinematic epic concept art, realistic matte painting, hyper photorealistic, epic, trending on artstation, movie concept art, \
    cinematic composition, ultra - detailed, realistic

here is the input: {input}
    """

photo_closeup_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, depth of field. bokeh. soft light. by Yasmin Albatoul, Harry Fayt. centered. \
    extremely detailed. Nikon D850, (35mm|50mm|85mm). award winning photography.

here is the input: {input}
    """

photo_portrait_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	portrait photo of `input goes here`, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, \
    Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography

here is the input: {input}
    """

postapocalyptic_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, fog, animals, birds, deer, bunny, postapocalyptic, overgrown with plant life and ivy, \
    artgerm, yoshitaka amano, gothic interior, 8k, octane render, unreal engine

here is the input: {input}
        """

schematic_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	23rd century scientific schematics for `input goes here`, blueprint, hyperdetailed vector technical documents, callouts, legend, patent registry

here is the input: {input}
    """

sketch_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, sketch, drawing, detailed, pencil, black and white by Adonna Khare, Paul Cadden, Pierre-Yves Riveau

here is the input: {input}
    """

space_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, by Andrew McCarthy, Navaneeth Unnikrishnan, Manuel Dietrich, photo realistic, 8 k, cinematic lighting, hd, \
    atmospheric, hyperdetailed, trending on artstation, deviantart, photography, glow effect

here is the input: {input}
    """

sprite_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	sprite of video games `input goes here`, icons, 2d icons, rpg skills icons, world of warcraft, league of legends, ability icon, fantasy, \
    potions, spells, objects, flowers, gems, swords, axe, hammer, fire, ice, arcane, shiny object, graphic design, high contrast, artstation

here is the input: {input}
        """

steampunk_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	`input goes here`, steampunk cybernetic biomechanical, 3d model, very coherent symmetrical artwork, unreal engine realistic render, 8k, \
    micro detail, intricate, elegant, highly detailed, centered, digital painting, artstation, smooth, sharp focus, illustration, artgerm, Caio Fantini, wlop

        
here is the input: {input}
"""

vehicles_template ="""You are going to be given an input, you only have to replace the input into the following template just like it is, don't add anything else and no aditional comments, 
it's very important that you don't modify the anything, just replace the input given and return the full template with the input on it:

here is the template:
	photograph of `input goes here`, photorealistic, vivid, sharp focus, reflection, refraction, sunrays, \
    very detailed, intricate, intense cinematic composition

here is the input: {input}
        """





prompt_infos = [
    {
        "name": "animals", 
        "description": "Good for represent detailed photographies of animals", 
        "prompt_template": animals_template
    },
    {
        "name": "archviz", 
        "description": "Good for represent detailed images of interior of houses, interiorism design", 
        "prompt_template": archviz_template
    },
    {
        "name": "building", 
        "description": "Good for represent detailed images of buildings, statues...", 
        "prompt_template": building_template
    },
    {
        "name": "cartoon", 
        "description": "Good for represent cartoon pictures and characters", 
        "prompt_template": cartoon_char_template
    },
    {
        "name": "design", 
        "description": "Good for represent detailed images of design art and concept art on a character sheet", 
        "prompt_template": concept_art_template
    },
    {
        "name": "cyberpunk", 
        "description": "Good for represent high quality images of a modern dystopian future", 
        "prompt_template": cyberpunk_template
    },
    {
        "name": "digital_art", 
        "description": "Good for represent detailed images of characters drawn digitally", 
        "prompt_template": digital_art_template
    },
    {
        "name": "digital_art_landscape", 
        "description": "Good for represent detailed images of landscapes drawn digitally", 
        "prompt_template": digital_art_landscape_template
    },
    {
        "name": "drawing", 
        "description": "Good for represent detailed watercolor funny drawings", 
        "prompt_template": drawing_template
    },
    {
        "name": "fashion", 
        "description": "Good for represent detailed full body photographies of a fashion model", 
        "prompt_template": fashion_template
    },
    {
        "name": "landscape", 
        "description": "Good for represent detailed photographies of landscapes", 
        "prompt_template": landscape_template
    },
    {
        "name": "photo_closeup", 
        "description": "Good for represent highly detailed closeup photographies of people", 
        "prompt_template": photo_closeup_template
    },
    {
        "name": "photo_portrait", 
        "description": "Good for represent highly detailed closeup photographies of objects and any stuff", 
        "prompt_template": photo_portrait_template
    },
    {
        "name": "postapocalyptic", 
        "description": "Good for represent photographies of abandoned, broken and overgrown buildings", 
        "prompt_template": postapocalyptic_template
    },
    {
        "name": "schematic", 
        "description": "Good for represent blueprint design schemas, as a patent registry", 
        "prompt_template": schematic_template
    },
    {
        "name": "sketch", 
        "description": "Good for represent a detailed pencil drawing sketch scene", 
        "prompt_template": sketch_template
    },
    {
        "name": "space", 
        "description": "Good for represent detailed images of the space and cosmos, full of stars and glowing effects", 
        "prompt_template": space_template
    },
    {
        "name": "sprite", 
        "description": "Good for represent images of videogames icons, like spells, potions, gems, swords, and so...", 
        "prompt_template": sprite_template
    },
    {
        "name": "steampunk", 
        "description": "Good for represent detailed images of cybernetic designs, steampunk design", 
        "prompt_template": steampunk_template
    },
    {
        "name": "vehicles", 
        "description": "Good for represent highly detailed photographies vehicles", 
        "prompt_template": vehicles_template
    }]


def auto_prompt(lyrics):

    chain = MultiPromptChain.from_prompts(OpenAI(openai_api_key='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'), 
                                                 prompt_infos, verbose=False)

    response = chain.run(lyrics)

    return response