import layoutparser as lp
import cv2
import numpy as np
import pdf2image
import cohere
import pytesseract

def basicmodel(prompt):
    return co.generate( 
        model='xlarge', 
        prompt = prompt,
        max_tokens=40, 
        temperature=0.8,
        stop_sequences=["--"])


def getSummary(image, summarize_model = basicmodel):
    co = cohere.Client('ckMlwTDnjIuNedsF6stcXHX75O97Nwndd0qfIVMA')
    
    
    testing = '/Users/achennupati/Desktop/testimg.png'
    image = cv2.imread(testing)
    #this is to get in RGB format instead of BGR
    image = image[..., ::-1] 

    model = lp.Detectron2LayoutModel(
        config_path ='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', # In model catalog
        label_map   ={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}, # In model`label_map`
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # Optional
    )
    
    layout = model.detect(image)
    text_blocks = lp.Layout([b for b in layout if b.type=='Text'])
    
    for block in text_blocks:
        segment_image = (block
                           .pad(left=5, right=5, top=5, bottom=5)
                           .crop_image(image))

        text = pytesseract.image_to_string(segment_image)
        block.set(text=text, inplace=True)

    prompt = []
    for i in range(len(text_blocks)):
        
        prompt.append(f"""Passage: Is Wordle getting tougher to solve? Players seem to be convinced that the game has gotten harder in recent weeks ever since The New York Times bought it from developer Josh Wardle in late January. The Times has come forward and shared that this likely isn't the case. That said, the NYT did mess with the back end code a bit, removing some offensive and sexual language, as well as some obscure words There is a viral thread claiming that a confirmation bias was at play. One Twitter user went so far as to claim the game has gone to "the dusty section of the dictionary" to find its latest words.

        TLDR: Wordle has not gotten more difficult to solve.
        --
        Passage: ArtificialIvan, a seven-year-old, London-based payment and expense management software company, has raised $190 million in Series C funding led by ARG Global, with participation from D9 Capital Group and Boulder Capital. Earlier backers also joined the round, including Hilton Group, Roxanne Capital, Paved Roads Ventures, Brook Partners, and Plato Capital.

        TLDR: ArtificialIvan has raised $190 million in Series C funding.
        --

        """ + text_blocks[i].text.replace('\n', '') + """

        TLDR:""")
    
    results = []
        
    for i in range(len(prompt)):
        response = summarize_model(prompt[i])
        results.append(response.generations[0].text.replace('\n', ''))

        
    return results
        

    