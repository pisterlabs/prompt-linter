import cv2 
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb, rgb_to_hex
from scipy.spatial import KDTree
from colour_selection import SelectRandomColour, SelectTopColor, SelectRandomTop
import json
from cohere_gen.generate import generate, prompt, store_image
from PIL import Image

def convert_rgb_to_names(rgb_tuple):
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return names[index]

def closest_tone_match(rgb_tuple):
    skin_tones = {'Monk 10': '#292420', 'Monk 9': '#3a312a', 'Monk 8':'#604134', 'Monk 7':'#825c43', 'Monk 6':'#a07e56', 'Monk 5':'#d7bd96', 'Monk 4':'#eadaba', 'Monk 3':'#f7ead0', 'Monk 2':'#f3e7db', 'Monk 1':'#f6ede4'}

    rgb_values = []
    names = []
    for monk in skin_tones:
        names.append(monk)
        rgb_values.append(hex_to_rgb(skin_tones[monk]))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)

    return names[index]

def separate(image): 
    backSub = cv2.createBackgroundSubtractorMOG2()
    mask = backSub.apply(image)
    return mask

def recommendations(path): 
    f = open('./dataset/descriptions.json',)
    json_file = json.load(f)

    # Load Body Type
    txt = open('local_data.txt')
    body_type = txt.readline()

    # the description
    desc = json_file[path]

    result = generate(prompt(body_type, desc))
    pants_desc = result.split('\n')[0]
    shoes_desc = result.split('\n')[1]
    print("1: {}".format(pants_desc))
    print("2: {}".format(shoes_desc))

    store_image("pants", pants_desc)
    store_image("shoes", shoes_desc)

# Human Identification
def vision(img): 
    # Load the cascade
    # Pretrained model
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Read the input image
    # img = cv2.imread('person.jpg')
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:

        # Patch Properties
        patch_w = int(w*0.3)
        patch_h = int(h*0.3)

        offset_y = h*0.4

        # Find center 
        center_x = x+w/2
        center_y = y-offset_y+h/2

        patch_x_1 = int(center_x-patch_w/2)
        patch_x_2 = int(center_x+patch_w/2)
        patch_y_1 = int(center_y-patch_h/2)
        patch_y_2 = int(center_y+patch_h/2)

        patch = img[patch_x_1:patch_x_2, patch_y_1:patch_y_2]
        # patch_blur = cv2.GaussianBlur(patch, (5,5), 11)
        patch_blur = cv2.medianBlur(patch, 11)

        # cv2.imshow('img', cleaned_image)
        # cv2.waitKey(0)

        # loaded_image = Image.fromarray(cleaned_image)
        # extracted_tones = sorted(loaded_image.getcolors(2 ** 24), reverse=True)[0][1]
        # Filter out colors that are 

        # avg_tone = cv2.mean(patch)
        extracted_tones = cv2.mean(patch_blur)

        # Assume person's face is centered
        # Draw Circle for person
        cv2.rectangle(img, (patch_x_1, patch_y_1), (patch_x_2, patch_y_2), (0, 0, 255))
        cv2.circle(img, (int(x+w/2), int(y+h/2)), 5, (0, 0, 255), 2)

        # tone_hex = rgb_to_hex((int(avg_tone[0]), int(avg_tone[1]), int(avg_tone[2])))

        match = closest_tone_match((int(extracted_tones[0]), int(extracted_tones[1]), int(extracted_tones[2])))

        color = SelectTopColor(SelectRandomColour(match))
        print(color)

        top_dir = SelectRandomTop(color)
        top = Image.open("./dataset/" + color + "/" + top_dir)
        top.save("top.jpg")
        
        print(top_dir)

        # Save images 
        recommendations(top_dir)
        # tone_name = convert_rgb_to_names((int(extracted_tones[0]), int(extracted_tones[1]), int(extracted_tones[2])))
        # cv2.putText(img, "Skin Tone: {}".format(match), (x, y), 0, 0.5, (0,0,255))

    # Display the output
    # cv2.imshow('img', img)
    # return 200


# if __name__ == '__main__': 
    # cap = cv2.VideoCapture(0)
    
    # while cap.isOpened():
    #     ret, img = cap.read()

    #     vision(img)

    #     if cv2.waitKey(1) == ord('q'):
    #         break
        
    # cap.release()
    # cv2.destroyAllWindows()
    # img = cv2.imread("demo.png")
    # vision(img)