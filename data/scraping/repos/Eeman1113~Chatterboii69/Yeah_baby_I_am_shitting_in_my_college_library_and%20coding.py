#setting up a delay 
import time 
time.sleep(5)

while True:
          ##############################################################################################################################################################
          #screenshot part starts here 
          import pyscreenshot as ImageGrab
          import numpy as np
          bbox=(493,791,984,824)#(left_x, top_y, right_x, bottom_y)
          screenshot = ImageGrab.grab(bbox=bbox, backend='mac_quartz')
          screenshot.save('/Users/eemanmajumder/code_shit/Chatterboii69/img/sample.png')
          ##############################################################################################################################################################
          #OCR Part
          
          # Import required packages how are you/ i'm good, thanks for asking. i've been busy lately with work and school, but I'm doing well. how about you/

          import cv2
          import pytesseract
          
          # Mention the installed location of Tesseract-OCR in your system
          pytesseract.pytesseract.tesseract_cmd = (r'/opt/homebrew/Cellar/tesseract/5.1.0/bin/tesseract')
          
          # Read image from which text needs to be extracted
          img = cv2.imread("/Users/eemanmajumder/code_shit/Chatterboii69/img/sample.png")
          
          # Preprocessing the image starts
          
          #negative of image 
          img_not = cv2.bitwise_not(img)
          cv2.imwrite("/Users/eemanmajumder/code_shit/Chatterboii69/img/img_not.png", img_not)
          # Convert the image to gray scale
          gray = cv2.cvtColor(img_not, cv2.COLOR_BGR2GRAY)
          # Performing OTSU threshold
          ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
          
          # Specify structure shape and kernel size.
          # Kernel size increases or decreases the area
          # of the rectangle to be detected.
          # A smaller value like (10, 10) will detect
          # each word instead of a sentence.
          rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
          
          # Applying dilation on the threshold image
          dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
          
          # Finding contours
          contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
          												cv2.CHAIN_APPROX_NONE)
          
          # Creating a copy of image
          im2 = img.copy()
          
          
          
          # Looping through the identified contours
          # Then rectangular part is cropped and passed on
          # to pytesseract for extracting text from it
          # Extracted text is then written into the text file
          for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
          	# Drawing a rectangle on copied image
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
          	
          	# Cropping the text block for giving input to OCR
            cropped = im2[y:y + h, x:x + w]
              
          # Open the file in append mode
          file = open("/Users/eemanmajumder/code_shit/Chatterboii69/text/recognized3.txt", "a")
          cv2.imwrite("/Users/eemanmajumder/code_shit/Chatterboii69/img/cropped.png", im2)
          # Apply OCR on the cropped image
          text = pytesseract.image_to_string(img_not, lang='eng')
          l=[]
          l.append(text)
          time.sleep(0.6)
          
          # Appending the text into file
          file.write(text)
          file.write("\n")
          
          # Close the file
          file.close
          #########################################################################################################################################
          # lets clean the data and remove the unwanted characters and add the text to a .csv file 
          file3=open("/Users/eemanmajumder/code_shit/Chatterboii69/text/recognized3.txt")
          lines=file3.readlines()
          print(lines)
          
          def seperate_words_in_a_sentence(lines):
              words=lines[-1].split()
              return words
          
          def remove_unwanted_characters(words):
            #delete last two words in the list
            del words[-2:]
          
          def words_to_sentence(words):
            sentence=''
            for word in words:
              sentence=sentence+word+' '
            return sentence
          
          
          def csv_file(sentence):
            file2=open("/Users/eemanmajumder/code_shit/Chatterboii69/text/recognized.csv",'w')
            file2.write(sentence)
            file2.close()
          
          def main(lines):
            words=seperate_words_in_a_sentence(lines)
            remove_unwanted_characters(words)
            sentence=words_to_sentence(words)
            csv_file(sentence)
            return sentence
          
          print(main(lines))
          #########################################################################################################################################
          #AI Part Starts here 
          # Importing the libraries
          import openai
          import os
          
          openai.api_key = "OPENAI_KEY_HERE"
          c=("Chat with me like a friend:\n "+main(lines))
          response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=c,
            temperature=0.7,
            max_tokens=64,
            top_p=1.0,
            frequency_penalty=0.0,

            presence_penalty=0.0
          )
          b=response.choices[0].text
          b=b.strip()
          print(b)
          #########################################################################################################################################
          #the reply mechanism starts here 
          import pyautogui
          
          pyautogui.write(b)
          pyautogui.hotkey('enter')
          time.sleep(0.8)
          #########################################################################################################################################
          #delete all the text stored in recognized3.txt
          file3=open("/Users/eemanmajumder/code_shit/Chatterboii69/text/recognized3.txt",'r+')
          file3.truncate(0)
          file3.close()
          #########################################################################################################################################
          #making a kill switch to stop the program
          import keyboard
          if keyboard.is_pressed(']'):
            exit()
          #########################################################################################################################################
