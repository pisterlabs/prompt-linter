from PIL import Image
import pytesseract
import PyPDF2
import os
import fitz  # PyMuPDF
from PIL import Image

def pdf_to_images(pdf_path, image_folder):
    
    # Ouvrir le fichier PDF
    pdf_document = fitz.open(pdf_path)

    # Créer le dossier pour stocker les images
    os.makedirs(image_folder, exist_ok=True)

    # Parcourir chaque page du PDF
    for page_number in range(pdf_document.page_count):
        # Extraire la page
        page = pdf_document[page_number]

        # Convertir la page en image
        image = page.get_pixmap()
        image_path = os.path.join(image_folder, f"page_{page_number + 1}.png")

        # Enregistrer l'image
        image.save(image_path)

    # Fermer le fichier PDF
    pdf_document.close()


from PIL import Image, ImageEnhance, ImageFilter

def preprocess_image(image_path):
    # Ouvrir l'image à l'aide de Pillow
    image = Image.open(image_path)

    # Appliquer des filtres pour améliorer la qualité de l'image
    enhanced_image = ImageEnhance.Contrast(image).enhance(2.0)  # Augmenter le contraste
    #enhanced_image = enhanced_image.filter(ImageFilter.MedianFilter())  # Appliquer un filtre médian

    # Convertir l'image en niveaux de gris pour la binarisation
    grayscale_image = enhanced_image.convert('L')

    # Appliquer la binarisation pour améliorer le contraste entre le texte et l'arrière-plan
    threshold = 150  # Ajustez cette valeur en fonction de votre image
    binary_image = grayscale_image.point(lambda p: p > threshold and 255)

    return binary_image

# Exemple d'utilisation
pdf_path = 'FACTURE.pdf'
image_folder = './'
pdf_to_images(pdf_path, image_folder)

# Spécifiez le chemin de l'image que vous souhaitez traiter
image_path = 'page_1.png'

# Ouvrir l'image à l'aide de Pillow
image = Image.open(image_path)


# Utiliser Tesseract pour extraire le texte
text = pytesseract.image_to_string(image)

# Afficher le texte extrait
print(text)




# "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"




# import openai

# openai.api_key = "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"

# prompt = f"Mon texte est rempli d'erreur corrige le :"+text

# completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo", 
#   messages=[{"role": "user", "content": prompt}]
# )

# print(completion['choices'][0]['message']['content'])



image_path = 'page-001.jpg'

# Ouvrir l'image à l'aide de Pillow
image = Image.open(image_path)
image.show()
enhanced_image = ImageEnhance.Contrast(image).enhance(2.0)

# Utiliser Tesseract pour extraire le texte
text = pytesseract.image_to_string(image)

# Afficher le texte extrait
print(text)



from pdf2image import convert_from_path
 
 
# Store Pdf with convert_from_path function
images = convert_from_path('FACTURE.pdf')
 
for i in range(len(images)):
   
      # Save pages as images in the pdf
    images[i].save('page'+ str(i) +'.jpg', 'JPEG')