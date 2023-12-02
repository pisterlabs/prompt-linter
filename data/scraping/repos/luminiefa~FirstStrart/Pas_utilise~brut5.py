import logging
import mariadb
import pytesseract
import csv
from telegram import __version__ as TG_VER
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton

from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)
from PIL import Image
import os
import ast
import traceback

from dotenv import dotenv_values
from openai import ChatCompletion

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Model
class Database:
    def __init__(self):
        self.connection = mariadb.connect(
            user="root",
            password="",
            host="localhost",
            port=3306,
            database="FirstStrat",
        )
        self.cursor = self.connection.cursor()
    
    def update_client_csv(self, client_id, document_id):
        """Update the fichier_csv_path for a client."""
        sql = "UPDATE clients SET fichier_csv_path = ? WHERE telegram_id = ?"
        self.cursor.execute(sql, (document_id, client_id))
        self.connection.commit()

    def create_documents_table(self):
        """Create the documents table if it doesn't exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                client_id INT,
                document_path VARCHAR(255),
                FOREIGN KEY (client_id) REFERENCES clients(telegram_id)
            )
        """)
        self.connection.commit()

    def create_clients_table(self):
        """Create the clients table if it doesn't exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                telegram_id INT PRIMARY KEY,
                nom VARCHAR(255),
                age INT,
                adresse VARCHAR(255),
                fichier_csv_path VARCHAR(255)
            )
        """)
        self.connection.commit()

    def create_requests_table(self):
        """Create the requests table if it doesn't exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INT AUTO_INCREMENT PRIMARY KEY,
                telegram_id INT,
                request_text TEXT,
                response_text TEXT,
                FOREIGN KEY (telegram_id) REFERENCES clients(telegram_id)
            )
        """)
        self.connection.commit()
        

    def insert_client(self, telegram_id, nom, age, adresse, fichier_csv_path):
        """Insert a client into the clients table."""
        sql = "INSERT INTO clients (telegram_id, nom, age, adresse, fichier_csv_path) VALUES (?, ?, ?, ?, ?)"
        self.cursor.execute(sql, (telegram_id, nom, age, adresse, fichier_csv_path))
        client_id = self.cursor.lastrowid
        self.connection.commit()

        if fichier_csv_path:
            document_id = self.insert_document(client_id, fichier_csv_path)
            self.update_client_csv(client_id, document_id)
    
    def insert_document(self, client_id, document_path):
        """Insert a document into the documents table."""
        sql = "INSERT INTO documents (client_id, document_path) VALUES (?, ?)"
        self.cursor.execute(sql, (client_id, document_path))
        document_id = self.cursor.lastrowid
        self.connection.commit()
        return document_id


    def insert_request(self, telegram_id, request_text, response_text):
        """Insert a request into the requests table."""
        sql = "INSERT INTO requests (telegram_id, request_text, response_text) VALUES (?, ?, ?)"
        self.cursor.execute(sql, (telegram_id, request_text, response_text))
        self.connection.commit()

    def get_documents_by_client_id(self, client_id):
        """Get all documents associated with a client based on the client ID."""
        sql = "SELECT * FROM documents WHERE client_id = ?"
        self.cursor.execute(sql, (client_id,))
        results = self.cursor.fetchall()
        return results

    def get_csv_path_by_telegram_id(self, telegram_id):
        # Exécuter la requête SQL pour récupérer le chemin du fichier CSV
        sql = "SELECT fichier_csv_path FROM clients WHERE telegram_id = ?"
        self.cursor.execute(sql, (telegram_id,))
        result = self.cursor.fetchone()
        csv_path = result[0] if result else None
        return csv_path

    def get_client_by_telegram_id(self, telegram_id):
        """Get the client based on the Telegram ID."""
        sql = "SELECT * FROM clients WHERE telegram_id = ?"
        self.cursor.execute(sql, (telegram_id,))
        result = self.cursor.fetchone()
        if result:
            return result
        else:
            return None

    def get_client_id_by_telegram_id(self, telegram_id):
        """Get the client ID based on the Telegram ID."""
        sql = "SELECT telegram_id FROM clients WHERE telegram_id = ?"
        self.cursor.execute(sql, (telegram_id,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        else:
            return None
        
    def delete_csv_path_by_telegram_id(self, telegram_id):
        """Delete the CSV file path for a given Telegram ID."""
        # Exécuter la requête SQL pour supprimer le chemin du fichier CSV
        sql = "UPDATE clients SET fichier_csv_path = '' WHERE telegram_id = ?"
        self.cursor.execute(sql, (telegram_id,))
        self.connection.commit()

# View
class BotView:
    def __init__(self, application, chatcompletion_api_key):
        self.application = application
        self.chatcompletion_api_key = chatcompletion_api_key
        self.awaiting_photo = False
        self.response = None

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Consentement : Obtenez le consentement explicite de l'utilisateur avant de collecter, traiter ou stocker ses données personnelles.
        Assurez-vous que l'utilisateur comprend clairement comment ses données seront utilisées.
        Durée de conservation : Ne conservez pas les données personnelles plus longtemps que nécessaire.
        Déterminez une période de conservation appropriée en fonction de la finalité du traitement.

        Send a message when the command /start is issued.
        Quand l'utilisateur se connecte, il doit être vérifié.
        Si il n'existe pas, il faut lui proposer un module de paiement et si il paye, on l'ajoute dans la base de données.
        Si il existe déjà, essayer de charger son fichier CSV s'il existe déjà, ensuite lui proposer le clavier pour charger l'historique et discuter directement avec GPT.
        Si il existe déjà mais n'a pas de fichier, lui proposer le menu d'upload CSV.
        Proposer un bouton pour télécharger son CSV.
        """
        user = update.effective_user
        telegram_id = user.id

        database = Database()
        client = database.get_client_by_telegram_id(telegram_id)
        csv_path = database.get_csv_path_by_telegram_id(telegram_id)

        if client:
            if csv_path:
                # Le bouton 1 doit s'afficher si  self.responseBase est = à None
                button1 = InlineKeyboardButton("Charger l'historique", callback_data="execute_sendGPT")
                button2 = InlineKeyboardButton("Télécharger les fichiers CSV", callback_data="execute_get_CSV")
                button3 = InlineKeyboardButton("Supprimer les fichiers CSV", callback_data="execute_delete_csv")
                reply_markup = InlineKeyboardMarkup([[button1],[button2], [button3]])
                await update.message.reply_text("Menu utilisateur.", reply_markup=reply_markup)
            else:
                button1 = InlineKeyboardButton("Send picture", callback_data="button1")
                button2 = InlineKeyboardButton("Send CSV", callback_data="button2")
                
                reply_markup = InlineKeyboardMarkup([[button1, button2]])
                await update.message.reply_text("Menu d'upload", reply_markup=reply_markup)
        else:
            # Insert new client into the clients table if they have paid
            # Show payment module to the user and insert the client into the database if they pay
            database.insert_client(telegram_id, "", 0, "", "")
            

            button1 = KeyboardButton('Option 1 🌟')
            button2 = KeyboardButton('Option 2 🌕')
            button3 = InlineKeyboardButton('Inline Option ⚡️', callback_data='inline_option')

            # Création du clavier avec les boutons
            keyboard_options = [[button1, button2], [button3]]
            keyboard = ReplyKeyboardMarkup(keyboard_options, resize_keyboard=True)

            # Utilisation du clavier lors de l'envoi d'un message
            await update.message.reply_html(
                rf"Hi {user.mention_html()}!",
                reply_markup=keyboard
            )


        
    async def handle_button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle button callback queries.
        """
        query = update.callback_query
        button = query.data
        
        if button == "button1":
            await query.answer("You pressed Button 1")
            self.awaiting_photo = True
            await query.message.reply_text("Envoyez la photo maintenant dans le chat")
        elif button == "button2":
            await query.answer("You pressed Button 2")
            self.awaiting_csv = True
            await query.message.reply_text("Envoyez le document maintenant dans le chat, le nom du fichier doit contenir l'endroit d'où viens le fichier. Ex: products_export_shopify.csv pour shopify ")
        elif button == "execute_extract":
            await query.answer("You pressed execute_extract")
            await self.extract_info(update, context)
            await query.message.reply_text("infos extraits")
        elif button == "execute_creation_csv":
            await query.answer("You pressed execute_creation_csv")
            await self.creation_client_csv(update, context)
            await query.message.reply_text("infos crées, une fois que vous allez appuiyer sur Exécuter l'envoi cela peux prendre un peux du temps en raison de traitement des informations par l'IA")
        elif button == "execute_sendGPT":
            await query.answer("You pressed execute_sendGPT")
            await self.send_infos_chatGPT(update, context)
            await query.message.reply_text("Traitement terminé, avez vous d'autres questions ? Si vous en avez tapez oui suivit de votre requête")
        elif button == "execute_get_CSV":
            await query.answer("You pressed execute_get_CSV")
            await self.send_file(update, context)
            await query.message.reply_text("Upload terminé.")
        elif button == "execute_delete_csv":
            await query.answer("You pressed execute_delete_csv")
            await self.delete_files(update, context)
            #aussi supp le csv dans documents
            # et les requêtes
            await query.message.reply_text("Suppression terminé.")
            

    async def send_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a file to the user."""
        try:
            telegram_id = update.effective_user.id  
            database = Database()
            csv_path_json = database.get_csv_path_by_telegram_id(telegram_id)
            print(csv_path_json)
            if csv_path_json:
                csv_paths = ast.literal_eval(csv_path_json)
                print(csv_paths)
            else:
                await update.callback_query.message.reply_text("No file paths found.")
                return
        except Exception as e:
            await update.callback_query.message.reply_text(f"Error getting the file paths: {str(e)}")
            traceback.print_exc()
            return
        
        try:
            # Specify the chat ID of the user you want to send the file to
            chat_id = update.effective_user.id
            for file_path in csv_paths:
                try:
                    print(f"Sending file: {file_path}")
                    # Send each file
                    await context.bot.send_document(chat_id=chat_id, document=open(file_path, 'rb'))
                    await update.callback_query.message.reply_text("File sent successfully.")
                except Exception as e:
                    await update.callback_query.message.reply_text(f"Error sending the file: {str(e)}")
                    traceback.print_exc()
        except Exception as e:
            await update.callback_query.message.reply_text(f"Error sending the file: {str(e)}")
            traceback.print_exc()


    async def delete_files(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Delete files."""
        try:
            telegram_id = update.effective_user.id  
            database = Database()
            csv_path_json = database.get_csv_path_by_telegram_id(telegram_id)
            
            if csv_path_json:
                csv_paths = ast.literal_eval(csv_path_json)
            else:
                await update.callback_query.message.reply_text("No file paths found.")
                return
        except Exception as e:
            await update.callback_query.message.reply_text(f"Error getting the file paths: {str(e)}")
            traceback.print_exc()
            return
        
        try:
            for file_path in csv_paths:
                try:
                    print(f"Deleting file: {file_path}")
                    if os.path.exists(file_path):
                        # Delete the file if it exists
                        os.remove(file_path)
                        print("File deleted successfully.")
                    else:
                        await update.callback_query.message.reply_text(f"File not found: {file_path}")
                        print("File not found.")
                except Exception as e:
                    await update.callback_query.message.reply_text(f"Error deleting the file: {str(e)}")
                    traceback.print_exc()
            
            # Delete the file paths from the database
            database.delete_csv_path_by_telegram_id(telegram_id)
            
            await update.callback_query.message.reply_text("Files deleted successfully.")
            
        except Exception as e:
            await update.callback_query.message.reply_text(f"Error deleting the files: {str(e)}")
            traceback.print_exc()


    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        await update.message.reply_text("Help!")

    async def echo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        C'est ici que le chat avec le bot serra géré.
        Quand l'utilisateur tape quelque chose dans le tchat, ça test si l'utilisateur a répondu oui,
        Si oui a été répondu il va envoyer la réponse
        Si non il va juste répondre Merci d'avoir utilisé nos services.
        Si il n'y a ni oui ni non l'utilisateur recevras le message d'aide.

        # Trouver un moyen d'ajouter un contexte global à mon avis créer 2 variables self.response différentes pour récupérer la première conv qui est celle qui contiens toutes les infos
        """
        if "oui" in update.message.text:
            message = update.message.text
            telegram_id = update.effective_user.id
            database = Database()
            client = database.get_client_by_telegram_id(telegram_id)
            if client:
                client_id = client[0]  # Get the client ID from the client tuple
                response_from_user = message
                database.insert_request(client_id, response_from_user, "")
                # Send message to ChatCompletion API
                api_key = self.chatcompletion_api_key
                # Faut récupérer le contexte des messages.
                if self.response is not None:
                    old_prompt = self.response
                completion = ChatCompletion.create(
                    model="gpt-3.5-turbo",  # Specify the model
                    api_key=api_key,
                    messages=[
                        {"role": "system", "content": old_prompt},
                        {"role": "user", "content": response_from_user},
                    ]
                )
                response = completion.choices[0].message.content.strip()
                database.insert_request(client_id, response_from_user, response)
                await update.message.reply_text(response)
                await update.message.reply_text("Veux tu poser une autre question ?")
                self.response = response
            else:
                await update.message.reply_text("Error: Client not found. Please start the bot by sending the /start command.")
        elif "non" in update.message.text:
            await update.message.reply_text("Merci d'avoir utilisé nos services.")
        else:
            await update.message.reply_text("N'oubliez pas que les commandes de bases sont: /start pour le menu d'accueil, .")
    
    async def process_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process the received document."""
        if self.awaiting_csv:
            document = update.message.document
            file = await document.get_file()
            # Specify the path where you want to save the document
            current_directory = os.path.abspath(os.getcwd())
            document_directory = os.path.join(current_directory, "documents")
            if not os.path.exists(document_directory):
                os.makedirs(document_directory)
            filename = document.file_name
            file_path = os.path.join(document_directory, filename)
            try:
                await file.download_to_drive(file_path)
                # Get the client ID based on the Telegram ID
                telegram_id = update.effective_user.id
                database = Database()
                client_id = database.get_client_id_by_telegram_id(telegram_id)
                if client_id:
                    # Insert the document into the documents table
                    database.insert_document(client_id, file_path)
                    await update.message.reply_text("Document saved successfully.")
                    button1 = InlineKeyboardButton("Exécuter l'extraction", callback_data="execute_extract")
                    reply_markup = InlineKeyboardMarkup([[button1]])
                    await update.message.reply_text("Menu d'extracion des infos", reply_markup=reply_markup)
                else:
                    await update.message.reply_text("Error: Client not found. Please start the bot by sending the /start command.")
            except Exception as e:
                await update.message.reply_text(f"Error processing the document: {str(e)}")
            self.awaiting_csv = False
        else:
            return
        
    async def extract_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Extract information from all the saved documents for a client."""
        # ça serrai bien de trouver le stock moyen
        def is_valid_filename(filename):
            if "shopify" in filename:
                return 1
            elif "autre_nom" in filename:
                return 2
            else:
                return 0

        def search_keywords(extracted_info):
            #Ajouter Handle et Variant Inventory Qty
            keywords = ["Cost per item", "Variant Price", "Handle", "Variant Inventory Qty"]
            found_rows = []
            for section_key, section_value in extracted_info.items():
                for row, row_data in section_value.items():
                    found_data = {}
                    for column_name, column_value in row_data.items():
                        if column_name in keywords:
                            found_data[column_name] = column_value
                    if found_data:
                        found_rows.append(found_data)
            return found_rows
        telegram_id = update.effective_user.id
        database = Database()
        client = database.get_client_by_telegram_id(telegram_id)
        if client:
            client_id = client[0]  # Get the client ID from the client tuple
            documents = database.get_documents_by_client_id(client_id)
            if documents:
                extracted_info = {}
                section_number = 1
                for document in documents:
                    document_path = document[2]  # Assuming the document_path column is at index 2
                    print(document_path)
                    section_key = f"section{section_number}"
                    section_number += 1
                    valid_filename = is_valid_filename(document_path)
                    if valid_filename == 1:
                        section_key = "shopify"
                    elif valid_filename == 2:
                        section_key = "autre_nom"
                    if valid_filename != 0:
                        with open(document_path, "r") as csv_file:
                            reader = csv.reader(csv_file, delimiter=',')
                            header = next(reader)  # Get the header row
                            extracted_info[section_key] = {}
                            for i, row in enumerate(reader):
                                row_dict = {}
                                for j, value in enumerate(row):
                                    column_name = header[j]
                                    row_dict[column_name] = value
                                extracted_info[section_key][i] = row_dict
                    #else:
                    finded_keywords = search_keywords(extracted_info)
                    # Do something with the extracted information
                    response = {}
                    for section, section_data in extracted_info.items():
                        response[section] = {}
                        for row_number, _ in section_data.items():
                            response[section][row_number] = finded_keywords[row_number]
                    print(response)                
                
                self.response = response
                # Ajouter button2 qui permet d'annuler l'exécution, donc drop colonne contiens le path de ce doc et self.responde = ""
                button1 = InlineKeyboardButton("Exécuter la création", callback_data="execute_creation_csv")
                reply_markup = InlineKeyboardMarkup([[button1]])
                await update.callback_query.message.reply_text("Menu création document", reply_markup=reply_markup)
            else:
                await update.callback_query.message.reply_text("No saved documents found.")
        else:
            await update.callback_query.message.reply_text("Error: Client not found. Please start the bot by sending the /start command.")
    
    async def creation_client_csv(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_id = update.effective_user.id  # Obtenez l'ID Telegram du client
        # Créez la fonction pour ces F:
        # 1. Rotation = Chiffre d'affaires / stock moyen
        # Ecc = Productivité/profitabilité
        if self.response:
            with open('client_data.csv', 'w', newline='') as csvfile:
                fieldnames = ['PV', 'Stock', 'PR', 'Nom', 'profit', 'marge', 'CFU', 'CVU', 'CA', 'CF', 'CV', 'Risque']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                CV_total = 0
                for section, section_data in self.response.items():
                    if "shopify" in section:
                        for row, row_data in section_data.items():
                            PV = 0.
                            stock = 0
                            PR = 0.
                            nom = ""
                            profit = 0.
                            CFU = 0.
                            CVU = 0.
                            CA = 0.
                            Risque = 0.
                            CF = 0.
                            CV = 0.
                            for cle, data in row_data.items():
                                if "Handle" in cle:
                                    nom = data
                                try:
                                    if PR is not None and PR != "":
                                        if "Cost per item" in cle:
                                            PR = float(data)
                                except Exception as e:
                                    print("le prix de revient na pas été indiqué sur le produit " + nom + " dans shopify, le calcul continue donc avec PR = 1")
                                    PR = 1
                                if "Variant Inventory Qty" in cle:
                                    stock = float(data)
                                try:
                                    if PV is not None and PV != "":
                                        if "Variant Price" in cle:
                                            PV = float(data)
                                except Exception as e:
                                    print("le prix de vente na pas été indiqué sur le produit " + nom + " dans shopify, le calcul continue donc avec PV = 1")
                                    PV = 1                                                                       
                            marge = PV - PR                           
                            Risque = marge / PR
                            profit = marge * stock
                            CA = PV * stock
                            CV = PR
                            CV_total += PR
                            writer.writerow({'PV': PV, 'Stock': stock, 'PR': PR, 'Nom': nom, 'marge': marge, 'profit': profit, 'CA': CA, 'CFU': 0, 'CVU': PR, 'CF': None, 'CV': None, 'Risque': Risque})
            #CV
            print("total CV: " + str(CV_total))
            #Créer nv csv
            with open('client_data_totaux.csv', 'w', newline='') as csvfile:
                fieldnames = ['CF', 'CV']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({'CF': None, 'CV': CV_total})
            # Obtenez le chemin absolu du fichier CSV
            csv_path = []
            csv_file_path = os.path.abspath('client_data.csv')
            csv_file_path_totaux = os.path.abspath('client_data_totaux.csv')
            csv_path.append(csv_file_path)
            csv_path.append(csv_file_path_totaux)           
            # Mettez à jour la colonne fichier_csv dans la table clients
            database = Database()
            database.update_client_csv(telegram_id, str(csv_path))
            await update.callback_query.message.reply_text("Data saved to client_data.csv")
            # Ajouter un button2 qui permet d'annuler l'action, donc supp le path csv de table client.
            button1 = InlineKeyboardButton("Exécuter l'envoi", callback_data="execute_sendGPT")
            reply_markup = InlineKeyboardMarkup([[button1]])    
            await update.callback_query.message.reply_text("Continuer", reply_markup=reply_markup)
        else:
            await update.callback_query.message.reply_text("No response available.")

    async def send_infos_chatGPT(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        def is_valid_filename(filename):
            if filename.endswith("client_data.csv"):
                return 1
            elif filename.endswith("client_data_totaux.csv"):
                return 2
            else:
                return 0

        def get_client_info_from_csv(file_paths):
            client_info = {}
            for path in file_paths:
                section_number = is_valid_filename(path)        
                client_info[section_number] = []
                if os.path.isfile(path):
                    with open(path, 'r') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            client_info[section_number].append(row)
            return client_info

        telegram_id = update.effective_user.id  
        database = Database()
        csv_path_json = database.get_csv_path_by_telegram_id(telegram_id)
        print(csv_path_json)
        if csv_path_json:
            csv_paths = ast.literal_eval(csv_path_json)
            print(csv_paths)
            client_info = get_client_info_from_csv(csv_paths)
            row_informations = ""
            for section, section_data in client_info.items():
                if section == 1:
                    for row_data in section_data:
                        PV = float(row_data.get('PV', 0))
                        stock = float(row_data.get('Stock', 0))
                        PR = float(row_data.get('PR', 0))
                        nom = row_data.get('Nom', '')
                        profit = float(row_data.get('profit', 0))
                        marge = float(row_data.get('marge', 0))
                        CFU = float(row_data.get('CFU', 0))
                        CVU = float(row_data.get('CVU', 0))
                        CA = float(row_data.get('CA', 0))
                        Risque = float(row_data.get('Risque', 0))
                        # Perform operations with PV, stock, PR, nom values
                        # Example: Print the values
                        print(f"PV: {PV}, Stock: {stock}, PR: {PR}, Nom: {nom}, profit: {profit}, marge: {marge}, CFU: {CFU}, CVU: {CVU}, CA: {CA}, Risque:{Risque}") 
                        row_informations += """
                        Voici un produit de shopify dons quelques informations, """ + "PV: " + str(PV) + "Nom: " + nom + "stock: " + str(stock) + " " + "PR: " + str(PR) + " " + "Profit: " + str(profit) + " " + "marge: " + str(marge) + " " + "CFU: "+ str(CFU) + " " + "CVU: " + str(CVU) + " " + "CA: " + str(CA) + " " + "Indicateur1: " + str(Risque)  + """
                        """

            for section, section_data in client_info.items():
                if section == 2:
                    for row_data in section_data:
                        CF = row_data.get('CF')
                        CV = row_data.get('CV')
                        # Perform operations with PV, stock, PR, nom values
                        # Example: Print the values
                        print(f"CV: {CV}, CF: {CF}")

            api_key = self.chatcompletion_api_key
        
        personnaliteGPT = """
        Je veux que vous agissiez comme un comptable et que vous trouviez des moyens créatifs de gérer les finances.
        """

        objectifFinalGPT = """
        Vous devrez tenir compte de la budgétisation, des stratégies d’investissement et de la gestion des risques lorsque vous créerez un plan financier pour votre client.
        """

        maniereProcceder = """
        Dans certains cas, vous devrez également fournir des conseils sur les lois et réglementations fiscales afin de les aider à maximiser leurs bénéfices. 
        """

        styleDuChat = """
        Dans le style de warren buffet
        """

        conditionGPT = """
        Dont voici les informations, """ + row_informations +   """
        Voici une liste des acronymes que j'utilise, PV = Prix de vente, PR = Prix de vente, CFU= Cout Fixe Unitaire, CVU = Cout Variable Unitaire, CA = Chiffre d'Affaires. Dans l'explications des variables, utilies ceux que je veux de te donner.
        """

        tacheAeffectuer = """
        Explique chaques variables et ce qu'elle représente, pour l'indicateur 1 base toi sur ça "Dans les informations d'un produit, la variable 'indicateur1' représente la marge divisée par le prix de revient, ce qui donne le rapport, plus le rapport est élevé, plus le pourcentage obtenu est élevé, ce qui indique un risque réduit. Cela signifie que le risque est inversement proportionnel au pourcentage obtenu. Donc si le rapport indicateur1 dépasse 1.0 cela est super !! oublie pas ça s'est important"
        Quand tu as le rapport de chaque produit pour indicateur1 refais un calcul pour vérifier quelle est le plus grand de tous et oublie pas c'est marge / PR et pas PV / PR !!.
        Maintenant que tu as compris la variable risque fait un exemple avec les infos obtenus des produit que je t'ai donné avant dans le texte. En bref tu dois chercher celui qui a le plus gros rapport rapport indicateur1 par exemple 1.666 et c'est celui qui a le plus grand nombre qui est le moin risqué. (Cela signifie que le risque est inversement proportionnel au pourcentage obtenu.")
        Quand tu parles du rapport indicateur1 oublie pas que le pourcentage ne veux pas dire la quantité de risque dans mon cas, ne te trompe pas et base toi jsute sur mon exemple rien d'autre pour le rapport indicateur1.
        devant chaque produit analysé donne le nom pour que on sache de quelle produit tu parles.
        """

        pre_prompt = personnaliteGPT + objectifFinalGPT + maniereProcceder + styleDuChat + conditionGPT + tacheAeffectuer
        prompt = pre_prompt

        completion = ChatCompletion.create(
            model="gpt-3.5-turbo-16k",  # Specify the model
            api_key=api_key,
            messages=[
                {"role": "system", "content": "You are an helpful accountant assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        try:
            # Vérifier le prompt dépasse pas un certains nombres de token.
            response = completion.choices[0].message.content.strip()
        except Exception as e:
                await update.message.reply_text(f"Error processing the request, re try again later: {str(e)}")
        print(response)
        self.response = response
        await update.callback_query.message.reply_text(response)
        
    async def process_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process the received image."""
        if self.awaiting_photo:
            photo = update.message.photo[-1]  # Get the last (highest resolution) photo
            file = await photo.get_file()
            # Specify the path where you want to save the image
            current_directory = os.path.abspath(os.getcwd())
            image_directory = os.path.join(current_directory, "images")
            if not os.path.exists(image_directory):
                os.makedirs(image_directory)
            filename = "image.jpg"
            file_path = os.path.join(image_directory, filename)
            try:
                await file.download_to_drive(file_path)
                text = self.extract_text_from_image(file_path)
                telegram_id = update.effective_user.id
                database = Database()
                client = database.get_client_by_telegram_id(telegram_id)
                if client:
                    client_id = client[0]  # Get the client ID from the client tuple
                    database.insert_request(client_id, text, "")
                    # Send message to ChatCompletion API
                    api_key = self.chatcompletion_api_key
                    intro = "Fais un tri qui te semble le plus logique et ignore les infos manquantes (j'ai besoin des nombres).\n"
                    data = "Voici quelques formules (Profit=Marge x Volume), (Marge= Prix de Vente (unitaire) – Prix de Revient (unitaire)), (Prix de Revient = Couts Fixes (unitaire) + Couts Variables (unitaire))), (Chiffres d’affaires = prix de vente (unitaire)x volume)  \n"
                    variable_1 = "Si tu peux essaye d'utiliser de voir quelles formules sont utilisable suivant les données que tu recevras."
                    variable_2 = "Voici une autre formule (Rotation= Chiffre d’affaires / stock moyen ).\n"
                    variable_3 = "Si tu peux essaye de trouver la réponse pour Rotation.\n"
                    pre_prompt = intro + data + variable_1 + variable_2 + variable_3
                    prompt = pre_prompt + text  # Use the extracted text as the prompt
                    completion = ChatCompletion.create(
                        model="gpt-3.5-turbo",  # Specify the model
                        api_key=api_key,
                        messages=[
                            {"role": "user", "content": "J'aimerais en savoir plus sur la variable 'Risque' dans les informations d'un produit."},
                            {"role": "system", "content": "Dans les informations d'un produit, la variable 'Risque' représente la marge divisée par le prix de revient, ce qui donne le rapport risque. Si le rapport obtenu pour le risque est élevé, cela indique un risque réduit. En d'autres termes, plus le rapport risque est élevé, plus le pourcentage obtenu est bas. Cela signifie que le risque est inversement proportionnel au pourcentage obtenu."},
                            {"role": "user", "content": "Merci pour l'explication ! Cela signifie-t-il que plus le pourcentage obtenu est bas, plus le risque est élevé ?"},
                            {"role": "system", "content": "Exactement ! Lorsque le pourcentage obtenu est bas, cela indique un risque élevé, ce qui peut être considéré comme un risque important associé au produit. À l'inverse, si le pourcentage obtenu est élevé, cela indique un risque réduit, ce qui est généralement préférable."},
                            {"role": "user", "content": "Compris ! Merci beaucoup pour ces informations, je vais te partager mes informations sur ma comptabilité: "},
                            {"role": "user", "content": "Voici mes informations" + prompt}
                        ]
                    )
                    response = completion.choices[0].message.content.strip()
                    database.insert_request(client_id, text, response)
                    await update.message.reply_text(response)
                    await update.message.reply_text("Text extracted and saved to the database.")
                else:
                    await update.message.reply_text("Error: Client not found. Please start the bot by sending the /start command.")
            except Exception as e:
                await update.message.reply_text(f"Error processing the image: {str(e)}")
            self.awaiting_photo = False
        else:
            return

    def extract_text_from_image(self, image_path):
        """Extract text from the given image using OCR."""
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text

# Controller
class BotController:
    def __init__(self, application, chatcompletion_api_key):
        self.view = BotView(application, chatcompletion_api_key)
        
    def register_handlers(self, application):
        application.add_handler(CommandHandler("start", self.view.start))
        application.add_handler(CallbackQueryHandler(self.view.handle_button_callback))
        application.add_handler(CommandHandler("help", self.view.help_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.view.echo))
        application.add_handler(MessageHandler(filters.PHOTO, self.view.process_image))
        application.add_handler(MessageHandler(filters.Document.MimeType('text/csv'), self.view.process_document))
        application.add_handler(CommandHandler("extract", self.view.extract_info))
        application.add_handler(CommandHandler("creation", self.view.creation_client_csv))
        application.add_handler(CommandHandler("sendGPT", self.view.send_infos_chatGPT))
        
def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    env_values = dotenv_values(".env")
    token = env_values["TOKEN"]
    application = Application.builder().token(token).build()
    # Get your ChatCompletion API key from environment variables or .env file
    chatcompletion_api_key = env_values["chatgpt_key"]
    controller = BotController(application, chatcompletion_api_key)
    controller.register_handlers(application)
    # Create the clients table if it doesn't exist
    database = Database()
    database.create_clients_table()
    database.create_requests_table()
    database.create_documents_table()
    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == "__main__":
    main()