# -*- coding: utf-8 -*-
"""
/***************************************************************************
        pyArchInit Plugin  - A QGIS plugin to manage archaeological dataset
                             stored in Postgres
                             -------------------
    begin                : 2007-12-01
    copyright            : (C) 2008 by Luca Mandolesi; Enzo Cocca <enzo.ccc@gmail.com>
    email                : mandoluca at gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from __future__ import absolute_import

import os
import re
from datetime import date

import sys
from builtins import range
from builtins import str

import openai
import requests
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtWidgets import QFileDialog,QDialog, QMessageBox,QCompleter,QComboBox,QInputDialog
from qgis.PyQt.uic import loadUiType
from qgis.core import QgsSettings
import csv, sqlite3
from ..modules.db.pyarchinit_conn_strings import Connection
from ..modules.db.pyarchinit_db_manager import Pyarchinit_db_management
from ..modules.db.pyarchinit_utility import Utility
from ..modules.gis.pyarchinit_pyqgis import Pyarchinit_pyqgis
from ..modules.utility.pyarchinit_error_check import Error_check
from ..modules.utility.askgpt import MyApp
from ..gui.sortpanelmain import SortPanelMain

MAIN_DIALOG_CLASS, _ = loadUiType(os.path.join(os.path.dirname(__file__), os.pardir, 'gui', 'ui', 'Thesaurus.ui'))


class pyarchinit_Thesaurus(QDialog, MAIN_DIALOG_CLASS):
    L=QgsSettings().value("locale/userLocale")[0:2]
    if L=='it':
        MSG_BOX_TITLE = "PyArchInit - Thesaurus"
    
    elif L=='de':
        MSG_BOX_TITLE = "PyArchInit - Thesaurus"
    else:
        MSG_BOX_TITLE = "PyArchInit - Thesaurus" 
    DATA_LIST = []
    DATA_LIST_REC_CORR = []
    DATA_LIST_REC_TEMP = []
    REC_CORR = 0
    REC_TOT = 0
    HOME = os.environ['PYARCHINIT_HOME']
    if L=='it':
        STATUS_ITEMS = {"b": "Usa", "f": "Trova", "n": "Nuovo Record"}
    else :
        STATUS_ITEMS = {"b": "Current", "f": "Find", "n": "New Record"}
    BROWSE_STATUS = "b"
    SORT_MODE = 'asc'
    if L=='it':
        SORTED_ITEMS = {"n": "Non ordinati", "o": "Ordinati"}
    else:
        SORTED_ITEMS = {"n": "Not sorted", "o": "Sorted"}
    SORT_STATUS = "n"
    UTILITY = Utility()
    DB_MANAGER = ""
    TABLE_NAME = 'pyarchinit_thesaurus_sigle'
    MAPPER_TABLE_CLASS = "PYARCHINIT_THESAURUS_SIGLE"
    NOME_SCHEDA = "Scheda Thesaurus Sigle"
    ID_TABLE = "id_thesaurus_sigle"
    if L=='it':
        CONVERSION_DICT = {
            ID_TABLE: ID_TABLE,
            "Nome tabella": "nome_tabella",
            "Sigla": "sigla",
            "Sigla estesa": "sigla_estesa",
            "Descrizione": "descrizione",
            "Tipologia sigla": "tipologia_sigla",
            "Lingua": "lingua"
        }

        SORT_ITEMS = [
            ID_TABLE,
            "Nome tabella",
            "Sigla",
            "Sigla estesa",
            "Descrizione",
            "Tipologia sigla",
            "Lingua"
        ]
    elif L=='de':
        CONVERSION_DICT = {
            ID_TABLE: ID_TABLE,
            "Tabellenname": "nome_tabella",
            "Abkürzung": "sigla",
            "Erweitertes Abkürzungszeichen": "sigla_estesa",
            "Beschreibung": "descrizione",
            "Abkürzungtyp": "tipologia_sigla",
            "Sprache": "lingua"
        }

        SORT_ITEMS = [
            ID_TABLE,
            "Tabellenname",
            "Abkürzung",
            "Erweitertes Abkürzungszeichen",
            "Beschreibung",
            "Abkürzungtyp",
            "Sprache"
        ]
    else:
        CONVERSION_DICT = {
            ID_TABLE: ID_TABLE,
            "Table name": "nome_tabella",
            "Code": "sigla",
            "Code whole": "sigla_estesa",
            "Description": "descrizione",
            "Code typology": "tipologia_sigla",
            "Language": "lingua"
        }

        SORT_ITEMS = [
            ID_TABLE,
            "Table name",
            "Code",
            "Code whole",
            "Description",
            "Code typology",
            "Language"
        ]   
    LANG = {
        "IT": ['it_IT', 'IT', 'it', 'IT_IT'],
        "EN_US": ['en_US','EN_US','en','EN','EN_EN'],
        "DE": ['de_DE','de','DE', 'DE_DE'],
        "FR": ['fr_FR','fr','FR', 'FR_FR'],
        "ES": ['es_ES','es','ES', 'ES_ES'],
        "PT": ['pt_PT','pt','PT', 'PT_PT'],
        "SV": ['sv_SV','sv','SV', 'SV_SV'],
        "RU": ['ru_RU','ru','RU', 'RU_RU'],
        "RO": ['ro_RO','ro','RO', 'RO_RO'],
        "AR": ['ar_AR','ar','AR', 'AR_AR'],
        "PT_BR": ['pt_BR','PT_BR'],
        "SL": ['sl_SL','sl','SL', 'SL_SL'],
    }

    TABLE_FIELDS = [
        "nome_tabella",
        "sigla",
        "sigla_estesa",
        "descrizione",
        "tipologia_sigla",
        "lingua"
    ]
    DB_SERVER = "not defined"  ####nuovo sistema sort

    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.pyQGIS = Pyarchinit_pyqgis(iface)
        self.setupUi(self)
        self.currentLayerId = None
        #self.comboBox_nome_tabella.currentTextChanged.connect(self.charge_n_sigla)
        try:
            self.on_pushButton_connect_pressed()
        except Exception as e:
            QMessageBox.warning(self, "Connection system", str(e), QMessageBox.Ok)
        self.comboBox_sigla_estesa.editTextChanged.connect(self.find_text)
        self.check_db()

    def read_api_key(self, path):
        with open(path, 'r') as f:
            return f.read().strip()

    def write_api_key(self, path, api_key):
        with open(path, 'w') as f:
            f.write(api_key)

    def is_valid_api_key(self,api_key):
        openai.api_key = api_key

        try:
            # Testa la chiave API effettuando una richiesta di completamento
            response = openai.Completion.create(
                engine="text-davinci-002",  # Sostituisci con il motore che intendi utilizzare
                prompt="Translate the following English text to French: '{}'",
                max_tokens=5
            )
            if response:
                return True
        except openai.Error as e:
            print(f"Received an error: {e}")
            return False

    def apikey_gpt(self):
        BIN = os.path.join(self.HOME, "bin")
        path_key = os.path.join(BIN, 'gpt_api_key.txt')

        if os.path.exists(path_key):
            api_key = self.read_api_key(path_key)

            if self.is_valid_api_key(api_key):
                return api_key
            else:
                reply = QMessageBox.question(None, 'Warning', 'Apikey non valida' + '\n'
                                             + 'Clicca ok per inserire la chiave',
                                             QMessageBox.Ok | QMessageBox.Cancel)
                if reply == QMessageBox.Ok:
                    api_key, ok = QInputDialog.getText(None, 'Apikey gpt', 'Inserisci apikey valida:')
                    if ok:
                        self.write_api_key(path_key, api_key)
                        return self.read_api_key(path_key)
                else:
                    return api_key

        else:
            api_key, ok = QInputDialog.getText(None, 'Apikey gpt', 'Inserisci apikey:')
            if ok:
                self.write_api_key(path_key, api_key)
                return self.read_api_key(path_key)
    def check_db(self):
        conn = Connection()
        conn_str = conn.conn_str()
        test_conn = conn_str.find('sqlite')

        if test_conn == 0:
            self.pushButton_import_csvthesaurus.setHidden(False)
        else:
            self.pushButton_import_csvthesaurus.setHidden(True)

    def contenuto(self, b):
        text = MyApp.ask_gpt(self,
                             f'forniscimi una descrizione e 3 link wikipidia riguardo a questo contenuto {b}, tenendo presente che il contesto è archeologico',
                             self.apikey_gpt())
        #url_pattern = r"(https?:\/\/\S+)"
        #urls = re.findall(url_pattern, text)
        return text#, urls
    def webview(self):
        description=str(self.textEdit_descrizione_sigla.toPlainText())

        links = re.findall("(?P<url>https?://[^\s]+)", description)

        # Create an HTML string to display in the QWebView
        html_string = "<html><body>"
        html_string += "<p>" + description.replace("\n", "<br>") + "</p>"
        html_string += "<ul>"
        for link in links:
            html_string += f"<li><a href='{link}'>{link}</a></li>"
        html_string += "</ul>"
        html_string += "</body></html>"
        vista=html_string
        QMessageBox.information(self, 'testo', str(html_string))
        # Display the HTML in a QWebView widget
        url=QUrl("about:blank")
        self.webView_adarte.setHtml(vista,url)


        # Show the QWebView widget
        self.webView_adarte.show()
    def handleComboActivated(self, index):
        selected_text = self.comboBox_sigla_estesa.itemText(index)
        generate_text = self.contenuto(selected_text)
        reply = QMessageBox.information(self, 'Info', generate_text, QMessageBox.Ok | QMessageBox.Cancel)
        if reply == QMessageBox.Ok:
            self.textEdit_descrizione_sigla.setText(generate_text)

    def on_suggerimenti_pressed(self):
        s = self.contenuto(self.comboBox_sigla_estesa.currentText())
        generate_text = s
        QMessageBox.information(self, 'info', str(generate_text), QMessageBox.Ok | QMessageBox.Cancel)

        if QMessageBox.Ok:
            self.textEdit_descrizione_sigla.setText(str(generate_text))
            self.webview()
        else:
            pass
    def find_text(self):

        if self.comboBox_sigla_estesa.currentText()=='':
            uri= 'https://vast-lab.org/thesaurus/ra/vocab/index.php?'
        else:

            uri= 'https://vast-lab.org/thesaurus/ra/vocab/index.php?ws=t&xstring='+ self.comboBox_sigla_estesa.currentText()+'&hasTopTerm=&hasNote=NA&fromDate=&termDeep=&boton=Conferma&xsearch=1#xstring'
            self.comboBox_sigla_estesa.completer().setCompletionMode(QCompleter.PopupCompletion)
            self.comboBox_sigla_estesa.setInsertPolicy(QComboBox.NoInsert)
        self.webView_adarte.load(QUrl(uri))
        self.webView_adarte.show()
    

    def  on_pushButton_import_csvthesaurus_pressed(self):
        '''funzione valida solo per sqlite'''
        
        s = QgsSettings()
        dbpath = QFileDialog.getOpenFileName(
            self,
            "Set file name",
            '',
            " csv pyarchinit thesaurus (*.csv)"
        )[0]
        filename=dbpath#.split("/")[-1]
        try:
            conn = Connection()
            conn_str = conn.conn_str()
            conn_sqlite = conn.databasename()
            
            sqlite_DB_path = '{}{}{}'.format(self.HOME, os.sep,
                                           "pyarchinit_DB_folder")
            
            con = sqlite3.connect(sqlite_DB_path +os.sep+ conn_sqlite["db_name"])
            cur = con.cursor()
            
            with open(str(filename),'r') as fin: 
                dr = csv.DictReader(fin) # comma is default delimiter
                to_db = [(i['nome_tabella'], i['sigla'], i['sigla_estesa'], i['descrizione'],i['tipologia_sigla'], i['lingua']) for i in dr]

            cur.executemany("INSERT INTO pyarchinit_thesaurus_sigle (nome_tabella, sigla, sigla_estesa, descrizione, tipologia_sigla, lingua ) VALUES (?, ?, ?,?,?,?);", to_db)
            con.commit()
            con.close()
            
        except AssertionError as e:
            QMessageBox.warning(self, 'error', str(e), QMessageBox.Ok)
        self.pushButton_view_all.click()  
    def enable_button(self, n):
        self.pushButton_connect.setEnabled(n)

        self.pushButton_new_rec.setEnabled(n)

        self.pushButton_view_all.setEnabled(n)

        self.pushButton_first_rec.setEnabled(n)

        self.pushButton_last_rec.setEnabled(n)

        self.pushButton_prev_rec.setEnabled(n)

        self.pushButton_next_rec.setEnabled(n)

        self.pushButton_delete.setEnabled(n)

        self.pushButton_new_search.setEnabled(n)

        self.pushButton_search_go.setEnabled(n)

        self.pushButton_sort.setEnabled(n)

    def enable_button_search(self, n):
        self.pushButton_connect.setEnabled(n)

        self.pushButton_new_rec.setEnabled(n)

        self.pushButton_view_all.setEnabled(n)

        self.pushButton_first_rec.setEnabled(n)

        self.pushButton_last_rec.setEnabled(n)

        self.pushButton_prev_rec.setEnabled(n)

        self.pushButton_next_rec.setEnabled(n)

        self.pushButton_delete.setEnabled(n)

        self.pushButton_save.setEnabled(n)

        self.pushButton_sort.setEnabled(n)

    def on_pushButton_connect_pressed(self):
        conn = Connection()
        conn_str = conn.conn_str()
        test_conn = conn_str.find('sqlite')

        if test_conn == 0:
            self.DB_SERVER = "sqlite"

        try:
            self.DB_MANAGER = Pyarchinit_db_management(conn_str)
            self.DB_MANAGER.connection()
            self.charge_records()  # charge records from DB
            # check if DB is empty
            if bool(self.DATA_LIST):
                self.REC_TOT, self.REC_CORR = len(self.DATA_LIST), 0
                self.DATA_LIST_REC_TEMP = self.DATA_LIST_REC_CORR = self.DATA_LIST[0]
                self.BROWSE_STATUS = 'b'
                self.label_status.setText(self.STATUS_ITEMS[self.BROWSE_STATUS])
                self.label_sort.setText(self.SORTED_ITEMS["n"])
                self.set_rec_counter(len(self.DATA_LIST), self.REC_CORR + 1)
                self.charge_list()
                self.fill_fields()
                #
            else:
                if self.L=='it':
                    QMessageBox.warning(self,"BENVENUTO", "Benvenuto in pyArchInit" + "Scheda Campioni" + ". Il database e' vuoto. Premi 'Ok' e buon lavoro!",
                                        QMessageBox.Ok)
                
                elif self.L=='de':
                    
                    QMessageBox.warning(self,"WILLKOMMEN","WILLKOMMEN in pyArchInit" + "Munsterformular"+ ". Die Datenbank ist leer. Tippe 'Ok' und aufgehts!",
                                        QMessageBox.Ok) 
                else:
                    QMessageBox.warning(self,"WELCOME", "Welcome in pyArchInit" + "Samples form" + ". The DB is empty. Push 'Ok' and Good Work!",
                                        QMessageBox.Ok)   
                self.charge_list()
                self.BROWSE_STATUS = 'x'
                self.on_pushButton_new_rec_pressed()
        except Exception as e:
            e = str(e)
            if e.find("no such table"):
            
                if self.L=='it':
                    msg = "La connessione e' fallita {}. " \
                          "E' NECESSARIO RIAVVIARE QGIS oppure rilevato bug! Segnalarlo allo sviluppatore".format(str(e))
                    self.iface.messageBar().pushMessage(self.tr(msg), Qgis.Warning, 0)
                
                    self.iface.messageBar().pushMessage(self.tr(msg), Qgis.Warning, 0)
                elif self.L=='de':
                    msg = "Verbindungsfehler {}. " \
                          " QGIS neustarten oder es wurde ein bug gefunden! Fehler einsenden".format(str(e))
                    self.iface.messageBar().pushMessage(self.tr(msg), Qgis.Warning, 0)
                else:
                    msg = "The connection failed {}. " \
                          "You MUST RESTART QGIS or bug detected! Report it to the developer".format(str(e))        
            else:
                if self.L=='it':
                    msg = "Attenzione rilevato bug! Segnalarlo allo sviluppatore. Errore: ".format(str(e))
                    self.iface.messageBar().pushMessage(self.tr(msg), Qgis.Warning, 0)
                
                elif self.L=='de':
                    msg = "ACHTUNG. Es wurde ein bug gefunden! Fehler einsenden: ".format(str(e))
                    self.iface.messageBar().pushMessage(self.tr(msg), Qgis.Warning, 0)  
                else:
                    msg = "Warning bug detected! Report it to the developer. Error: ".format(str(e))
                    self.iface.messageBar().pushMessage(self.tr(msg), Qgis.Warning, 0)

    def charge_list(self):
        #pass
        self.comboBox_lingua.clear()
        lingua = []
        for key, values in self.LANG.items():
            lingua.append(key)
        self.comboBox_lingua.addItems(lingua)
    def charge_n_sigla(self):    
        self.comboBox_tipologia_sigla.clear()
        self.comboBox_tipologia_sigla.update()
        list1=['1.1']
        list2=['2.1','2.2','2.3','2.4','2.5','2.6','2.7','2.8','2.9','2.10','2.11','2.12','2.13','2.14','2.15','2.16','2.17','2.18','2.19','2.20','2.21','2.22','2.23','2.24','2.25','2.26','2.27','2.28','2.29','2.30','2.31','2.32','2.33','2.34','2.35','2.36','2.37','2.38','2.39','2.40','2.41','2.42','201.201','202.202','203.203']
        list3=['3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','301.301']
        list4=['4.1']
        list5=['5.1','5.2','5.3']
        list6=['6.1','6.2','6.3','6.4','6.5','6.6','6.7','6.8']
        list7=['7.1','7.2','7.3','7.4','7.5','7.6','7.7','701.701','702.702']
        list8=['8.1','8.2','8.3','8.4','8.5','801.801']
        list9 = ['9.1', '9.2']
        if self.comboBox_nome_tabella.currentText()=='site_table':
            
            self.comboBox_tipologia_sigla.clear()
            self.comboBox_tipologia_sigla.addItems(list1)
        if self.comboBox_nome_tabella.currentText()=='us_table':
            self.comboBox_tipologia_sigla.clear()
            self.comboBox_tipologia_sigla.addItems(list2)
        if self.comboBox_nome_tabella.currentText()=='inventario_materiali_table':
            self.comboBox_tipologia_sigla.clear()
            self.comboBox_tipologia_sigla.addItems(list3)
        if self.comboBox_nome_tabella.currentText()=='campioni_table':
            self.comboBox_tipologia_sigla.clear()
            self.comboBox_tipologia_sigla.addItems(list4)
        if self.comboBox_nome_tabella.currentText()=='inventario_lapidei_table':
            self.comboBox_tipologia_sigla.clear()
            self.comboBox_tipologia_sigla.addItems(list5)
        if self.comboBox_nome_tabella.currentText()=='struttura_table':
            self.comboBox_tipologia_sigla.clear()
            self.comboBox_tipologia_sigla.addItems(list6)
        if self.comboBox_nome_tabella.currentText()=='tomba_table':
            self.comboBox_tipologia_sigla.clear()
            self.comboBox_tipologia_sigla.addItems(list7)
        if self.comboBox_nome_tabella.currentText()=='individui_table':
            self.comboBox_tipologia_sigla.clear()
            self.comboBox_tipologia_sigla.addItems(list8)
        if self.comboBox_nome_tabella.currentText()=='documentazione_table':
            self.comboBox_tipologia_sigla.clear()
            self.comboBox_tipologia_sigla.addItems(list9)
    
    def on_pushButton_sort_pressed(self):
        if self.check_record_state() == 1:
            pass
        else:
            dlg = SortPanelMain(self)
            dlg.insertItems(self.SORT_ITEMS)
            dlg.exec_()

            items, order_type = dlg.ITEMS, dlg.TYPE_ORDER

            self.SORT_ITEMS_CONVERTED = []
            for i in items:
                self.SORT_ITEMS_CONVERTED.append(self.CONVERSION_DICT[str(i)])

            self.SORT_MODE = order_type
            self.empty_fields()

            id_list = []
            for i in self.DATA_LIST:
                id_list.append(eval("i." + self.ID_TABLE))
            self.DATA_LIST = []

            temp_data_list = self.DB_MANAGER.query_sort(id_list, self.SORT_ITEMS_CONVERTED, self.SORT_MODE,
                                                        self.MAPPER_TABLE_CLASS, self.ID_TABLE)

            for i in temp_data_list:
                self.DATA_LIST.append(i)
            self.BROWSE_STATUS = "b"
            self.label_status.setText(self.STATUS_ITEMS[self.BROWSE_STATUS])
            if type(self.REC_CORR) == "<type 'str'>":
                corr = 0
            else:
                corr = self.REC_CORR

            self.REC_TOT, self.REC_CORR = len(self.DATA_LIST), 0
            self.DATA_LIST_REC_TEMP = self.DATA_LIST_REC_CORR = self.DATA_LIST[0]
            self.SORT_STATUS = "o"
            self.label_sort.setText(self.SORTED_ITEMS[self.SORT_STATUS])
            self.set_rec_counter(len(self.DATA_LIST), self.REC_CORR + 1)
            self.fill_fields()

    def on_pushButton_new_rec_pressed(self):
        
        if bool(self.DATA_LIST):
            if self.data_error_check() == 1:
                pass
            else:
                if self.BROWSE_STATUS == "b":
                    if bool(self.DATA_LIST):
                        if self.records_equal_check() == 1:
                            if self.L=='it':
                                self.update_if(QMessageBox.warning(self, 'Errore',
                                                                   "Il record e' stato modificato. Vuoi salvare le modifiche?",QMessageBox.Ok | QMessageBox.Cancel))
                            elif self.L=='de':
                                self.update_if(QMessageBox.warning(self, 'Error',
                                                                   "Der Record wurde geändert. Möchtest du die Änderungen speichern?",
                                                                   QMessageBox.Ok | QMessageBox.Cancel))
                                                                   
                            else:
                                self.update_if(QMessageBox.warning(self, 'Error',
                                                                   "The record has been changed. Do you want to save the changes?",
                                                                   QMessageBox.Ok | QMessageBox.Cancel))

                            # set the GUI for a new record
        if self.BROWSE_STATUS != "n":
            self.BROWSE_STATUS = "n"
            self.label_status.setText(self.STATUS_ITEMS[self.BROWSE_STATUS])
            self.empty_fields()
            self.label_sort.setText(self.SORTED_ITEMS["n"])

            self.setComboBoxEditable(["self.comboBox_sigla"], 1)
            self.setComboBoxEditable(["self.comboBox_sigla_estesa"], 1)
            self.setComboBoxEditable(["self.comboBox_tipologia_sigla"], 1)
            self.setComboBoxEditable(["self.comboBox_nome_tabella"], 1)
            self.setComboBoxEditable(["self.comboBox_lingua"], 1)

            self.setComboBoxEnable(["self.comboBox_sigla"], "True")
            self.setComboBoxEnable(["self.comboBox_sigla_estesa"], "True")
            self.setComboBoxEnable(["self.comboBox_tipologia_sigla"], "True")
            self.setComboBoxEnable(["self.comboBox_nome_tabella"], "True")
            self.setComboBoxEnable(["self.comboBox_lingua"], "True")

            self.set_rec_counter('', '')
            self.enable_button(0)

    def on_pushButton_save_pressed(self):
        # save record
        if self.BROWSE_STATUS == "b":
            if self.data_error_check() == 0:
                if self.records_equal_check() == 1:
                    if self.L=='it':
                        self.update_if(QMessageBox.warning(self, 'Errore',
                                                           "Il record e' stato modificato. Vuoi salvare le modifiche?",QMessageBox.Ok | QMessageBox.Cancel))
                    elif self.L=='de':
                        self.update_if(QMessageBox.warning(self, 'Error',
                                                           "Der Record wurde geändert. Möchtest du die Änderungen speichern?",
                                                           QMessageBox.Ok | QMessageBox.Cancel))
                                                    
                    else:
                        self.update_if(QMessageBox.warning(self, 'Error',
                                                           "The record has been changed. Do you want to save the changes?",
                                                           QMessageBox.Ok | QMessageBox.Cancel))
                    self.SORT_STATUS = "n"
                    self.label_sort.setText(self.SORTED_ITEMS[self.SORT_STATUS])
                    self.enable_button(1)
                    self.fill_fields(self.REC_CORR)
                else:
                    if self.L=='it':
                        QMessageBox.warning(self, "ATTENZIONE", "Non è stata realizzata alcuna modifica.", QMessageBox.Ok)
                    elif self.L=='de':
                        QMessageBox.warning(self, "ACHTUNG", "Keine Änderung vorgenommen", QMessageBox.Ok)
                    else:
                        QMessageBox.warning(self, "Warning", "No changes have been made", QMessageBox.Ok) 
        else:
            if self.data_error_check() == 0:
                test_insert = self.insert_new_rec()
                if test_insert == 1:
                    self.empty_fields()
                    self.label_sort.setText(self.SORTED_ITEMS["n"])
                    self.charge_list()
                    self.charge_records()
                    self.BROWSE_STATUS = "b"
                    self.label_status.setText(self.STATUS_ITEMS[self.BROWSE_STATUS])
                    self.REC_TOT, self.REC_CORR = len(self.DATA_LIST), len(self.DATA_LIST) - 1
                    self.set_rec_counter(self.REC_TOT, self.REC_CORR + 1)

                    self.setComboBoxEditable(["self.comboBox_sigla"], 1)
                    self.setComboBoxEditable(["self.comboBox_sigla_estesa"], 1)
                    self.setComboBoxEditable(["self.comboBox_tipologia_sigla"], 1)
                    self.setComboBoxEditable(["self.comboBox_nome_tabella"], 1)
                    self.setComboBoxEditable(["self.comboBox_lingua"], 1)

                    self.setComboBoxEnable(["self.comboBox_sigla"], "False")
                    self.setComboBoxEnable(["self.comboBox_sigla_estesa"], "False")
                    self.setComboBoxEnable(["self.comboBox_tipologia_sigla"], "False")
                    self.setComboBoxEnable(["self.comboBox_nome_tabella"], "False")
                    self.setComboBoxEnable(["self.comboBox_lingua"], "False")

                    self.fill_fields(self.REC_CORR)
                    self.enable_button(1)
                else:
                    pass

    def data_error_check(self):
        test = 0
        EC = Error_check()
        if self.L=='it':
            if EC.data_is_empty(str(self.comboBox_sigla.currentText())) == 0:
                QMessageBox.warning(self, "ATTENZIONE", "Campo Sigla \n Il campo non deve essere vuoto", QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.comboBox_sigla_estesa.currentText())) == 0:
                QMessageBox.warning(self, "ATTENZIONE", "Campo Sigla estesa \n Il campo non deve essere vuoto",
                                    QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.comboBox_tipologia_sigla.currentText())) == 0:
                QMessageBox.warning(self, "ATTENZIONE", "Tipologia sigla. \n Il campo non deve essere vuoto",
                                    QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.comboBox_nome_tabella.currentText())) == 0:
                QMessageBox.warning(self, "ATTENZIONE", "Campo Nome tabella \n Il campo non deve essere vuoto",
                                    QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.comboBox_lingua.currentText())) == 0:
                QMessageBox.warning(self, "ATTENZIONE", "Campo lingua \n Il campo non deve essere vuoto",
                                    QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.textEdit_descrizione_sigla.toPlainText())) == 0:
                QMessageBox.warning(self, "ATTENZIONE", "Campo Descrizione \n Il campo non deve essere vuoto",
                                    QMessageBox.Ok)
                test = 1
        
        
            """controllo lunghezza campo alfanumerico"""
            sigla = self.comboBox_sigla.currentText()
            if sigla!='':
                if EC.data_lenght(sigla, 3) == 0:
                    QMessageBox.warning(self, "ATTENZIONE",
                                        "Campo Sigla. \n Il valore non deve superare i 3 caratteri alfabetici",
                                        QMessageBox.Ok)
                                        
                    test = 1                    
        elif self.L=='de':
            if EC.data_is_empty(str(self.comboBox_sigla.currentText())) == 0:
                QMessageBox.warning(self, "ACHTUNG", "Feld Abkürzung \n Das Feld darf nicht leer sein", QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.comboBox_sigla_estesa.currentText())) == 0:
                QMessageBox.warning(self, "ACHTUNG", "Feld Erweitertes Abkürzungszeichen \n Das Feld darf nicht leer sein",
                                    QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.comboBox_tipologia_sigla.currentText())) == 0:
                QMessageBox.warning(self, "ACHTUNG", "Abkürzungtyp. \n Das Feld darf nicht leer sein",
                                    QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.comboBox_nome_tabella.currentText())) == 0:
                QMessageBox.warning(self, "ACHTUNG", "Tabellenname \n Das Feld darf nicht leer sein",
                                    QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.comboBox_lingua.currentText())) == 0:
                QMessageBox.warning(self, "ACHTUNG", "Feld Sprache \n Das Feld darf nicht leer sein",
                                    QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.textEdit_descrizione_sigla.toPlainText())) == 0:
                QMessageBox.warning(self, "ACHTUNG", "Feld Beschreibung \n Das Feld darf nicht leer sein",
                                    QMessageBox.Ok)
                test = 1
        else:
            if EC.data_is_empty(str(self.comboBox_sigla.currentText())) == 0:
                QMessageBox.warning(self, "WARNING", "Code Field \n The field must not be empty", QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.comboBox_sigla_estesa.currentText())) == 0:
                QMessageBox.warning(self, "WARNING", "Code whole field \n The field must not be empty",
                                    QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.comboBox_tipologia_sigla.currentText())) == 0:
                QMessageBox.warning(self, "WARNING", "Code typology \n The field must not be empty",
                                    QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.comboBox_nome_tabella.currentText())) == 0:
                QMessageBox.warning(self, "WARNING", "Table name \n The field must not be empty",
                                    QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.comboBox_lingua.currentText())) == 0:
                QMessageBox.warning(self, "WARNING", "Language field \n The field must not be empty",
                                    QMessageBox.Ok)
                test = 1

            if EC.data_is_empty(str(self.textEdit_descrizione_sigla.toPlainText())) == 0:
                QMessageBox.warning(self, "WARNING", "Description field \n The field must not be empty",
                                    QMessageBox.Ok)
                test = 1        
        return test

    def insert_new_rec(self):
        try:
            data = self.DB_MANAGER.insert_values_thesaurus(
                self.DB_MANAGER.max_num_id(self.MAPPER_TABLE_CLASS, self.ID_TABLE) + 1,
                str(self.comboBox_nome_tabella.currentText()),  # 1 - nome tabella
                str(self.comboBox_sigla.currentText()),  # 2 - sigla
                str(self.comboBox_sigla_estesa.currentText()),  # 3 - sigla estesa
                str(self.textEdit_descrizione_sigla.toPlainText()),  # 4 - descrizione
                str(self.comboBox_tipologia_sigla.currentText()),  # 5 - tipologia sigla
                str(self.comboBox_lingua.currentText()))  # 6 - lingua

            try:
                self.DB_MANAGER.insert_data_session(data)
                return 1
            except Exception as e:
                e_str = str(e)
                if e_str.__contains__("IntegrityError"):
                    
                    if self.L=='it':
                        msg = self.ID_TABLE + " gia' presente nel database"
                        QMessageBox.warning(self, "Error", "Error" + str(msg), QMessageBox.Ok)
                    elif self.L=='de':
                        msg = self.ID_TABLE + " bereits in der Datenbank"
                        QMessageBox.warning(self, "Error", "Error" + str(msg), QMessageBox.Ok)  
                    else:
                        msg = self.ID_TABLE + " exist in db"
                        QMessageBox.warning(self, "Error", "Error" + str(msg), QMessageBox.Ok)  
                else:
                    msg = e
                    QMessageBox.warning(self, "Error", "Error 1 \n" + str(msg), QMessageBox.Ok)
                return 0
        except Exception as e:
            QMessageBox.warning(self, "Error", "Error 2 \n" + str(e), QMessageBox.Ok)
            return 0
    def check_record_state(self):
        ec = self.data_error_check()
        if ec == 1:
            return 1  # ci sono errori di immissione
        elif self.records_equal_check() == 1 and ec == 0:
            if self.L=='it':
                self.update_if(
                
                    QMessageBox.warning(self, 'Errore', "Il record e' stato modificato. Vuoi salvare le modifiche?",
                                        QMessageBox.Ok | QMessageBox.Cancel))
            elif self.L=='de':
                self.update_if(
                    QMessageBox.warning(self, 'Errore', "Der Record wurde geändert. Möchtest du die Änderungen speichern?",
                                        QMessageBox.Ok | QMessageBox.Cancel))
            else:
                self.update_if(
                    QMessageBox.warning(self, "Error", "The record has been changed. You want to save the changes?",
                                        QMessageBox.Ok | QMessageBox.Cancel))
            # self.charge_records()
            return 0  # non ci sono errori di immissione

            # records surf functions

    def on_pushButton_view_all_pressed(self):
        if self.check_record_state() == 1:
            pass
        else:
            self.empty_fields()
            self.charge_records()
            self.fill_fields()
            self.BROWSE_STATUS = "b"
            self.label_status.setText(self.STATUS_ITEMS[self.BROWSE_STATUS])
            if type(self.REC_CORR) == "<type 'str'>":
                corr = 0
            else:
                corr = self.REC_CORR
            self.set_rec_counter(len(self.DATA_LIST), self.REC_CORR + 1)
            self.REC_TOT, self.REC_CORR = len(self.DATA_LIST), 0
            self.DATA_LIST_REC_TEMP = self.DATA_LIST_REC_CORR = self.DATA_LIST[0]
            self.label_sort.setText(self.SORTED_ITEMS["n"])

            # records surf functions

    def on_pushButton_first_rec_pressed(self):
        if self.check_record_state() == 1:
            pass
        else:
            try:
                self.empty_fields()
                self.REC_TOT, self.REC_CORR = len(self.DATA_LIST), 0
                self.fill_fields(0)
                self.set_rec_counter(self.REC_TOT, self.REC_CORR + 1)
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e), QMessageBox.Ok)

    def on_pushButton_last_rec_pressed(self):
        if self.check_record_state() == 1:
            pass
        else:
            try:
                self.empty_fields()
                self.REC_TOT, self.REC_CORR = len(self.DATA_LIST), len(self.DATA_LIST) - 1
                self.fill_fields(self.REC_CORR)
                self.set_rec_counter(self.REC_TOT, self.REC_CORR + 1)
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e), QMessageBox.Ok)

    def on_pushButton_prev_rec_pressed(self):
        if self.check_record_state() == 1:
            pass
        else:
            self.REC_CORR = self.REC_CORR - 1
            if self.REC_CORR == -1:
                self.REC_CORR = 0
                if self.L=='it':
                    QMessageBox.warning(self, "Attenzione", "Sei al primo record!", QMessageBox.Ok)
                elif self.L=='de':
                    QMessageBox.warning(self, "Achtung", "du befindest dich im ersten Datensatz!", QMessageBox.Ok)
                else:
                    QMessageBox.warning(self, "Warning", "You are to the first record!", QMessageBox.Ok)        
            else:
                try:
                    self.empty_fields()
                    self.fill_fields(self.REC_CORR)
                    self.set_rec_counter(self.REC_TOT, self.REC_CORR + 1)
                except Exception as e:
                    QMessageBox.warning(self, "Error", str(e), QMessageBox.Ok)

    def on_pushButton_next_rec_pressed(self):
        if self.check_record_state() == 1:
            pass
        else:
            self.REC_CORR = self.REC_CORR + 1
            if self.REC_CORR >= self.REC_TOT:
                self.REC_CORR = self.REC_CORR - 1
                if self.L=='it':
                    QMessageBox.warning(self, "Attenzione", "Sei all'ultimo record!", QMessageBox.Ok)
                elif self.L=='de':
                    QMessageBox.warning(self, "Achtung", "du befindest dich im letzten Datensatz!", QMessageBox.Ok)
                else:
                    QMessageBox.warning(self, "Error", "You are to the first record!", QMessageBox.Ok)  
            else:
                try:
                    self.empty_fields()
                    self.fill_fields(self.REC_CORR)
                    self.set_rec_counter(self.REC_TOT, self.REC_CORR + 1)
                except Exception as e:
                    QMessageBox.warning(self, "Error", str(e), QMessageBox.Ok)

    def on_pushButton_delete_pressed(self):
        if self.L=='it':
            msg = QMessageBox.warning(self, "Attenzione!!!",
                                      "Vuoi veramente eliminare il record? \n L'azione è irreversibile",
                                      QMessageBox.Ok | QMessageBox.Cancel)
            if msg == QMessageBox.Cancel:
                QMessageBox.warning(self, "Messagio!!!", "Azione Annullata!")
            else:
                try:
                    id_to_delete = eval("self.DATA_LIST[self.REC_CORR]." + self.ID_TABLE)
                    self.DB_MANAGER.delete_one_record(self.TABLE_NAME, self.ID_TABLE, id_to_delete)
                    self.charge_records()  # charge records from DB
                    QMessageBox.warning(self, "Messaggio!!!", "Record eliminato!")
                except Exception as e:
                    QMessageBox.warning(self, "Messaggio!!!", "Tipo di errore: " + str(e))
                if not bool(self.DATA_LIST):
                    QMessageBox.warning(self, "Attenzione", "Il database è vuoto!", QMessageBox.Ok)
                    self.DATA_LIST = []
                    self.DATA_LIST_REC_CORR = []
                    self.DATA_LIST_REC_TEMP = []
                    self.REC_CORR = 0
                    self.REC_TOT = 0
                    self.empty_fields()
                    self.set_rec_counter(0, 0)
                    # check if DB is empty
                if bool(self.DATA_LIST):
                    self.REC_TOT, self.REC_CORR = len(self.DATA_LIST), 0
                    self.DATA_LIST_REC_TEMP = self.DATA_LIST_REC_CORR = self.DATA_LIST[0]
                    self.BROWSE_STATUS = "b"
                    self.label_status.setText(self.STATUS_ITEMS[self.BROWSE_STATUS])
                    self.set_rec_counter(len(self.DATA_LIST), self.REC_CORR + 1)
                    self.charge_list()
                    self.fill_fields()
        elif self.L=='de':
            msg = QMessageBox.warning(self, "Achtung!!!",
                                      "Willst du wirklich diesen Eintrag löschen? \n Der Vorgang ist unumkehrbar",
                                      QMessageBox.Ok | QMessageBox.Cancel)
            if msg == QMessageBox.Cancel:
                QMessageBox.warning(self, "Message!!!", "Aktion annulliert!")
            else:
                try:
                    id_to_delete = eval("self.DATA_LIST[self.REC_CORR]." + self.ID_TABLE)
                    self.DB_MANAGER.delete_one_record(self.TABLE_NAME, self.ID_TABLE, id_to_delete)
                    self.charge_records()  # charge records from DB
                    QMessageBox.warning(self, "Message!!!", "Record gelöscht!")
                except Exception as e:
                    QMessageBox.warning(self, "Messagge!!!", "Errortyp: " + str(e))
                if not bool(self.DATA_LIST):
                    QMessageBox.warning(self, "Achtung", "Die Datenbank ist leer!", QMessageBox.Ok)
                    self.DATA_LIST = []
                    self.DATA_LIST_REC_CORR = []
                    self.DATA_LIST_REC_TEMP = []
                    self.REC_CORR = 0
                    self.REC_TOT = 0
                    self.empty_fields()
                    self.set_rec_counter(0, 0)
                    # check if DB is empty
                if bool(self.DATA_LIST):
                    self.REC_TOT, self.REC_CORR = len(self.DATA_LIST), 0
                    self.DATA_LIST_REC_TEMP = self.DATA_LIST_REC_CORR = self.DATA_LIST[0]
                    self.BROWSE_STATUS = "b"
                    self.label_status.setText(self.STATUS_ITEMS[self.BROWSE_STATUS])
                    self.set_rec_counter(len(self.DATA_LIST), self.REC_CORR + 1)
                    self.charge_list()
                    self.fill_fields()
        else:
            msg = QMessageBox.warning(self, "Warning!!!",
                                      "Do you really want to break the record? \n Action is irreversible.",
                                      QMessageBox.Ok | QMessageBox.Cancel)
            if msg == QMessageBox.Cancel:
                QMessageBox.warning(self, "Message!!!", "Action deleted!")
            else:
                try:
                    id_to_delete = eval("self.DATA_LIST[self.REC_CORR]." + self.ID_TABLE)
                    self.DB_MANAGER.delete_one_record(self.TABLE_NAME, self.ID_TABLE, id_to_delete)
                    self.charge_records()  # charge records from DB
                    QMessageBox.warning(self, "Message!!!", "Record deleted!")
                except Exception as e:
                    QMessageBox.warning(self, "Message!!!", "error type: " + str(e))
                if not bool(self.DATA_LIST):
                    QMessageBox.warning(self, "Warning", "the db is empty!", QMessageBox.Ok)
                    self.DATA_LIST = []
                    self.DATA_LIST_REC_CORR = []
                    self.DATA_LIST_REC_TEMP = []
                    self.REC_CORR = 0
                    self.REC_TOT = 0
                    self.empty_fields()
                    self.set_rec_counter(0, 0)
                    # check if DB is empty
                if bool(self.DATA_LIST):
                    self.REC_TOT, self.REC_CORR = len(self.DATA_LIST), 0
                    self.DATA_LIST_REC_TEMP = self.DATA_LIST_REC_CORR = self.DATA_LIST[0]
                    self.BROWSE_STATUS = "b"
                    self.label_status.setText(self.STATUS_ITEMS[self.BROWSE_STATUS])
                    self.set_rec_counter(len(self.DATA_LIST), self.REC_CORR + 1)
                    self.charge_list()
                    self.fill_fields()  
            
            
            
            self.SORT_STATUS = "n"
            self.label_sort.setText(self.SORTED_ITEMS[self.SORT_STATUS])

    def on_pushButton_sigle_pressed(self):
        if self.L=='it':    
            filepath = os.path.dirname(__file__)
            filepath = os.path.join(filepath, 'codici_it.html')
            #os.startfile(filepath)
            self.webView_adarte.load(QUrl.fromLocalFile(filepath))
            self.webView_adarte.show()
        elif self.L=='de':  
            filepath = os.path.dirname(__file__)
            filepath = os.path.join(filepath, 'codici_de.html')
            os.startfile(filepath)
        else:
            filepath = os.path.dirname(__file__)
            filepath = os.path.join(filepath, 'codici_en.html')
            os.startfile(filepath)
    def on_pushButton_new_search_pressed(self):
        if self.check_record_state() == 1:
            pass
        else:
            self.enable_button_search(0)

            # set the GUI for a new search
            if self.BROWSE_STATUS != "f":
                self.BROWSE_STATUS = "f"
                self.label_status.setText(self.STATUS_ITEMS[self.BROWSE_STATUS])
                ###
                self.setComboBoxEditable(["self.comboBox_sigla"], 1)
                self.setComboBoxEditable(["self.comboBox_sigla_estesa"], 1)
                self.setComboBoxEditable(["self.comboBox_tipologia_sigla"], 1)
                self.setComboBoxEditable(["self.comboBox_nome_tabella"], 1)
                self.setComboBoxEditable(["self.comboBox_lingua"], 1)

                self.setComboBoxEnable(["self.comboBox_sigla"], "True")
                self.setComboBoxEnable(["self.comboBox_sigla_estesa"], "True")
                self.setComboBoxEnable(["self.comboBox_tipologia_sigla"], "True")
                self.setComboBoxEnable(["self.comboBox_nome_tabella"], "True")
                self.setComboBoxEnable(["self.comboBox_lingua"], "True")

                self.setComboBoxEnable(["self.textEdit_descrizione_sigla"], "False")
                ###
                self.label_status.setText(self.STATUS_ITEMS[self.BROWSE_STATUS])
                self.set_rec_counter('', '')
                self.label_sort.setText(self.SORTED_ITEMS["n"])
                self.charge_list()
                self.empty_fields()

    def on_pushButton_search_go_pressed(self):
        if self.BROWSE_STATUS != "f":
            if self.L=='it':
                QMessageBox.warning(self, "ATTENZIONE", "Per eseguire una nuova ricerca clicca sul pulsante 'new search' ",
                                    QMessageBox.Ok)
            elif self.L=='de':
                QMessageBox.warning(self, "ACHTUNG", "Um eine neue Abfrage zu starten drücke  'new search' ",
                                    QMessageBox.Ok)
            else:
                QMessageBox.warning(self, "WARNING", "To perform a new search click on the 'new search' button ",
                                    QMessageBox.Ok) 
        else:
            search_dict = {
                self.TABLE_FIELDS[0]: "'" + str(self.comboBox_nome_tabella.currentText()) + "'",  # 1 - Nome tabella
                self.TABLE_FIELDS[1]: "'" + str(self.comboBox_sigla.currentText()) + "'",  # 2 - sigla
                self.TABLE_FIELDS[2]: "'" + str(self.comboBox_sigla_estesa.currentText()) + "'",  # 3 - sigla estesa
                self.TABLE_FIELDS[4]: "'" + str(self.comboBox_tipologia_sigla.currentText()) + "'",
                self.TABLE_FIELDS[5]: "'" + str(self.comboBox_lingua.currentText()) + "'"
                # 3 - tipologia sigla
            }

            u = Utility()
            search_dict = u.remove_empty_items_fr_dict(search_dict)

            if not bool(search_dict):
                if self.L=='it':
                    QMessageBox.warning(self, "ATTENZIONE", "Non è stata impostata nessuna ricerca!!!", QMessageBox.Ok)
                elif self.L=='de':
                    QMessageBox.warning(self, "ACHTUNG", "Keine Abfrage definiert!!!", QMessageBox.Ok)
                else:
                    QMessageBox.warning(self, " WARNING", "No search has been set!!!", QMessageBox.Ok)      
            else:
                res = self.DB_MANAGER.query_bool(search_dict, self.MAPPER_TABLE_CLASS)
                if not bool(res):
                    if self.L=='it':
                        QMessageBox.warning(self, "ATTENZIONE", "Non è stato trovato nessun record!", QMessageBox.Ok)
                    elif self.L=='de':
                        QMessageBox.warning(self, "ACHTUNG", "Keinen Record gefunden!", QMessageBox.Ok)
                    else:
                        QMessageBox.warning(self, "WARNING," "No record found!", QMessageBox.Ok) 

                    self.set_rec_counter(len(self.DATA_LIST), self.REC_CORR + 1)
                    self.DATA_LIST_REC_TEMP = self.DATA_LIST_REC_CORR = self.DATA_LIST[0]

                    self.fill_fields(self.REC_CORR)
                    self.BROWSE_STATUS = "b"
                    self.label_status.setText(self.STATUS_ITEMS[self.BROWSE_STATUS])

                    self.setComboBoxEditable(["self.comboBox_sigla"], 1)
                    self.setComboBoxEditable(["self.comboBox_sigla_estesa"], 1)
                    self.setComboBoxEditable(["self.comboBox_tipologia_sigla"], 1)
                    self.setComboBoxEditable(["self.comboBox_nome_tabella"], 1)
                    self.setComboBoxEditable(["self.comboBox_lingua"], 1)

                    self.setComboBoxEnable(["self.comboBox_sigla"], "False")
                    self.setComboBoxEnable(["self.comboBox_sigla_estesa"], "False")
                    self.setComboBoxEnable(["self.comboBox_tipologia_sigla"], "False")
                    self.setComboBoxEnable(["self.comboBox_nome_tabella"], "False")
                    self.setComboBoxEnable(["self.comboBox_lingua"], "False")

                    self.setComboBoxEnable(["self.textEdit_descrizione_sigla"], "True")

                else:
                    self.DATA_LIST = []

                    for i in res:
                        self.DATA_LIST.append(i)

                    self.REC_TOT, self.REC_CORR = len(self.DATA_LIST), 0
                    self.DATA_LIST_REC_TEMP = self.DATA_LIST_REC_CORR = self.DATA_LIST[0]
                    self.fill_fields()
                    self.BROWSE_STATUS = "b"
                    self.label_status.setText(self.STATUS_ITEMS[self.BROWSE_STATUS])
                    self.set_rec_counter(len(self.DATA_LIST), self.REC_CORR + 1)

                    if self.L=='it':
                        if self.REC_TOT == 1:
                            strings = ("E' stato trovato", self.REC_TOT, "record")
                        else:
                            strings = ("Sono stati trovati", self.REC_TOT, "records")
                    elif self.L=='de':
                        if self.REC_TOT == 1:
                            strings = ("Es wurde gefunden", self.REC_TOT, "record")
                        else:
                            strings = ("Sie wurden gefunden", self.REC_TOT, "records")
                    else:
                        if self.REC_TOT == 1:
                            strings = ("It has been found", self.REC_TOT, "record")
                        else:
                            strings = ("They have been found", self.REC_TOT, "records")

                        self.setComboBoxEditable(["self.comboBox_sigla"], 1)
                        self.setComboBoxEditable(["self.comboBox_sigla_estesa"], 1)
                        self.setComboBoxEditable(["self.comboBox_tipologia_sigla"], 1)
                        self.setComboBoxEditable(["self.comboBox_nome_tabella"], 1)
                        self.setComboBoxEditable(["self.comboBox_lingua"], 1)

                        self.setComboBoxEnable(["self.comboBox_sigla"], "False")
                        self.setComboBoxEnable(["self.comboBox_sigla_estesa"], "False")
                        self.setComboBoxEnable(["self.comboBox_tipologia_sigla"], "False")
                        self.setComboBoxEnable(["self.comboBox_nome_tabella"], "False")
                        self.setComboBoxEnable(["self.comboBox_lingua"], "False")

                        self.setComboBoxEnable(["self.textEdit_descrizione_sigla"], "True")

                    QMessageBox.warning(self, "Message", "%s %d %s" % strings, QMessageBox.Ok)

        self.enable_button_search(1)

    def on_pushButton_test_pressed(self):
        pass

    ##      data = "Sito: " + str(self.comboBox_sito.currentText())
    ##
    ##      test = Test_area(data)
    ##      test.run_test()

    def on_pushButton_draw_pressed(self):
        pass

        # self.pyQGIS.charge_layers_for_draw(["1", "2", "3", "4", "5", "7", "8", "9", "10", "12"])

    def on_pushButton_sites_geometry_pressed(self):
        pass

    ##      sito = unicode(self.comboBox_sito.currentText())
    ##      self.pyQGIS.charge_sites_geometry(["1", "2", "3", "4", "8"], "sito", sito)

    def on_pushButton_rel_pdf_pressed(self):
        pass

    ##      check=QMessageBox.warning(self, "Attention", "Under testing: this method can contains some bugs. Do you want proceed?",QMessageBox.Cancel,1)
    ##      if check == 1:
    ##          erp = exp_rel_pdf(unicode(self.comboBox_sito.currentText()))
    ##          erp.export_rel_pdf()

    def update_if(self, msg):
        rec_corr = self.REC_CORR
        if msg == QMessageBox.Ok:
            test = self.update_record()
            if test == 1:
                id_list = []
                for i in self.DATA_LIST:
                    id_list.append(eval("i." + self.ID_TABLE))
                self.DATA_LIST = []
                if self.SORT_STATUS == "n":
                    temp_data_list = self.DB_MANAGER.query_sort(id_list, [self.ID_TABLE], 'asc',
                                                                self.MAPPER_TABLE_CLASS,
                                                                self.ID_TABLE)  # self.DB_MANAGER.query_bool(self.SEARCH_DICT_TEMP, self.MAPPER_TABLE_CLASS) #
                else:
                    temp_data_list = self.DB_MANAGER.query_sort(id_list, self.SORT_ITEMS_CONVERTED, self.SORT_MODE,
                                                                self.MAPPER_TABLE_CLASS, self.ID_TABLE)
                for i in temp_data_list:
                    self.DATA_LIST.append(i)
                self.BROWSE_STATUS = "b"
                self.label_status.setText(self.STATUS_ITEMS[self.BROWSE_STATUS])
                if type(self.REC_CORR) == "<type 'str'>":
                    corr = 0
                else:
                    corr = self.REC_CORR
                return 1
            elif test == 0:
                return 0

                # custom functions

    def charge_records(self):
        self.DATA_LIST = []

        if self.DB_SERVER == 'sqlite':
            for i in self.DB_MANAGER.query(self.MAPPER_TABLE_CLASS):
                self.DATA_LIST.append(i)
        else:
            id_list = []
            for i in self.DB_MANAGER.query(self.MAPPER_TABLE_CLASS):
                id_list.append(eval("i." + self.ID_TABLE))

            temp_data_list = self.DB_MANAGER.query_sort(id_list, [self.ID_TABLE], 'asc', self.MAPPER_TABLE_CLASS,
                                                        self.ID_TABLE)

            for i in temp_data_list:
                self.DATA_LIST.append(i)

    def datestrfdate(self):
        now = date.today()
        today = now.strftime("%d-%m-%Y")
        return today

    def table2dict(self, n):
        self.tablename = n
        row = eval(self.tablename + ".rowCount()")
        col = eval(self.tablename + ".columnCount()")
        lista = []
        for r in range(row):
            sub_list = []
            for c in range(col):
                value = eval(self.tablename + ".item(r,c)")
                if bool(value):
                    sub_list.append(str(value.text()))
            lista.append(sub_list)
        return lista

    def empty_fields(self):
        self.comboBox_sigla.setEditText("")  # 1 - Sigla
        self.comboBox_sigla_estesa.setEditText("")  # 2 - Sigla estesa
        self.comboBox_tipologia_sigla.setEditText("")  # 1 - Tipologia sigla
        self.comboBox_nome_tabella.setEditText("")  # 2 - Nome tabella
        self.textEdit_descrizione_sigla.clear()  # 4 - Descrizione
        self.comboBox_lingua.setEditText("")

    def fill_fields(self, n=0):
        self.rec_num = n



        str(self.comboBox_sigla.setEditText(self.DATA_LIST[self.rec_num].sigla))  # 1 - Sigla
        str(self.comboBox_sigla_estesa.setEditText(self.DATA_LIST[self.rec_num].sigla_estesa))  # 2 - Sigla estesa
        str(self.comboBox_tipologia_sigla.setEditText(
            self.DATA_LIST[self.rec_num].tipologia_sigla))  # 3 - tipologia sigla
        str(self.comboBox_nome_tabella.setEditText(self.DATA_LIST[self.rec_num].nome_tabella))  # 4 - nome tabella
        str(str(
            self.textEdit_descrizione_sigla.setText(self.DATA_LIST[self.rec_num].descrizione)))  # 5 - descrizione sigla
        str(self.comboBox_lingua.setEditText(self.DATA_LIST[self.rec_num].lingua))  # 6 - lingua

    def set_rec_counter(self, t, c):
        self.rec_tot = t
        self.rec_corr = c
        self.label_rec_tot.setText(str(self.rec_tot))
        self.label_rec_corrente.setText(str(self.rec_corr))

    def set_LIST_REC_TEMP(self):
        lingua=""
        l = self.comboBox_lingua.currentText()
        for key,values in self.LANG.items():
            if values.__contains__(l):
                lingua = key
        # data
        self.DATA_LIST_REC_TEMP = [
            str(self.comboBox_nome_tabella.currentText()),  # 1 - Nome tabella
            str(self.comboBox_sigla.currentText()),  # 2 - sigla
            str(self.comboBox_sigla_estesa.currentText()),  # 3 - sigla estesa
            str(self.textEdit_descrizione_sigla.toPlainText()),  # 4 - descrizione
            str(self.comboBox_tipologia_sigla.currentText()),  # 3 - tipologia sigla
            str(lingua)  # 6 - lingua
        ]

    def set_LIST_REC_CORR(self):
        self.DATA_LIST_REC_CORR = []
        for i in self.TABLE_FIELDS:
            self.DATA_LIST_REC_CORR.append(eval("unicode(self.DATA_LIST[self.REC_CORR]." + i + ")"))

    def setComboBoxEnable(self, f, v):
        field_names = f
        value = v

        for fn in field_names:
            cmd = '{}{}{}{}'.format(fn, '.setEnabled(', v, ')')
            eval(cmd)

    def setComboBoxEditable(self, f, n):
        field_names = f
        value = n

        for fn in field_names:
            cmd = '{}{}{}{}'.format(fn, '.setEditable(', n, ')')
            eval(cmd)

    def rec_toupdate(self):
        rec_to_update = self.UTILITY.pos_none_in_list(self.DATA_LIST_REC_TEMP)
        return rec_to_update

    def records_equal_check(self):
        self.set_LIST_REC_TEMP()
        self.set_LIST_REC_CORR()

        if self.DATA_LIST_REC_CORR == self.DATA_LIST_REC_TEMP:
            return 0
        else:
            return 1

    def update_record(self):
        try:
            self.DB_MANAGER.update(self.MAPPER_TABLE_CLASS,
                                   self.ID_TABLE,
                                   [eval("int(self.DATA_LIST[self.REC_CORR]." + self.ID_TABLE + ")")],
                                   self.TABLE_FIELDS,
                                   self.rec_toupdate())
            return 1
        except Exception as e:
            str(e)
            save_file='{}{}{}'.format(self.HOME, os.sep,"pyarchinit_Report_folder") 
            file_=os.path.join(save_file,'error_encodig_data_recover.txt')
            with open(file_, "a") as fh:
                try:
                    raise ValueError(str(e))
                except ValueError as s:
                    print(s, file=fh)
            if self.L=='it':
                QMessageBox.warning(self, "Messaggio",
                                    "Problema di encoding: sono stati inseriti accenti o caratteri non accettati dal database. Verrà fatta una copia dell'errore con i dati che puoi recuperare nella cartella pyarchinit_Report _Folder", QMessageBox.Ok)
            
            
            elif self.L=='de':
                QMessageBox.warning(self, "Message",
                                    "Encoding problem: accents or characters not accepted by the database were entered. A copy of the error will be made with the data you can retrieve in the pyarchinit_Report _Folder", QMessageBox.Ok) 
            else:
                QMessageBox.warning(self, "Message",
                                    "Kodierungsproblem: Es wurden Akzente oder Zeichen eingegeben, die von der Datenbank nicht akzeptiert werden. Es wird eine Kopie des Fehlers mit den Daten erstellt, die Sie im pyarchinit_Report _Ordner abrufen können", QMessageBox.Ok)
            return 0

    def testing(self, name_file, message):
        f = open(str(name_file), 'w')
        f.write(str(message))
        f.close()


## Class end

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = pyarchinit_US()
    ui.show()
    sys.exit(app.exec_())
