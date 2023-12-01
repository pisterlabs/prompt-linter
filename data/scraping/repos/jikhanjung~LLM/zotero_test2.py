from openai import OpenAI
from dotenv import load_dotenv # pip install python-dotenv
from pyzotero import zotero
import cmd
import os

load_dotenv()

class ZWrapper():
    def __init__(self):
        zotero_api_key = os.environ.get("ZOTERO_API_KEY")
        zotero_user_id = os.environ.get("ZOTERO_USER_ID")
        self.zot = zotero.Zotero(zotero_user_id, 'user', zotero_api_key)
        self._key_list = []
        self._collection_list = []
        self._zcollection_list = []
        self._zcollection_tree = []

    def get_collection(self, collection_id):
        for zcol in self._zcollection_list:
            if zcol._collection['data']['key'] == collection_id:
                return zcol
        return None
    
    def build_tree(self):
        collection = self.zot.collection('M5EN26AJ')
        self._collection_list.append(collection)
        #self._collection_list = self.zot.all_collections()
        for collection in self._collection_list:
            zcol = ZCollection(self.zot, collection)
            self._zcollection_list.append(zcol)
            self._key_list.append(collection['data']['key'])

        for zcol in self._zcollection_list:
            if 'parentCollection' in zcol._collection['data'] and zcol._collection['data']['parentCollection'] == False:
                self._zcollection_tree.append(zcol)
                #print(collection['data']['name'], collection['data']['parentCollection'])
            else:
                for parent in self._zcollection_list:
                    if parent._collection['data']['key'] == zcol._collection['data']['parentCollection']:
                        parent.addChild(zcol)
                        #print(collection['data']['name'], collection['data']['parentCollection'])
                        break

    def dump(self, item_key, filename, filepath):
        return self.zot.dump(item_key, filename, filepath)

    def print_tree(self):
        for zcol in self._zcollection_tree:
            self.print_tree_helper(zcol, 0)
    
    def print_tree_helper(self, zcol, level):
        print(" "*level, zcol._collection['data']['name'], zcol._collection['data']['key'])
        for child in zcol.child_collections:
            self.print_tree_helper(child, level+1)

class ZCollection():
    def __init__(self, zot, collection):
        self.zot = zot
        self._collection = collection
        self.child_collections = []
        self.child_items = []
        self.item_tree = []
        self.parent = None
    
    def addChildCollection(self, child):
        self.child_collections.append(child)
        child.setParent(self)
    
    def setParent(self, parent):
        self.parent = parent

    def get_parent(self):
        return self.parent

    def read_items(self):
        items = self.zot.collection_items(self._collection['data']['key'])
        for item in items:
            child_item = ZItem(self.zot, item)
            self.child_items.append(child_item)

        for zitem in self.child_items:
            if 'parentItem' not in zitem._item['data']:
                self.item_tree.append(zitem)
            else:
                for parent in self.child_items:
                    if parent._item['data']['key'] == zitem._item['data']['parentItem']:
                        parent.add_child_item(zitem)
                        break

    def print_items(self):
        for zitem in self.item_tree:
            if zitem._item['data']['itemType'] == 'attachment':
                print("  -",zitem._item)
            else:
                print(zitem._item)
                #print(zitem._item['data']['key'], zitem._item['data']['version'], zitem._item['data']['itemType'], zitem._item['data']['title'])

    def print_item_tree(self):
        for zitem in self.item_tree:
            self.print_tree_helper(zitem, 0)
    
    def print_tree_helper(self, zitem, level):
        print(" "*level, zitem._item['data']['title'], zitem._item['data']['key'])
        for child in zitem.child_item_list:
            self.print_tree_helper(child, level+1)

class ZItem():
    def __init__(self, zot, item):
        self.zot = zot
        self._item = item
        self.child_item_list = []

    def add_child_item(self, zitem):
        self.child_item_list.append(zitem)
        
client = OpenAI()
def get_or_create_assistant( asst_name ):
    asst_list = client.beta.assistants.list( order="desc", limit="20", )
    #print(asst_list.data)

    if len(asst_list.data) == 0:
        print("no assistant")
        assistant = client.beta.assistants.create(
            name=asst_name,
            instructions="You are a research assistant in paleontology.",
            tools=[{"type": "code_interpreter"}],
            model="gpt-4-1106-preview"
        )
        asst_list = client.beta.assistants.list( order="desc", limit="20", )

    for asst in asst_list.data:
        if asst.name == asst_name:
            return asst

def get_or_create_thread( thread_id = 'thread_cZLk7hjIlsR1uhGYttIAG2T9' ):
    if not thread_id:
        thread = client.beta.threads.create()
    else:
        thread = client.beta.threads.retrieve(thread_id)
    return thread


z = ZWrapper()
z.build_tree()
z.print_tree()
zcol = z.get_collection('M5EN26AJ')
if zcol:
    zcol.read_items()
    #zcol.print_items()
    zcol.print_item_tree()
    #for item in zcol.item_tree:
    #    print(item._item['data']['key'])
