import pandas as pd
import numpy as np
import re
import itertools
import streamlit as st
from collections import Counter
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
sns.set_context("talk")
from scipy import stats
from pprint import pprint #Manipulacion de datos
import datetime
import dateutil
from streamlit import components
#LDA MODEL FOR OBSERVACIONES
#quitar mas profundamente stop_words
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import unicodedata
import tqdm
import spacy.cli
from spacy.lang.es.stop_words import STOP_WORDS
#descargamos los modelos
from nltk.corpus import stopwords
nltk.download('stopwords')
spacy.cli.download("es_core_news_md")
from sklearn.feature_extraction.text import CountVectorizer
import sys
import plotly.express as px
import matplotlib.pyplot as plt
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import preprocessor as p
import plotly.express as px
import os
from sklearn.preprocessing import MinMaxScaler
sys.setrecursionlimit(10000)
#############LSTM
import sklearn
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#clasificacion de sentimientos, en español
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from classifier import *
clf = SentimentClassifier()
from pylab import rcParams
## APLLY LDA MODEL TO OBSERVACIONES
#Gensim para modelado de temas, indexación de documentos y recuperación de similitudes con grandes corpus
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
#Spacy para la lemmatization
import spacy
# Herramientas de graficado
import pyLDAvis
import pyLDAvis.gensim
# Habilitado de logging para gensim (opcional)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.colors as mcolors



class nlp_pqr:
    def __init__(self, df):
        a_name = ['aarn','aaron','abad','abadia','abadiano','abalo','abaunza','abbigail','abdias','abdubal','abdul','abdulio','abel','abelanid','abelardo','abella','abello','abigail','abimel','abogados','abonce','abondano','abraham','abrahan','abreo','absalon','ac','acalo','accion','acciones','acebedo','acebo','acened','acenet','aceneth','acero','acevedo','acevedoq','achicanoy','achinte','achipiz','achly','acosta','acuña','ada','adaine','adalberto','adalena','adalgisa','adalgiza','adalia','adaluz','adames','adan','adarbe','adarve','adda','adecco','adecio','adel','adela','adelaida','adelayda','adelcia','adelfa','adelina','adelinda','adelmo','adelsaina','ademir','adey','adiel','adiela','adila','adisson','admiautos','administracion','administraciones','administradora','adnelice','adolfo','adonis','adoracion','adrada','adrian','adrian','adriana','adriana','adriano','aduanas','afanador','agatha','agaton','agencia','agnelio','agrace','agraces','agrada','agreda','agredo','agregado','agricolas','agricolas','agron','agropecuaria','aguacia','aguado','agualimpia','aguas','aguayo','agudelo','agueda','aguelo','aguiar','aguilar','aguilera','aguillon','aguinaga','aguiño','aguirre','agustin','agustina','agusto','ahedo','ahumada','aiba','aicardo','aicedo','aida','aidaly','aide','aidee','aidy','aily','ailyn','aimer','aimy','ainosas','airlyng','aissa','aitana','aiza','aizmeth','ala','alaba','aladino','alaguna','alaia','alain','alan','alana','alarcon','alarcon','alava','alba','alban','albaran','albaro','albarracin','albear','albeir','albeiro','albert','alberta','alberto','albery','albey','albino','albornoz','alcalde','alcaldia','alcaldia','alcaraz','alcazar','alcibiades','alcides','alcira','alda','aldana','aldarriaga','aldemar','alderete','alderis','aldery','aldineber','alean','alechor','alegria','alegria','alegrias','aleiber','aleicy','aleida','alejandra','alejandrina','alejandro','alejjandro','alejnadro','aleman','alemesa','alencia','ales','alessandra','alex','alexa','alexander','alexandra','alexandre','alexei','alexi','alexis','alexy','aleyda','aleyi','alezander','alfa','alfaro','alferez','alfonso','alfredo','ali','alia','alianza','alicia','alicio','alicua','alida','alimentarias','alimentos','alina','alinton','aliria','alirio','alis','alisa','alisamar','alison','alisson','alix','aliz','aljadi','aljure','allison','allisson','alliver','alma','almanza','almario','almeida','almendra','almendres','almendrez','alnoimer','alomia','alondra','alonso','alos','alpaca','alpala','alquedan','alquive','alsate','altamar','altamirano','altura','alulema','aluminio','alva','alvarado','alvaran','alvarez','alvarez','alvaro','alvear','alvedia','alveiro','alvenis','alves','alviar','alvis','alviz','alyn','alysson','alzamora','alzate','alzle','ama','amada','amadeo','amado','amador','amaia','amalci','amalf','amalfi','amalfy','amalia','amall','amancio','amanda','amarfi','amariles','amaris','amaury','amaya','amazo','ambuila','ambuz','amelia','amelie','amer','america','americo','amezquita','amibia','amilcar','amilvia','amin','amina','aminta','amira','amor','amorocho','amortegui','amortiteca','amparo','ampudia','amu','amy','ana','ana|','anabeiba','anabeiva','anabel','anabela','anabolena','anacona','anahi','anahia','anaibe','anali','analida','analy','anatalia','anavela','anaya','anayaeli','anayibe','anayiber','anchico','ancizar','and','ander','anderson','andersson','andica','andrade','andre','andrea','andreina','andres','andres','andrew','andrey','andry','anduquia','andy','anely','anganoy','angarita','angel','angel','angela','angeles','angeli','angeli','angelica','angelica','angelik','angelina','angeline','angelino','angello','angelly','angelo','angely','angge','anggy','angi','angie','angiee','angisay','angola','angrino','angucho','angulo','angy','anibal','anidisney','anirley','anita','anjie','anllely','anlly','anna','anni','annie','annirley','anny','anonio','anrrango','ante','antero','anthonella','anthony','antia','antidio','antimo','antolinez','antonella','antoni','antonia','antonieta','antonio','antony','antuan','anturi','anuar','anyel','anyela','anyeli','anyella','anyelli','anyelo','anyely','anyi','anyuri','anzola','añasco','apache','aparicio','apellidos','apolinar','apolinaria','apolindar','aponte','aponza','apostol','apraez','aptos','aqueli','aquilina','ara','araceli','aracelly','aracely','aragon','aragon','aramburo','araminta','arana','aranda','arandia','arando','aranelly','arango','aranguren','arantxa','aranza','aranzazu','araque','arara','ararat','arasmith','araujo','arbeiro','arbelaez','arbey','arboleda','arcadio','arce','arcelia','arcelio','arcenio','arces','arcesio','arcila','arcini','arciniega','arciniegas','arcira','arcley','arcon','arcos','ardila','aredondo','areiza','arelis','arellano','arena','arenales','arenas','arevalo','argelia','argemiro','argenis','argenys','argenzola','argermiro','argidia','argote','argoti','argotte','argoty','arguello','ariadna','ariana','ariane','arias','aricapa','aricia','ariel','ariela','aris','arismendi','arismendy','arist','aristides','aristizabal','aristizabal','aristobulo','ariza','arizabalate','arizabaleta','arizala','arizmendi','ariztizabal','arlency','arleny','arles','arlesa','arlex','arley','arlony','arly','armando','armenia','armero','armida','arnedo','arnobia','arnobio','arnoby','arnoldo','arnolfo','arnolia','arnoris','arnory','arnubia','arnulfo','aroca','arocemena','arosa','arosemena','arpesod','arquimedes','arrechea','arredondo','arrerondo','arriaga','arrias','arriero','arrieta','arris','arrollo','arroyabe','arroyave','arroyo','arrubia','arrubla','arsayus','arsecio','arsenio','arsesio','arteaga','artemio','artemo','artiaga','artunduaga','arturo','arvey','ary','arzayus','arzayuz','asbeidy','ascanio','ascencio','asceneth','ascension','ascuntar','asdrubal','asencion','aseneth','aseo','asesorias','ashley','ashli','ashlly','ashly','asis','aslan','asly','asmed','asmucol','asnoraldo','asociacion','asociados','asprilla','astaiza','astañeda','astella','asteria','asterio','astolfo','astrid','astros','astudillo','atanasio','ategue','atehortua','athala','atias','atuesta','audberto','audini','audivert','audrey','audry','augusto','aulestia','aulia','aulin','aura','auraelina','aurelena','aurelia','aureliano','aurelina','aurelinas','aurelio','auristella','aurita','aurora','ausecha','ausique','austin','autocorp','autonoma','aux','auxilio','avalo','avalos','avella','avendaño','avieser','avila','avila','aviles','avirama','avril','axel','aya','ayala','ayazo','aychell','ayda','ayde','aydee','aydivy','ayerbe','ayerve','ayili','aylin','aymer','ayner','ayora','ayovi','ayuda','aza','azael','azcarate','azucena','azyade','b.','babativa','baca','bacca','badillo','bados','baena','bahamon','bahamon','bahos','baicue','bairon','balanta','balantalina','balbin','balbina','balcazar','balcazar','baldemar','baldomero','baldrich','balladalez','ballen','ballestas','ballesteros','balmes','balmory','baltan','baltazar','balvin','bambague','banderas','banessa','banguera','banguero','banol','banoy','banquet','bañol','baos','baquero','barahona','barandica','barbara','barbetti','barbosa','barcenas','barco','barlaham','barney','baron','barona','barragan','barrantes','barreiro','barreneche','barrera','barrero','barreto','barrientos','barriga','barrios','barroso','barrrera','bartola','bartolo','bartolome','basante','basilia','basilio','bastidas','basto','bastos','batero','batioja','bautista','bayardo','bayer','bayona','bayron','bazan','bazantes','bbva','beatris','beatriz','beca','becca','beccy','becerra','becoche','bedolla','bedon','bedoya','beiba','beimer','bejarano','beky','bela','belalcazar','belarmina','belavenko','belen','belinda','belisa','belisario','belizabet','bella','bellaizan','bellalid','bello','belly','belquiz','beltran','beltran','benachi','benalcazar','benardo','benavides','benavidez','benedicto','benemerito','benhur','benicia','benicio','benigno','benilda','benilde','benites','benitez','benito','benjamin','benjamin','benjumea','benvides','benzon','berardo','berenice','berfalia','bermeo','bermudez','bermudez','bermuidez','bernabe','bernal','bernales','bernarda','bernardo','bernate','bernelio','berney','beron','beronica','berrio','berta','bertha','berthalia','bertil','bertilda','bertilde','bertulfo','besrnal','bessy','betancour','betancourt','betancourth','betancur','betancurluis','betancurt','betancurth','betes','bethy','betsabe','betsabeth','betsy','betty','betulia','beyanil','beylin','bibian','bibiana','biggitti','billy','biojo','birmany','biscue','biscunda','bislealdo','bismar','biteli','biuza','biviam','bivian','biviana','blackburn','bladimir','blainer','blanca','blanco','blandon','blandon','bleik','bleisi','blexon','bobadilla','bocanegra','bogota','bohorquez','bojaca','bojorge','bolaños','bolañoz','bolena','bolivar','bolivia','bolney','bomberos','bone','bonifacia','bonilla','booder','borda','boris','borja','bornachera','borrais','borras','borrero','botache','botello','botero','botina','bouzas','bpo','brady','brahant','brahiam','brahian','brahyam','bram','bran','brand','brandon','brasilia','bravo','brayam','brayan','braynny','bredy','breiner','brenda','brenyer','breychell','breydy','breyner','brian','briceida','bricella','briceño','briches','bridge','bridgee','brigette','briggitte','brigid','brigida','brigitte','brilli','brillit','brillith','brilly','brillyth','brindarly','briñez','brito','britsy','brittany','britto','briyid','briyith','briyitt','broquis','bruce','brunilda','bryan','bryllitte','btancourth','bubu','bucanero','bucheli','buchelli','buelvas','buenaventura','buendia','bueno','buesaquillo','buga','buila','builes','buitrago','buitron','bula','burbano','burgos','burgues','buritica','buritica','busscar','bustamante','bustillos','bustos','byron','c','cabal','caballero','cabanillas','cabanzo','cabezas','cabral','cabrales','cabrera','cacais','caceres','caceres','cadavid','cadena','caez','cafe','cagua','caguasango','cagueñas','caicedo','caipe','cairasco','caisa','caja','cajares','cajas','cajiado','cajiao','cajica','cajigas','calambas','calapsu','caldas','calderon','calderon','caldon','caleb','caleño','calero','calibio','calima','caliredes','calle','callejas','calles','calonge','calonje','caludia','calvache','calvaran','calvo','calzada','camacho','camargo','camayo','cambindo','camelo','camila','camilo','camla','campaña','campas','campaz','campeon','campiño','campo','campoelias','campos','campuzano','canabal','canache','canal','canaz','canchala','canchon','cancimance','cancimanci','candamil','candel','candela','candelo','candida','cando','canencio','canga','canizales','canizalez','cano','cansimanse','cantero','cantillo','cantor','canvindo','caña','cañar','cañarte','cañas','cañaveral','cañizalez','cañola','cañon','capera','capote','capurro','carabali','carabalica','caraballo','caracas','carbonell','carbonero','cardenas','cardenas','cardeño','cardiel','cardona','cardoso','cardoza','cardozo','cardula','caren','cargues','caridb','carillo','carime','carina','carla','carlina','carlo','carlomar','carlos','carlosama','carlota','carmelina','carmelita','carmelo','carmen','carmensa','carmenza','carmeza','carminia','carmiña','carmona','carnes','caro','carol','carola','carolan','carolay','carolina','caroll','carranza','carrascal','carrasquilla','carreño','carrera','carrero','carretero','carriillo','carrillo','cartagena','carton','carvajal','carvjal','casa','casadiego','casallan','casallas','casamachin','casanova','casañas','casaran','casas','casierra','casilimas','caso','casseta','cassetta','casso','castaneda','castano','castañeda','castaño','casteblanco','castelblanco','castellano','castellanos','casteñeda','castiblanco','castilla','castillo','castilo','castiño','castrillo','castrillon','castro','castsñeda','catacoli','catacoly','catalina','catalino','catamuscay','cataño','caterin','cathalina','catherin','catherine','cathy','catuche','cauca','caucayo','caviche','caviedes','caycedo','cc','cdem&cdeb','ceballos','cebastian','cecilia','cecilio','cedano','cedeño','cediel','ceferino','ceforo','celada','celades','celen','celene','celeni','celeny','celeste','celestino','celi','celia','celimo','celina','celis','celiz','celmira','celorio','celsa','celso','celvia','cely','cenaida','cenaira','cenayda','cened','ceneida','cenelia','cenelli','cenelly','cenen','ceneyda','cenide','cenider','cenobia','cenon','centeno','centro','cepeda','cerezo','ceron','ceron','cerpa','cerquera','cerrato','cersain','certuche','cervantes','cervera','cesar','cespedes','cess','cexprom','cha','chachanoy','chachinoy','chacon','chacua','chacue','chaguendo','chagueza','chaia','chaire','chala','chalacan','chalarca','chalare','chalarga','chambo','chambueta','chamizas','chamizo','chamorro','champutiz','chanchi','chantre','chañag','chapal','chaparro','chapid','chapurri','chara','charelin','charfuelan','chariana','charlotte','charon','charria','charrupi','charry','chasqui','chate','chaura','chausa','chaux','chavarria','chavarriaga','chavarro','chaverra','chaves','chavez','chayrla','chazatar','chazy','chec','checa','chelsea','chen','chenche','chepe','cheverry','chia','chiappe','chica','chicaiza','chicangana','chicue','chila','chilama','chilehuit','chilito','chilo','chima','chimachana','chimunja','chincha','chinchila','chinchilla','chinguad','chipichape','chiquito','chirley','chirpofer','chistian','chitan','chitiva','chito','choco','choconta','chocue','choi','cholo','chore','christiam','christian','christophe','christopher','christother','chrystian','chucrala','chud','chujfi','churi','cia','cidre','cielo','cifuentes','cilene','cileny','cilia','cindrid','cindy','cinia','cinthia','cinthya','cintya','cirley','ciro','cisley','cisneros','cisto','civiles','clair','claire','clara','clarena','clarencio','claret','claribel','claritza','clarivel','clariza','claros','claudia','claudina','claudineth','claudio','claver','clavijo','clayde','cleiser','cleiver','clelia','clemencia','clemente','clementina','cleofe','cleofelina','cleotilde','climaco','clinica','clinica','cliniserra','co','cobo','cobos','cock','cocoma','cocuñame','cocuy','codognotto','cogollo','cohecha','colina','coll','collantes','collazos','collo','colmenares','colo','colombia','colombiana','colombo','colonia','colorado','colpensiones','colviseg','combariza','comercio','cometa','comfamiliar','compañia','compuser','con','concai','concejo','concepcion','concepcion','concha','conda','conde','coneo','confecciones','congo','conindex','conny','conrado','consolacion','constain','constantino','constanza','construcciones','constructora','consuelo','consuleo','consultores','conto','contratando','contreras','cook','coolaborar','coomulsap','coopetativa','copete','coqueco','coques','coquis','coral','corbeta','corcino','cordero','cordoba','cordoba','cordobes','cordovez','corina','corleone','cornejo','cornelia','cornelio','cornelito','corodoba','coromoto','corona','coronado','coronel','corporacion','corral','corrales','correa','correal','correales','corredor','cortazar','cortes','cortez','cortinas','corzo','cosecha','cosme','cossio','costa','cotazo','cote','cotrina','covaleda','coy','cresencia','crespo','crhistian','criollo','crisdayan','crisostomo','crispin','cristal','cristancho','cristhiam','cristhian','cristhofer','cristi','cristiam','cristian','cristina','cristo','cristobal','cristofer','cristopher','cros','cruz','cruzana','crystal','crystian','cuaccialpu','cuacialpu','cuadrado','cuadros','cuaero','cuaical','cuaican','cuaicuan','cuaran','cuartas','cuasapud','cuasmayan','cuaspa','cuaspud','cuastumal','cuatin','cuaycan','cubides','cubillos','cucaita','cucalon','cuchillo','cuchimba','cuci','cucuma','cucuñame','cuellar','cuenca','cuenu','cuero','cueros','cuerpo','cuervo','cuesta','cuestas','cuetia','cueto','cueva','cuevas','cujar','cuji','cultid','cumbal','cumbe','cunda','cundumi','cupajita','cupitra','cusi','cuspian','cuspoca','custodia','custodio','cuta','cutiva','cynthia','da','dabeiba','dabeyba','dacareth','dadi','dagoberto','dagua','dahian','dahiana','dahiann','daian','daiana','daich','daifeny','daihana','dailin','dainer','daira','dairo','dairon','dais','daissy','daisy','daiyi','daizury','daladier','dalfeni','dalia','daliana','dalila','dalilly','daliri','dallana','dallys','damari','damaris','damelines','damian','damiro','damizuri','dana','dandiney','daneida','danelia','danelly','daneris','danery','daneyi','dani','dania','danicela','daniel','daniela','daniella','danilo','danilsa','danir','danis','danissa','danitza','danllely','danna','danniela','danny','danober','danora','danover','dante','dany','danyerlin','danyi','dar','daraviña','darbey','darcy','dari','daria','dariana','dario','dario','daris','darley','darli','darlid','darlin','darlina','darling','darlinton','darly','darlyn','darnelly','darwin','dary','daryery','dasly','dasy','datasae','dativa','dauqui','davalos','david','daviddavid','davila','davinson','davismach','day','dayan','dayana','dayanery','dayanna','dayhana','dayhann','dayra','dayron','daysuri','daysury','dayzury','daz','daza','dazza','deantonio','deaza','debie','debora','deborah','decio','deiba','deiber','deibi','deiby','deibys','deicy','deifa','deifan','deifeny','deimi','deiny','deiro','deisi','deison','deisy','deiva','deivi','deivid','deivis','deivy','del','dela','delayda','delbasto','delcida','delfa','delfamira','delfina','delgadillo','delgado','delgiza','delia','delina','delio','delkin','delliponti','delly','delma','delmy','delsy','delver','dely','deni','denice','denis','denise','denisse','dennis','denny','dennys','densy','denudetid','deny','denys','denyse','denzel','deossa','departamental','derazo','derek','derian','derli','derlin','derling','derlis','derlly','derly','dery','desideria','deverson','devia','devian','dey','deyaneri','deyanira','deybi','deyby','deyci','deyi','deylan','deymer','deysi','deysy','dharren','dhayana','diago','dian','diana','dianely','dianeth','dianey','dianid','diannet','dianny','diany','dias','diaz','diaz','diby','dickson','dictto','dicue','didier','didimo','diego','diela','diesel','dievely','diez','digitex','digna','dignery','dignier','dignora','dik','dila','dilan','dileny','dilfa','dilia','dillan','dilma','dilson','dilvia','dimar','dimarchy','dimas','dimate','dina','dinas','dinelli','diney','dinori','diocele','diogenes','diogne','diomedes','diomelina','diomira','dione','dioney','dionisia','dionisio','dionne','diony','dios','diosa','dioselina','diossa','diozahara','dique','direccion','directv','diriam','dirsa','disnery','disney','disneyda','dissan','dissu','distribuidora','distrital','diuza','diva','diver','divia','divian','divier','diyi','dizu','docier','dogman','dolcey','doli','dolly','dolmi','dolores','doly','domico','domiguez','dominga','domingo','dominguez','dominguez','dominic','donaciano','donatello','donatila','doney','donna','donneys','donny','donoso','dora','doracela','dorado','doralba','doralia','doralice','doralis','dorance','dorani','dorey','doreysi','doria','dorian','doriela','dorila','doris','dorlam','dorley','dorman','dorronsoro','dory','doryhan','dorys','dositeo','dosman','douglas','drews','driana','droguerias','drombo','duarte','dubal','duban','dubelly','duber','duberley','dubernal','duberney','dubian','dubys','dudli','dueñas','dufay','dulce','dulcey','dulfady','dulfary','dulfay','dulnelly','duque','duran','durango','durfay','durfey','durlandy','durley','dussan','duval','duvan','duver','duverney','dwight','dydier','dylan','dyro','dyssetth','dyvi','e','eastman','ebelia','eberney','ebet','eblin','ecco','ecdiver','echabarria','echavarria','echavrria','echegoyen','echeverri','echeverry','echevery','eda','eddie','eddier','eddna','eddy','edelmar','edelmira','edelvin','eder','ederson','edgar','edgardo','edi','ediber','edic','edid','edie','edier','edificaciones','edil','edilbe','edilberto','edilia','edilma','edilmer','edilsa','edilso','edilson','edima','edinarco','edinson','edinsonn','edinsson','edison','edisory','edit','edith','editha','editjh','edixon','edna','ednna','edoin','edson','edsson','edu','eduar','eduard','eduardo','educativa','eduerd','eduin','edulfa','edulfaris','edusmildo','edwar','edward','edwardo','edwars','edwarth','edwin','edy','efectivas','eficacia.','efigenia','efigenio','efrain','efrain','efred','efredy','efren','ega','egberto','egidio','egile','egna','egricelda','ehernan','ehivar','eida','eiden','eider','eiderman','eidi','eilin','eilyn','eiman','eimy','einar','eine','einer','eisenhooer','eiser','eivar','eiver','eiverth','el','eladio','elan','elba','elbar','elber','elbia','elcie','elcira','elcy','elda','elder','elean','eleana','eleany','eleazar','electricos','electrogenos','eledxader','elejalde','elena','elenith','eleonardo','eleonora','eleuterio','eli','elia','elian','eliana','elias','elibardo','eliberto','eliced','eliceo','elicer','elida','elider','elidier','eliecer','elina','elio','eliobel','elisa','elisabet','elisabeth','eliseo','eliud','eliver','eliza','elizabet','elizabeth','eljure','elkin','ella','elmar','elmer','elodia','eloina','eloisa','eloise','eloy','elpidia','elquin','elsa','elsi','elsidie','elsin','elsis','elson','elsy','eltidio','elva','elver','elvia','elvin','elvio','elvira','elvis','ely','elza','ema','emaida','emanuel','emaris','ember','emberto','embus','emelda','emely','emerita','emerson','emid','emidio','emigdio','emilce','emilda','emili','emilia','emiliana','emiliano','emilio','emilli','emilse','emilsen','emilssen','emily','emir','emira','emiro','emlia','emma','emmanuel','emmily','emperatriz','emplear','empresa','en','enais','enaliced','encarnacion','enciso','encizo','ened','enedis','eneida','enelia','eneriet','enerieth','eneyda','engracia','enid','enides','enidia','enidt','enilda','enio','enit','enith','enna','enny','enoc','enoe','enoris','enrioue','enrique','enriqueta','enriquez','enriquez','enrquez','enrrique','ensueño','enuar','enyell','epaminondas','epm','erasmo','eraso','erazmo','erazo','ercelly','ercilia','ercy','erdilia','eric','erica','erick','erik','erika','erlendy','ermeliza','ermila','erminda','erminia','erminso','erminsul','ermira','erned','ernedys','ernestina','ernesto','erney','ernilso','ervin','ery','esau','escalante','escallon','escamilla','escandon','escarpetta','escarraga','escarragas','escarria','escilda','esclava','escobar','escobedo','escolastica','escovar','escudero','escuela','esdrube','ese','esguerra','esilia','esimirey','esleida','eslendi','eslin','esly','esme','esmeralda','esmerlin','esmilda','esmira','esnayder','esneda','esnedy','esneider','esnelda','esnelia','esnery','esneth','esney','esneyder','esnna','españa','esparza','especialista','espejo','espeña','esperanza','espin','espinal','espinel','espinosa','espinoza','espitia','esquivel','essica','est','estacio','estafany','estanislao','esteba','estebali','esteban','estefam','estefani','estefania','estefanny','estefany','esteffany','estefhany','estefi','estela','estelia','estella','estephania','ester','esterling','estevan','esteven','esthefania','esther','estiben','estiven','estrada','estrella','estrellita','estrid','estupiñan','etayo','etelberto','etelvina','ethan','etm','etna','etsahin','eucaris','euclides','eudalia','eudocia','eufemia','eufemiano','eufracio','eufrasino','eugenia','eugenio','eugenioa','eugradis','eulalia','euli','eulises','eulofia','eulogio','eumir','eunice','eunise','euripides','euscategui','euse','eusebia','eusebio','eusse','eustacia','eustorgio','euvid','eva','evadista','evan','evangelina','evangelista','evarbo','evelia','evelier','evelin','evelina','evelio','evelly','evelsi','evelyn','evencio','evening','ever','everney','evert','eveyiseth','evolet','exdubar','exel','exela','expain','exposito','eybar','eyber','eyder','eydi','eydryan','eyiseth','eyleen','eylin','eylyn','eyzlen','ezequiel','ezray','fabara','faber','fabert','fabian','fabian','fabiano','fabio','fabioa','fabiola','fabricio','facundo','faency','fager','fahissury','faissury','faisully','faisuri','faisury','faizury','fajardo','falla','falon','fals','falya','fandino','fandiño','fanery','fanni','fannor','fanny','fanor','fanory','farfan','farid','faride','farley','farnery','farnesio','farney','fary','fatima','faustina','faustino','fausto','faver','favian','favid','favio','fayneri','faysiury','faysuli','fayzuly','fda','feder','federico','federman','felicia','felicidad','felicitas','felipa','felipe','felisa','felix','feneida','feniber','fenifer','feniver','fenner','ferannda','ferdinando','fergie','feria','ferla','ferley','fermin','fernada','fernadez','fernado','fernan','fernana','fernanada','fernanda','fernandez','fernandez','fernando','fernely','ferney','ferraro','ferreira','ferrer','ferrerosa','ferrin','ferro','fiallo','fichica','fidel','fideligna','fidelina','fidencio','fides','fieber','fierro','fiesco','figueroa','figureroa','filena','filiberto','filigrana','filippo','filomena','fiquitiva','fiscalia','fisely','fitzgeral','fitzgerald','flaquer','flavio','flechas','flecher','flerida','flor','flora','floralba','florancy','florangela','florencia','florencio','florentina','florentino','flores','floresmilo','florez','florez','flori','florian','floriano','floridana','florido','florines','floripes','flover','flower','flowers','floyd','fluvia','folleco','fonnegra','fonseca','fontal','fontalvo','fontecha','forero','fori','forigua','formit','fortich','fory','fracturas','francedys','francelia','francelina','franceline','franceny','francery','frances','francesco','franci','francia','francined','francis','francisca','francisco','franciso','franco','francois','francy','frank','franklin','frankly','franky','franquelina','franquis','franz','fraysury','frecia','freddy','fredericson','fredi','fredy','fredys','freidel','freider','freides','freire','freitas','fremat','fresanelia','fresneda','freyre','frimer','frimet','frncisco','froy','fruit','fuelantala','fuelpas','fuelpaz','fueltala','fuenmayor','fuentes','fuertes','fuerzas','fujiy','fuller','fulvia','fundacion','fundacion','fuquene','furio','g','gabalan','gabi','gabina','gabino','gabriel','gabriela','gabriella','gabrill','gael','gaez','gaitan','galan','galarraga','galarza','galeano','galia','galicia','galindez','galindo','gallady','gallardo','gallego','galliadi','gallo','gallon','gallon','galo','galvan','galves','galvez','galvis','galviz','gamba','gamboa','ganan','gañan','garaviña','garavito','garay','garcera','garces','garces','garci','garcia','garcia','gardel','garibello','garnica','garrido','gary','garzon','garzon','gasca','gaspar','gaviria','gazo','gean','gebara','gefrey','geiber','geiler','geiman','geiner','geinova','geinson','geison','gelsomina','gelsys','gelvez','gema','gembuel','genaro','gency','general','genesis','genesis','geni','genid','genis','genith','genivora','genny','genova','genoveva','gente','gentil','geny','geobana','geohana','geomar','george','georgina','geovani','geovanni','geovanny','geovany','geradina','geraldin','geraldine','gerardina','gerardine','gerardo','gercy','gerhemy','gerji','gerley','german','germania','geromito','geronimo','gerrado','gersain','gersey','gerson','gersson','gertrudis','gerviz','gesni','gestion','getial','geurrero','gezenia','gheraldine','gian','gianella','gianfranco','gianina','giannino','gianny','giany','gibel','gicela','gicell','giesenow','gil','gilbardo','gilbert','gilberto','gildardo','gilde','gildo','gilma','gilmer','gilon','gina','ginet','gineth','ginette','ginna','ginneth','gino','giobanna','giomar','giovana','giovann','giovanna','giovanni','giovannie','giovanny','gipsy','giral','girald','giraldo','girlado','giron','gironza','gisel','gisela','gisell','gisella','giselle','giselth','giset','gissel','gissell','gissella','gisselle','gissette','giuliano','giuseppe','gizet','gladid','gladis','gladiz','gladuys','gladys','gleidys','gleydin','gloria','glubis','godoy','goez','golu','gomez','gomez','gongola','gongolino','gongora','gonzaga','gonzales','gonzales','gonzalez','gonzalez','gonzalias','gonzaliaz','gonzalo','gordillo','gordon','gorlato','gourmet','govinda','goyeneche','goyes','gracia','graciela','graciliana','graciliano','gracilis','grajales','gran','granada','granado','granados','granda','grande','grandez','granja','granobles','granosle','grece','grecia','gregori','gregoria','gregorio','greicy','greisy','gretty','grey','greyns','gricela','gricelda','gricelly','grijalba','grijalda','grincelly','grisales','grisalez','griselda','grobien','groelfi','grueso','gruesso','grupo','guaca','guacaneme','guacas','guacha','guacheta','guadalupe','guadir','guahuña','guaido','guainas','gualdron','gualmatan','gualtero','gualteros','guamanga','guamiolamag','guampe','guancha','guañarita','guapacha','guapache','guapacho','guapi','guaqueta','guaquez','guar','guaran','guardia','guarin','guarni','guarnizo','guasaquillo','guasca','guaspud','guastar','guata','guatame','guayabo','guayal','guayara','guaza','gudiela','guegia','gueiby','guendica','guengue','guepud','guerao','guerra','guerrero','guerron','guetio','guette','guevara','guezaquillo','gugu','guido','guillen','guillermina','guillermo','guilllermo','guilombo','guinand','guince','guinchin','guiomar','guiovani','guiral','guisel','guiseppe','guitierrez','gullermo','gumenrsindo','gumersindo','gurrute','gustavo','gustin','gustinez','gutiererez','gutierrez','gutierrez','gutierrez','guzman','guzman','habib','habid','hadassa','haddad','hader','hadith','hael','haider','haidy','hailen','haimer','hamel','hamer','hames','hamilthon','hamilton','hanae','handersson','haner','hanna','hannah','hanndri','hanner','hanrryr','hans','hansel','hansen','hanz','haouchar','harbey','harby','harinera','harlington','harol','harold','harrison','harry','harryson','hartunduaga','harvey','harvi','harvy','hary','hatta','haussman','hayda','haydee','haymer','hazkary','heber','hebert','heberth','heberto','heblyn','hector','hector','heda','heder','hedil','heduin','heibel','heiber','heider','heidi','heidy','heilen','heiling','heindember','heisson','heladio','helbert','helda','helder','helen','helena','heli','heliberto','helica','helida','heliodoro','heliut','hellen','helly','helmer','helver','hemel','hemilson','henan','henao','hendes','henio','henor','henri','henriquez','henry','heraldo','herbey','hercelia','hercilia','heredia','heriberto','herica','herleny','herlex','herley','herlinda','herly','herman','hermann','hermelina','hermelinda','hermelizelda','hermen','hermencia','hermenson','hermes','hermila','herminda','herminia','herminio','herminson','herminsul','herminzul','hermogenes','hermosa','hermoso','hermoza','hernadez','hernado','hernan','hernan','hernandez','hernandez','hernando','hernesley','hernesto','herney','heroina','herran','herreño','herrer','herrera','herreros','herrique','hersain','herson','hervin','herweanis','herzon','hesair','hesilda','hever','hevers','heyder','heydy','heynier','heyson','hibor','hidalgo','hidrobo','higinio','higuera','higuita','hija','hijo','hilario','hilarion','hilary','hilaryn','hilda','hildaura','hilde','hildebrando','hilder','hillary','hilman','hilton','hincapie','hincapie','hinestrosa','hinestroza','hinojosa','hipasio','hipolito','hirosi','hislena','hissami','hobana','hoffman','hoffmann','holanda','holguin','holguin','holman','holmer','holmes','holver','home','homero','homes','honorio','hoover','hooverth','horacio','hordubay','horley','hormaza','horta','hortencia','hortensia','hortua','hospital','hoya','hoyola','hoyos','hoz','hrrl','huaza','hubeny','huber','huberley','hubert','huerfano','huerta','huertas','hugo','huila','humbarila','humberto','hungria','hunter','hurmendiz','hurtado','huseim','hutter','huver','huverney','i','ia','ian','ibagon','ibague','iban','ibañez','ibarbo','ibarguen','ibarra','ibarvo','ibata','iber','ibeth','ibett','ibis','ibon','ico','icopo','ida','idacio','idalba','idale','idali','idalia','idalid','idaly','idarraga','idarraga','idelbert','idelga','idelse','ideneo','ider','iderlan','idrobo','ifalia','igar','iglesias','ignacio','igua','iguaran','iguita','ijaji','ilbia','ilda','ildelides','ilder','ildira','ilduara','ileana','iles','ilia','iliana','ililian','illegas','illera','illya','ilma','ilner','ilsa','ilse','ilva','imat','imbachi','imbago','imbajoa','imelda','impormaderas','impuestos','inagan','incapie','incauca','incaucaservicios','inchima','indira','industriale','inelda','ines','inestrosa','infanta','infante','ingenio','ingenios','ingrib','ingrid','ingridt','ingris','ingrit','ingritd','ingry','inides','inirida','inmaculada','inocencia','insa','insandara','institucio','institucion','instituto','insuasti','insuasty','integrado','integral','integrales','interfisica','internacional','inversiones','inyaely','ipia','ipiales','ipila','ipujan','iragorri','iralda','irene','iriarte','irina','iris','irlan','irlanda','irlena','irleny','irley','irma','irurita','isaac','isaacs','isabel','isabela','isabelina','isabell','isabella','isabeth','isadora','isaias','isajar','isamara','isamel','isanoa','isanora','isaura','isaza','isaziga','isco','isdar','ises','ishibashi','isidora','isidoro','isidro',]
        b_name = ['eva','isis','islena','isleny','ismael','ismenia','isneire','isnelda','isnely','isolina','israel','isrrael','italia','italo','ithan','itsahiana','ituyan','itzel','ivan','ivan','ivana','ivaniel','ivanna','iver','iverson','ives','iveth','ivette','ivon','ivone','ivonn','ivonne','ixchel','iza','izan','izquierdo','jabela','jacinta','jacinto','jackelina','jackeline','jackelinne','jacob','jacobo','jacome','jacqueline','jader','jadiel','jady','jael','jagua','jahaira','jahiana','jahir','jahumer','jaiber','jaidar','jaide','jaider','jaidi','jaidiver','jaidivy','jaime','jaimer','jaimes','jainer','jair','jairo','jaiseep','jaiver','jajoy','jak','jakeline','jalid','jalie','jalil','jalile','jalin','jalitza','jaller','jama','jamauca','jamel','jamer','james','jamiledh','jamileth','jamilton','jamioy','jamir','jan','jana','janer','janet','janeth','janey','jani','jania','janice','janier','janine','janira','janner','jannet','janneth','jannier','janniher','janny','jansasoy','jansenio','jaquelin','jaquelina','jaqueline','jara','jaramillo','jaramilo','jarbin','jarby','jared','jarinzon','jarky','jarlin','jarly','jarme','jarminson','jarminton','jaro','jarrinson','jarvi','jasbleidy','jasel','jasmin','jasned','jason','jativa','jauime','javela','javer','javiano','javier','jaycol','jayder','jayko','jazmin','jean','jeanet','jeanneth','jeannette','jefer','jeferson','jefferson','jeffeson','jeffrey','jeffry','jehison','jeibi','jeicob','jeider','jeidy','jeimmy','jeimy','jeinny','jeins','jeison','jeisson','jeiver','jelianna','jembuel','jency','jeneth','jenffer','jenidia','jenifer','jeniffer','jeninnfer','jenith','jenkryfer','jenneffer','jennifer','jenniffer','jennilled','jenny','jennye','jennyfer','jensin','jeny','jenyfer','jenyffer','jeovana','jeraldin','jerdey','jeremias','jeremie','jeremih','jeremy','jerez','jeris','jermain','jerny','jeronimo','jeronimo','jerson','jesabeth','jesica','jesid','jessi','jessica','jessika','jesucita','jesus','jesus','jeykob','jeymy','jeyns','jeynsson','jeyson','jhair','jharolyn','jharrison','jheison','jhenyfer','jheraldin','jherien','jherlin','jhessenia','jhoan','jhoana','jhoanna','jhoel','jhoetmy','jhohana','jhohanna','jhohanny','jhojainer','jhojan','jhojana','jhojanes','jholfady','jhon','jhonalex','jhonatan','jhonathan','jhonattan','jhond','jhonier','jhonn','jhonnatan','jhonnathan','jhonnier','jhonny','jhonson','jhony','jhorfan','jhorjan','jhorlan','jhorman','jhoselin','jhosselyn','jhosua','jhosue','jhovana','jhovang','jhulian','jhuliana','jhurany','jicel','jihan','jill','jimena','jimenes','jimenez','jimenez','jimeno','jimmi','jimmy','jimy','jineth','jirado','jireh','jirlesa','jisell','jiseth','joahana','joan','joana','joanna','joany','joao','joaqui','joaquin','joaquin','joaquina','joe','joel','johan','johana','johann','johanna','johanny','johany','john','johnatan','johnathan','johnier','johnn','johnnatan','johnnier','johnny','johny','johon','johs','johvana','joimer','joiner','jojana','jojoa','jonatan','jonathan','jonattan','jonh','jonnatan','jonnathan','jonny','jony','jordan','jordany','jordy','jorge','jorladys','jorman','joronda','jos','josa','jose','jose','josefa','josefina','joselin','joselyn','josemaria','joseph','joseth','joshep','joshua','joshue','josias','joslany','josse','josselin','josselyn','josser','jossie','josue','josue','jova','jovana','jovanna','jovanni','jovanny','jovany','jovel','jualian','juan','juana','juanillo','juanita','juber','judit','judith','judy','juith','juleidy','julen','julia','julialba','julian','julian','juliana','julianna','julicue','julie','julier','juliet','julieta','julieth','juliett','julio','julissa','jully','julvia','july','junier','junior','junnior','jurado','jurany','juri','juridica','juspian','jusseth','justin','justina','justiniano','justinico','justino','justo','justyn','juvenal','juverney','jynei','kacterine','kahory','kahterine','kaled','kalindi','kamila','kamily','karem','karen','karent','karime','karin','karina','karla','karlo','karlos','karol','karolayn','karolina','karoll','kassandra','katalina','kateherine','katerin','katerine','kathalina','katherin','katherine','katheryn','katheryne','katiza','katlin','katterine','kattering','kattie','katty','katy','keila','keiner','kelen','keler','kelin','kelis','kelly','kely','kemberle','kenay','kendall','kener','keneth','kenia','kennedy','kennet','kenneth','kenny','keren','kerly','ketherine','ketsender','kety','kevin','kevinn','kevyn','kewin','keyla','keyner','kiliang','kimberly','kirlyn','kiyomi','klinger','koba','korina','koure','krafft','kristian','krysthol','l','la','labao','labio','laborales','labrada','labrador','laddy','ladecol','ladino','ladis','lady','ladys','lagarejo','lago','lagos','laguna','laia','laidys','laila','laines','lais','lajas','lalinde','lam','lame','lamilla','lamos','lamprea','lamus','lanchero','lancheros','landaeta','landazabal','landazary','landazuri','landazury','landinez','landino','landy','lañas','lara','larbin','lareo','largacha','largo','larrahondo','larrañaga','larrarte','larrea','larrosa','larry','las','lasa','laserna','lasprilla','lasso','latorre','laudino','laura','laureano','lauren','laurens','laurentina','laurido','laurnt','lauzier','lavao','laverde','lavrea','lay','layden','laydi','lazaro','lazo','leady','leah','leal','leandra','leandro','leannny','leany','lebaza','ledesma','ledezma','ledis','ledys','lee','leeam','legal','legarda','legria','legro','leguizamo','leguizamos','leiba','leibia','leiby','leida','leide','leider','leidi','leido','leidy','leidy','leila','leina','leiner','leirin','leisi','leison','leiton','leiva','leiver','leivy','lejandra','lelia','lema','lemos','lemus','lena','lengi','lenid','lenin','lenis','lenix','leny','leocadia','leofrande','leomina','leon','leon','leonarda','leonardo','leoncio','leondenis','leonel','leonela','leones','leonidas','leonila','leonilde','leonisa','leoniza','leonor','leonora','leontina','leopoldina','leopoldo','lerma','lernes','lesbia','lesby','lesdy','leslie','lesly','lesmes','lessdy','lesvi','lesvia','leticia','letrado','leudo','levi','levis','levy','lewinson','lewis','leyda','leydi','leydy','leyes','leyla','leyrin','leyton','leyva','lezama','lezcano','lia','liam','lian','liana','libad','libaniel','libardo','liber','libia','liboria','liborio','librada','libreros','liced','licenia','liceth','liciardy','licinia','lid','lida','lider','liderman','lidermina','lidia','lidier','lidis','lievano','lifuyed','ligia','liham','lila','lili','lilia','liliam','lilian','liliana','lilianaandreacalero','lilibeth','lilino','lilley','lilly','liloy','lily','lima','limbania','lina','linares','lince','linda','lindanyi','lindelia','lindo','lindsay','lineth','liney','linggren','linian','lino','linsay','linthon','linton','linz','liria','lis','lisa','lisandro','lisbe','lisbeth','lisbey','liscano','lisdaris','liseht','liselmo','liset','liseth','lisette','lisimaco','lisney','lisset','lisseth','lissette','liver','livia','liviam','lix','liz','liza','lizanyuri','lizarazo','lizbeth','lizcano','lizeth','lizette','llague','llamosa','llano','llanos','llanten','llarida','llirly','lloreda','llovany','loaiza','lobaton','lobello','loboa','lobon','local','lodt','lody','logan','logistica','lola','lombana','lombo','londono','londoño','londoñoo','londooo','longa','longana','lopeda','lopera','lopez','lopez','lora','lore','lorean','loren','lorena','lorensa','lorenza','lorenzo','lores','lorieth','lorin','lorna','lorza','los','losada','loteria','lotero','lourdes','lourido','lovera','loza','lozada','lozads','lozano','ltada','ltda','ltda.','luan','lubby','lubdibia','lubelly','lubian','lubin','lubo','lucano','lucas','luccy','lucedy','luceida','luceli','lucelia','lucelida','lucelli','lucelly','lucely','lucena','lucenid','luceny','lucero','luci','lucia','lucia','luciana','luciano','lucidia','luciela','lucila','lucinda','lucio','luciola','luciria','luciveli','lucrecia','lucresia','lucumi','lucy','ludeina','luder','ludibia','ludivia','ludy','luengas','lugil','lugo','luider','luis','luisa','luiyith','lujanith','luke','luligo','luly','luna','lupe','lus','lusaida','lusby','luseyne','luvier','luz','luzaida','luzbian','luzby','luzcelly','luzdary','luzmedia','luzmila','lyam','lyda','lyons','lysbeth','mabeisi','mabel','mabeli','maber','maca','macca','machado','macias','macuace','macuase','madahy','madan','madelaine','madeleine','madeleyne','madeleys','madelyn','mader','madrid','madrigal','madriñan','madroñero','maestre','mafier','mafla','magali','magalli','magally','magaly','magaña','magda','magdalena','magdalida','maglioni','magnalila','magnely','magnol','magnolia','magnory','magnus','magola','magon','magy','mahecha','maibel','maicol','maigual','maikol','maile','mailen','maillely','maily','mainaguez','mainguez','maira','mairena','mairongo','maithe','majin','malagon','malambo','malcon','maldonado','malena','males','malfis','malfitano','mallama','mallarino','mallely','mallorquin','malori','malory','malpud','mambague','mambuscay','mamian','mancera','manchabajoy','manchola','mancilla','manco','manfredy','manizales','manjarres','manquillo','manrique','manrrique','manso','mantilla','manuel','manuela','manufacturas','manyi','manyoma','manzano','manzo','manzour','mañosca','mañozca','mañunga','mañuzca','mao','mapura','maquilon','mar','marahevel','marcela','marceliano','marcelino','marcelo','marcia','marcial','marciana','marciano','marco','marcos','marcucci','mardoris','mardory','marduk','mare','mareina','marelbi','marely','marelyn','marenco','margalida','margareth','margarita','margid','margie','margori','margorie','margory','margot','margoth','margy','marha','mari','maria','maria','mariaca','mariah','mariajose','marialit','mariam','marian','mariana','marianela','marianella','mariangel','marianita','marianna','mariano','mariantonia','mariany','mariapaz','maribed','maribel','maricel','maricela','maricelly','maricely','marie','mariela','mariella','marielly','marielvy','marieth','mariette','marihtza','marilene','marilu','mariluz','marilyn','marin','marin','marina','marinc','marincano','marinela','marinella','marinez','marinilse','marino','mariño','mario','marion','maris','marisel','marisela','marisell','marisella','marisinelly','marisol','marisolani','maritza','maritzabell','mariu','marjely','marjy','mark','marlen','marlene','marleni','marlenis','marleny','marles','marley','marleyi','marlin','marlit','marlly','marlody','marlon','marly','marlyn','marlyng','marmol','marmolejo','marny','marñolejo','marolanda','marquez','marquez','marquinez','marriaga','marroquin','marta','martha','martin','martin','martina','martines','martinez','martinez','martiniano','martino','martir','martnez','martos','maruja','marulanda','marvin','marwin','mary','marya','marybel','maryeri','maryi','maryluz','maryoli','maryori','maryory','marysol','marystella','maryuri','maryury','mashacuri','masivo','masmela','massiel','masso','mata','matabanchoy','matallan','matallana','matavanchoy','mateo','materano','materon','matess','mateus','matew','mathias','mathius','matias','matilde','matiz','matoma','matta','matthew','maturana','mauna','maura','maureen','mauren','mauricio','maurico','maurieth','mauro','maury','mauselen','mavel','maver','maveryng','maximiliano','maximo','maximus','maxwell','maya','mayarlih','mayckol','maycol','mayela','mayeli','mayely','mayeni','mayer','mayerli','mayerlin','mayerling','mayerlis','mayerly','mayerlyn','mayers','mayileth','mayli','mayolin','mayor','mayorca','mayorga','mayorin','mayorquin','mayra','mayrin','mayuli','mayuri','mazabuel','mazo','mazorra','mazuera','mc','mclea','mecias','medardo','medellin','medicina','medina','meek','megan','meibe','meiby','meili','meivy','mejia','mejia','mejias','melan','melania','melanie','melanny','melany','melba','melchor','melecio','melendes','melendez','melenge','melenje','melfi','melia','melida','melindres','melisa','melissa','meliton','meliza','mellizo','melo','melqui','melquisedec','melva','melvy','mely','mena','menandro','mendes','mendez','mendez','mendieta','mendinueta','mendivelso','mendoza','meneces','menecez','menelly','menese','meneses','mensa','menza','meñaca','mera','meraly','mercado','mercedes','mercenario','merchan','merchancano','merchant','merci','mercy','merida','merino','merizalde','merle','merlin','merlly','merly','merlyn','mery','mesa','mesias','messa','mesti','mestil','mestizo','mestra','mesu','metal','metales','metaute','metromovil','meyer','meza','mezu','mia','micedith','michael','michaell','michel','micheli','michell','michelle','michelly','micolta','middleton','mieles','mier','migdalia','migdonia','miguel','mikol','mila','miladis','milady','miladys','milagro','milagros','milan','milanes','milbia','milciades','milde','mildred','mildreth','mildrey','mile','miled','miledy','mileidy','mileisy','milena','milenis','mileny','miler','milet','mileth','milexy','mileydy','milgen','milindres','militares','millan','millar','millenium','miller','millerlandy','milley','miloy','milton','milvia','mimalchi','mimi','mina','minda','minota','minu','minyeli','mira','miralba','mirama','miramag','miranda','mirbean','mireizon','mirelda','mirelia','mirella','mirelly','mireya','mirialba','miriam','mirian','mirillo','mirley','mirna','mirquez','mirta','mirtha','miryam','miryan','miryeny','misael','misaela','misaelina','misas','mischele','mishell','miyerlay','modesta','mogollon','mogollon','mogrovejo','mohamet','moises','moises','mojarrango','mojica','molano','moledo','molina','molineros','mompotes','mona','monar','moncada','moncaleano','moncayo','mondragon','mondragon','monedero','monica','monica','monique','monje','monroy','monrroy','monsalve','monserrat','monserrath','montacargas','montalvo','montana','montaña','montañez','montaño','montealegre','montegranario','montehermoso','montenegro','montero','monterrey','montes','montezuma','montiel','montilla','montillo','montoya','montserrat','moquera','mora','morales','moralez','moran','morant','morante','morantes','morcillo','morea','moreira','morelia','moreno','morera','moriano','morillo','moriones','morostoque','mosca','moscoso','moscote','mosorongo','mosquera','mossos','mostacilla','mota','motato','motoa','motor','motta','movil','moya','moyano','mr','mucua','muebles','muelas','mueses','mujer','mulato','mulcue','muller','multiactiva','multiservicios','munares','munebar','munera','munevar','municipal','municipio','munoz','muñeton','muño','muñoz','mur','murcia','murgas','murgueitio','muriel','murillas','murillo','muskus','mustafa','mutis','mutual','mykel','myllan','myriam','myrian','myryam','nabollan','nabor','naboyan','nacianceno','nacion','nacional','nacionales','nacivar','nader','nadia','nagles','nahia','nahiara','nahir','nahomi','nahum','naiala','naiara','naidu','naikira','nain','nampia','nanci','nancy','napoleon','naranjo','narcilo','narcisa','narciso','nardy','naren','nariño','narly','narradondo','narvaez','narvaez','narvez','nasamuez','nasly','nastacuaz','natacha','natali','natalia','natalie','natally','nataly','nates','nathalia','nathalie','nathaly','nathasha','natib','natividad','naufa','nauffal','nava','naval','navarrete','navarreto','navarro','navas','naveros','navia','nayady','nayancy','nayarith','nayda','naydu','nayeli','nayely','nayerli','nayibe','nayiby','nayive','nayla','naymar','nazareth','nazari','nazario','nazarit','nazly','nazlyn','nebrijo','neby','nedier','nedy','nefer','neftali','negret','negro','neicet','neicy','neida','neifer','neiger','neila','neiman','neira','neire','neisa','neison','neithan','neiva','neiver','neiza','nel','nela','nelci','nelcy','nelffi','nelfi','neli','nelia','nelida','nella','nelli','nelly','nellyreth','nelsi','nelson','nelsy','nelver','nely','nelys','nemecio','nemesio','nemu','nena','nereyda','nerieth','nerihe','nerly','neronimo','nery','nesly','nesmy','nestor','nestor','netty','neubedy','neuver','ney','neyci','neyired','neyler','neymar','neyra','neytan','nhora','nia','nibia','nickincyn','nicol','nicolas','nicolas','nicolasa','nicolay','nicold','nicole','nicoll','nicolle','nidia','nidya','niebles','nieto','nieva','nieves','nikol','nikolai','nikolas','nikole','nikoll','nilen','nille','nillereth','nilo','nilsa','nilsen','nilson','nilton','nilvi','ninco','ninfa','nini','niny','niño','nipaz','nipcela','nirama','nircy','niria','nirza','nivia','nixon','niyereth','niyiret','niyireth','niyirey','no','noe','noe','noebia','noel','noelba','noelia','noelva','noelvia','noemi','noemy','nogardo','noguera','nogues','nohelia','nohellys','nohely','nohemi','nohemy','nohira','nohora','nohra','noira','nolberto','nolmar','nombre','nomelin','nomina','nomina.com','nonietzko','nora','noralba','noraldo','norayda','norbelly','norberto','norbey','norby','norela','norelia','norely','norena','noreña','norfa','norfalia','norfi','norha','norhelia','nori','noriega','norles','norley','norma','norman','norvey','nory','noscue','novoa','noy','nubia','nubiola','numael','nunez','nunila','nuñez','nupan','nupia','nur','nurelba','nury','nuvia','nuvidia','nydia','ñañez','ñuscue','ñustes','oamira','oarles','obando','obdulia','obdulio','obed','obeida','oberi','obersen','obertulio','obonaga','obras','obregon','ocampo','ocando','occidente','ochoa','ociel','ocoro','octalivar','octavila','octavio','ocupar','oderay','odila','odilfo','odilia','odilma','odin','odriguez','oelty','ofelia','ofelmira','offir','ofir','oidor','oime','ojeda','olano','olariz','olarte','olave','olaya','olegario','oleida','olga','olguin','oliday','olimpia','olimpica','olinda','olinde','oliria','oliva','olivar','olivares','oliver','oliverio','oliveros','olivia','olivio','olivo','olma','olman','olmedo','olmer','olmes','olmos','olver','omaira','omar','ome','omen','omery','omes','omez','oneida','oney','oneyda','onid','onofre','oquendo','oralia','orbegozo','orcue','ordenoñez','ordonez','ordoñes','ordoñez','ordoñez','orduz','oreidy','orejarena','orejuela','orellana','orfa','orfalina','orfanery','orfany','orfelina','orfely','orfilia','orfindey','organizacion','oriana','orielly','origua','oriol','orjuela','orlady','orlain','orlando','orlania','orlay','orley','orlinda','orlivia','ormeño','orobio','orosco','orozco','orporacion','orrego','orsileny','ortega','ortegon','ortiz','ortunduaga','osa','osbaldo','oscar','ose','osidis','osiel','osiris','osleyda','osma','osman','osmer','osoario','osorio','osorno','ospima','ospina','ospitia','ossa','ossiris','osso','osvaldo','oswaldo','otalbaro','otalivar','otalora','otalvaro','otalvora','otavo','otaya','otero','otilia','otoniel','otto','ovalle','ovalles','oved','oveimar','overimar','oveth','ovidio','oviedo','ovirley','oweimar','owens','oyola','pabla','pablo','pabon','pabon','pacateque','pacheco','pachene','pachichana','pacho','pachon','pacue','padilla','padro','padua','paez','paez','paguatian','pahola','paja','pajarito','pajaro','pajoy','palacio','palaciodos','palacios','paladines','paladinez','palau','palechor','palecia','palencia','palma','palmezano','palmina','palmira','paloma','palomeque','palomino','palomo','palta','pamela','pamo','pamplona','panameño','panesso','paniagua','pantoja','paola','papamija','papel','papeles','papsy','paque','paracios','paramo','pardey','pardo','paredes','pareja','paris','parra','parrado','parraga','parrales','partner','paruma','pascagaza','pascual','pascuala','pascuas','pascuaza','pascumal','pasos','pasquel','pastor','pastora','pastrana','pasuy','patarroyo','paternina','patichoy','paticia','patino','patiño','patrcia','patrica','patricia','patrocinia','patrocinio','paubla','paucar','paul','paula','paulina','pauline','paulo','pava','pavel','pavi','pavon','paya','payan','payares','paz','pazmiño','pazos','pazu','peada','pealez','pechene','pechucue','pecillo','pedraza','pedrero','pedreros','pedro','pedroza','pedrozo','pelaez','pelaez','penagos','pencua','penilla','peña','peñafiel','peñalosa','peñaloza','peñaranda','peñarete','peñuela','pepicano','perafan','peralta','perdomo','perdonmo','perea','pereañez','pereira','perenguez','peres','perez','perez','perez','perico','perilla','perla','perlaza','pernia','perrea','persides','personeria','pescador','pesellin','pestana','peter','petrel','petsain','petter','petuma','phanor','pherson','piamba','pianda','pichica','pichimata','pico','picon','piedad','piedra','piedrahita','pierre','pieruchini','pilar','pilar','pilcue','pillimue','pimienta','pinch','pinchao','pineda','pinilla','pinillo','pinillos','pino','pinos','pinto','pinzon','pinzon','piña','piñerete','piñerez','piñeros','piño','pipicano','piquitero','piracoca','piraneque','piraquive','piscal','pita','pito','piza','pizarro','pizo','plata','playonero','plaza','plazas','plinio','plutarco','polanco','polania','policarpa','policarpo','polindara','pollos','polo','poloche','polonia','pomares','pombo','pomeo','pompilio','ponbo','ponton','pool','popayan','popo','porfirio','porras','portela','portilla','portillo','portocarrero','posada','poscue','posse','posso','possu','poter','potes','potosi','poveda','pracedes','prada','pradilla','prado','preciado','presentacion','presiga','prestadora','pretel','pretelt','prez','pricelis','prieto','primero','primitivo','primo','primoris','priscila','proaños','productora','proing','providencia','proyeccion','proyectos','ptrocinia','puablo','pubenza','publica','puchana','puente','puentes','puerta','puertas','puerto','puetate','puetes','pulecio','pulgar','pulgarin','pulgarin','pulido','pulpa','puni','pupiales','purificacion','pusil','q','q.v','quebrada','quebradas','quejada','quelal','quenguan','quenoran','quesada','quetama','quevedo','quezada','quiceno','quick','quiguanas','quiguanaz','quijano','quilcue','quilindo','quimbaya','quimbayo','quina','quinayas','quinceno','quinche','quinonez','quintana','quintero','quinteroval','quintin','quinto','quiñones','quiñonez','quique','quirama','quiroga','quiros','quiroz','quisoboni','quistanchala','quistial','quitiaquez','quitombo','quitumbo','quiza','quizamano','r','raba','racines','rada','rafael','rafaela','raga','raigosa','raigoza','ralphfi','ramirez','ramirez','ramirezx','ramiro','ramo','ramon','ramos','rangel','raphael','raquejo','raquel','rasha','raudino','raul','raul','rave','raymond','rayo','rcn','realpe','reaños','rebeca','rebellon','rebeyon','rebolleda','rebolledo','recalde','recio','redemer','redes','redondo','refiere','regina','regional','reina','reinaldo','reinario','reinel','reinelio','reinell','reinelly','reinerio','reinier','reinilfa','reinosa','reinoso','remedios','remeo','remicio','remigio','remolina','renata','rendon','rendon','rene','rened','renfijo','rengifo','renteria','renteria','renza','resguzman','restrepo','resurreccion','retavisca','retrepo','revellon','revelo','rey','reyes','reynaldo','reynel','reza','reznik','riaño','riaños','riasco','riascos','riascos&nbsp;','riazcos','ribon','ricardina','ricardo','ricaurte','ricaute','ricci','riccio','richard','richer','rico','rigoberto','rincon','rincon','rincones','rio','rioja','riomalo','riorrecio','rios','rios','riovo','ripe','ripoll','risaralda','risso','rita','rivas','rivasq','rivera','rivero','riveros','rivillas','rizo','roa','roballegas','robayo','robeiro','robelto','rober','robert','roberteau','roberth','roberto','robertulio','robin','robins','robinson','robledo','robles','rocha','roche','rocio','rock','rodallega','rodallegas','rodas','rodolfo','rodrigez','rodrigo','rodriguez','rodriguez','rogelia','rogelio','rogelis','roger','rogerio','roin','rojas','rojo','rolando','roldan','roldanillo','roman','roman','romaña','romel','romelia','romeo','romero','romiliana','rommy','romo','romulo','romy','ronal','ronald','roncancio','rondon','roney','ronny','roosevelt','roostvelh','roque','rosa','rosabel','rosada','rosalba','rosalbina','rosales','rosalia','rosalinda','rosaline','rosana','rosangela','rosario','rosas','rosaura','rose','roselia','roseline','roselver','rosely','rosember','rosemberg','rosen','rosendo','rosero','rosillo','rosina','rosinda','rosio','rosiver','rosmary','rosmery','rosmira','roso','rossemary','rossi','rotavisky','rotavista','roure','rovinson','rovira','roxana','royero','rozo','rta','rua','ruales','ruano','rubby','rubelia','ruben','ruber','rubert','rubi','rubia','rubialba','rubiano','rubidia','rubiel','rubiela','rubier','rubilia','rubio','rubria','rubriche','ruby','ruco','rudas','rudy','rueda','rufina','rufino','ruge','rugeles','ruiz','ruiz','rumaña','rumualdo','ruperto','ruque','rusbel','rusby','rusca','russi','rutber','ruth','ruverth','rvera','rym','sas','s','s.a','s.a.','s.a.s','s.a.s.','sa','saa','saavedra','sabarain','sabas','sabath','sabi','sabina','sabogal','sabrina','sacramento','sadi','sady','saenz','sahiary','said','saida','saide','saiduta','saira','saitemp','salamanca','salamando','salas','salazar','salcedo','saldaña','saldariaga','saldarriaga','salem','salgado','salguero','salima','salinas','salma','salom','salome','salome','salomon','salud','salustino','salvador','sam','samanta','samantha','samara','samboni','sambony','sambrano','sameco','samir','samira','sammer','sammy','sampayo','sampedro','samuel','samuell','samy','san','sana','sanabria','sanches','sanchez','sanchez','sanclemente','sanders','sandoval','sandra','sandro','sangiovanni','sanin','sanjuan','sanmiguel','sanna','sanson','sant','santa','santacoloma','santacruz','santafe','santamaria','santamaria','santana','santander','santanilla','santiagho','santiago','santibañez','santigo','santillana','santofimio','santos','sanz','sapuyes','sara','sarabia','sarah','sarai','sarasa','sarasti','sarasty','sarat','saray','sarita','sarmiento','sarrea','sarria','sarzosa','sas','sastoque','sastre','satisabal','sativa','satizabal','saturia','saucea','saucedo','saul','saulo','savine','sayledt','sayuri','scarpeta','scarpetta','sebastian','sebastian','secretaria','sedano','sedas','sediel','sedoc','seferina','segovia','segrera','segunda','segundo','segura','seguridad','seguro','seira','seiza','selemin','selene','seleny','selorio','semanate','senayda','sendoya','senel','señor','sepulveda','sergina','sergio','serna','serra','serrano','serrato','services','servicio','servicios','serviconstrucciones','serviespeciales','servio','severino','severo','sevilla','sevillano','seydel','shadday','shadid','shaiel','shaik','shailler','shaira','shalom','shara','sharay','sharick','sharit','sharom','sharon','sharyk','sharyth','sheccid','sheila','shek','shelly','sherley','shermie','shery','sheyla','shirledy','shirley','shti','siarah','sibaja','sicard','sicaroni','sicua','sidney','sierra','siesly','sigifredo','siglo','sigrid','silena','silfredo','silva','silvana','silvia','silvio','simanca','simbaqueba','simon','sinar','sindi','sindia','sindicato','sindy','sinisterra','sintrainagro','sinza','sion','sir','sirlene','sirley','sissa','sivoney','sixta','sixto','siyey','slanne','sleither','slith','smart','smc','smit','smith','snayder','sneider','sniper','soacha','sobeida','sociedad','socorro','soffi','sofi','sofia','sofia','sofonias','sofya','sogamoso','sohir','soide','sol','solandria','solandy','solange','solangel','solangy','solanilla','solanlly','solano','solanyi','solar','solarte','soledad','soledis','soleibe','soleine','soleiny','solemy','soler','solery','soley','solfemira','solhangelly','soliman','solis','solita','solla','solomoflex','solorsano','solorzano','soluciones','sonia','sonnia','sons','sophia','sophie','sor','soraida','sorane','sorangel','sorany','soraya','sorayda','sorenydht','sorey','sorfiria','sori','soriano','sorin','sorlanda','sorley','sorleyda','sorvay','sory','sosa','soscue','sossa','soteldo','sotelo','sotero','soto','sotomayor','sperez','spitia','sprintis','steban','stefani','stefania','stefanie','stefannia','stefannia','stefanny','stefannya','stefano','stefany','stefanya','steffany','stefhani','stefhania','stefhany','stehiner','stela','stella','stephania','stephanie','stephanny','stephany','stephanya','sterling','stevan','steve','steveen','steven','stevens','stevenson','steward','stf','sthefani','sthefania','sthefannia','sthefany','sthella','sthepanie','sthepany','sther','sthevan','sthevens','stiveen','stiven','sttid','studio','styles','styven','styward','suarez','suarez','suaza','sucrala','sucre','sucroal','suescun','sugey','sujely','sujey','sulay','suldery','suleima','suley','suleydi','suleydy','sulgey','sully','sulma','suluaga','suly','sumitep','superior','supermercado','supermercados','supertiendas','supremo','surata','suray','surly','surtifamiliar','susan','susana','susy','suzarte','suzuki','sylvana','sylvia','taba','tabares','tabarez','tabarquino','tabary','tabima','taborda','tacan','tadalinka','tafur','tafurt','taguado','taicus','taimal','tairy','takata','talaga','talento','taliana','tamara','tamayo','tami','tandeoy','tandioy','tangarife','tania','tao','tapasco','tape','tapia','tapias','tapie','tapiero','tapieros','taquez','taquinas','taramuel','tarapues','tarazona','tarcila','tarquino','tasama','tascon','tascon','tatiana','tautas','tavera','tax','taxis','teamco','tecnohigiene','tedesco','tejada','telcos','teleche','telesforo','tellez','tello','temporal','temporales','tengono','tenorio','teodocia','teodolfo','teodolinda','teodomira','teodora','teodoro','teodosia','teofila','teofilo','tepud','tequin','teran','teresa','teresita','terma','terranova','terreros','terresa','territorial','tersila','tesillo','teuta','tex','tezna','thailin','thalia','thaliana','thalya','thamara','thanairy','thevenet','thiago','thomas','thommas','tibaduiza','tiberio','ticora','tifanny','tigreros','timana','timaran','timon','timoteo','tinallas','tinoco','tintinago','tique','tirado','tirsa','titimbo','tito','toapanta','tobar','tobias','tobito','tobon','toledo','tolosa','toloza','tomas','tombe','tonusco','tonuzco','topa','toquica','torcoroma','torijano','toro','toronto','torre','torrente','torres','torrez','torrijos','toscano','tosse','tovar','trabajadores','transito','transporte','transportes','tranzit','travi','treghetti','trejo','trejos','triana','tribiño','tri-fit','trineyda','trinida','trinidad','triny','triviño','troche','troches','trochez','trujillo','trullo','truque','tuberquia','tucanes','tulande','tulandy','tulcan','tulia','tulio','tulua','tumbaqui','tumbo','tumiña','tunubala','tupue','tuquerres','turisticas','turizo','turjillo','turriago','tusarma','tuta','tutacha','tutistar','tys','ubaldina','ubaldo','ubeimar','ubelny','ubeny','ubenyde','uberley','uberlinda','ubernel','uberney','uchima','uchinba','uchur','udislay','uintero','ul','ulabarry','ulbrainer','ulchate','ulchur','ulcue','ulda','uldrey','uleivar','ulises','ulloa','ullos','ullune','umaña','umbarila','unas','ungria','unigarro','union','unispan','universidad','uno','upegui','uran','urania','urbano','urbertino','urbina','urcue','urcuqui','urdaneta','urdinola','urfania','uribe','uricoechea','uriel','urieles','urmendez','urmendiz','urquijo','urquina','urrea','urrego','urresta','urreste','urresti','urresty','urriago','urrutia','urueña','usa','useche','usma','usme','ussa','ustariz','usuga','usuriaga','usurriaga','uzuriaga','uzurriaga','v','vaca','vacca','valasco','valbuena','valdeblanquez','valderrama','valderruten','valdes','valdez','valdiri','valdivia','valdivieso','valdubino','valencia','valenciano','valens','valentierra','valentin','valentina','valentino','valenzuela','valeri','valeria','valerie','valero','valery','valeryn','valezco','valicia','valladales','valladares','valle','vallecilla','vallejo','valles','vallesilla','valnery','valois','valor','valquidia','valverde','valvula','vander','vanegas','vanesa','vaness','vanessa','vaneza','vannessa','vaquero','vaquiro','varanzeta','varela','vargas','varon','varon','varona','vasco','vasques','vasquez','vasquez','vasto','vda','vecsit','vega','veimar','veimer','vejarano','vela','velandia','velarde','velasco','velasquez','velasquez','velazco','velazquez','velez','velez','velosa','veloza','venachi','venegas','veneranda','venicia','venites','ventas','vente','ventura','venus','vera','verde','verdugo','verenguez','verganzo','vergara','vergel','vernaza','veronica','vesga','via','viafara','viafara','viagnery','viana','vianey','viany','vicenta','vicente','victor','victor','victoria','victoriano','vicuña','vida','vidal','vidales','vidarte','vidaura','vieda','viedma','viedman','viera','vigoya','vilamar','vilgia','villa','villabon','villabona','villacorte','villada','villadiego','villafañ','villafañe','villafuerte','villalba','villalobos','villamarin','villamil','villamizar','villamuez','villan','villan','villani','villano','villanueva','villaquiran','villaquiran','villareal','villarraga','villarreal','villarruel','villasmil','villavicencio','villegas','villeth','villota','vilma','vilmary','viloria','vinasco','vinculo','vinseth','viñas','violet','violeta','violetta','vique','virgelina','virgen','virgilia','virgilio','virginia','virisimo','viscaya','viscue','visitacion','visscher','vitalina','viteri','viuda','viva','vivana','vivas','vivero','viveros','viviam','viviama','vivian','viviana','viviany','vivienda','viviiana','vladimir','vmanrique','vnastar','voluntarios','volveras','von','vozmedia']
        c_name  =['ana', 'julia','ariza','dawer','wagner','walberto','waldina','waldir','walter','waltero','walteros','walther','wanessa','warner','wbaldino','wednesday','weimar','weizlen','wenceslao','wendy','wg','wheymar','whilinton','white','wiky','wilber','wilberto','wilches','wilchez','wildemar','wilder','wilderman','wilfer','wilford','wilfredo','wilfrido','wiliams','willan','willberth','willen','william','willian','willians','willington','willinton','wilma','wilman','wilmar','wilmer','wilmert','wilson','wilter','wilton','winy','wisberth','wisnton','wiston','wlabarry','wladimir','wm','wolf','wtk4','x','xavier','xenia','xilena','ximena','xiomahara','xiomara','xiury','xxi','y','ya','yaca','yackeline','yacnury','yacuma','yacumal','yaddy','yader','yaderli','yaderlyn','yadilin','yadir','yadira','yady','yaen','yaffah','yague','yahanna','yailan','yailin','yaily','yaima','yaira','yairmil','yaisa','yaisseidi','yajaira','yakeline','yala','yalanda','yaledit','yalena','yalila','yalileth','yalit','yalmira','yama','yamanaka','yamarlyn','yamid','yamil','yamila','yamile','yamiled','yamileida','yamilet','yamileth','yampuezan','yan','yancy','yande','yandeivis','yandi','yandri','yandun','yandy','yaned','yaneida','yanela','yanes','yaneska','yanet','yaneth','yaney','yanguas','yanguma','yanid','yanier','yanince','yanira','yanire','yaniria','yanisa','yannell','yanneth','yannier','yanten','yanten','yanza','yañez','yaquelin','yaqueline','yaqueno','yara','yarby','yareli','yarely','yaribelly','yarilet','yarine','yaritza','yariz','yaroline','yascuaran','yasiris','yasmid','yasmin','yasminy','yasno','yasnury','yasodhara','yatacue','yate','yavely','yazmin','yazmira','yeberxon','yecenia','yecid','yeferson','yeferzon','yefferson','yehisuni','yei','yeiber','yeibi','yeibig','yeico','yeidi','yeidy','yeiko','yeiler','yeimar','yeime','yeimi','yeimmi','yeimmy','yeimy','yeiner','yeini','yeinson','yeiny','yeisi','yeison','yeisson','yela','yelaine','yelexis','yelitza','yelleany','yelly','yemir','yenci','yency','yendi','yeni','yenifer','yeniffer','yenireth','yenis','yenith','yenni','yennifer','yenniffer','yenny','yennyfer','yennys','yensi','yensy','yeny','yepes','yepez','yeraldi','yeraldin','yeraldine','yeraldini','yerardin','yeris','yerislady','yerladys','yerlany','yerlez','yeron','yerson','yesenia','yeserid','yeshua','yesica','yesid','yesika','yesquen','yessenia','yessica','yessid','yessika','yestin','yesviley','yetzy','yeyenny','yezenia','ygnolia','yheiner','yhirman','yhon','yhonalde','yhordan','yiam','yibi','yiced','yicel','yicela','yicele','yiceth','yidi','yilber','yilena','yiliana','yilma','yilmer','yimara','yimmi','yina','yine','yined','yinehy','yinet','yinetc','yineth','yinna','yinneth','yiomara','yira','yiret','yiria','yirlene','yirlesa','yisel','yisela','yiseli','yisell','yiseth','yisnei','ynes','yoan','yoana','yoani','yoanna','yoany','yobani','yobanny','yobany','yocue','yodin','yoel','yofer','yoger','yohana','yohandri','yohanna','yohn','yoiner','yoisy','yojana','yoladys','yolanda','yolando','yolehidy','yoleine','yolena','yolima','yoline','yolisbeth','yoliveth','yolvy','yoly','yomaira','yomara','yomay','yon','yonatan','yonathan','yonda','yonny','yony','yor','yordy','yorfany','york','yorladis','yorladys','yorleni','yorley','yorly','yorman','yormensy','yoryani','yosando','yoseph','yoshef','yoshua','yosimar','yosselyn','yotengo','yotoco','youblin','yovana','yovani','yovanna','yovanni','yovany','ysis','yubel','yubiza','yubleny','yuceth','yuco','yucuma','yudi','yudy','yuladith','yulady','yulani','yulder','yule','yuleidy','yuleni','yuleny','yuli','yulian','yuliana','yulianna','yuliany','yulied','yulier','yuliet','yuliete','yulieth','yulisa','yulisnedy','yulissa','yulith','yuliza','yullieth','yully','yulmery','yuly','yumarly','yunda','yuraima','yurani','yuranny','yurany','yureidi','yuri','yurian','yuridya','yurledy','yurley','yurly','yury','yusbeli','yusef','yusell','yuseth','yusledi','yusmery','yustes','yusti','yustres','yusty','yuttajeidi','yuttiel','yuviza','yvonne','z','zabala','zacha','zaday','zafiro','zafra','zahira','zaida','zailem','zaitler','zalamanca','zamara','zambran','zambrano','zamir','zamora','zamorano','zamudio','zandra','zapata','zape','zarah','zarama','zarante','zarate','zaray','zareth','zaria','zayda','zazipa','zea','zegarra','zeida','zein','zemanate','zemyaxe','zenaida','zenon','zharick','zharik','zipagauta','zmin','zoe','zoila','zonal','zonia','zoraida','zoray','zoraya','zorayda','zorelly','zorilla','zorrilla','zoyla','zulay','zuleima','zulema','zuleta','zuley','zuleyma','zuleyme','zulima','zully','zulma','zulman','zuluaga','zuly','zulyma','zuñiga','zuñiga','zuri','zurita','zury']
        names=a_name+b_name+c_name
        stop_num=['uno','dos','tres','cuatro','cinco','seis','siete','ocho','diez','once','doce','trece','catorce','quince','dieciseis','diecisiete','dieciocho','diecinueve','veinte','veintiuno','veintidos','ventitres','veinticuatro','veinticinco','veintiseis','veintisiete','veintiocho','veintinueve','treinta','treinta y uno','treinta y dos','treinta y tres','treinta y cuatro','treinta y cinco','treinta y seis','treinta y siete','treinta y ocho','treinta y nueve','cuarenta','cuarenta y uno', 'cuarentay dos', 'cuarentay tres','cuarenta y cuatro','cuarenta y cinco', 'cuarenta y seis', 'cuarenta y siete', 'cuarentay ocho', 'cuarentay nueve', 'cincuenta', 'cincuenta y uno','cincuenta y dos', 'cincuenta y tres', 'cincuenta y cuatro', 'cincuenta y cinco', 'cincuenta y seis', 'cincuenta y siete', 'cincuenta y ocho', 'cincuenta y nueve', 'sesenta', 'sesenta y uno', 'sesenta y dos', 'sesenta y tres', 'sesenta y cuatro', 'sesenta y cinco', 'sesenta y seis', 'sesenta y siete', 'sesenta y ocho', 'sesenta y nueve', 'setenta','setenta y uno', 'setenta y dos', 'setenta y tres', 'setenta y cuatro', 'setenta y cinco', 'setenta y seis', 'setenta y siete', 'setenta y ocho', 'setenta y nueve', 'ochenta', 'ochenta y uno', 'ochenta y dos', 'ochenta y tres', 'ochenta y cuatro', 'ochenta y cinco', 'ochenta y seis', 'ochenta y siete', 'ochenta y ocho', 'ochenta y nueve', 'noventa', 'noventa y uno', 'noventa y dos', 'noventa y tres', 'noventa y cuatro', 'noventa y cinco', 'noventa y seis', 'noventa y siete', 'noventa y siete', 'noventa y ocho', 'noventa y nueve', 'cien', 'doscientos','trescientos','cuatrocientos','quinientos','seiscientos','setecientos','ochocientos', 'novecientos','mil','dosmil', 'tresmil','cuatromil','cincomil','seismil','sietemil','ochomil','nuevemil','diezmil', 'oncemil','docemil','trecemil','catorcemil','quincemil','dieciseismil','diecisietemil','dieciochomil','diecinuevemil','veintemil'  ]
        sw_basico = ['hola','dias','buenas','ahorita','ahora','siguiente','mucho','traves','pues','dijeron','dice','gracias','senora','senor','señora','señor','buena','buen','buenos','buenas','dias','dia','noche','noches','tarde','tardes','a','al','algo','algunas','algunos','ante','antes','como','con','contra','cual','cuando','del','desde','donde','durante','e','el','ella','ellas','ellos','en','entre','era','erais','eramos','eran','eras','eres','es','esa','esas','ese','eso','esos','esta','estaba','estabais','estabamos','estaban','estabas','estad','estada','estadas','estado','estados','estais','estamos','estan','estando','estar','estara','estaran','estaras','estare','estareis','estaremos','estaria','estariais','estariamos','estarian','estarias','estas','este','esteis','estemos','esten','estes','esto','estos','estoy','estuve','estuviera','estuvierais','estuvieramos','estuvieran','estuvieras','estuvieron','estuviese','estuvieseis','estuviesemos','estuviesen','estuvieses','estuvimos','estuviste','estuvisteis','estuvo','fue','fuera','fuerais','fueramos','fueran','fueras','fueron','fuese','fueseis','fuesemos','fuesen','fueses','fui','fuimos','fuiste','fuisteis','ha','habeis','habia','habiais','habiamos','habian','habias','habida','habidas','habido','habidos','habiendo','habra','habran','habras','habre','habreis','habremos','habria','habriais','habriamos','habrian','habrias','han','has','hasta','hay','haya','hayais','hayamos','hayan','hayas','he','hemos','hube','hubiera','hubierais','hubieramos','hubieran','hubieras','hubieron','hubiese','hubieseis','hubiesemos','hubiesen','hubieses','hubimos','hubiste','hubisteis','hubo','la','las','le','les','lo','los','mas','me','mi','mia','mias','mio','mios','mis','mucho','muchos','muy','nada','ni','no','nos','nosotras','nosotros','nuestra','nuestras','nuestro','nuestros','o','os','otra','otras','otro','otros','para','pero','poco','por','porque','que','quien','quienes','se','sea','seais','seamos','sean','seas','sentid','sentida','sentidas','sentido','sentidos','sera','seran','seras','sere','sereis','seremos','seria','seriais','seriamos','serian','serias','si','siente','sin','sintiendo','sobre','sois','somos','son','soy','su','sus','suya','suyas','suyo','suyos','tambien','tanto','te','tendra','tendran','tendras','tendre','tendreis','tendremos','tendria','tendriais','tendriamos','tendrian','tendrias','tened','teneis','tenemos','tenga','tengais','tengamos','tengan','tengas','tengo','tenia','teniais','teniamos','tenian','tenias','tenida','tenidas','tenido','tenidos','teniendo','ti','tiene','tienen','tienes','todo','todos','tu','tus','tuve','tuviera','tuvierais','tuvieramos','tuvieran','tuvieras','tuvieron','tuviese','tuvieseis','tuviesemos','tuviesen','tuvieses','tuvimos','tuviste','tuvisteis','tuvo','tuya','tuyas','tuyo','tuyos','un','una','uno','unos','vosotras','vosotros','vuestra','vuestras','vuestro','vuestros','y','ya','yo','a','aca','ahi','ajena','ajenas','ajeno','ajenos','al','algo','algun','alguna','algunas','alguno','algunos','alla','alli','alli','ambos','ampleamos','ante','antes','aquel','aquella','aquellas','aquello','aquellos','aqui','aqui','arriba','asi','atras','aun','aunque','bajo','bastante','bien','cabe','cada','casi','cierta','ciertas','cierto','ciertos','como','como','con','conmigo','conseguimos','conseguir','consigo','consigue','consiguen','consigues','contigo','contra','cual','cuales','cualquier','cualquiera','cualquieras','cuan','cuan','cuando','cuanta','cuanta','cuantas','cuantas','cuanto','cuanto','cuantos','cuantos','de','dejar','del','demas','demas','demasiada','demasiadas','demasiado','demasiados','dentro','desde','donde','dos','el','el','ella','ellas','ello','ellos','empleais','emplean','emplear','empleas','empleo','en','encima','entonces','entre','era','eramos','eran','eras','eres','es','esa','esas','ese','eso','esos','esta','estaba','estado','estais','estamos','estan','estar','estas','este','esto','estos','estoy','etc','fin','fue','fueron','fui','fuimos','gueno','ha','hace','haceis','hacemos','hacen','hacer','haces','hacia','hago','hasta','incluso','intenta','intentais','intentamos','intentan','intentar','intentas','intento','ir','jamas','junto','juntos','la','largo','las','lo','los','mas','mas','me','menos','mi','mia','mia','mias','mientras','mio','mio','mios','mis','misma','mismas','mismo','mismos','modo','mucha','muchas','muchisima','muchisimas','muchisimo','muchisimos','mucho','muchos','muy','nada','ni','ningun','ninguna','ningunas','ninguno','ningunos','no','nos','nosotras','nosotros','nuestra','nuestras','nuestro','nuestros','nunca','os','otra','otras','otro','otros','para','parecer','pero','poca','pocas','poco','pocos','podeis','podemos','poder','podria','podriais','podriamos','podrian','podrias','por que','por','porque','primero desde','primero','puede','pueden','puedo','pues','que','que','querer','quien','quien','quienes','quienesquiera','quienquiera','quiza','quizas','sabe','sabeis','sabemos','saben','saber','sabes','se','segun','ser','si','si','siempre','siendo','sin','sin','sino','so','sobre','sois','solamente','solo','somos','soy','sr','sra','sres','sta','su','sus','suya','suyas','suyo','suyos','tal','tales','tambien','tambien','tampoco','tan','tanta','tantas','tanto','tantos','te','teneis','tenemos','tener','tengo','ti','tiempo','tiene','tienen','toda','todas','todo','todos','tomar','trabaja','trabajais','trabajamos','trabajan','trabajar','trabajas','trabajo','tras','tu','tu','tus','tuya','tuyo','tuyos','ultimo','un','una','unas','uno','unos','usa','usais','usamos','usan','usar','usas','uso','usted','ustedes','va','vais','valor','vamos','van','varias','varios','vaya','verdad','verdadera','vosotras','vosotros','voy','vuestra','vuestras','vuestro','vuestros','y','ya','yo']

        stop_words= names+stopwords.words('spanish')+stop_num+sw_basico

        self.df=df
        self.patterns1= self.patterns = [(r'indutrial','industrial'),(r'preautoriaciones','preautorizaciones'),(r'tralado', 'traslado'),(r'lamada', 'llamada'), (r'tamite', 'tramites'), (r'procedimientos', 'procedimiento'), (r'ortos', 'otros')]
        self.lista_sw=stop_words
        self.n= [(r'consiliacion','consiliación'),(r'reposicion','reposición'),(r'matricula','matrícula'),(r'practicas','prácticas'),(r'ocupacion','ocupación'),(r'construccion','construcción'),(r'instalacion','instalación'),(r'invasion','invasión'),(r'rapidas','rápidas'),(r'prescripcion','prescripción'),(r'escenarios','escenario'),(r'vinculacion','vinculación'),(r'cesantias','cesantías'),(r'cesantia','cesantias'),(r'reclamacion','reclamación'),(r'economica','económica'),(r'pension','pensión'),(r'evaluacion','evaluación'),(r'arboles','arbol'),(r'certificados','certificado'),(r'certificacion','certificado'),(r'sanitaria','sanitario'),(r'capacitacion','capacitación'),(r'acompanamiento','acompañamiento'), (r'ninas','niñas'),(r'donacion','donación'),(r'analisis','análisis'),(r'autorizacion','autorización'),(r'examenes','exámenes'),(r'renovacion','renovación'),(r'inclusion','inclusión'),(r'economico','económico'),(r'atencion','atención'),(r'informacion','información'),(r'tecnica','técnica'),(r'canicas','canecas'),(r'esterilizacion','esterilización'),(r'organico','órganico'),(r'disposicion','disposición'),(r'prestamo','prestámo'),(r'recoleccion','recolección'),(r'extension','extensión'),(r'critico','crítico'),(r'problematica','problemática'),(r'ecologico','ecológico'), (r'intervencion','intervención'),(r'bario','barrio'),(r'basuras','basura'), (r'árboles','árbol'),(r'arbol','árbol'),(r'arboles','árbol')]
        
      
    ########################################
    #          NUEVO
    #########################################
    def patrones(self, sector):

        lista_corr=[(r'bario','barrio'), (r'ambiental','ambiente'), (r'sanitariovisita','sanitario visita'), (r'favorablevisita','favorable visita')]
        STOPWORDS=["espacio","permiso","horario","extension","visita","informacion","derecho","peticion", "favorable", "comprendo", "concepto", "definitiva","definitivas"]
        stopwords = set(STOPWORDS)
        stopwords.update(["poner"])
        df=self.df
        df=df[df.sector==sector]
        df.reset_index(inplace=True, drop=True)

        def s_correction( text, patterns):
            patterns1 = [(re.compile(regex), repl) for (regex, repl) in   patterns]
            s = text
            for (pattern, repl) in patterns1:
                (s, count) = re.subn(pattern, repl, s)
            return s

        df['Contenido0']=df['Contenido0'].apply(lambda x: s_correction(x, lista_corr))
        df['Contenido0']=df['Contenido0'].apply(lambda x: x.split())
        lista_mensajeuser=[]
        for i in range(len(df)):
            lista_mensajeuser+=df['Contenido0'][i]
        long_string=','.join(lista_mensajeuser)
        #Creacion lista stop words
        wordcloud = WordCloud(width=1000, height=500,background_color="white",min_font_size=3, max_font_size=150, max_words=700, contour_width=80, contour_color='steelblue', margin=15, stopwords=stopwords)
        #Crear el word cloud
        wordcloud.generate(long_string)
        #Visualizar el word cloud
        plt.figure(figsize=(15,10))
        plt.imshow(wordcloud, interpolation='bilinear', resample=True)
        plt.axis("off")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write("**Nube de palabras sector {}:**".format(sector))
        st.write("Permite visualizar las palabras más representativas en cada uno de los sectores. ")
        st.pyplot()
        

    def palabras_ngramas(self,sector):
        df=self.df
        df.reset_index(inplace=True, drop=True)
        lista_stop_words=self.lista_sw

        def s_correction(text, patterns):
            patterns1 = [(re.compile(regex), repl) for (regex, repl) in   patterns]
            s = text
            for (pattern, repl) in patterns1:
                   (s, count) = re.subn(str(pattern), str(repl), str(s))
            return s

        def plot_10_most_common_words(count_data, count_vectorizer):
            #Grafico de frecuencias
            words = count_vectorizer.get_feature_names()
            total_counts = np.zeros(len(words))
            for t in count_data:
                total_counts+=t.toarray()[0]

            count_dict = (zip(words, total_counts))
            count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:30]
            words = [w[0] for w in count_dict]
            counts = [w[1] for w in count_dict]
            df0 = pd.DataFrame()
            x_pos = np.arange(len(words))
            df0['words'] = words
            df0['counts'] = counts

            df0 = df0.rename(columns={'words': 'Palabra clave', 'counts': 'Frecuencia'})
            st.write("**Términos más comunes en {}:**".format(segmento))
            st.write("Permite visualizar las palabras que aparecen conjuntas, es decir permite contextualizar la palabra, por ejemplo la palabra certificado puede aparecer con: bancario, trabajo, tributario. Por lo tanto, esta gráfica permite estudiar más en detalle los temas que se tratan en cada segmento.")
            fig = px.bar(df0, x='Palabra clave', y='Frecuencia', color_discrete_sequence=["lightseagreen","gold","yellowgreen"],   labels=dict(x="Palabra Clave", y="Frecuencia", color="Place"))
            
            st.plotly_chart(fig)

        if sector=='ambiental':
            df=df[df.sector=='ambiental']

            lista=self.n
            df['Contenido0']=df['Contenido0'].apply(lambda x: s_correction(x, lista))
            segmento = str.upper("EN EL SECTOR MEDIO AMBIENTE (N-gramas)")

            count_vectorizer = CountVectorizer(stop_words=['sonidos','secretario','presente','agradecemos','dedicado','permiso','derecho','dejarlo','entiendo','pregunta','desempena','quincena']+lista_stop_words,ngram_range=(2, 2))
            # Ajustar y transformar los términos procesados
            count_data = count_vectorizer.fit_transform(df['Contenido0'])
            #Visualizar los términos mas comunes
            return plot_10_most_common_words(count_data, count_vectorizer)

        if sector=='salud':
            df=df[df.sector==sector]

            lista=self.n


            df['Contenido0']=df['Contenido0'].apply(lambda x: s_correction(x, lista))
            segmento = str.upper("EN EL SECTOR SALUD (N-gramas)")

            count_vectorizer = CountVectorizer(stop_words=['poner','concepto','favorable','visita','sonidos','secretario','presente','agradecemos','dedicado','derecho','dejarlo','entiendo','pregunta','desempena','quincena']+lista_stop_words,ngram_range=(2, 2))
            # Ajustar y transformar los términos procesados
            count_data = count_vectorizer.fit_transform(df['Contenido0'])
            #Visualizar los términos mas comunes
            return plot_10_most_common_words(count_data, count_vectorizer)

        if sector=='educacion':
            df=df[df.sector==sector]

            lista=self.n


            df['Contenido0']=df['Contenido0'].apply(lambda x: s_correction(x, lista))
            segmento = str.upper("EN EL SECTOR EDUCACIÓN (N-gramas)")

            count_vectorizer = CountVectorizer(stop_words=['cancion','permita','medio','poner','favorable','visita','sonidos','secretario','presente','agradecemos','dedicado','derecho','dejarlo','entiendo','pregunta','desempena','quincena']+lista_stop_words,ngram_range=(2, 2))
            # Ajustar y transformar los términos procesados
            count_data = count_vectorizer.fit_transform(df['Contenido0'])
            #Visualizar los términos mas comunes
            return plot_10_most_common_words(count_data, count_vectorizer)


        if sector=='justicia y seguridad':
            df=df[df.sector==sector]
            lista=self.n


            df['Contenido0']=df['Contenido0'].apply(lambda x: s_correction(x, lista))
            segmento = str.upper("EN EL SECTOR JUSTICIA Y SEGURIDAD (N-gramas)")

            count_vectorizer = CountVectorizer(stop_words=['meses','solicita','trasto','solicito','fecha','horario','comprendo','cancion','permita','medio','poner','visita','sonidos','presente','agradecemos','dedicado','dejarlo','entiendo','pregunta','desempena','quincena']+lista_stop_words,ngram_range=(2, 2))
            # Ajustar y transformar los términos procesados
            count_data = count_vectorizer.fit_transform(df['Contenido0'])
            #Visualizar los términos mas comunes
            return plot_10_most_common_words(count_data, count_vectorizer)



    def palabras_freq(self,sector):
        df=self.df
        df.reset_index(inplace=True, drop=True)
        lista_stop_words=self.lista_sw
        lista=[(r'árboles','árbol')]

        def s_correction(text, patterns):
            patterns1 = [(re.compile(regex), repl) for (regex, repl) in   patterns]
            s = text
            for (pattern, repl) in patterns1:
                   (s, count) = re.subn(pattern, repl, s)
            return s

        def plot_10_most_common_words(count_data, count_vectorizer):
            #Grafico de frecuencias
            words = count_vectorizer.get_feature_names()
            total_counts = np.zeros(len(words))
            for t in count_data:
                total_counts+=t.toarray()[0]

            count_dict = (zip(words, total_counts))
            count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:30]
            words = [w[0] for w in count_dict]
            counts = [w[1] for w in count_dict]
            x_pos = np.arange(len(words))

            fig = px.bar( x=words, y=counts, color_discrete_sequence=["lightseagreen","gold","yellowgreen"], title='TÉRMINOS MÁS COMUNES EN:'+' '+segmento,  labels=dict(x="Palabra Clave", y="Frecuencia", color="Place"))

       
            st.plotly_chart(fig)

        if sector=='ambiental':
            df=df[df.sector=='ambiental']

            lista=self.n


            df['Contenido0']=df['Contenido0'].apply(lambda x: s_correction(x, lista))

            segmento = str.upper("EL SECTOR {}".format(sector))
            # Iniciar el count vectorizer con stop words personalizado en español
            count_vectorizer = CountVectorizer(stop_words=['personas','solicito','llega','hablar','nuevamente','envie','diciendo','llamada','envio','aparece','necesito','respuesta','informacion','entiendo','solicitud','problema','pregunta', 'haciendo']+lista_stop_words)
            # Ajustar y transformar los términos procesados
            count_data = count_vectorizer.fit_transform(df_personal['Contenido0'])
            #Visualizar los términos mas comunes
            plot_10_most_common_words(count_data, count_vectorizer)



        if sector=='salud':
            df=df[df.sector==sector]

            lista=self.n


            df['Contenido0']=df['Contenido0'].apply(lambda x: s_correction(x, lista))

            segmento = str.upper("EL SECTOR SALUD")
            # Iniciar el count vectorizer con stop words personalizado en español
            count_vectorizer = CountVectorizer(stop_words=['información','entrega','favorable','concepto','derecho','visita','poner','personas','solicito','llega','hablar','nuevamente','envie','diciendo','llamada','envio','aparece','necesito','respuesta','informacion','entiendo','solicitud','problema','pregunta', 'haciendo']+lista_stop_words)
            # Ajustar y transformar los términos procesados
            count_data = count_vectorizer.fit_transform(df['Contenido0'])
            #Visualizar los términos mas comunes
            plot_10_most_common_words(count_data, count_vectorizer)


        if sector=='educacion':
            df=df[df.sector==sector]

            lista=self.n


            df['Contenido0']=df['Contenido0'].apply(lambda x: s_correction(x, lista))

            segmento = str.upper("EL SECTOR EDUCACIÓN")
            # Iniciar el count vectorizer con stop words personalizado en español
            count_vectorizer = CountVectorizer(stop_words=['permita','opina','cesantias','definitiva','solicita','cancion','definitivas','información','entrega','favorable','concepto','derecho','visita','poner','personas','solicito','llega','hablar','nuevamente','envie','diciendo','llamada','envio','aparece','necesito','respuesta','informacion','entiendo','solicitud','problema','pregunta', 'haciendo']+lista_stop_words)
            # Ajustar y transformar los términos procesados
            count_data = count_vectorizer.fit_transform(df['Contenido0'])
            #Visualizar los términos mas comunes
            plot_10_most_common_words(count_data, count_vectorizer)


        if sector=='justicia y seguridad':
            df=df[df.sector==sector]

            lista=self.n

            df['Contenido0']=df['Contenido0'].apply(lambda x: s_correction(x, lista))

            segmento = str.upper("EL SECTOR JUSTICIA Y SEGURIDAD")
            # Iniciar el count vectorizer con stop words personalizado en español
            count_vectorizer = CountVectorizer(stop_words=['evento','meses','solicita','trasto','solicito','fecha','horario','comprendo','cancion','permita','medio','poner','visita','sonidos','presente','agradecemos','dedicado','dejarlo','entiendo','pregunta','desempena','quincena','permita','opina','cesantias','definitiva','solicita','cancion','definitivas','información','entrega','favorable','concepto','derecho','visita','poner','personas','solicito','llega','hablar','nuevamente','envie','diciendo','llamada','envio','aparece','necesito','respuesta','informacion','entiendo','solicitud','problema','pregunta', 'haciendo']+lista_stop_words)
            # Ajustar y transformar los términos procesados
            count_data = count_vectorizer.fit_transform(df['Contenido0'])
            #Visualizar los términos mas comunes
            plot_10_most_common_words(count_data, count_vectorizer)


    def lda_corpus(self, sector):

        df=self.df
        df.reset_index(inplace=True, drop=True)
        lista_stop_words=self.lista_sw
        def corpus(sector, dff):
            df=dff
            df=df[df.sector==sector]
            nlp = spacy.load('es_core_news_md')

            def sent_to_words(sentences):
                for sentence in sentences:
                    yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True remueve puctuacion

                dw = list(sent_to_words(df[['Contenido']]['Contenido']))

            def remove_stopwords(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in swseg] for doc in texts]

            def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]

            def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]

            def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
                texts_out = []
                for sent in texts:
                    doc = nlp(" ".join(sent))
                    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
                return texts_out

            #CREACION DEL CORPUS Y FACTORIZACION DE TERMINOS
            #Analizando los mensajes del usuario en general
            data=df['Contenido'].tolist()

            data_words = list(sent_to_words(data))# Se crea una lista de listas

            #BRIGRAMAS
            # Build the bigram and trigram models
            bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # Umbral superior de pocas frases.
            trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

            bigram_mod = gensim.models.phrases.Phraser(gensim.models.Phrases(dw, min_count=5, threshold=100))
            trigram_mod = gensim.models.phrases.Phraser(gensim.models.Phrases(bigram[dw], threshold=100))

            def remove_stopwords(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in lista_stop_words] for doc in texts]

            def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]

            def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]

            def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
                texts_out = []
                for sent in texts:
                    doc = nlp(" ".join(sent))
                    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
                return texts_out

                #Definicion de segmento a analizar

                data_words_bigrams = make_bigrams(remove_stopwords(dw))
                data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
                id2word = corpora.Dictionary(data_lemmatized)
                texts = data_lemmatized
                corpus = [id2word.doc2bow(text) for text in texts]
                return id2word, texts, corpus

        parameters=corpus(sector,df)
        return parameters

    def lda_corpus(self, sector):

        df=self.df
        df.reset_index(inplace=True, drop=True)
        lista_stop_words=self.lista_sw
    
        def corpus(sector, dff):
          df=dff
          df=df[df.sector==sector]
          nlp = spacy.load('es_core_news_md')
    
          def sent_to_words(sentences):
              for sentence in sentences:
                  yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True remueve puctuacion
          
          dw = list(sent_to_words(df[['Contenido']]['Contenido']))
          
          def remove_stopwords(texts):
              return [[word for word in simple_preprocess(str(doc)) if word not in swseg] for doc in texts]
    
          def make_bigrams(texts):
              return [bigram_mod[doc] for doc in texts]
    
          def make_trigrams(texts):
              return [trigram_mod[bigram_mod[doc]] for doc in texts]
    
          def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
              texts_out = []
              for sent in texts:
                  doc = nlp(" ".join(sent)) 
                  texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
              return texts_out
    
          #CREACION DEL CORPUS Y FACTORIZACION DE TERMINOS
          #Analizando los mensajes del usuario en general
          data=df['Contenido'].tolist()
    
          data_words = list(sent_to_words(data))# Se crea una lista de listas
    
          #BRIGRAMAS
          # Build the bigram and trigram models
          bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # Umbral superior de pocas frases.
          trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    
          bigram_mod = gensim.models.phrases.Phraser(gensim.models.Phrases(dw, min_count=5, threshold=100))
          trigram_mod = gensim.models.phrases.Phraser(gensim.models.Phrases(bigram[dw], threshold=100))
    
          def remove_stopwords(texts):
              return [[word for word in simple_preprocess(str(doc)) if word not in lista_stop_words] for doc in texts]
    
          def make_bigrams(texts):
              return [bigram_mod[doc] for doc in texts]
    
          def make_trigrams(texts):
              return [trigram_mod[bigram_mod[doc]] for doc in texts]
    
          def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
              texts_out = []
              for sent in texts:
                  doc = nlp(" ".join(sent)) 
                  texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
              return texts_out
    
          #Definicion de segmento a analizar
          
          data_words_bigrams = make_bigrams(remove_stopwords(dw))
          data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
          id2word = corpora.Dictionary(data_lemmatized)
          texts = data_lemmatized
          corpus = [id2word.doc2bow(text) for text in texts]
          return id2word, texts, corpus
    
        parameters=corpus(sector,df)
        return parameters
    
   
    def lda_model(self, sector):

        parameters=self.lda_corpus(sector)
        id2word= parameters[0]
        texts=   parameters[1]
        corpus=  parameters[2]
        lista_stop_words=self.lista_sw
        
        #num-topics,alpha, beta
        p_model_ambiente=[3,0.61,0.909999999999999]
        p_model_salud=[3, 0.9099999999999999, 0.9099999999999999]
        p_model_educacion=[4, 0.9099999999999999, 0.31]
        p_model_justicia=[4, 0.61, 0.61]

        def lda_model1(n_topic, a, b):
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=n_topic,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b,
                                           per_word_topics=True)
            return lda_model
            
        if sector=='ambiental':
            lda_model=lda_model1(p_model_ambiente[0],p_model_ambiente[1], p_model_ambiente[2])
            #Visualizacion de los tópicos generados en el modelo LDA
            pprint(lda_model.print_topics())
            doc_lda = lda_model[corpus]

            #Prerrequisitos minimos para la representacion grafica del modelado de topicos
            d = id2word
            c = corpus
            lda = lda_model

            #Parametros de visualizacion
            data = pyLDAvis.gensim.prepare(lda, c, d)
            string_html = pyLDAvis.prepared_data_to_html(data)
                                    
            st.write("Explicación de como entender este grafico")
            components.v1.html(string_html, width=800, height=700, scrolling=True)


            cols = [color for name, color in mcolors.XKCD_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

            cloud = WordCloud(stopwords=lista_stop_words,
                          background_color='white',
                          width=800,
                          height=700,
                          max_words=15,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)

            topics = lda_model.show_topics(formatted=False)

            fig, axes = plt.subplots(1, 3, figsize=(10,10), sharex=True, sharey=True)

            for i, ax in enumerate(axes.flatten()):
                fig.add_subplot(ax)
                topic_words = dict(topics[i][1])
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=15))
                plt.gca().axis('off')


            plt.subplots_adjust(wspace=0, hspace=0)
            plt.axis('off')
            plt.margins(x=0, y=0)
            plt.tight_layout()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write("Topicos de xxkfoejfoewjfowe")
            st.pyplot()

        if sector=='salud':
            lda_model=lda_model1(p_model_salud[0],p_model_salud[1], p_model_salud[2])
            #Visualizacion de los tópicos generados en el modelo LDA
            pprint(lda_model.print_topics())
            doc_lda = lda_model[corpus]

            #Prerrequisitos minimos para la representacion grafica del modelado de topicos
            d = id2word
            c = corpus
            lda = lda_model

            #Parametros de visualizacion
            data = pyLDAvis.gensim.prepare(lda, c, d)
            string_html = pyLDAvis.prepared_data_to_html(data)
                                    
            st.write("Explicación de como entender este grafico")
            components.v1.html(string_html, width=800, height=700, scrolling=True)


            cols = [color for name, color in mcolors.XKCD_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

            cloud = WordCloud(stopwords=lista_stop_words,
                          background_color='white',
                          width=800,
                          height=700,
                          max_words=15,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)

            topics = lda_model.show_topics(formatted=False)

            fig, axes = plt.subplots(1, 3, figsize=(10,10), sharex=True, sharey=True)

            for i, ax in enumerate(axes.flatten()):
                fig.add_subplot(ax)
                topic_words = dict(topics[i][1])
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=15))
                plt.gca().axis('off')


            plt.subplots_adjust(wspace=0, hspace=0)
            plt.axis('off')
            plt.margins(x=0, y=0)
            plt.tight_layout()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write("Topicos de xxkfoejfoewjfowe")
            st.pyplot()

        if sector=='educacion':
            lda_model=lda_model1(p_model_educacion[0],p_model_educacion[1], p_model_educacion[2])
            #Visualizacion de los tópicos generados en el modelo LDA
            pprint(lda_model.print_topics())
            doc_lda = lda_model[corpus]

            #Prerrequisitos minimos para la representacion grafica del modelado de topicos
            d = id2word
            c = corpus
            lda = lda_model

            #Parametros de visualizacion
            data = pyLDAvis.gensim.prepare(lda, c, d)
            string_html = pyLDAvis.prepared_data_to_html(data)
                                    
            st.write("Explicación de como entender este grafico")
            components.v1.html(string_html, width=800, height=700, scrolling=True)


            cols = [color for name, color in mcolors.XKCD_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

            cloud = WordCloud(stopwords=lista_stop_words,
                          background_color='white',
                          width=800,
                          height=700,
                          max_words=15,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)

            topics = lda_model.show_topics(formatted=False)

            fig, axes = plt.subplots(1, 4, figsize=(10,10), sharex=True, sharey=True)

            for i, ax in enumerate(axes.flatten()):
                fig.add_subplot(ax)
                topic_words = dict(topics[i][1])
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=15))
                plt.gca().axis('off')


            plt.subplots_adjust(wspace=0, hspace=0)
            plt.axis('off')
            plt.margins(x=0, y=0)
            plt.tight_layout()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write("Topicos de xxkfoejfoewjfowe")
            st.pyplot()

        if sector=='justicia y seguridad':
            lda_model=lda_model1(p_model_justicia[0],p_model_justicia[1], p_model_justicia[2])
            #Visualizacion de los tópicos generados en el modelo LDA
            pprint(lda_model.print_topics())
            doc_lda = lda_model[corpus]

            #Prerrequisitos minimos para la representacion grafica del modelado de topicos
            d = id2word
            c = corpus
            lda = lda_model

            #Parametros de visualizacion
            data = pyLDAvis.gensim.prepare(lda, c, d)
            string_html = pyLDAvis.prepared_data_to_html(data)
                                    
            st.write("Explicación de como entender este grafico")
            components.v1.html(string_html, width=800, height=700, scrolling=True)


            cols = [color for name, color in mcolors.XKCD_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

            cloud = WordCloud(stopwords=lista_stop_words,
                          background_color='white',
                          width=800,
                          height=700,
                          max_words=15,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)

            topics = lda_model.show_topics(formatted=False)

            fig, axes = plt.subplots(1, 4, figsize=(10,10), sharex=True, sharey=True)

            for i, ax in enumerate(axes.flatten()):
                fig.add_subplot(ax)
                topic_words = dict(topics[i][1])
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=15))
                plt.gca().axis('off')


            plt.subplots_adjust(wspace=0, hspace=0)
            plt.axis('off')
            plt.margins(x=0, y=0)
            plt.tight_layout()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write("Topicos de xxkfoejfoewjfowe")
            st.plotly_chart(fig)


    def experiencia_user(self, sector):

        df=self.df
        df=df[df.sector==sector]

        def polaridad(x):
            return np.nan if str(x) == '' else clf.predict(x)
        df['polaridad_usuario'] = df['texto_completo'].apply(polaridad)

        #Asignacion del sentimiento
        df['sentimiento_cliente'] = df['polaridad_usuario'].apply(lambda x:'Negativos' if x<=0.4 else('Positivos' if x>=0.65 else 'Neutrales'))
        names = 'Negativos', 'Positivos', 'Neutrales'

        b = len(df[df.sentimiento_cliente=='Positivos'])
        a = len(df[df.sentimiento_cliente=='Negativos'])
        c = len(df[df.sentimiento_cliente=='Neutrales'])

        sizes=[a, b, c]

        df0=pd.DataFrame()
        df0['experiencia']=[ 'Negativos', 'Positivos', 'Neutrales']
        df0['cantidad']=sizes
        
        fig, ax = plt.subplots(figsize=(12, 5))
        fig = px.pie(df0, values='cantidad', names='experiencia',hole=0.5,title='ANÁLISIS DE LA PERCEPCIÓN DEL USUARIO EN EL SECTOR:'+' '+str.upper(sector),color_discrete_sequence=["lightseagreen","palegoldenrod","darkseagreen"])
        
        st.plotly_chart(fig)
        
