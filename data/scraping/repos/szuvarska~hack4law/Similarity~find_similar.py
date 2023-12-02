#coding=UTF-8
import pprint
import pandas as pd

from pathlib import Path

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document


# Potrzebne pakiety to: 'langchain[chromadb], sentence-transformer i chromadb


def return_df_with_similarities(query:str, tags = [], data_path = 'Data/output_for_frontend_2.csv' ):

    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    chroma_db_filepath = Path("./chroma_db")

    df = pd.read_csv(data_path, sep = ";")

    if chroma_db_filepath.exists():
        db = Chroma(embedding_function=embeddings, persist_directory=str(chroma_db_filepath))
    else:
        documents = df.apply(lambda row: Document(page_content = row['uzasadnienie']
                                                , metadata = {'source' : row['id']}), axis=1)

        text_splitter = CharacterTextSplitter(chunk_size=3000 , chunk_overlap=300)
        docs = text_splitter.split_documents(documents)
        db = Chroma.from_documents(docs, embeddings, persist_directory=str(chroma_db_filepath))

    sim = db.similarity_search_with_score(
        query, k=len(df)
        
    )

    results = [(score, doc.metadata["source"], doc.page_content) for (doc, score) in sim]
    results.sort(key=lambda x: x[0])

    # pprint.pprint(results)
    
    df_results = pd.DataFrame(results, columns = ['similarity', 'id', 'uzasadnienie'])
    df_results['id'] = df_results['id'].astype('str')
    df['id'] = df['id'].astype('str')
    df = df[[col for col in df.columns if col!="uzasadnienie"]]
    merged = df_results.merge(df, on='id', how='inner')

    condition = merged['tags'].apply(lambda row_tags: all(tag in row_tags for tag in tags))
    filtered_merged = merged.loc[condition]
    filtered_merged = filtered_merged.sort_values(by=['similarity'], ascending=True).iloc[:10, :]
    return filtered_merged


if __name__ == '__main__':
    # query = 'powód company SA B wnosić stwierdzić nieważność uchwała zgromadzenie Wspólników spółka ograniczyć odpowiedzialność B podjąć 23122002 rok moc zmienić umowa spółka dodać § 11 nowy uregulowanie usta 4 5 przewidujący umorzyć udział wspólnik powziąć uchwała zgromadzenie Wspólnik ogłoszeć upadłość wynagrodzenie stanowiącym równowartość wartość księgowy udział podstawa powództwo wskazywać przepis artykuł 252§1 zw artykuł 250 pkt4 ksh k3 nadto podnosić legitymacja wyprowadzić możnaba artykuł 189 k 41 Pozwan sp oo B wnosić oddalić powództwo sąd okręgowy – sąd gospodarczy Białystok wyrokie dzień 05122003 rok ustalić uchwała zgromadzenie wspólnik Spółka ograniczyć odpowiedzialność B podjąć dzień 23122002 rok moc § 11 umowa spółka dodać ustęp 4 5 przewidywać umorzyć udział wspólnik powziąć uchwała zgromadzenie Wspólnik ogłosić upadłość wynagrodzenie stanowiącym równowartość wartość księgowy udział nieważny wpis ostateczny ustalić kwota 250 złoty obciążyć pozwany obowiązek uiszczeć wpis rzecz skarb państwo sąd ustalić umowa dzień 3 lipiec 2001 rok zawiązać zostać spółka ograniczyć odpowiedzialność B duży udziałowiec zostać COMPANY SA B obejmować udział 2960 wartość 1 480 00000 złoty § 11 umowa ustalić udział spółka móc umarzać podstawa uchwała zgromadzenie Wspólnik podjętej większość 23 głos usta 1 umorzeć udział móc nastąpić czysty zysk obniżyć kapitał zakładowy umorzenie równomierć wszystek udział decydować zgromadzenie wspólnik umorzyć udział nierównomierny niezbędny zgoda wszystek wspólnik usta 2 przypadek umorzyć udział wspólniek przysługiwać zwrot równowartość umorzonych udział wartość bilansowy dzień określić uchwała umorzenie termin miesiąc dzień podjąć uchwała chyba uchwała określić termin dzień 23122002 rok odbyć Nadzwyczajne walny zgromadzenie Wspólników spółka Zgromadzenie reprezentować kapitał zakładowy porządek obrady przewidywać minuta dyskusja przedmiot podwyższyć kapitał zakładowy spółka zmiana umowa spółka podjąć przedmiot uchwać tok obrady głosowanie jawć jednomyślnie brak sprzeciw podjąć uchwała moc § 11 umowa spółka dodać zostać usta 4 5 pozwalać umorzyć udział wspólnik powziąć uchwała zgromadzenie Wspólnik ogłoszeć upadłość wynagrodzenie stanowiącym równowartość wartość księgowy udział dzień 16012002 rok company SA B złożyć podanie otworzyć postępowanie układowy dzień 21022003 rok ogłoszenie upadłość dzień 28042003 rok sąd ogłosić upadłość spółka ustaloć stan faktyczny sąd okręgowy wskazać rozwiązanie przewidziać przepis artykuł 199§4 ksh mieć cel przeciwdziałać niepożądany sytuacja naruszać uzasadniony interes niektóry wspólnik zwłaszcza mieć spółka wystarczająco mocny pozycja prawny faktyczny Rozwiązanie zapewnić wspólnik możliwość wycofać spółka znaczniejszy strata wskazać umorzyć udział ustawodawca wyszczególnić rodzaj zdarzenie objąć zostać dyspozycja artykuł 199§4 ksh określić zdarzenie móc objąć zwany automatyczny umorzenie ocena sąd okręgowy oznaczać zainteresować móc dokonać zmiana umowa spółka przyjąć zdarzenie prawny fakt ogłosić upadłość skutkuć umorzeniem udział skutka zdarzenie odmienny regulować obowiązywać przepis prawny istota uchwała wspólnik móc zmierzać pokrzywdzeć wierzyciel wspólnik pozwany spółka wobec mieć zostać ogłoszić upadłość zdanie sąd syndyk należeć krąg podmiot mowa artykuł 250 ksha ustawodawca dać legitymacja wytoczeć powództwo stwierdzić nieważność uchwała wspólnik sprzeczć ustawa podstawa artykuł 252 ksh Wskazywać roszczenie sprowadzać żądać stwierdzeć nieważność uchwała sprzeczny przepis prawo sąd związać sformułować konkluzja pozwu kwalifikacja prawny dochodzoć żądać zmieniać wyrok brzmieć konkluzja pozwu przyjmować kwalifikacja prawny dochodzony żądać ramy zgłoszony podstawa faktyczny powództwo naruszać przepis artykuł 321§1 sąd okręgowy uznać potrzebny rozważenie roszczenie powód móc znaleźć oparcie przepis artykuł 189 ramy podstawa wskazać chwila ogłosić upadłość upaść tracić rzecz syndyk prawo zarząd rozporządzać swój majątek Syndyk działać rzecz masa upadłość interes rzecz wszystek wierzyciel przepis artykuł 67 prawo upadłościow odbierać poszczególny wierzyciel prawo wytoczeć powództwo przyznawać syndykowy nadto przepis artykuł 56 prawo upadłościowego cel zaskarżyć czynność prawny upadłego zdziałany szkoda wierzyciel dopuszczać odpowiedni stosować przepis kodeks postępowanie cywilny mieć umożliwić syndyk odpowiedni skorzystać klasyczny instrument prawo cywilny cel zapewnić bezpośredni ochrona masa upadłość sąd wskazać doktryna orzecznictwo dopuszczać powództwo ustalić nieważność uchwała wypadek bezwzględny nieważność podstawa artykuł 189 kpca zasada ogólny zdanie sąd powód – syndyk zaskarżać uchwała interes wszystek wierzyciel mieć interes prawny żądanie ustalenie sporny uchwała nieważna naruszać uregulowanie zawrzeć artykuł 20 90 prawo upadłościow mieć cel obejść syndyk wyłącznie uprawniyć dysponować udziałami posiadać upaść pozwany spółka Umorzenie automatyczny udział upadłego spółka ogłoszeć upadłość powodować wygaśniąć uczestnictwo spółka równowartość odpowiadać wartość księgowy odbiegać istotny wartość wolnorynkowy ocena sąd argument przemawiać przyjąć sporny uchwała sprzeczny powołany wysoko przepis prawo mieć cel obejśt ustawa Godzić interes przyszły wierzyciel upadłego wspólnik wyłączać uprawnienie syndyka dysponować przedmiotowy udziałami ogłoszeć upadłość wyłącnie uprawniyć ocena sąd syndyk móc dochodzić ochrona podstawa artykuł 189 niezależnie przysługujący roszczenie przykład droga skarga pauliański tryb artykuł 67 prawo upadłościowego wzmocnić argumentacja sąd okręgowy powołać przepis artykuł 83 84 ustawa dzień 28022003 rok prawo upadłościowy naprawcze DzU numer 60 poz 535 wskazywać nieważny postanowienie umowa zastrzegać wypadek ogłosić upadłość zmiana rozwiązać stosunek prawny strona upadły ogłoszeć upadłość zmiana wygaśniąć stosunek prawny strona upaść możliwy tylkko przepis ustawa czynność prawny dokonać naruszeć ustawa bezskuteczny wobec masa upadłość umowa strona przewidywać skutek Powołując zatem podstawa swój rozstrzygnięci artykuł 189 zw artykuł 58 kc sąd powództwo uwzględnić Apelacja wyrok wnieść pozwany spółka Zarzucając naruszyć przepis artykuł 321§1 przepis prawo materialny tj artykuł 252§1 zdanie drugi ksh poprzez zastosowanie rozpoznawać sprawa artykuł 189 artykuł 536 prawo upadłościowy naprawczy zw artykuł 620§1 ksh wnosić zmiana zaskarżyć wyrok oddaleć powództwo zasądzić koszt proces oba instancja'
    query = "rozwod"
    print(return_df_with_similarities(query))