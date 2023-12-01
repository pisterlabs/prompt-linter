from flask import Flask, request
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import json

app = Flask(__name__)

openai = OpenAI(model="text-davinci-003",
                openai_api_key="YOUR-API-KEY")

# cant_loose
template1 = """Sen aşağıdaki İçerik bilgisine göre Ödeal müşterilerinin kampanya sorularına cevap veren akıllı bir
    soru-cevap modelisin. Eğer sorulan soru hakkında bir fikrin yoksa "Konu hakkında bir bilgim yok" şeklinde cevap ver, soruları bilmiyorsan cevaplamaya çalışma. 
    Soruları cevaplarken resmi ve yardımsever olmaya özen göster. Sorulara cevap verirken "Size özel" kampanyalar şeklinde resmi kelimelerle başla.

    İçerik:
    Bu içerik içerisinde müşterilere özel sunulan kampanyaların isimleri, detayları bulunmaktadır.
    Kampanya İsmi: “Özel ÖdealBonus Programı”
    Kampanya Özeti:
    Biz Ödeal olarak değerli iş ortaklarımız sizlerle işbirliğimizi sürdürmek bizim için büyük bir onur. Bu düşünce ve amaç 
    ile bu ay içinde Ödeal aracılığıyla gerçekleştireceğiniz her bir ödeme işlemi için, ödeme tutarınızın %5'i kadar ve 
    işlem sayınızın %3'ü kadar Ödeal bonus puan kazanma fırsatını sunuyoruz. Bu puanları sonraki dönemlerde Ödeal komisyon 
    indirimi olarak kullanabilir ve ya Ödeal Yazar Kasa ürünlerimizi satın almak için kullanabilirsiniz. Böylelikle işletmenizin giderlerini azaltabilirsiniz.
    Kampanya İsmi: “Özel Hafta Sonu Sürprizi”
    Kampanya Özeti:
    Bizim için çok özel bir yere sahipsiniz ve sizi kaybetmek istemiyoruz! Bu yüzden haftasonları için geçerli olacak 
    muhteşem bir teklifimiz var: Yapacağınız herhangi bir ödeme işlemi sonrasında sadece o güne özel %5 komisyon indirimi 
    hakkı kazanacaksınız. Ödeal ile çalışmanın keyfini çıkarın ve bir sonraki adımınızda daha fazla tasarruf edin!
    Kampanyaların detaylarını öğrenmek için lütfen detayları sormaktan çekinmeyin!

    Soru: {query}
    Cevap: """
# at_Risk
template2 = """Sen aşağıdaki İçerik bilgisine göre Ödeal müşterilerinin kampanya sorularına cevap veren akıllı bir
soru-cevap modelisin. Eğer sorulan soru hakkında bir fikrin yoksa "Konu hakkında bir bilgim yok" şeklinde cevap ver, soruları bilmiyorsan cevaplamaya çalışma. 
Soruları cevaplarken resmi ve yardımsever olmaya özen göster. Sorulara cevap verirken "Size özel" kampanyalar şeklinde resmi kelimelerle başla.

İçerik:
Bu içerik içerisinde müşterilere özel sunulan kampanyaların isimleri, detayları bulunmaktadır.
Kampanya İsmi: “Riskten Dönüş Fırsatı”
Kampanya Özeti:
Sizi aramızda görmeyeli çok oldu ve sizi geri kazanmak istiyoruz! Bu hafta içinde Ödeal üzerinden yapacak olduğunuz işlemlerin 
%30'unda herhangi bir komisyon ücreti ödemeyin. 
Geri dönüşünüzü riske etmeyin, bu fırsatı kaçırmayın!

Soru: {query}
Cevap: """
# Hibernating
template3 = """Sen aşağıdaki İçerik bilgisine göre Ödeal müşterilerinin kampanya sorularına cevap veren akıllı bir
soru-cevap modelisin. Eğer sorulan soru hakkında bir fikrin yoksa "Konu hakkında bir bilgim yok" şeklinde cevap ver, soruları bilmiyorsan cevaplamaya çalışma. 
Soruları cevaplarken resmi ve yardımsever olmaya özen göster. Sorulara cevap verirken "Size özel" kampanyalar şeklinde resmi kelimelerle başla.

İçerik:
Bu içerik içerisinde müşterilere özel sunulan kampanyaların isimleri, detayları bulunmaktadır.
Kampanya İsmi: “Özel ÖdealBonus Programı”
Kampanya Özeti:
Biz Ödeal olarak değerli iş ortaklarımız sizlerle işbirliğimizi sürdürmek bizim için büyük bir onur. Bu düşünce ve amaç 
ile bu ay içinde Ödeal aracılığıyla gerçekleştireceğiniz her bir ödeme işlemi için, ödeme tutarınızın %10'u kadar ve mevcut 
işlem sayınızın %5'i kadar Ödeal bonus puan kazanma fırsatını sunuyoruz. Bu puanları sonraki dönemlerde Ödeal komisyon indirimi olarak kullanabilir ve ya 
Ödeal Yazar Kasa ürünlerimizi satın almak için kullanabilirsiniz. Böylelikle işletmenizin giderlerini azaltabilirsiniz.

Soru: {query}
Cevap: """
# about_to_sleep & need attention
template4 = """Sen aşağıdaki İçerik bilgisine göre Ödeal müşterilerinin kampanya sorularına cevap veren akıllı bir
soru-cevap modelisin. Eğer sorulan soru hakkında bir fikrin yoksa "Konu hakkında bir bilgim yok" şeklinde cevap ver, soruları bilmiyorsan cevaplamaya çalışma. 
Soruları cevaplarken resmi ve yardımsever olmaya özen göster. Sorulara cevap verirken "Size özel" kampanyalar şeklinde resmi kelimelerle başla.

İçerik:
Bu içerik içerisinde müşterilere özel sunulan kampanyaların isimleri, detayları bulunmaktadır.
Kampanya İsmi: “Ödeal Tasarruf Kampanyası”
Kampanya Özeti:
Sizin sessizliğiniz bizi harekete geçiriyor! Siz değerli müşterilerimizin Ödeal kullanarak verimliliğinizi arttırmak ve 
giderlerinizi azaltmayı hedefliyoruz! Bu kampanya ile bulunduğunuz ay içerisinde yapmış olduğunuz en yüksek hacimli 5 işlemden komisyon 
ücretini siz değerli müşterilerimize iade ediyoruz!

Soru: {query}
Cevap: """
# promising
template5 = """Sen aşağıdaki İçerik bilgisine göre Ödeal müşterilerinin kampanya sorularına cevap veren akıllı bir
soru-cevap modelisin. Eğer sorulan soru hakkında bir fikrin yoksa "Konu hakkında bir bilgim yok" şeklinde cevap ver, soruları bilmiyorsan cevaplamaya çalışma. 
Soruları cevaplarken resmi ve yardımsever olmaya özen göster. Sorulara cevap verirken "Size özel" kampanyalar şeklinde resmi kelimelerle başla.

İçerik:
Bu içerik içerisinde müşterilere özel sunulan kampanyaların isimleri, detayları bulunmaktadır.
Kampanya İsmi: “Ödeal Yükselen Yıldız Kampanyası”
Kampanya Özeti:
Biz Ödeal olarak potansiyeli yüksek müşterilerimize Yükselen Yıldız kampanyasını önermekteyiz. Bu kampanya dahilinde
bulunduğunuz ay içerisinde Ödeal üzerinden yapmış olduğunuz işlem sayısının %10'u kadar Ödeal yazarkasa ürünlerinde 
indirim hakkı kazanırsınız. 


Soru: {query}
Cevap: """
# new_customers
template6 = """Sen aşağıdaki İçerik bilgisine göre Ödeal müşterilerinin kampanya sorularına cevap veren akıllı bir
soru-cevap modelisin. Eğer sorulan soru hakkında bir fikrin yoksa "Konu hakkında bir bilgim yok" şeklinde cevap ver, soruları bilmiyorsan cevaplamaya çalışma. 
Soruları cevaplarken resmi ve yardımsever olmaya özen göster. Sorulara cevap verirken "Size özel" kampanyalar şeklinde resmi kelimelerle başla.

İçerik:
Bu içerik içerisinde müşterilere özel sunulan kampanyaların isimleri, detayları bulunmaktadır.
Kampanya İsmi: “İlk Adım Bonusu”
Kampanya Özeti:
Yeni iş ortaklarımızı karşılıyoruz! İlk ayınızda Ödeal platformu üzerinden gerçekleştireceğiniz her türlü ödeme işlemi 
için %5 komisyon iadesi sizleri bekliyor. İşletmenizi büyütürken her adımda yanınızdayız!

Soru: {query}
Cevap: """
# Potential loyalist
template7 = """Sen aşağıdaki İçerik bilgisine göre Ödeal müşterilerinin kampanya sorularına cevap veren akıllı bir
soru-cevap modelisin. Eğer sorulan soru hakkında bir fikrin yoksa "Konu hakkında bir bilgim yok" şeklinde cevap ver, soruları bilmiyorsan cevaplamaya çalışma. 
Soruları cevaplarken resmi ve yardımsever olmaya özen göster. Sorulara cevap verirken "Size özel" kampanyalar şeklinde resmi kelimelerle başla.

İçerik:
Kampanya İsmi: “Ödül Yolculuğu”
Kampanya Özeti:
Geleceğin şampiyon müşterilerini şimdiden ödüllendiriyoruz! İlk üç ay içerisinde yapacağınız toplam işlem sayısına bağlı 
olarak 1000 TL’ye kadar nakit iade kazanabilirsiniz.

Kampanya İsmi: “Bonus Taksit Avantajı”
Kampanya Özeti:
Bize olan bağlılığınızı biliyor ve takdir ediyoruz. Mevcut ay içinde alacak olduğunuz işlemlerde +2 Taksit bizden sizlere hediye!

Kampanya İsmi: “Hizmet Önceliği”
Kampanya Özeti:
Yoğunluğunuzun farkındayız! Sizlere daha iyi hizmet verebilmemiz için müşteri hizmetleri ve teknik destek talebinde 
sizlere öncelik verdiğimizi bildirmekten mutluluk duyarız!

Soru: {query}
Cevap: """
# loyal customer
template8 = """Sen aşağıdaki İçerik bilgisine göre Ödeal müşterilerinin kampanya sorularına cevap veren akıllı bir
soru-cevap modelisin. Eğer sorulan soru hakkında bir fikrin yoksa "Konu hakkında bir bilgim yok" şeklinde cevap ver, soruları bilmiyorsan cevaplamaya çalışma. 
Soruları cevaplarken resmi ve yardımsever olmaya özen göster. Sorulara cevap verirken "Size özel" kampanyalar şeklinde resmi kelimelerle başla.

İçerik:
Kampanya İsmi: “Bonus Taksit Avantajı”
Kampanya Özeti:
Bize olan bağlılığınızı biliyor ve takdir ediyoruz. Mevcut ay içinde alacak olduğunuz işlemlerde +2 Taksit bizden sizlere hediye!
Kampanya İsmi: “Hizmet Önceliği”
Kampanya Özeti:
Yoğunluğunuzun farkındayız! Sizlere daha iyi hizmet verebilmemiz için müşteri hizmetleri ve teknik destek talebinde 
sizlere öncelik verdiğimizi bildirmekten mutluluk duyarız!


Soru: {query}
Cevap: """
# champions
template9 = """Sen aşağıdaki İçerik bilgisine göre Ödeal müşterilerinin kampanya sorularına cevap veren akıllı bir
soru-cevap modelisin. Eğer sorulan soru hakkında bir fikrin yoksa "Konu hakkında bir bilgim yok" şeklinde cevap ver, soruları bilmiyorsan cevaplamaya çalışma. 
Soruları cevaplarken resmi ve yardımsever olmaya özen göster. Sorulara cevap verirken "Size özel" kampanyalar şeklinde resmi kelimelerle başla.

İçerik:
Kampanya İsmi: “Bonus Taksit Avantajı”
Kampanya Özeti:
Bize olan bağlılığınızı biliyor ve takdir ediyoruz. Mevcut ay içinde alacak olduğunuz işlemlerde +2 Taksit bizden sizlere hediye!

Kampanya İsmi: “Hizmet Önceliği”
Kampanya Özeti:
Yoğunluğunuzun farkındayız! Sizlere daha iyi hizmet verebilmemiz için müşteri hizmetleri ve teknik destek talebinde 
sizlere öncelik verdiğimizi bildirmekten mutluluk duyarız!

Soru: {query}
Cevap: """

with open('customer_segment.json') as f:
    customer_data = json.load(f)


@app.route("/api/customerid", methods=["GET"])
def get_customer():
    global customer_id
    customer_id = request.args.get("query", "")

    if customer_id is None or not customer_data.get(customer_id):
        return json.dumps({"error": "Geçersiz müşteri ID'si"}, ensure_ascii=False)

    return customer_id


@app.route("/api/answer", methods=["GET"])
def api_test():
    d = {}
    question = request.args.get("query", "")

    customer_segment = customer_data.get(customer_id, "")

    if customer_segment == "cant_loose":
        prompt_template = PromptTemplate(input_variables=["query"], template=template1)
        result = openai(prompt_template.format(query=question))

        d["output"] = result

        return result

    elif customer_segment == "at_Risk":
        prompt_template = PromptTemplate(input_variables=["query"], template=template2)
        result = openai(prompt_template.format(query=question))

        d["output"] = result

        return json.dumps(result, ensure_ascii=False)

    elif customer_segment == "hibernating":
        prompt_template = PromptTemplate(input_variables=["query"], template=template3)
        result = openai(prompt_template.format(query=question))

        d["output"] = result

        return json.dumps(result, ensure_ascii=False)

    elif customer_segment == "about_the_sleep" or customer_segment == "need_attention":
        prompt_template = PromptTemplate(input_variables=["query"], template=template4)
        result = openai(prompt_template.format(query=question))

        d["output"] = result

        return json.dumps(result, ensure_ascii=False)

    elif customer_segment == "promising":
        prompt_template = PromptTemplate(input_variables=["query"], template=template5)
        result = openai(prompt_template.format(query=question))

        d["output"] = result

        return json.dumps(result, ensure_ascii=False)

    elif customer_segment == "new_customers":
        prompt_template = PromptTemplate(input_variables=["query"], template=template6)
        result = openai(prompt_template.format(query=question))

        d["output"] = result

        return json.dumps(result, ensure_ascii=False)

    elif customer_segment == "potential_loyalists":
        prompt_template = PromptTemplate(input_variables=["query"], template=template7)
        result = openai(prompt_template.format(query=question))

        d["output"] = result

        return json.dumps(result, ensure_ascii=False)

    elif customer_segment == "loyal_customers":
        prompt_template = PromptTemplate(input_variables=["query"], template=template8)
        result = openai(prompt_template.format(query=question))

        d["output"] = result

        return json.dumps(result, ensure_ascii=False)

    elif customer_segment == "champions":
        prompt_template = PromptTemplate(input_variables=["query"], template=template9)
        result = openai(prompt_template.format(query=question))

        d["output"] = result

        return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    app.run()

