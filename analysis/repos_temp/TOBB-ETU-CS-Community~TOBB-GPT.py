"""Bu görevde yapman gereken bu şey, kullanıcı sorularını arama sorgularına dönüştürmektir. Bir kullanıcı
     soru sorduğunda, soruyu, kullanıcının bilmek istediği bilgileri getirecek bir Google arama sorgusuna dönüştürmelisin. Soru bir fiil
     içeriyorsa bu fiili kaldırarak onu bir isime dönüştürmen gerekiyor. Eğer soru türkçe ise türkçe, ingilizce ise ingilizce
     bir cevap üret ve cevabı json formatında döndür. Json formatı şöyle olmalı:
     {"query": output}
     """f"""Dönüştürmen gereken soru, tek tırnak işaretleri arasındadır:
     '{question}'
     Verdiğin cevap da yalnızca arama sorgusu yer almalı, başka herhangi bir şey yazmamalı ve tırnak işareti gibi
     bir noktalama işareti de eklememelisin. Sonucu json formatında dönmelisin.
     Json formatı şöyle olmalı:
     {{"query": output}}"""