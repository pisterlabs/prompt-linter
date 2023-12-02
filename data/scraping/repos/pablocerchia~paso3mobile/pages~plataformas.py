import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

def propuestas():

  import base64
  from dotenv import load_dotenv
  from PyPDF2 import PdfReader
  from langchain.text_splitter import CharacterTextSplitter
  from langchain.embeddings.openai import OpenAIEmbeddings
  from langchain.vectorstores import FAISS
  from langchain.chains.question_answering import load_qa_chain
  from langchain.llms import OpenAI
  from langchain.callbacks import get_openai_callback
  import os
  from fpdf import FPDF
  from PIL import Image
  st.markdown("""<style>.css-zt5igj svg{display:none}</style>""", unsafe_allow_html=True)


  cvmilei1 = Image.open('data/cvmilei.jpg')
  cvmilei = cvmilei1.resize((650, 900))
  cvmassa1 = Image.open('data/cvmassa.jpg')
  cvmassa = cvmassa1.resize((650, 900))
  cvpato1 = Image.open('data/cvpato.jpg')
  cvpato = cvpato1.resize((650, 900))
  cvschia1 = Image.open('data/cvschia.jpg')
  cvschia = cvschia1.resize((650, 900))
  cvbregman1 = Image.open('data/cvbregman.jpg')
  cvbregman = cvbregman1.resize((650, 900))

  # st.set_page_config(page_title = 'Elecciones 2023 - Sitio de consulta',
  #                     layout='wide', initial_sidebar_state='collapsed')
  # st.markdown(
  #         """
  #       <style>
  #       [data-testid="stSidebar"][aria-expanded="true"]{
  #           min-width: 200px;
  #           max-width: 200px;
  #       }
  #       """,
  #         unsafe_allow_html=True,
  #     )
  # st.markdown("""<style>.css-zt5igj svg{display:none}</style>""", unsafe_allow_html=True)

  #st.title('Plataformas electorales nacionales de las agrupaciones politicas')

  # with open("data/LLA.pdf","rb") as f:
  #     base64_pdf = base64.b64encode(f.read()).decode('utf-8')

  # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="650" height="1000" type="application/pdf"></iframe>'

  # with open("data/JxC.pdf","rb") as f:
  #     base64_pdf2 = base64.b64encode(f.read()).decode('utf-8')

  # pdf_display2 = f'<iframe src="data:application/pdf;base64,{base64_pdf2}" width="650" height="1000" type="application/pdf"></iframe>'

  # with open("data/UxP.pdf","rb") as f:
  #     base64_pdf3 = base64.b64encode(f.read()).decode('utf-8')

  # pdf_display3 = f'<iframe src="data:application/pdf;base64,{base64_pdf3}" width="650" height="1000" type="application/pdf"></iframe>'


  with open("data/LLA.pdf", "rb") as pdf_file:
      PDF_milei = pdf_file.read()

  with open("data/JxC.pdf", "rb") as pdf_file:
      PDF_JxC = pdf_file.read()

  with open("data/UxP.pdf", "rb") as pdf_file:
      PDF_UxP = pdf_file.read()

  with open("data/FIT.pdf", "rb") as pdf_file:
      PDF_FIT = pdf_file.read()

  
  c1, c2, c3 = st.columns([0.1,0.8,0.1])

  milei_fmi_1 = "https://www.infobae.com/politica/2023/09/14/javier-milei-hablo-de-su-reunion-con-el-fmi-fui-muy-claro-la-deuda-se-paga/"
  milei_fmi_2 = "https://www.infobae.com/economia/2023/08/18/javier-milei-le-prometio-al-fmi-que-si-es-presidente-va-pagar-la-deuda-y-que-no-habra-default-con-los-acreedores-privados/"
  milei_fmi_3 = "https://www.cronista.com/economia-politica/javier-milei-aseguro-que-hara-con-el-fmi-en-caso-de-llegar-a-la-presidencia/"
  milei_laboral = "https://www.infobae.com/politica/2023/09/13/debemos-adaptarnos-a-los-nuevos-tiempos-laborales-los-detalles-de-la-reunion-entre-javier-milei-y-luis-barrionuevo/"
  milei_agro = "https://www.perfil.com/noticias/politica/de-que-se-trata-la-reforma-agropecuaria-que-propone-javier-milei.phtml"
  milei_edu1 = "https://www.cronista.com/economia-politica/como-funciona-el-sistema-de-vouchers-similar-al-que-propone-javier-milei/"
  milei_edu_2 = "https://www.lanacion.com.ar/sociedad/es-viable-el-sistema-de-vouchers-la-opinion-de-los-expertos-sobre-la-propuesta-clave-del-plan-nid23082023/"
  milei_edu_3 = "https://derechadiario.com.ar/economia/sistema-de-vouchers-para-la-educacion-asi-funciona-la-propuesta-de-milei-para-terminar-con-la-corrupcion-y-promover-la-competencia"
  milei_edu_4 = "https://cenital.com/por-que-ni-milei-ni-nadie-podra-implementar-los-vouchers-educativos-11-puntos-para-entenderlo/"
  milei_genero = "https://corta.com/genero-el-papa-y-cambio-climatico-las-definiciones-de-milei-con-tucker-carlson/"
  milei_seguridad = "https://www.lanacion.com.ar/politica/el-plan-de-javier-milei-sobre-seguridad-bajar-la-edad-de-imputabilidad-y-frenar-los-piquetes-nid22082023/"
  milei_salud1 = "https://www.a24.com/politica/javier-milei-exclusivo-a24-como-es-el-proyecto-educacion-y-salud-publica-que-piensa-implementar-n1194671"
  milei_salud2 = "https://www.lanacion.com.ar/sociedad/el-plan-sanitario-de-javier-milei-es-posible-subsidiar-a-los-pacientes-no-a-los-hospitales-y-la-nid20082023/"
  milei_ddhh1 = "https://www.instagram.com/p/CxK4DhGuepC/?img_index=1"
  milei_ddhh2 = "https://www.perfil.com/noticias/politica/que-piensa-victoria-villarruel-del-terrorismo-de-estado-y-como-lo-uso-como-plataforma-politica.phtml"
  milei_malvinas = "https://www.perfil.com/noticias/actualidad/milei-entre-el-shabat-en-miami-y-el-encuentro-con-luis-barrionuevo.phtml"
  milei_fantino = "https://www.youtube.com/watch?v=5Z8JRRIhRAo"

  with c2:
    st.markdown("<h3 style='text-align: center;'>PROPUESTAS DE LOS CANDIDATOS PRESIDENCIALES<br></h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; font-weight: normal;'>Conocé qué es lo que propone cada candidato sobre los temas más importantes. A esto se le suma una sección donde se recopilan distintas notas en los medios que dieron en las últimas semanas.<br> A su vez podrás consultar el curriculum de cada uno y acceder a la plataforma electoral que cada candidato propone, así como también a sus redes sociales. <br><br></h5>", unsafe_allow_html=True)

      
    st.markdown("<h2 style='text-align: center;'>Javier Milei (La Libertad Avanza)</h2>", unsafe_allow_html=True)
    with st.expander('📈 **Economía**'):
          st.markdown('<div style="text-align: justify;"><br> <b>Para concretar la reforma integral que se propone se necesitarán 35 años, divididos en tres etapas sucesivas</b>.<br><br> <b>Primera etapa:</b> fuerte recorte del gasto público del Estado y una <b>reforma tributaria que empuje una baja de los impuestos</b>, la <b>flexibilización laboral</b> para la creación de empleos en el sector privado y una <b>apertura unilateral al comercio internacional</b>. Ello acompañado por una reforma financiera impulse una banca libre y desregulada junto a la libre competencia de divisas. <br><br> <b>Segunda etapa:</b> se propone una <b>reforma previsional para recortar el gasto del estado en jubilaciones y pensiones</b> de los ítems que más empujan el déficit fiscal alentando un sistema de capitalización privado, junto a un programa de retiros voluntarios de empleados públicos y achicamiento del estado. <br> Por otro lado se propone <b>reducir el número de ministerios a 8.</b>  En esta etapa <b>comenzarán a eliminarse de forma progresiva los planes sociales</b> a medida que se generen otros ingresos como consecuencia de la creación de puestos de trabajos en el sector privado, liquidación del Banco Central de la República Argentina, estableciendo un sistema de banca Simons, con encajes al 100% para depósitos a la vista. <br><br> <b>Tercera etapa:</b> reforma profunda del sistema de salud con impulso del sistema privado, competitividad libre entre empresas del sector, una reforma del sistema educativo y la ampliación de un sistema de seguridad no invasivo para la población y la eliminación de la coparticipación.<br></div>', unsafe_allow_html=True)
          #st.write("<br> **Para concretar la reforma integral que se propone se necesitarán 35 años, divididos en tres etapas sucesivas**.<br><br> **Primera etapa:** fuerte recorte del gasto público del Estado y una **reforma tributaria que empuje una baja de los impuestos**, la **flexibilización laboral** para la creación de empleos en el sector privado y una **apertura unilateral al comercio internacional**. Ello acompañado por una reforma financiera impulse una banca libre y desregulada junto a la libre competencia de divisas. <br><br> **Segunda etapa:** se propone una **reforma previsional para recortar el gasto del estado en jubilaciones y pensiones** de los ítems que más empujan el déficit fiscal alentando un sistema de capitalización privado, junto a un programa de retiros voluntarios de empleados públicos y achicamiento del estado. <br> Por otro lado se propone **reducir el número de ministerios a 8.**  En esta etapa **comenzarán a eliminarse de forma progresiva los planes sociales** a medida que se generen otros ingresos como consecuencia de la creación de puestos de trabajos en el sector privado, liquidación del Banco Central de la República Argentina, estableciendo un sistema de banca Simons, con encajes al 100% para depósitos a la vista. <br><br> **Tercera etapa:** reforma profunda del sistema de salud con impulso del sistema privado, competitividad libre entre empresas del sector, una reforma del sistema educativo y la ampliación de un sistema de seguridad no invasivo para la población y la eliminación de la coparticipación.", unsafe_allow_html=True)
    with st.expander('⚕️ **Salud**'):
      st.markdown('<div style="text-align: justify;"><br> <b>Los ministerios de Salud, Desarrollo Social, Trabajo y Educación de la Nación serían condensados en una única cartera llamada Capital Humano.</b> <br><br> Para LLA, <b>el mejor sistema de salud posible es uno privado donde cada argentino pague sus servicios</b>. Reformar profundamente el sistema de salud con impulso del sistema privado, competitividad libre entre empresas del sector y mejorar la estructura edilicia hospitalaria.<br><br>  <b>Arancelar todas las prestaciones</b> y auto gestionar el servicio de salud el servicio de salud en trabajos compartidos con la salud privada.<br><br>  <b>Proteger al niño desde la concepción:</b> Están en contra de la Ley de Interrupción Voluntaria del Embarazo y planean hacer un plebiscito para ver si mantener o eliminar la ley.<br><br>  <b>Modificar la Ley de Salud Mental</b> y desarrollar programas de prevención para los trastornos adictivos, educativos y de la personalidad.<br><br> <b>Obras sociales:</b> se buscará a nivel nacional liberar la cautividad de los afiliados a los seguros sociales de salud (obras sociales nacionales y PAMI, entre otras) para que la gente elija libremente, estableciendo una libre competencia. Por ejemplo, un jubilado podría elegir cualquier otra obra social que no sea el PAMI.<br><br> <b>Regular la documentación de extranjeros que trabajen en salud</b> y exigir a los turistas extranjeros que cuenten con un seguro de salud.<br></div>', unsafe_allow_html=True)
      #st.write("<br> **Los ministerios de Salud, Desarrollo Social, Trabajo y Educación de la Nación serían condensados en una única cartera llamada Capital Humano.** <br><br> Para LLA, **el mejor sistema de salud posible es un sistema de salud privado donde cada argentino pague sus servicios**. Reformar profundamente el sistema de salud con impulso del sistema privado, competitividad libre entre empresas del sector y mejorar la estructura edilicia hospitalaria.<br><br>  **Arancelar todas las prestaciones** y auto gestionar el servicio de salud el servicio de salud en trabajos compartidos con la salud privada.<br><br>  **Proteger al niño desde la concepción:** Están en contra de la Ley de Interrupción Voluntaria del Embarazo y planean hacer un plebiscito para ver si mantener o eliminar la ley.<br><br>  **Modificar la Ley de Salud Mental** y desarrollar programas de prevención para los trastornos adictivos, educativos y de la personalidad.<br><br> **Obras sociales:** se buscará a nivel nacional liberar la 'cautividad' de los afiliados a los seguros sociales de salud (obras sociales nacionales y PAMI, entre otras) para que la gente elija libremente, estableciendo una libre competencia. Por ejemplo, un jubilado podría elegir cualquier otra obra social que no sea el PAMI.<br><br> **Regular la documentación de extranjeros que trabajen en salud** y exigir a los turistas extranjeros que cuenten con un seguro de salud. ", unsafe_allow_html=True)
    with st.expander('📖 **Educación**'):
      st.markdown('<div style="text-align: justify;"><br> <b>Sistema de vouchers.</b> Descentralizar la educación entregando el presupuesto a los padres, en lugar de dárselo al ministerio, financiando la demanda.<br><br> <b>Eliminar la obligatoriedad de la ESI</b> en todos los niveles de enseñanza. <br><br> Generar competencia entre las instituciones educativas desde lo curricular en todos los niveles de educación, incorporando más horas de materias como matemática, lengua, ciencias, tic o por la orientación y/o infraestructura.<br><br>  <b>Promover una transformación curricular donde se promueva un enfoque pedagógico por habilidades</b>, que va más allá de la simple transmisión del conocimiento. Aplicando modificaciones que orienten a los estudiantes a las profesiones necesarias para el país (ingenieros, informáticos).<br><br>  Crear la carrera docente de nivel universitario y la carrera de directivos y supervisores.<br></div>', unsafe_allow_html=True)
      #st.write("<br> **Sistema de vouchers.** Descentralizar la educación entregando el presupuesto a los padres, en lugar de dárselo al ministerio, financiando la demanda.<br><br> **Eliminar la obligatoriedad de la ESI** en todos los niveles de enseñanza. <br><br> Generar competencia entre las instituciones educativas desde lo curricular en todos los niveles de educación, incorporando más horas de materias como matemática, lengua, ciencias, tic o por la orientación y/o infraestructura.<br><br>  **Promover una transformación curricular donde se promueva un enfoque pedagógico por habilidades**, que va más allá de la simple transmisión del conocimiento. Aplicando modificaciones que orienten a los estudiantes a las profesiones necesarias para el país (ingenieros, informáticos).<br><br>  Crear la carrera docente de nivel universitario y la carrera de directivos y supervisores.", unsafe_allow_html=True)
    with st.expander('♻️ **Ambiente**'):
      st.markdown('<div style="text-align: justify;"><br> <b>El cambio climático es parte de la agenda socialista.</b> En realidad el mundo ha tenido otros picos de altas temperaturas como ahora. Es un comportamiento cíclico independientemente del hombre. Si los precios fueran libres, esos problemas se arreglarían.<br><br> <b>Invertir en el mantenimiento del sistema energético actual y promover nuevas fuentes de energías renovables y limpias</b> (solar, eólica, hidrógeno verde, etc).<br><br> Fomentar la creación de centros de reciclaje de residuos para su transformación en energía y materiales reutilizables.<br><br> <b>Profundizar la investigación en energía nuclear</b> a fin de elaborar generadores nucleares de industria nacional para la generación de energía y exportación.<br><br> Promover una agricultura de buenas prácticas contemplando sustentabilidad del suelo y la preservación del medioambiente.<br><br> Reformular el sistema de Emergencia Agropecuaria y <b>cuidar nuestro patrimonio marítimo evitando el aprovechamiento indiscriminado e ilegal.</b><br></div>', unsafe_allow_html=True)
      #st.write("<br> **El cambio climático es parte de la agenda socialista.** En realidad el mundo ha tenido otros picos de altas temperaturas como ahora. Es un comportamiento cíclico independientemente del hombre. Si los precios fueran libres, esos problemas se arreglarían.<br><br> **Invertir en el mantenimiento del sistema energético actual y promover nuevas fuentes de energías renovables y limpias** (solar, eólica, hidrógeno verde, etc).<br><br> Fomentar la creación de centros de reciclaje de residuos para su transformación en energía y materiales reutilizables.<br><br> **Profundizar la investigación en energía nuclear** a fin de elaborar generadores nucleares de industria nacional para la generación de energía y exportación.<br><br> Promover una agricultura de buenas prácticas contemplando sustentabilidad del suelo y la preservación del medioambiente.<br><br> Reformular el sistema de Emergencia Agropecuaria y **cuidar nuestro patrimonio marítimo evitando el aprovechamiento indiscriminado e ilegal.**", unsafe_allow_html=True)
    with st.expander('👮‍♂️ **Seguridad**'):
      st.markdown('<div style="text-align: justify;"><br> <b>Ley para reglamentar el derecho a manifestarse.</b> El derecho a manifestarse está reconocido por la Constitución. Pero cortar rutas, incendiar gomas, impedir la circulación y llevar niños a una manifestación para usarlos como escudo no puede ocurrir.<br><br> A pesar de que Milei se pronunció en favor de “la libre portación de armas” y luego dijo que este asunto no está incluido en su plataforma, Villarruel indicó que no hay intención de modificar la ley actual sobre armamento de la población civil.<br><br> Construir establecimientos penitenciarios con sistema de gestión público-privada y eliminar los salarios de los reclusos a través de una modificación de la legislación.<br><br> Prestar especial <b>atención a la lucha contra el narcotráfico</b>, atacando cada una de las células y organizaciones delictivas, controlando límites provinciales y espacio aéreo con radares y personal calificado, dotando a su personal de conocimiento, herramientas de trabajo, de protección junto con el manejo de y aplicación de las nuevas tecnologías.<br><br> Disminuir la dificultad del accionar policial a través de la modificación de las leyes y la presentación de nuevos proyectos de ley y sanear todas las fuerzas de seguridad, haciendo eje en la lucha contra la corrupción.<br><br> Prohibir el ingreso al país de extranjeros con antecedentes penales y deportar inmediatamente a extranjeros que cometan delitos en el país.<br></div>', unsafe_allow_html=True)
      #st.write("<br> **Ley para reglamentar el derecho a manifestarse.** El derecho a manifestarse está reconocido por la Constitución. Pero cortar rutas, incendiar gomas, impedir la circulación y llevar niños a una manifestación para usarlos como escudo no puede ocurrir.<br><br> A pesar de que Milei se pronunció en favor de “la libre portación de armas” y luego dijo que este asunto no está incluido en su plataforma, Villarruel indicó que no hay intención de modificar la ley actual sobre armamento de la población civil.<br><br> Construir establecimientos penitenciarios con sistema de gestión público-privada y eliminar los salarios de los reclusos a través de una modificación de la legislación.<br><br> Prestar especial **atención a la lucha contra el narcotráfico**, atacando cada una de las células y organizaciones delictivas, controlando límites provinciales y espacio aéreo con radares y personal calificado, dotando a su personal de conocimiento, herramientas de trabajo, de protección junto con el manejo de y aplicación de las nuevas tecnologías.<br><br> Disminuir la dificultad del accionar policial a través de la modificación de las leyes y la presentación de nuevos proyectos de ley y sanear todas las fuerzas de seguridad, haciendo eje en la lucha contra la corrupción.<br><br> Prohibir el ingreso al país de extranjeros con antecedentes penales y deportar inmediatamente a extranjeros que cometan delitos en el país.", unsafe_allow_html=True)
    with st.expander('📰 **Notas en los medios**'):
      #st.markdown("<h3 style='text-align: center;'>Es importante tener en cuenta que los medios hegemónicos no son apartidarios y pueden presentar visiones parciales sobre los temas tratados.</h3>", unsafe_allow_html=True)
      st.write("<br> **FMI:** <br> [Milei sobre la deuda del FMI (Infobae)](%s)" % milei_fmi_1, unsafe_allow_html=True)
      st.write("[Milei: 'Nosotros no vamos a defaultear ni al FMI ni la deuda soberana' (Infobae)](%s)" % milei_fmi_2, unsafe_allow_html=True)
      st.write("[Su plan de austeridad: ajuste más fuerte que el que propone el FMI (El Cronista)](%s)" % milei_fmi_3, unsafe_allow_html=True)
      st.write("<br> **Derechos Humanos/Negacionismo:** <br> [¿Por qué se dice que Victoria Villaruel (candidata a vicepresidenta por LLA) es 'negacionista' de la última dictadura? (Klio y la Turba)](%s)" % milei_ddhh1, unsafe_allow_html=True)
      st.write("[Qué piensa Victoria Villarruel del terrorismo de estado (Perfil)](%s)" % milei_ddhh2, unsafe_allow_html=True)
      st.write("<br> **Reforma laboral:** <br> [Reunión con Barrionuevo: “Debemos adaptarnos a los nuevos tiempos laborales” (Infobae)](%s)" % milei_laboral, unsafe_allow_html=True)
      st.write("<br> **Agro:** <br> [Detalles de su reforma agraria (Perfil)](%s)" % milei_agro, unsafe_allow_html=True)
      st.write("<br> **Educación (sistema de vouchers):** <br> [Descripción del sistema de vouchers (El Cronista)](%s)" % milei_edu1, unsafe_allow_html=True)
      st.write("[Opinión de los 'expertos' (La Nación)](%s)" % milei_edu_2, unsafe_allow_html=True)
      #st.write("[Opinión a favor: por qué sería viable (La Derecha Diario)](%s)" % milei_edu_3, unsafe_allow_html=True)
      st.write("[Opinión en contra: por qué no sería viable (Cenital)](%s)" % milei_edu_4, unsafe_allow_html=True)
      st.write("<br> **Género y ambiente:** <br> [Entrevista con Tucker Carlson - Sus definiciones sobre cuestiones de género y el cambio climático (Corta)](%s)" % milei_genero, unsafe_allow_html=True)
      st.write("<br> **Seguridad:** <br> [Detalles de su doctrina de seguridad (La Nación)](%s)" % milei_seguridad, unsafe_allow_html=True)
      st.write("<br> **Salud:** <br> [Entrevista - Su plan para la salud pública (A24)](%s)" % milei_salud1, unsafe_allow_html=True)
      st.write("[Viabilidad de su plan sanitario' (La Nación)](%s)" % milei_salud2, unsafe_allow_html=True)
      st.write("<br> **Malvinas:** <br> [Milei avaló los dichos de Diana Mondino sobre las islas Malvinas y los derechos de los isleños (Perfil)](%s)" % milei_malvinas, unsafe_allow_html=True)
      st.write("<br> **Entrevista con Fantino:** <br> [Entrevista con Alejandro Fantino (Neura Media)](%s)" % milei_fantino, unsafe_allow_html=True)
    with st.expander('📄 **Currículum Vitae**'):
      c144, c244, c44 = st.columns([0.2,0.6,0.2])
      with c244:
        st.image(cvmilei)  
    with st.expander('🔍 **Si querés saber más sobre sus propuestas...**'):
      st.write("<br> **Sitio oficial:** https://lalibertadavanza.com.ar/<br><br> **Youtube:** https://www.youtube.com/@JavierMileiOK/videos<br><br> **Descargá la plataforma electoral del partido:**", unsafe_allow_html=True)
      st.download_button(label="Descargar PDF",
                        data=PDF_milei,
                        file_name="LLA.pdf",
                        mime='application/octet-stream')
      
              #url = "https://cenital.com/paso-una-eleccion-de-cuatro-cuartos/"
              #st.write("[link](%s)" % url)
              #st.markdown('<div style="text-align: justify;"></div>', unsafe_allow_html=True)

    massa_medidas1= "https://www.perfil.com/noticias/politica/massa-si-tienen-que-ahorrar-compren-un-autito-no-me-vayan-a-comprar-dolares.phtml"
    massa_medidas2 = "https://www.perfil.com/noticias/economia/sergio-massa-anuncio-que-sube-el-piso-de-ganancias-a-1500000-a-partir-de-octubre.phtml"
    massa_medidas3 = "https://www.telam.com.ar/notas/202309/640236-massa-medidas.html"
    massa_medidas4 = "https://www.pagina12.com.ar/587317-massa-anuncio-un-nuevo-proyecto-de-ley-de-financiamiento-edu"
    massa_agro1 = "https://www.infocampo.com.ar/una-por-una-cuales-son-las-nuevas-medidas-de-sergio-massa-para-el-agro/"
    massa_malvinas = "https://www.cronista.com/economia-politica/el-canciller-de-massa-cruzo-fuerte-a-mondino-por-sus-dichos-sobre-malvinas-que-le-dijo/"
    massa_malvinas2 = "https://www.lanacion.com.ar/politica/malvinas-los-referentes-de-politica-exterior-de-patricia-bullrich-y-sergio-massa-salieron-al-cruce-nid11092023/"
    massa_ypf = 'https://www.ambito.com/politica/sergio-massa-critico-el-fallo-estadounidense-contra-argentina-ypf-viola-y-vulnera-nuestra-soberania-n5816884'
    massa_propuestas = "https://cnnespanol.cnn.com/2023/08/11/propuestas-sergio-massa-elecciones-argentina-orix-arg/"
    massa_fmi = "https://www.perfil.com/noticias/economia/sergio-massa-agosto-fue-el-peor-mes-de-los-ultimos-25-anos-de-la-economia-argentina.phtml"
    massa_fmi2 = "https://www.cronista.com/economia-politica/massa-ventilo-mas-entretelones-de-la-negociacion-con-el-fmi-y-conto-anecdotas-picantes/"
    massa_fmi3 = "https://www.clarin.com/economia/fmi-massa-ganancias-deficit_0_0LcdjwkPqq.html"
    massa_fmi4 = "https://www.letrap.com.ar/economia/la-tijera-sergio-massa-que-piensa-recortar-alcanzar-el-deficit-cero-n5403093"
    massa_fmi5 = "https://www.infobae.com/economia/2023/09/13/massa-responsabilizo-al-fmi-por-la-inflacion-de-agosto/"
    massa_fmi6 = "https://www.perfil.com/noticias/politica/sergio-massa-le-vamos-a-pagar-al-fmi-para-que-se-vaya-de-argentina-y-nos-dejen-decidir-de-manera-soberana.phtml"
    massa_seguridad = "https://www.diarioelzondasj.com.ar/296767-exclusivo-el-plan-seguridad-inteligente-de-massa"
    massa_vivienda = "https://www.tiempodesanjuan.com/economia/el-gobierno-abrio-un-registro-familias-que-necesiten-lotes-viviendas-y-agricultura-n358368"
    massa_fantino = "https://www.youtube.com/watch?v=n22g0O_q2A8"

    st.markdown("<h2 style='text-align: center;'>Sergio Massa (Unión por la Patria)</h2>", unsafe_allow_html=True)
    with st.expander('📈 **Economía**'):
          st.markdown('<div style="text-align: justify;"><br> <b>Sostener el nivel de inversión pública</b> para poder seguir haciendo obras de desarrollo. <br><br> Tener <b>equilibrio fiscal y superávit comercial</b> con acumulación de reservas y con recuperación de ingresos. En el 2024 la inflación no será el principal problema de la economía Argentina.<br><br> <b>La solución más grande que tiene Argentina es vender lo que hace al mundo.</b> Que cada vez se venda más al mundo porque <b>eso va a dar los dólares para ser libres, para ser soberanos</b>. Aumentar los volúmenes de exportación con simplificación de impuestos para las PyMes exportadoras para juntar dólares para <b>cumplir el acuerdo con el FMI, pero defendiendo los intereses de Argentina.</b> <br><br> <b>Fomentar la energía y el sector minero como activo estratégico</b> para el desarrollo. Planificar la matriz energética argentina para hacer un cambio para <b>que Argentina deje de ser importador para ser exportador.</b><br></div>', unsafe_allow_html=True)
          #st.write("<br> **Sostener el nivel de inversión pública** para poder seguir haciendo obras de desarrollo. <br><br> Tener **equilibrio fiscal y superávit comercial** con acumulación de reservas y con recuperación de ingresos. En el 2024 la inflación no será el principal problema de la economía Argentina.<br><br> **La solución más grande que tiene Argentina es vender lo que hace al mundo.** Que cada vez se venda más al mundo porque **eso va a dar los dólares para ser libres, para ser soberanos**. Aumentar los volúmenes de exportación con simplificación de impuestos para las PyMes exportadoras para juntar dólares para **cumplir el acuerdo con el FMI, pero defendiendo los intereses de Argentina.** <br><br> **Fomentar la energía y el sector minero como activo estratégico** para el desarrollo. Planificar la matriz energética argentina para hacer un cambio para **que la argentina deje de ser importador para ser exportador.**", unsafe_allow_html=True)
    with st.expander('⚕️ **Salud**'):
      st.markdown('<div style="text-align: justify;"><br> Tener un <b>sistema de salud público de calidad bien equipado</b>, considerando que no todos los argentinos tienen acceso a una prepaga.<br><br> Sostener el sistema de medicamentos gratuitos para jubilados.<br></div>', unsafe_allow_html=True)
      #st.write("<br> Tener un **sistema de salud público de calidad bien equipado**, considerando que no todos los argentinos tienen acceso a una prepaga.<br><br> Sostener el sistema de medicamentos gratuitos para jubilados.", unsafe_allow_html=True)
    with st.expander('📖 **Educación**'):
      st.markdown('<div style="text-align: justify;"><br> <b>La formación terciaria y universitaria es lo que garantiza el progreso.</b> Mejorar la educación pública, la inversión en universidades y crear más universidades. Lograr una universidad pública gratuita de calidad inclusiva.<br><br> <b>Incorporar la capacitación en tecnología</b> y herramientas relacionadas al “nuevo mercado de trabajo”. Cuarto y quinto año tienen que tener programación y robótica obligatoriamente.<br></div>', unsafe_allow_html=True)
      #st.write("<br> **La formación terciaria y universitaria es lo que garantiza el progreso.** Mejorar la educación pública, la inversión en universidades y crear más universidades. Lograr una universidad pública gratuita de calidad inclusiva.<br><br> **Incorporar la capacitación en tecnología** y herramientas relacionadas al “nuevo mercado de trabajo”. Cuarto y quinto año tienen que tener programación y robótica obligatoriamente.", unsafe_allow_html=True)
    with st.expander('♻️ **Ambiente**'):
      st.markdown('<div style="text-align: justify;"><br> <b>Desarrollo y ambiente pueden convivir y crecer.</b> Es necesario establecer los parámetros que de alguna manera que no aparezcan como contradictorios sino complementarios. <br><br> Desarrollo de una <b>política ambiental de adaptación y mitigación al cambio climático</b> <br><br> Se busca acompañar el <b>desarrollo de la minería sostenible</b>, como puntal del desarrollo regional y nacional preservando el cuidado del ambiente<br><br> <b>Residuos sólidos urbanos:</b> erradicación de basurales a cielo abierto y creación de 10 centros ambientales en el país. <br><br> <b>Pérdida de biodiversidad y deforestación:</b>  frenar la deforestación ilegal de bosques nativos y aumentar progresivamente el financiamiento previsto en la Ley 26.331 de Presupuestos Mínimos. <br><br> <b>Cambio climático:</b> cumplir con el Plan Nacional de Adaptación y Mitigación al Cambio Climático, exigir a los países centrales cumplir con el compromiso de financiar en US$ 100 mil millones a los países en desarrollo y el <b>canje de deuda por naturaleza</b>; esto es, transacciones voluntarias en las que un acreedor cancela o reduce la deuda de un gobierno a cambio de que este tome compromisos ambientales.<br></div>', unsafe_allow_html=True)
      #st.write(" <br> **Desarrollo y ambiente pueden convivir y crecer.** Es necesario establecer los parámetros que de alguna manera que no aparezcan como contradictorios sino complementarios. <br><br> Desarrollo de una **política ambiental de adaptación y mitigación al cambio climático** <br><br> Se busca acompañar el **desarrollo de la minería sostenible**, como puntal del desarrollo regional y nacional preservando el cuidado del ambiente<br><br> **Residuos sólidos urbanos:** erradicación de basurales a cielo abierto y creación de 10 centros ambientales en el país. <br><br> **Pérdida de biodiversidad y deforestación:**  frenar la deforestación ilegal de bosques nativos y aumentar progresivamente el financiamiento previsto en la Ley 26.331 de Presupuestos Mínimos. <br><br> **Cambio climático:** cumplir con el Plan Nacional de Adaptación y Mitigación al Cambio Climático, exigir a los países centrales cumplir con el compromiso de financiar en US$ 100 mil millones a los países en desarrollo y el **canje de deuda por naturaleza**; esto es, transacciones voluntarias en las que un acreedor cancela o reduce la deuda de un gobierno a cambio de que este tome compromisos ambientales.", unsafe_allow_html=True)
    with st.expander('👮‍♂️ **Seguridad**'):
      st.markdown('<div style="text-align: justify;"><br> <b>Acompañar la inversión en seguridad de varias provincias</b>, para seguir consolidando la calidad de vida de la gente.<br><br> Coordinar fuerzas de seguridad nacional con Unidad Información Financiera, Banco Central, para no solamente ocuparse de la persecución de quienes comercializan drogas, sino también seguir el dinero de grandes distribuidores.<br><br> Poner un <b>acento más fuerte en la lucha contra la inseguridad y el narcotráfico</b>. El Gran Buenos Aires y Rosario son la prioridad pero también la frontera. Fortalecer el interior y pelear en la frontera.<br><br> <b>Agregar sistemas de videovigilancia</b> porque la prevención en la lucha contra la seguridad es fundamental.<br></div>', unsafe_allow_html=True)
      #st.write("<br> **Acompañar la inversión en seguridad de varias provincias**, para seguir consolidando la calidad de vida de la gente.<br><br> Coordinar fuerzas de seguridad nacional con Unidad Información Financiera, Banco Central, para no solamente ocuparse de la persecución de quienes comercializan drogas, sino también seguir el dinero de grandes distribuidores.<br><br> Poner un **acento más fuerte en la lucha contra la inseguridad y el narcotráfico**. El Gran Buenos Aires y Rosario son la prioridad pero también la frontera. Fortalecer el interior y pelear en la frontera.<br><br> **Agregar sistemas de videovigilancia** porque la prevención en la lucha contra la seguridad es fundamental.", unsafe_allow_html=True)
    with st.expander('📰 **Notas en los medios**'):
      st.write("<br> **Medidas:** <br> [Devolución del IVA, Ganancias, créditos, inversión educativa y quita de retenciones a las economías regionales (Telám)](%s)" % massa_medidas3, unsafe_allow_html=True)
      st.write("[Suba del piso de Ganancias (Perfil)](%s)" % massa_medidas2, unsafe_allow_html=True)
      st.write("[Massa: 'Si tienen que ahorrar compren un autito, no me vayan a comprar dólares' (Perfil)](%s)" % massa_medidas1, unsafe_allow_html=True)
      st.write("<br> **Vivienda:** <br> [Se abrió un registro para familias que necesiten lotes para viviendas y agricultura (El Tiempo de San Juan)](%s)" % massa_vivienda, unsafe_allow_html=True)
      st.write("<br> **Propuestas:** <br> [Resumen de sus propuestas (CNN)](%s)" % massa_propuestas, unsafe_allow_html=True)
      st.write("<br> **FMI/Inflación:** <br> [Massa: 'Agosto fue el peor mes de los últimos 25 años de la economía argentina' (Perfil)](%s)" % massa_fmi, unsafe_allow_html=True)
      st.write("[Massa responsabilizó al FMI por la inflación de agosto (Infobae)](%s)" % massa_fmi5, unsafe_allow_html=True)
      st.write("[Massa: “Le vamos a pagar al FMI para que se vaya de Argentina y nos dejen decidir” (Perfil)](%s)" % massa_fmi2, unsafe_allow_html=True)
      st.write("[Negociaciones con el FMI (El Cronista)](%s)" % massa_fmi2, unsafe_allow_html=True)
      st.write("[Advierten que las medidas de Massa van en contra de la aceleración del ajuste que pide el FMI (Clarín)](%s)" % massa_fmi3, unsafe_allow_html=True)
      st.write("[Presupuesto 2024 - La tijera de Sergio Massa: qué piensa recortar para alcanzar el déficit cero (Letra P)](%s)" % massa_fmi4, unsafe_allow_html=True)
      st.write("<br> **Educación:** <br> [Nuevo proyecto de Ley de Financiamiento Educativo (Página 12)](%s)" % massa_medidas4, unsafe_allow_html=True)
      st.write("<br> **Seguridad:** <br> [Su plan de seguridad en detalle (El Zonda)](%s)" % massa_seguridad, unsafe_allow_html=True)
      st.write("<br> **Agro:** <br> [Medidas anunciadas para el campo (InfoCampo)](%s)" % massa_agro1, unsafe_allow_html=True)
      st.write("<br> **Malvinas:** <br> [El 'canciller' de Massa le respondió a la candidata de Milei: 'Ellos piden que se respete a los kelpers, nosotros pedimos que se respete a los caídos'. (El Cronista)](%s)" % massa_malvinas, unsafe_allow_html=True)
      st.write("[Respuesta desde el gobierno a Diana Mondino (La Nación)](%s)" % massa_malvinas2, unsafe_allow_html=True)
      st.write("<br> **Fallo de YPF:** <br> [Massa criticó el fallo estadounidense contra Argentina por YPF: 'Viola y vulnera nuestra soberanía' (Ámbito)](%s)" % massa_ypf, unsafe_allow_html=True)
      st.write("<br> **Entrevista con Fantino:** <br> [Entrevista con Alejandro Fantino (Neura Media)](%s)" % massa_fantino, unsafe_allow_html=True)
    with st.expander('📄 **Currículum Vitae**'):
      c1445, c2445, c445 = st.columns([0.2,0.6,0.2])
      with c2445:
        st.image(cvmassa)
    with st.expander('🔍 **Si querés saber más sobre sus propuestas...**'):
      st.write("<br> **Sitio oficial:** https://porlapatria.org/ <br><br> **Youtube:** https://www.youtube.com/@porlapatria <br><br> **Descargá la plataforma electoral del partido:**", unsafe_allow_html=True)
      st.download_button(label="Descargar PDF",
                        data=PDF_UxP,
                        file_name="UxP.pdf",
                        mime='application/octet-stream')

    bullrich_fmi = "https://www.bloomberglinea.com/latinoamerica/argentina/fmi-asegura-que-nuevo-acuerdo-con-argentina-cuenta-con-el-apoyo-de-milei-y-bullrich/"
    bullrich_fmi2 = "https://www.cronista.com/economia-politica/carlos-melconian-se-reunio-con-el-fmi-para-presentar-las-propuestas-de-patricia-bullrich-de-cara-a-octubre/"
    bullrich_fmi3 = "https://www.baenegocios.com/economia/Dolarizacion-Ganancias-y-reforma-laboral-los-ejes-del-plan-economico-de-Patricia-Bullrich-20230906-0005.html"
    bullrich_fmi4 = "https://www.eldiarioar.com/economia/bullrich-propuesta-blindaje-serian-consecuencias-nuevo-prestamo-fmi_1_10418070.html"
    bullrich_prop = "https://chequeado.com/el-explicador/esto-propone-patricia-bullrich-sobre-politica-cambiaria-retenciones-planes-sociales-inseguridad-y-educacion-que-dicen-los-datos-y-los-especialistas/"
    bullrich_edu = "https://elintra.com.ar/2023/09/14/necesitamos-educacion-universitaria-gratuita-calificada-y-preparada-patricia-bullrich/"
    bullrich_edu2 = "https://www.ambito.com/politica/bullrich-hablo-contra-la-universidad-publica-y-el-gobierno-la-desmintio-n5758767"
    bullrich_agro = "https://news.agrofy.com.ar/noticia/206295/patricia-bullrich-reunio-su-equipo-economico-se-hablo-reduccion-acelerada-retenciones"
    bullrich_agro2 = "https://tn.com.ar/campo/2023/08/18/cuales-son-las-principales-propuestas-para-el-agro-de-los-5-candidatos-presidenciales/"
    bullrich_ypf = "https://www.letrap.com.ar/politica/el-fallo-ypf-ring-un-duro-cruce-patricia-bullrich-axel-kicillof-n5402979"
    bullrich_ypf2 = "https://www.clarin.com/politica/patricia-bullrich-fallo-ypf-kicillof-hizo-canchero-dijo-pagaba-pesos-ahora-cuesta-16-mil-millones-dolares_0_Yisbg8Z0ct.html"
    bimonetarismo = "https://www.ambito.com/finanzas/bimonetarismo-como-es-el-modelo-que-proponen-bullrich-y-melconian-n5807299"
    bullrich_fantino = "https://www.youtube.com/watch?v=4b-gyeFQw_M&t=6s"



    st.markdown("<h2 style='text-align: center;'>Patricia Bullrich (Juntos por el Cambio)</h2>", unsafe_allow_html=True)
    with st.expander('📈 **Economía**'):
          st.markdown('<div style="text-align: justify;"><br> <b>Ir a un bimonetarismo y sacar el cepo cambiario</b> que disminuye las reservas e impide exportar e importar en una economía libre que promueva las inversiones productivas. <b>Negociar blindaje con el FMI que evite una explosión de la economía.</b><br><br> <b>Bajar la inflación bajando la emisión monetaria</b>, con solidez fiscal, un Banco Central independiente, una reforma del Estado y laboral, cambios impositivos y un pacto federal.<br><br> <b>Convertir los planes sociales en seguros de desempleo y que en 4 años dejen de existir los planes sociales, reemplazandolos con empleo</b>. Fomentar la dispersión territorial hacia ciudades más amables con la generación de polos de desarrollo por fuera del Conurbano.<br> Crear un <b>Pacto Federal para reducir impuestos improductivos para economías regionales y discutir la distribución de los subsidios hacia la Capital Federal.</b><br><br> <b>Terminar con las restricciones a las importaciones</b> para evitar que retrase y estanque el progreso. Promover que el campo crezca libremente y aumente las exportaciones. <b>Promover las exportaciones de la pesca, la minería, la industria y todos los sectores.</b><br><br> <b>Achicar el Estado y congelar contrataciones. Pasar de 23 ministerios ineficientes a 8-10.</b> Reducir la burocracia estatal, el gasto en personal sin tareas específicas y mejorar salarios del personal esencial como policías, médicos y docentes. <br>Definir un plazo para que las empresas públicas presenten un plan de negocios con déficit cero. Si no, serán privatizadas o reconvertidas en cooperativas.<br></div>', unsafe_allow_html=True)
          #st.write("<br> Ir a un [bimonetarismo](%s) y **sacar el cepo cambiario** que disminuye las reservas e impide exportar e importar en una economía libre que promueva las inversiones productivas. **Negociar blindaje con el FMI que evite una explosión de la economía.**<br><br> **Bajar la inflación bajando la emisión monetaria**, con solidez fiscal, un Banco Central independiente, una reforma del Estado y laboral, cambios impositivos y un pacto federal.<br><br> **Convertir los planes sociales en seguros de desempleo y que en 4 años dejen de existir los planes sociales, reemplazandolos con empleo**. Fomentar la dispersión territorial hacia ciudades más amables con la generación de polos de desarrollo por fuera del Conurbano.<br> Crear un **Pacto Federal para reducir impuestos improductivos para economías regionales y discutir la distribución de los subsidios hacia la Capital Federal.**<br><br> **Terminar con las restricciones a las importaciones** para evitar que retrase y estanque el progreso. Promover que el campo crezca libremente y aumente las exportaciones. **Promover las exportaciones de la pesca, la minería, la industria y todos los sectores.**<br><br> **Achicar el Estado y congelar contrataciones. Pasar de 23 ministerios ineficientes a 8-10.** Reducir la burocracia estatal, el gasto en personal sin tareas específicas y mejorar salarios del personal esencial como policías, médicos y docentes. <br>Definir un plazo para que las empresas públicas presenten un plan de negocios con déficit cero. Si no, serán privatizadas o reconvertidas en cooperativas." % bimonetarismo, unsafe_allow_html=True)
    with st.expander('⚕️ **Salud**'):
      st.markdown('<div style="text-align: justify;"><br> <b>Asegurar el acceso a las prestaciones de salud según el ciclo de vida de las personas.</b> Implementando acciones de promoción y prevención, como son los controles médicos periódicos, lavacunación y la difusión de hábitos saludables.<br><br> <b>Plan integral de salud materno infantil.</b> Disminución de la brecha de la mortalidad materno-infantil entre las provincias. Asegurar el pleno cumplimiento de la Ley de atención y cuidado integral de la salud durante el embarazo y la primera infancia. Garantizar que todas las maternidades públicas sean seguras. <br><br><b>Transformación digital del sistema de salud.</b> Historia clínica electrónica interoperable, receta electrónica, telemedicina en todo el territorio nacional, para tener un sistema más eficiente, integrado y que facilite el acceso a la atención de los población.<br><br> <b>Cuidar a los que nos cuidan.</b> Asegurando a los profesionales de la salud una remuneración acorde a sus responsabilidades y promoviendo su capacitación continua. Implementando políticas que incentiven la formación en especialidades críticas.<br><br> <b>Libertad de elección.</b> Los trabajadores podrán optar por la obra social de su preferencia sin ningún tipo de limitación. También quienes se jubilen y quieran permanecer en su obra social podrán hacerlo sin restricciones.<br></div>', unsafe_allow_html=True)
      #st.write("<br> **Asegurar el acceso a las prestaciones de salud según el ciclo de vida de las personas.** Implementando acciones de promoción y prevención, como son los controles médicos periódicos, lavacunación y la difusión de hábitos saludables.<br><br> **Plan integral de salud materno infantil.** Disminución de la brecha de la mortalidad materno-infantil entre las provincias. Asegurar el pleno cumplimiento de la Ley de atención y cuidado integral de la salud durante el embarazo y la primera infancia. Garantizar que todas las maternidades públicas sean seguras. <br><br>**Transformación digital del sistema de salud.** Historia clínica electrónica interoperable, receta electrónica, telemedicina en todo el territorio nacional, para tener un sistema más eficiente, integrado y que facilite el acceso a la atención de los población.<br><br> **Cuidar a los que nos cuidan.** Asegurando a los profesionales de la salud una remuneración acorde a sus responsabilidades y promoviendo su capacitación continua. Implementando políticas que incentiven la formación en especialidades críticas.<br><br> **Libertad de elección.** Los trabajadores podrán optar por la obra social de su preferencia sin ningún tipo de limitación. También quienes se jubilen y quieran permanecer en su obra social podrán hacerlo sin restricciones.", unsafe_allow_html=True)
    with st.expander('📖 **Educación**'):
      st.markdown('<div style="text-align: justify;"><br> <b>Declarar la educación servicio esencial.</b> El derecho de huelga no debe estar por encima del derecho de los chicos a aprender. El servicio esencial es que cada año los chicos aprendan lo que deben aprender, nunca menos de 190 días de clase. <br><br> <b>Programa Nacional de Alfabetización</b> para que todos los niños lean, comprendan textos básicos en primer grado y manejen las operaciones matemáticas básicas al inicio de la escuela primaria. <br><br> <b>Implementar un examen de ingreso para docentes secundarios y universitarios</b>, con exámenes regulares a docentes en función para asegurar un buen nivel educativo. Desregular la actividad docente para incorporar otros formadores profesionales en cuestiones como informática y nuevas tecnologías, incorporándolos a la actividad docente con un curso de pedagogía.<br><br> <b>Ampliación de la cobertura del nivel inicial</b>, enfocada en la estimulación temprana durante la primera infancia y articulada con programas sociales. El proceso educativo debe iniciarse en la primera infancia con la estimulación temprana y el cuidado adecuado para que los chicos puedan encarar en igualdad de condiciones su educación. <br><br> <b>Modernización del sistema universitario</b>, adecuando la extensión de las carreras de grado y facilitando la inserción laboral. Se ampliarán las becas para carreras estratégicas y estudiantes de sectores vulnerables, y se dará impulso a carreras más cortas y títulos intermedios.<br></div>', unsafe_allow_html=True)
      #st.write("<br> **Declarar la educación servicio esencial.** El derecho de huelga no debe estar por encima del derecho de los chicos a aprender. El servicio esencial es que cada año los chicos aprendan lo que deben aprender, nunca menos de 190 días de clase. <br><br> **Programa Nacional de Alfabetización** para que todos los niños lean, comprendan textos básicos en primer grado y manejen las operaciones matemáticas básicas al inicio de la escuela primaria. <br><br> **Implementar un examen de ingreso para docentes secundarios y universitarios**, con exámenes regulares a docentes en función para asegurar un buen nivel educativo. Desregular la actividad docente para incorporar otros formadores profesionales en cuestiones como informática y nuevas tecnologías, incorporándolos a la actividad docente con un curso de pedagogía.<br><br> **Ampliación de la cobertura del nivel inicial**, enfocada en la estimulación temprana durante la primera infancia y articulada con programas sociales. El proceso educativo debe iniciarse en la primera infancia con la estimulación temprana y el cuidado adecuado para que los chicos puedan encarar en igualdad de condiciones su educación. <br><br> **Modernización del sistema universitario**, adecuando la extensión de las carreras de grado y facilitando la inserción laboral. Se ampliarán las becas para carreras estratégicas y estudiantes de sectores vulnerables, y se dará impulso a carreras más cortas y títulos intermedios.", unsafe_allow_html=True)
    with st.expander('♻️ **Ambiente**'):
      st.markdown('<div style="text-align: justify;"><br> El mundo está atravesado por un fenómeno de cambio climático que indiscutiblemente constituye un grave riesgo. Es importante la <b>protección del medio ambiente frente a la contaminación y a la depredación de los recursos naturales.</b> <br><br> El <b>cumplimiento del acuerdo de París</b> es clave para luchar contra el cambio climático.<br><br> <b>El incremento de la producción y la ampliación de la infraestructura para energías renovables y de transición</b> en Argentina, como la energía solar o el gas.<br><br> <b>La producción de minerales críticos para la transición energética</b>, como el litio o el cobre en la Argentina. <br><br> Perseguir en forma efectiva el tráfico de vida silvestre, laexplotación ilegal de recursos naturales, y evitar la depredación u ocupación ilegal de Parques Nacionales.<br></div>', unsafe_allow_html=True)
      #st.write("<br> El mundo está atravesado por un fenómeno de cambio climático que indiscutiblemente constituye un grave riesgo. Es importante la **protección del medio ambiente frente a la contaminación y a la depredación de los recursos naturales.** <br><br> El **cumplimiento del acuerdo de París** es clave para luchar contra el cambio climático.<br><br> **El incremento de la producción y la ampliación de la infraestructura para energías renovables y de transición** en Argentina, como la energía solar o el gas.<br><br> **La producción de minerales críticos para la transición energética**, como el litio o el cobre en la Argentina. <br><br> Perseguir en forma efectiva el tráfico de vida silvestre, laexplotación ilegal de recursos naturales, y evitar la depredación u ocupación ilegal de Parques Nacionales.", unsafe_allow_html=True)
    with st.expander('👮‍♂️ **Seguridad**'):
      st.markdown('<div style="text-align: justify;"><br> <b>Orden Público.</b> Terminar con los bloqueos, las ocupaciones y los cortes ilegales, para que los argentinos puedan moverse, trabajar y estudiar sin que les hagan la vida imposible. Definir un marco legal para la realización ordenada de manifestaciones. <br><br> <b>Combatir el narcotráfico y el crimen organizado</b>, retomando el control del territorio, desplegando fuerzas federales con apoyo logístico de las Fuerzas Armadas en zonas críticas como Rosario y el Conurbano, y desarticular las organizaciones y mercados ilegales. <br><br> Tener <b>mano firme en el combate con el delito:</b> modificar el código penal elevando las penas de narcotráfico, abusos, homicidios y todos los delitos graves.<br><br> <b>Bajar la edad de imputabilidad a los 14 años de edad</b> y que exista para chicos con menos de catorce años un tratamiento en el que se trabaje la toma de conciencia del daño cometido en el delito, para que no arruinen sus vidas, ni la de los otros.<br><br> <b>Protección y Control de Fronteras / Plan Escudo Fronterizo.</b> Ampliar y fortalecer los mecanismos de control de fronteras a fin de proteger a nuestro país del delito trasnacional con un fuerte uso de tecnología para el control del espacio aéreo, mediante radares, aprobación de una Ley de Derribo y fortalecimiento del control de vías navegables. <br><br> <b>Darle a las fuerzas armadas el lugar que les corresponde</b> para que dejen de estar de lado y trabajen al servicio del país.<br></div>', unsafe_allow_html=True)
      #st.write("<br> **Orden Público.** Terminar con los bloqueos, las ocupaciones y los cortes ilegales, para que los argentinos puedan moverse, trabajar y estudiar sin que les hagan la vida imposible. Definir un marco legal para la realización ordenada de manifestaciones. <br><br> **Combatir el narcotráfico y el crimen organizado**, retomando el control del territorio, desplegando fuerzas federales con apoyo logístico de las Fuerzas Armadas en zonas críticas como Rosario y el Conurbano, y desarticular las organizaciones y mercados ilegales. <br><br> Tener **mano firme en el combate con el delito:** modificar el código penal elevando las penas de narcotráfico, abusos, homicidios y todos los delitos graves.<br><br> **Bajar la edad de imputabilidad a los 14 años de edad** y que exista para chicos con menos de catorce años un tratamiento en el que se trabaje la toma de conciencia del daño cometido en el delito, para que no arruinen sus vidas, ni la de los otros.<br><br> **Protección y Control de Fronteras / Plan Escudo Fronterizo.** Ampliar y fortalecer los mecanismos de control de fronteras a fin de proteger a nuestro país del delito trasnacional con un fuerte uso de tecnología para el control del espacio aéreo, mediante radares, aprobación de una Ley de Derribo y fortalecimiento del control de vías navegables. <br><br> **Darle a las fuerzas armadas el lugar que les corresponde** para que dejen de estar de lado y trabajen al servicio del país.", unsafe_allow_html=True)   
    with st.expander('📰 **Notas en los medios**'):
      st.write("<br> **FMI:** <br> [FMI asegura que Milei y Bullrich “apoyan el compromiso continúo con el Fondo” (Bloomberg)](%s)" % bullrich_fmi, unsafe_allow_html=True)
      st.write("[Reunión Melconian con el FMI (El Cronista)](%s)" % bullrich_fmi2, unsafe_allow_html=True)
      st.write("[El plan económico: Acuerdo con el FMI, Ganancias y reforma laboral (BAE Negocios)](%s)" % bullrich_fmi3, unsafe_allow_html=True)
      st.write("[Bullrich y su propuesta de “blindaje” para salir del cepo lo antes posible (El Diario)](%s)" % bullrich_fmi4, unsafe_allow_html=True)
      st.write("<br> **Propuestas:** <br> [Resumen de sus propuestas y la opinión de especialistas (Chequeado)](%s)" % bullrich_prop, unsafe_allow_html=True)
      st.write("<br> **Educación:** <br> [Bullrich: “Necesitamos educación universitaria gratuita, calificada y preparada” (El Intra)](%s)" % bullrich_edu, unsafe_allow_html=True)
      st.write("[Bullrich: “casi la mitad de los alumnos de las universidades públicas son extranjeros” - La respuesta del gobierno (Ámbito)](%s)" % bullrich_edu2, unsafe_allow_html=True)
      st.write("<br> **Agro:** <br> [Reducción acelerada de las retenciones (Agrofy)](%s)" % bullrich_agro, unsafe_allow_html=True)
      st.write("[Comparación con lo que proponen los otros 4 candidatos (TN)](%s)" % bullrich_agro2, unsafe_allow_html=True)
      st.write("<br> **Fallo YPF:** <br> [Bullrich: 'Que la guita la ponga el kirchnerismo' (Letra P)](%s)" % bullrich_ypf, unsafe_allow_html=True)
      st.write("[Bullrich: 'Kicillof se hizo el canchero y dijo que lo pagaba en pesos y ahora cuesta 16 mil millones de dólares' (Clarín)](%s)" % bullrich_ypf2, unsafe_allow_html=True)
      st.write("<br> **Entrevista con Fantino:** <br> [Entrevista con Alejandro Fantino (Neura Media)](%s)" % bullrich_fantino, unsafe_allow_html=True)
    with st.expander('📄 **Currículum Vitae**'):
      c1446, c2446, c446 = st.columns([0.2,0.6,0.2])
      with c2446:
        st.image(cvpato)
    with st.expander('🔍 **Si querés saber más sobre sus propuestas...**'):
      st.write("<br> **Sitio oficial:** https://patriciabullrich.com.ar/ <br><br> **Youtube:** https://www.youtube.com/@PatriciaBullrich<br><br> **Plataforma electoral:** https://drive.google.com/file/d/17-8oPtMWYgoRYU3UyBOO26Nm912HQAUS/edit <br><br> **Descargá la plataforma electoral del partido:**", unsafe_allow_html=True)
      st.download_button(label="Descargar PDF",
                        data=PDF_JxC,
                        file_name="JxC.pdf",
                        mime='application/octet-stream')

    schia_fmi = "https://www.cba24n.com.ar/cordoba/schiaretti---el-fmi-quiere-que-argentina-no-tenga-mas-dificultades-_a642be9b0deb55ebae7c768ec"
    schia_fmi2 = "https://www.eldiariocba.com.ar/locales/2023/7/27/hay-que-negociar-para-no-pagar-al-fmi-hasta-que-no-crezcamos-100244.html"
    schia_edu = "https://www.lavoz.com.ar/politica/juan-schiaretti-la-educacion-debe-ser-publica-y-gratuita-y-una-obligacion-del-estado/"
    schia_ypf = "https://www.lavozdesanjusto.com.ar/schiaretti-tildo-de-mamarrachada-kirchnerista-la-expropiacion-de-ypf"
    schia_agro = "https://www.ambito.com/politica/schiaretti-junto-miembros-la-mesa-enlace-si-soy-presidente-eliminare-las-retenciones-n5757980"
    schia_seguridad = "https://www.letrap.com.ar/politica/esta-inseguridad-no-es-mia-cordoba-y-la-nacion-niegan-saqueos-y-escala-la-pelea-politica-n5402618"

    st.markdown("<h2 style='text-align: center;'>Juan Schiaretti (Hacemos por Nuestro País)</h2>", unsafe_allow_html=True)
    with st.expander('📈 **Economía**'):
          st.markdown('<div style="text-align: justify;"><br> <b>Bajar la inflación con un plan integral de estabilización:</b> prudencia y disciplina en la emisión monetaria, equilibrio fiscal, recomposición de reservas, defensa de la competencia, incentivo a las inversiones y una inteligente política de ingresos.<br><br><b>Establecer un sistema tributario simple, estable y progresivo:</b> conseguir un sistema tributario que no penalice la producción con malos impuestos, eliminar de manera gradual las retenciones a exportaciones y alentar con los ingresos fiscales la inversión productiva con creación de empleo.<br><br><b>Generar una moneda nacional:</b> una moneda nacional sana y fuerte como condición estructural para la estabilidad económica.<br><br><b>Realizar un proyecto de desarrollo integral:</b> permitir desplegar la potencialidad de nuestros recursos humanos y naturales, impulsar la capacidad innovadora de las personas como base para un compromiso armonioso con las empresas, sindicatos, universidades y complejo científico-tecnológico.<br><br><b>Garantizar una nueva política tributaria para las Pymes y establecer un sistema simple y accesible de créditos a las producción:</b> alentar la inversión con empleo, incentivos tributarios para que se reviertan utilidades en bienes de capital<br></div>', unsafe_allow_html=True)
          #st.write("<br> **Bajar la inflación con un plan integral de estabilización:** prudencia y disciplina en la emisión monetaria, equilibrio fiscal, recomposición de reservas, defensa de la competencia, incentivo a las inversiones y una inteligente política de ingresos.<br><br>**Establecer un sistema tributario simple, estable y progresivo:** conseguir un sistema tributario que no penalice la producción con malos impuestos, eliminar de manera gradual las retenciones a exportaciones y alentar con los ingresos fiscales la inversión productiva con creación de empleo.<br><br>**Generar una moneda nacional:** una moneda nacional sana y fuerte como condición estructural para la estabilidad económica.<br><br>**Realizar un proyecto de desarrollo integral:** permitir desplegar la potencialidad de nuestros recursos humanos y naturales, impulsar la capacidad innovadora de las personas como base para un compromiso armonioso con las empresas, sindicatos, universidades y complejo científico-tecnológico.<br><br>**Garantizar una nueva política tributaria para las Pymes y establecer un sistema simple y accesible de créditos a las producción:** alentar la inversión con empleo, incentivos tributarios para que se reviertan utilidades en bienes de capital", unsafe_allow_html=True)
    with st.expander('⚕️ **Salud**'):
      st.markdown('<div style="text-align: justify;"><br> <b>Avanzar en la digitalización de la medicina:</b> Promover la telemedicina para lograr un acceso real de forma eficaz y rápida, brindando una atención permanente y especializada a pesar de las distancias en todas las ciudades del país. También se debe avanzar en un sistema que permita pedir un turno o consultar resultados desde el celular o computadora, para no tener que hacer colas en la madrugada para conseguir un turno. Además se debe generar una única historia clínica digital, esta mejora la atención y baja los costos de salud al dar acceso a toda la información en todo momento.<br><br> <b>Mejorar la atención de los problemas de salud mental y bienestar emocional:</b> Se va a promover la modificación de la ley de salud mental para adaptarla a las necesidades actuales. Además es fundamental que los profesionales que están en la primera línea de fuego para la detección de los problemas de salud mental, se capaciten para mejorar la detección, resolución o derivación a centros especializados para su tratamiento o rehabilitación.<br><br> <b>Jerarquizar a los profesionales de salud:</b> Establecer estándares mínimos de condiciones laborales para los profesionales de salud, generando incentivos específicos para la radicación de profesionales en todo el país, en particular las áreas más desatendidas, como las zonas rurales, así como promover el desarrollo de las especialidades cuyo déficit es cada vez más pronunciado, lo que está afectando la atención médica tanto en el sector público como en el privado a lo largo y ancho de nuestro territorio.<br><br> <b>Mejorar la atención primaria en salud:</b> Invertir en infraestructura, profesionales, tecnología y financiamiento para que cada argentino pueda acceder a un centro de salud con cuidados continuados y se favorezca la participación social en el cuidado de la salud. Poner foco en la salud materno-infantil, la vacunación y nutrición, las enfermedades transmisibles, la adopción de estilos de vida saludables y la prevención y control de enfermedades crónicas, que hoy dan cuenta de más de 7 de cada 10 muertes en nuestro país.<br><br> <b>Promover la articulación e integración de los subsectores de la salud:</b> Garantizar en un mismo paquete todas las prestaciones con eficiencia, equidad y acreditación de la calidad, tanto en el sector público de las provincias, como en todas las obras sociales, las empresas de medicina prepaga, la red de hospitales, las clínicas y los sanatorios privados.<br></div>', unsafe_allow_html=True)
      #st.write("<br> **Avanzar en la digitalización de la medicina:** Promover la telemedicina para lograr un acceso real de forma eficaz y rápida, brindando una atención permanente y especializada a pesar de las distancias en todas las ciudades del país. También se debe avanzar en un sistema que permita pedir un turno o consultar resultados desde el celular o computadora, para no tener que hacer colas en la madrugada para conseguir un turno. Además se debe generar una única historia clínica digital, esta mejora la atención y baja los costos de salud al dar acceso a toda la información en todo momento.<br><br> **Mejorar la atención de los problemas de salud mental y bienestar emocional:** Se va a promover la modificación de la ley de salud mental para adaptarla a las necesidades actuales. Además es fundamental que los profesionales que están en la primera línea de fuego para la detección de los problemas de salud mental, se capaciten para mejorar la detección, resolución o derivación a centros especializados para su tratamiento o rehabilitación.<br><br> **Jerarquizar a los profesionales de salud:** Establecer estándares mínimos de condiciones laborales para los profesionales de salud, generando incentivos específicos para la radicación de profesionales en todo el país, en particular las áreas más desatendidas, como las zonas rurales, así como promover el desarrollo de las especialidades cuyo déficit es cada vez más pronunciado, lo que está afectando la atención médica tanto en el sector público como en el privado a lo largo y ancho de nuestro territorio.<br><br> **Mejorar la atención primaria en salud:** Invertir en infraestructura, profesionales, tecnología y financiamiento para que cada argentino pueda acceder a un centro de salud con cuidados continuados y se favorezca la participación social en el cuidado de la salud. Poner foco en la salud materno-infantil, la vacunación y nutrición, las enfermedades transmisibles, la adopción de estilos de vida saludables y la prevención y control de enfermedades crónicas, que hoy dan cuenta de más de 7 de cada 10 muertes en nuestro país.<br><br> **Promover la articulación e integración de los subsectores de la salud:** Garantizar en un mismo paquete todas las prestaciones con eficiencia, equidad y acreditación de la calidad, tanto en el sector público de las provincias, como en todas las obras sociales, las empresas de medicina prepaga, la red de hospitales, las clínicas y los sanatorios privados.", unsafe_allow_html=True)
    with st.expander('📖 **Educación**'):
      st.markdown('<div style="text-align: justify;"><br> <b>Reconstruir un verdadero sistema nacional de educación y garantizar calidad educativa:</b> fortalecer el federalismo integrando diferencias y poniendo equilibrio donde hay desigualdad.<br><br>  <b>Garantizar la ampliación de la jornada extendida en nivel primario:</b> incluyendo idiomas y robótica en todas las escuelas. Volver esencial a la tecnología digital en el aprendizaje y fortalecer el aprendizaje de las ciencias básicas.<br><br> <b>Jerarquizar la función docente:</b> con remuneraciones justas y formación permanente.<br><br>  <b>Crear un “cuarto nivel educativo”:</b> destinado a jóvenes y adultos, asociado al mundo del trabajo y la formación laboral.<br></div>', unsafe_allow_html=True)
      #st.write("<br> **Reconstruir un verdadero sistema nacional de educación y garantizar calidad educativa:** fortalecer el federalismo integrando diferencias y poniendo equilibrio donde hay desigualdad.<br><br>  **Garantizar la ampliación de la jornada extendida en nivel primario:** incluyendo idiomas y robótica en todas las escuelas. Volver esencial a la tecnología digital en el aprendizaje y fortalecer el aprendizaje de las ciencias básicas.<br><br> **Jerarquizar la función docente:** con remuneraciones justas y formación permanente.<br><br>  **Crear un “cuarto nivel educativo”:** destinado a jóvenes y adultos, asociado al mundo del trabajo y la formación laboral.", unsafe_allow_html=True)
    with st.expander('♻️ **Ambiente**'):
      st.markdown('<div style="text-align: justify;"><br> <b>Sostener esfuerzos por frenar la deforestación:</b> llevar a ceros en términos netos en el año 2030. Garantizar la eliminación de plásticos de un solo uso de manera inmediata.<br><br> <b>Cumplir los objetivos de emisiones de GEI del Acuerdo de París:</b> aumentar gradualmente el corte de combustibles tradicional con bioetanol y con biodiesel.<br><br> <b>Revalorizar al Mar Argentino en la economía nacional:</b> incorporar a la matriz productiva e industrial al Mar como actor protagónico, en el desarrollo y explotación de la plataforma marítima, sin dejar de lado el cuidado del ambiente.<br></div>', unsafe_allow_html=True)
      #st.write("<br> **Sostener esfuerzos por frenar la deforestación:** llevar a ceros en términos netos en el año 2030. Garantizar la eliminación de plásticos de un solo uso de manera inmediata.<br><br> **Cumplir los objetivos de emisiones de GEI del Acuerdo de París:** aumentar gradualmente el corte de combustibles tradicional con bioetanol y con biodiesel.<br><br> **Revalorizar al Mar Argentino en la economía nacional:** incorporar a la matriz productiva e industrial al Mar como actor protagónico, en el desarrollo y explotación de la plataforma marítima, sin dejar de lado el cuidado del ambiente.", unsafe_allow_html=True)
    with st.expander('👮‍♂️ **Seguridad**'):
      st.markdown('<div style="text-align: justify;"><br> <b>Promover el desarme de la ciudadanía:</b> evitar la circulación de armas ilegales.<br><br> <b>Crear un Consejo Federal para el mejoramiento de la Justicia Federal:</b> poner en marcha el sistema acusatorio en todo el territorio nacional, inmediata implementación del juicio por jurado, entre otras.<br><br> <b>Contar con más efectivos en las fuerzas federales:</b> cuidar la frontera y combatir al narcotráfico.<br><br> <b>Reformular el trabajo de inteligencia criminal y análisis del delito:</b> incorporar tecnología y coordinación entre las fuerzas de seguridad, justicia y diversas agencias del Estado.<br></div>', unsafe_allow_html=True)
      #st.write("<br> **Promover el desarme de la ciudadanía:** evitar la circulación de armas ilegales.<br><br> **Crear un Consejo Federal para el mejoramiento de la Justicia Federal:** poner en marcha el sistema acusatorio en todo el territorio nacional, inmediata implementación del juicio por jurado, entre otras.<br><br> **Contar con más efectivos en las fuerzas federales:** cuidar la frontera y combatir al narcotráfico.<br><br> **Reformular el trabajo de inteligencia criminal y análisis del delito:** incorporar tecnología y coordinación entre las fuerzas de seguridad, justicia y diversas agencias del Estado.", unsafe_allow_html=True)   
    with st.expander('📰 **Notas en los medios**'):
      st.write("<br> **FMI:** <br> [Schiaretti: 'El FMI quiere que Argentina no tenga más dificultades' (CBA24N)](%s)" % schia_fmi, unsafe_allow_html=True)
      st.write("[Schiaretti: 'Hay que negociar para no pagar al FMI hasta que no crezcamos' (El Diario CBA)](%s)" % schia_fmi2, unsafe_allow_html=True)
      st.write("<br> **Educación:** <br> [Schiaretti: 'La educación debe ser pública y gratuita, y una obligación del Estado' (La Voz)](%s)" % schia_edu, unsafe_allow_html=True)
      st.write("<br> **YPF:** <br> [Schiaretti: La expropiación es 'una mamarrachada kirchnerista' (La Voz de San Justo)](%s)" % schia_ypf, unsafe_allow_html=True)
      st.write("<br> **Agro:** <br> [Schiaretti: 'Si soy presidente, eliminaré las retenciones' (Ámbito)](%s)" % schia_agro, unsafe_allow_html=True)
      st.write("<br> **Seguridad:** <br> [Córdoba y la Nación niegan saqueos y escala la pelea política (Letra P)](%s)" % schia_seguridad, unsafe_allow_html=True)
    with st.expander('📄 **Currículum Vitae**'):
      c14467, c24467, c4467 = st.columns([0.2,0.6,0.2])
      with c24467:
        st.image(cvschia)
    with st.expander('🔍 **Si querés saber más sobre sus propuestas...**'):
      st.write("<br> **Sitio oficial:** https://www.hacemospornuestropais.ar/ <br><br> **Youtube:** https://www.youtube.com/@HacemosporNuestroPais<br><br> **Plataforma electoral:** <br>", unsafe_allow_html=True)
      st.link_button("Descargar PDF", "https://drive.google.com/drive/folders/1HFzPjCLfXTwg7kyEOZGb35Dx-jdMj59l?usp=sharing")

    breg_fmi = "https://jacobinlat.com/2023/08/11/un-mensaje-contundente-contra-el-ajuste-y-el-sometimiento-al-fmi/"
    breg_fmi2 = "https://www.laizquierdadiario.com/Del-Cano-Aunque-como-candidato-diga-lo-contrario-Massa-aplica-el-ajuste-para-complacer-al-FMI"
    breg_agro = "https://news.agrofy.com.ar/noticia/206195/propuestas-agro-milei-bullrich-massa-schiaretti-y-bregman-mirada-que-aplicaran-partir"
    breg_ambiente = "https://www.laizquierdadiario.cl/La-lista-que-encabeza-Bregman-propone-pelear-por-una-transicion-energetica-desde-abajo"
    breg_segu = "https://chequeado.com/el-explicador/elecciones-paso-2023-esto-proponen-los-precandidatos-presidenciales-sobre-seguridad/"
    breg_edu = "https://avellanedahoy.com.ar/nota/18701/bregman-en-avellaneda-la-izquierda-siempre-va-a-estar-en-cada-pelea-en-defensa-de-la-educacion-publica/"
    breg_edu2 = "https://www.laizquierdadiario.com/Myriam-Bregman-cruzo-a-Milei-por-su-rechazo-a-la-ESI"
    breg_edu3 = "https://avellanedahoy.com.ar/nota/18401/se-realizo-la-jornada-de-debate-propuesta-educativa-del-frente-de-izquierda-unidad/"


    st.markdown("<h2 style='text-align: center;'>Myriam Bregman (Frente de Izquierda)</h2>", unsafe_allow_html=True)

    with st.expander('📈 **Economia**'):
          st.write("<br> **Romper con el FMI:** decirle no al pago de la deuda y usar esa plata para pagar salarios, generar trabajo y garantizar el acceso a la salud, educación y vivienda.<br><br> **Nacionalizar la banca y el comercio exterior, estatizar todas las privatizadas:** evitar la fuga de capitales y estatizar privadas de servicios bajo el control de sus trabajadores y usuarios junto con técnicos y especialistas de la universidades públicas. Cuidar a los pequeños ahorristas y brindar créditos baratos. <br><br> **Aumentar salarios, jubilaciones, anular la reforma previsional y prohibir despidos y suspensiones:** el ingreso mensual debe cubrir las necesidades básicas. Expropiar y estatizar empresas en crisis para que sean puestas a producir, bajo el control de sus trabajadores y trabajadoras. Eliminar trabajo precario y en negro, todos y todas a planta permanente. Rechazar nuevas formas de explotación laboral a través de plataformas virtuales. 82% móvil para los jubilados y jubiladas. <br><br> **Eliminar el IVA de la canasta familiar:** Abolir el impuesto al salario.", unsafe_allow_html=True)
    with st.expander('⚕️ **Salud**'):
      st.write("<br> **Garantizar el acceso a la salud:** usando la plata del pago de la deuda al FMI<br><br> **Unificar y centralizar el sistema de salud:** reunir la totalidad de los recursos del sistema público, privado, de obras sociales y de la Universidad, bajo control de los trabajadores y profesionales. Implementar comités de emergencia central y locales, con participación de los y las trabajadores/as. ", unsafe_allow_html=True)
    with st.expander('📖 **Educación**'):
      st.write("<br> **Garantizar el acceso a la educación:** usando la plata del pago de la deuda al FMI", unsafe_allow_html=True)
    with st.expander('♻️ **Ambiente**'):
      st.write("<br> **Eliminar la minería y el uso indiscriminado de agrotóxicos:** rechazar el fracking y la megaminería. Anular el acuerdo YPF-Chevron. Expropiar esas firmas sin indemnización y que reparen los daños causados.<br><br> **Producir y distribuir energía según las necesidades populares fundamentales:** la renta petrolera y minera debe financiar la transición hacia una matriz energética sustentable y diversificada, desarrollando las energías renovables y/o de bajo impacto ambiental en consulta con las comunidades locales. Prohibir las fumigaciones aéreas y el uso indiscriminado de agrotóxicos.", unsafe_allow_html=True)
    with st.expander('👮‍♂️ **Seguridad**'):
      st.write("<br> No se encontraron propuestas en la fuente seleccionada", unsafe_allow_html=True)
    with st.expander('📰 **Notas en los medios**'):
      st.write("<br> **FMI:** <br> [Entrevista: Definiciones sobre el gobierno actual, la interna de la izquierda y la situación con el FMI (Jacobin)](%s)" % breg_fmi, unsafe_allow_html=True)
      st.write("[Del Caño: 'Aunque como candidato diga lo contrario, Massa aplica el ajuste para complacer al FMI' (La Izquierda Diario)](%s)" % breg_fmi2, unsafe_allow_html=True)
      st.write("<br> **Agro:** <br> [Comparación de sus propuestas con la de otros candidatos (Agrofy)](%s)" % breg_agro, unsafe_allow_html=True)
      st.write("<br> **Ambiente:** <br> [Transición energética y ecológica desde abajo: estatización de toda la industria energética, bajo control de sus trabajadores (La Izquierda Diario)](%s)" % breg_ambiente, unsafe_allow_html=True)
      st.write("<br> **Seguridad:** <br> [Comparación de sus propuestas con la de otros candidatos (Chequeado)](%s)" % breg_segu, unsafe_allow_html=True)
      st.write("<br> **Educación:** <br> [Bregman: 'La Izquierda siempre va a estar en cada pelea en defensa de la educación pública' (Avellaneda Hoy)](%s)" % breg_edu, unsafe_allow_html=True)
      st.write("[Myriam Bregman cruzó a Milei por su rechazo a la ESI (La Izquierda Diario)](%s)" % breg_edu2, unsafe_allow_html=True)
      st.write("[Jornada de debate 'Propuesta educativa del Frente de Izquierda Unidad' (Avellaneda Hoy)](%s)" % breg_edu3, unsafe_allow_html=True)
    with st.expander('📄 **Currículum Vitae**'):
      c144678, c244678, c44678 = st.columns([0.2,0.6,0.2])
      with c244678:
        st.image(cvbregman)
    with st.expander('🔍 **Si querés saber más sobre sus propuestas...**'):
      st.write("<br> **Sitio oficial:** https://www.myriambregman.com.ar/index.php <br><br> **Youtube:** https://www.youtube.com/@ptsargentina<br><br> **Descargá la plataforma electoral del partido:**", unsafe_allow_html=True)
      st.download_button(label="Descargar PDF",
                        data=PDF_FIT,
                        file_name="FIT.pdf",
                        mime='application/octet-stream')



    st.write("**Fuente: https://merepresenta.info/propuestas** <br><br>", unsafe_allow_html=True)

    #with st.expander("**La Libertad Avanza**"):
        #st.markdown(pdf_display, unsafe_allow_html=True)

    #with st.expander("**Juntos por el Cambio**"):
        #st.markdown(pdf_display2, unsafe_allow_html=True)

    #with st.expander("**Union por la Patria**"):
        #st.markdown(pdf_display3, unsafe_allow_html=True)

    # Load environment variables
    load_dotenv()

    # st.markdown("<h1 style='text-align: center;'>Plataformas electorales nacionales de las agrupaciones politicas<br><br></h1>", unsafe_allow_html=True)


    # plat1, plat3, plat2 = st.columns([0.4, 0.1, 0.5])

    # with plat1:
    #     with st.expander("**La Libertad Avanza**"):
    #       st.markdown(pdf_display, unsafe_allow_html=True)

    #     with st.expander("**Juntos por el Cambio**"):
    #       st.markdown(pdf_display2, unsafe_allow_html=True)

        # with st.expander("**Union por la Patria**"):
        #   st.markdown(pdf_display3, unsafe_allow_html=True)

    # with plat2:
      
    st.subheader("**Consulta las plataformas electorales con ChatGPT**", anchor=False)

    def main():
        load_dotenv()

        pdf_mapping = {
        'La Libertad Avanza': 'data/LLA.pdf',
        'Juntos por el Cambio': 'data/JxC.pdf',
        'Union por la Patria': 'data/UxP.pdf'
        
    }
        
        custom_names = list(pdf_mapping.keys())

        selected_custom_name = st.selectbox('Selecciona la plataforma', ['', *custom_names])
        pdf = pdf_mapping.get(selected_custom_name)

        
        if pdf is not None:
          pdf_reader = PdfReader(pdf)
          text = ""
          for page in pdf_reader.pages:
            text += page.extract_text()
            
          
          text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
          )
          chunks = text_splitter.split_text(text)
          
          
          embeddings = OpenAIEmbeddings()
          knowledge_base = FAISS.from_texts(chunks, embeddings)
          
          
          user_question = st.text_input("Pregunta sobre la plataforma electoral:")
          if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
              response = chain.run(input_documents=docs, question=user_question)
              print(cb)
              
            st.write(response)
        

    if __name__ == '__main__':
        main()
