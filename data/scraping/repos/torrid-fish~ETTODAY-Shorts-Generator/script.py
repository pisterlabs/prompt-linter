import openai
import json

# Example Text
example_input_text =\
    '''
    input text:
    嘉義縣民雄鄉菁埔村一處三合院今(16)日上午發生火警，4歲、2歲的一對何姓小姊弟遭反鎖留在家中受困，雙雙喪命，何姓夫妻得知後分頭趕往醫院，將孩子們送往殯儀館，妻子更是當場癱軟，泣不成聲。
    初步了解，何家一家7口住在該處三合院中，夫妻育有5個孩子，今上午父親何男(33歲)外出工作，3名孩童也到學校上課，只留下母親張女(32歲)在家照顧4歲、2歲的小姊弟，未料張女為了申辦低收入戶的資料暫時外出，將大門反鎖獨留2幼童在家，卻發生火災悲劇。
    夫妻倆得知2子女命喪火窟，分別前往大林慈濟醫院、嘉義基督教醫院，悲痛送孩子到殯儀館，而張女在簽署文件時手扶額頭不斷流淚，難以接受孩子頓時消失的事實；何男難過透露，他負責外出工作，並將錢交給妻子管理，因此家務及孩子生活主要也是妻子在處理。
    民雄鄉公所表示，將協助家屬辦理殯葬事宜，包括納骨塔等事宜，又因本件疑似涉及兒少法，家長不得讓未滿6歲的幼兒獨自一人在家，現則由警政、消防、社會局等單位調查火災發生細節。
    '''
example_input_img_decription= \
'''
input image description:
img1:家屬悲慟簽屬文件的照片
img2:火災現場照片，消防人願正在撲滅火焰，房子一片焦黑
'''
example_input_time = "input length 30s"
example_output = \
'''
[
{
    "title":"嘉義幼童火災喪命 家庭瞬間破碎",
    "keywords":["幼童喪命"],
    "outline":["最後提醒各位觀眾注意家庭安全","讓我們期許悲劇不再發生"],
    "imageDescription":"火災場景的照片"
},
{
    "script": "今早，嘉義縣民雄鄉菁埔村的何家遭遇不幸，家中突發火警",
    "imageDescription": "img2",
    "keywords": ["嘉義縣民雄鄉菁埔村的何家"]
},
{
    "script": "家中的4歲和2歲小姊弟因被反鎖在家中而受困，最終雙雙喪命",
    "imageDescription": "4歲和2歲小姊弟被反鎖在家中受困",
    "keywords": ["雙雙喪命"]
},
{
    "script": "何姓夫妻得知這個噩耗，急忙分頭趕往不同的醫院",
    "imageDescription": "何姓夫妻急忙奔向醫院",
    "keywords": ["悲痛欲絕"]
},
{
    "script": "最終只能將孩子們送往殯儀館，妻子在殯儀館前悲痛欲絕，泣不成聲",
    "imageDescription": "img1",
    "keywords": ["殯儀館"]
},
{
    "script": "回顧事發，何家一家7口，夫妻育有5個孩子。何先生外出工作，3名孩童上學",
    "imageDescription": "一張照片，照片裡有何家7口，一男一女和5名小孩",
    "keywords": ["外出工作"]
},
{
    "script": "只留下何夫人照顧家中兩幼童。未料何夫人為申辦低收入戶資料短暫外出，反鎖大門，沒想到竟釀成悲劇",
    "imageDescription": "何夫人急忙出門，面色焦急，兩名孩童待在家中的示意圖",
    "keywords": ["反鎖大門"]
},
{
    "script": "何先生痛訴，家務和孩子的生活主要是夫人負責，這次事件讓他深感自責和無助",
    "imageDescription": "何先生雙手抱頭，淚水在眼眶打轉",
    "keywords": ["痛訴"]
},
{
    "script": "民雄鄉公所表示，將全力協助何家辦理後事，並因此事件涉嫌違反兒少法，警消和社會局已介入調查火災發生的細節",
    "imageDescription": "警消和社會局人員在火警現場進行仔細調查，記錄每個細節",
    "keywords": ["違反兒少法"]
}
]
'''

# Practice Text
practice_text =  \
    '''
    啪一聲就沒電！清大生怨「10天停電6次」：不被當人看

    「請問我們是住在什麼難民窟嗎？」一名清大女同學抱怨，學校短短10天就停電6次，有3次還是無預警，學生們冰箱食物全壞、各種電器受損、報告沒存到檔全沒了、睡到一半被熱醒，學生不得已只好到外面旅館住宿。對此清大回應，停電主因是高壓站電纜老舊及受潮，已啟動供電改善工程，預計10月29日解除限電。
    清大女同學在Dcard上發文，「清大短短10天停電6次，根本活在難民營」，從9日無預警停電開始，10天內就停電6次，其中有3次是無預警，學生們冰箱食物全壞、各種電器因停電受損甚至直接死機、報告沒存到檔直接飛灰煙滅、工作站死機根本沒辦法寫程式交作業、睡到一半被斷電熱醒。
    不少住宿生只好向租屋的同學尋求協助，有些人甚至直接住在旅館，原PO向學校反映，卻感到不被重視，「校方至今沒有提出具體的補償措施，只有寫了一封含糊其辭的安撫信給大家，說可以依據停電天數補償住宿費」。
    她無奈嘆，「這是一個台灣前幾志願的學術殿堂，應該出現的樣子嗎？」有一次無預警停電，甚至在停電5分鐘後才寄通知信，「這一個月以來，我能感受到的就是，清大這個學校完全不把學生當人看，時不時就可以體驗一下全黑的世界，請問我們是住在什麼難民窟嗎？」
    貼文一出，清大生紛紛在留言區抱怨，「同校幫推，行政處室開一半電燈、教室處室都不開冷氣，天氣一熱起來根本要人命，完全不把學生、教職員當人看」、「禮拜一凌晨停電完，早上不知道為什麼我硬碟就炸了，看到學校跟我們說要共體時艱阿，感覺好像是我們故意浪費電的樣子，就覺得超不爽」、「晚上突然啪一聲就沒電的最可惡！欸！我有繳住宿費耶！為什麼必須到處流浪，還要到處問住校外或交大的同學，有沒有空間可以借住借洗澡」、「停電整個學校黑漆漆，宿舍門禁完全解除，住宿生在校園內的安全超級有問題，校方有在管嗎」、「這10天比我過去20年經歷過的停電次數還多」。
    還有清大生指出，「校方目前的態度就是，1、要我賠償你的損失，除非你很窮，活不下去，2、學生說宿舍區突然斷電很糟，那我就直接用天氣預報預測明天會用電超標，先告訴你要斷電」。
    【15:30更新清大回應】
    清華大學表示，近期停電的主因是供應全校約一半電力的第二高壓站電纜因老舊及受潮，10月9日晚間突然故障。緊急修復後，10月16日又有另一處電纜故障。
    校方已啟動兩項供電改善工程，包括儘速修復故障區段電纜，並將原定於114年進行、把原來埋在地下的電纜改走共同管道工程提前為立即施工，完工後將大幅提升供電穩定性。
    但在施工期間，必須由第一高壓站來負擔原第二高壓站供電區域的供電。為免電力負荷過大，造成全校跳電，影響研究與教學，目前採取限電措施，希望師生不要開冷氣。昨天及今天氣溫升高，用電量突增，因此，下午2點到5點部分宿舍也不得不停電。
    清華大學預計在10月29日可修復第二高壓站電纜，解除限電情況。
    校方除多次發信、公告向師生說明，提出宿費的減免方案，及因此受影響學生急難救助。校長也與宿舍生代表開會，盡最大誠意溝通。
    '''
practice_input_time =  "input length 30s"
practice__description = \
'''
    img0:"清華大學正門照片"
    img1:"清華大學停電時間記錄表"
    img2:"清華大學的停電公告"
    img3:"同學在網路上張貼關於停電的文章"
'''
practice_answer = \
'''
[
    {
    "title":"清大學生爆「10天停電6次」慘況",
    "keywords":["清華大學","停電慘況"],
    "outline":["最後，我們必須強調學生權益的重要性","一起來看看他們經歷了什麼"],
    "imageDescription":"清華大學停電的圖片"
    },
    {
    "script": "最近，清大一名女生在網上爆料，學校10天內竟然停電六次",
    "imageDescription": "img0",
    "keywords": ["停電六次"]
    },
    {
    "script": "有三次停電甚至是完全無預警的，結果學生們的電器受損，食物變壞",
    "imageDescription": "帶有冰箱食物變質的照片",
    "keywords": ["無預警停電"]
    },
    {
    "script": "連報告都沒存到，中途斷電，檔案全沒了，甚至有人質疑「我們這是住在難民窟嗎？」",
    "imageDescription": "一台電腦突然關機，屏幕變黑",
    "keywords": ["檔案意外遺失"]
    },
    {
    "script": "這個情況讓學生們不得不求助外面，有些甚至直接住到旅館去",
    "imageDescription": "學生背著行李走出校園的照片",
    "keywords": ["暫住旅館"]
    },
    {
    "script": "同學們對學校無奈，表示校方並未提出具體的補償措施",
    "imageDescription": "img3",
    "keywords": ["無具體補償措施"]
    },
    {
    "script": "疑難，本來只是一場停電，為什麼變得這麼困擾？學生們不禁大費苦心，嘗試理解為什麼會變成這樣",
    "imageDescription": "學生擔憂的臉部特寫",
    "keywords": ["學生困惑"]
    },
    {
    "script": "清大回應認為，停電的主因是安裝有問題的電纜老舊，已運行改善工程",
    "imageDescription": "img2",
    "keywords": ["電纜老舊"]
    },
    {
    "script": "預計將在10月29日完成，屆時應可以解除限電",
    "imageDescription": "清大工程預計完成的公告",
    "keywords": ["10月29日"]
    }
]
'''

def script_generator(text: str, length: int, imgsDescription: list[str]) -> list[dict]:
    """
    Use given text and description of image to generate script.
    ## Args
    - length: The length of the video.
    - text: The text of the news.
    - imgsDescription: The list of string that describe the images.
    ## Return
    - setences(list[dist]): A list contains multiple sentence.
                            Each sentence have several keys as example below:
    """
    input_description = ''
    for i, element in enumerate(imgsDescription):
        input_description += f"img{i}:" + element + "\n"
    input_length = f"input length {length}s\n"

    openai.api_key = "<YOUR OPENAI API KEY>"
    
    completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Your are a professional news writer in ettoday(news company). Your job is to write the script for short video in Mandarin. Try your best to make a attractive script."},
        {"role": "user",  "content": '''Your mission：
                        1.I will give you an article of news, length of the script(sec), and some image descriptions in Mandarin.
                        2.You have to output the srcipts which include title, keywords, outline and images.
                        3.The content should be based on the input text. Don't fabricate fake news.
                        4.It is basically retrieve some sentensces from each paragraph.
                        5.numbers(Dates,amount),Proprietary are usually appear with important information.
                        6.the script should be colloquial and arouse curiosity.
                        '''},
        {"role": "user", "content": '''Here is some standard：
                        Please follow output format strictly. The output should be availbel to converted to list of dict by python code, json.loads.
                        each combinaiton of{"script","imageDescription","keywords"} is called section. For each section the length of script should be average 30 charaters, no more than 50 charaters.
                        title：at most 2 sentences, should be Sensational. Each sentence at most 8 Mandarin charaters. the keyword of the title should be the word in the title.
                        outline: two sentences and should be short with 25 characters. Leave and emotional comment. The sentence should begin with "最後" ex: "最後我想說停電實在是太不方便了，如果是你會怎麼做呢？". Please do not copy this sentence.
                        scripts: please modeify the content with input parameter,length of the script(sec). Speaking speed is about 400 characters per minutes(Mandarin).
                                The first part of the scripts should introduce main point of the news, 5W, and so on. Try your  best to arouse people's curiosity.
                                The first word of first sentence should be "最近","近日","近期","近年"... depends on the content.
                        image descriptions：please describe the image as detailed a possible. Try to make people can imagine the scene, if you think provided image is good enough. You can just use it. However, you can use at most 2 images.
                        keywords: 1 keyword  for each srction except for the first section.
                        keywords: places, names,times numbers,and so on, are usually the keyword. Otherwise, choose emotional words.
                        '''},
        {"role": "user", "content": "I will now give you an example, the input time, the input text, input image description, output data will be given below."},
        {"role": "user", "content": example_input_time},
        {"role": "user", "content": example_input_text},
        {"role": "user", "content": example_input_img_decription},
        {"role": "user", "content": example_output},
        {"role": "user", "content": "now I will give you an article, its your time to generate the script"},
        {"role": "user", "content": practice_input_time +  practice_text + practice__description},
        {"role": "assistant", "content": practice_answer},
        {"role": "user", "content": "This script is very good, now try another one"},
        {"role": "user", "content": input_length + text + input_description},
    ]
    )

    sentences = completion.choices[0].message["content"]
    sentences = json.loads(sentences)

    return sentences