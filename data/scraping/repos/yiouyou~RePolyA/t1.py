from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
                                 
llm = Ollama(model="yi:34b-q4_K_M", 
             callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))

llm("### System:\n\n假定你是情报分析员，请从给定新闻句子中，抽取如下实体：'时间'，'地点'，'人物'，'军队'，'武器'，'伤亡'。下面是一些csv格式的新闻句子分析示例，其中新闻句子已用双引号括起来，逗号后面是句子的分析结果：\n\n\"202306032020，乌克兰防空部队编号11316部队，在奥德赛市奥德赛特镇检测到导弹袭击，立即分析上报。\", {\"时间\":\"202306032020\", \"地点\":\"奥德赛市奥德赛特镇\", \"人物\": \"\", \"军队\":\"乌克兰防空部队编号11316部队\", \"武器\": \"\", \"伤亡\": \"\"}\n\n\"202306032025，导弹在乌克兰基辅市沙拉克镇发生爆炸，造成两座大楼炸毁，约160名平民伤亡。\", {\"时间\":\"202306032025\", \"地点\":\"基辅市沙拉克镇\", \"人物\": \"\", \"军队\":\"\", \"武器\": \"\", \"伤亡\":\"160名平民伤亡\"}\n\n\"202306032025，乌克兰军事指挥中心瓦列里·扎卢日内将军，做出紧急预案，《军事入侵防御紧急方案》（乌-防空10586号），命令乌克兰陆军部队编号11316部队，乌克兰陆军部队编号11316部队，立即做出应对措施，立即处理。\", {\"时间\":\"202306032025\", \"地点\": \"\", \"人物\":\"瓦列里·扎卢日内将军\", \"军队\":\"乌克兰陆军部队编号11316部队,乌克兰陆军部队编号11316部队\", \"武器\": \"\", \"伤亡\": \"\"}\n\n\"2023年6月3日20时10分，乌克兰防空部队编号1136部队，在哈尔科夫市布朗尼镇检测到导弹袭击，立即分析上报。\", {\"时间\":\"2023年6月3日20时10分\", \"地点\":\"哈尔科夫市布朗尼镇\", \"人物\": \"\", \"军队\":\"乌克兰防空部队编号1136部队\", \"武器\": \"\", \"伤亡\": \"\"}\n\n\"还有消息称，清水河多个据点已被同盟军占领，禁止通行。\", {\"时间\": \"\", \"地点\":\"清水河\", \"人物\": \"\", \"军队\": \"同盟军\", \"武器\": \"\", \"伤亡\": \"\"}\n\n\"据当地居民消息，今日凌晨4点开始，有武装团体同时袭击掸邦北部腊戌、登尼、贵概、达莫尼、南非卡、木姐、南坎、瑟兰、滚弄、货班、清水河、老街地军事部署区。\", {\"时间\":\"凌晨4点\", \"地点\":\"掸邦北部腊戌,登尼,贵概,达莫尼,南非卡,木姐,南坎,瑟兰,滚弄,货班,清水河,老街\", \"人物\": \"\", \"军队\": \"\", \"武器\": \"\", \"伤亡\": \"\"}\n\n\"据确切消息，从10月27日凌晨4点开始，以上多个地区军事部署区、收费站、警察哨所、所有的警察局均同时受到相关武装力量攻击。\", {\"时间\":\"10月27日凌晨4点\", \"地点\": \"军事部署区,收费站,警察哨所,警察局\", \"人物\": \"\", \"军队\": \"\", \"武器\": \"\", \"伤亡\": \"\"}\n\n\"据央视军事消息，据缅甸媒体报道，10月27日，缅北腊戌、贵慨等多地的缅军军事据点遭武装袭击并爆发交火，战事激烈。\", {\"时间\":\"10月27日\", \"地点\":\"缅北腊戌,缅北贵慨\", \"人物\": \"\", \"军队\": \"\", \"武器\": \"\", \"伤亡\": \"\"}\n\n\"据中国侨网消息，据称，果敢民族民主同盟军发布公告对此次武装袭击负责，称行动旨在打击老街的电诈民团，因此从即日起封锁腊戌至清水河、木姐的道路。\", {\"时间\": \"\", \"地点\":\"腊戌,清水河,木姐\", \"人物\": \"\", \"军队\": \"果敢民族民主同盟军\", \"武器\": \"\", \"伤亡\": \"\"}\n\n\"指挥中心派遣陆军第三保障部队前往。\", {\"时间\":\"\", \"地点\":\"\", \"人物\": \"\", \"军队\":\"陆军第三保障部队\", \"武器\": \"\", \"伤亡\":\"\"}\n\n### Instruction: \n\n请分析\"2023年6月3日20时10分，乌克兰防空预警检测外籍导弹入境，乌克兰军事指挥中心依据《军事入侵防御紧急方案》（乌-防空10586号），对该事件做出紧急应对措施，导弹途径基辅市、哈尔科夫市、奥德赛市、最终20时35分在顿涅茨克市发生爆炸，造成两座大楼炸毁，约160名平民伤亡，出动乌克兰防空1军和防空13军，共计10辆防空导弹装甲车，50枚防空导弹，150名士兵，对袭击时间做出紧急处理。\"\n以json格式( {\"时间\":\"\", \"地点\":\"\", \"人物\": \"\", \"军队\":\"\", \"武器\":\"\", \"伤亡\":\"\"} )输出；如果某个实体里包含多项信息，请将它们用'，'隔开。\n\n### Response:")

