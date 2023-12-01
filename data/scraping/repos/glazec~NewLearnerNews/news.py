import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import datetime
from icecream import ic
import sentry_sdk
import telegram
import os
import asyncio

sentry_sdk.init(
    dsn="https://fc880ea6ee11c5613ad2eb62d9eb2bf1@o262884.ingest.sentry.io/4505684111785984",

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0
)
tg_id_dict = {"@glazecl": "828090678", "@kris_michiel": "5616258966",
              "@Newlearner365": "614953732", "张启": "5156033499", "Aaron Lou": "1149527409"}
content1 = "Apple is reportedly prepping for an iPhone 15 Pro event on September 13. According to information seen by 9to5Mac, mobile carriers have been asking employees not to take days off on September 13 due to a major smartphone announcement. The event is expected to announce the new iPhones, which will feature a new design with slightly curved edges and thinner bezels around the display, Dynamic Island and USB-C instead of Lightning, and a new periscope lens for better optical zoom on the Pro models. Pre-orders are expected to begin on September 15, with the official launch a week later on September 22. Prices of the new iPhones may rise by up to $200 compared to the current generation."  # noqa: E501

content2 = """Harmonyos 放肆玩亠起玩 有个性 全新的设计让你的照片、心情 甚至是自然天气都能成为你个性的完美表达 艺术主角’ 智能识别画面主体 多样艺术背景随心搭 海报标题’ 字体设计更多元 (i26 全新Emoji表情’ 情绪表达更多样 时尚画册’ 杂志化布局 0g:0g 趣味心情主题’ 把你的心情设省成主题 全景天气壁纸” 将自然天气装进手机，实时感知天气变化 Harmonyos 4趣味主题‘ 可爱胜胀风，更立体更好玩 (08:0g 更高效 信息处理和内容分享变得更加高效便捷 实况窗 任务进度随时掌握，可大可小可交互 更多设备更高效 弦月窗° 实时任务状态，拾腕便知 生道费中 𣊭 全新通知中心 “置顶"关键信息 轻重缓急一目了然 超级中转站° 双指长按 提取文本图片 更多应用随手拖入 多屏同享 音画同步 座舱也能一起看大片 演示批注° 可圈点可标注，多种文档不限应用 手写批注实时展示，会议交流高效轻松 游戏流转’ 痤舱畅玩手澼 航拍流转 手机航拍流转座舱 “大智慧 全新小艺，更聪明、更能干、更贴心 更聪明 智慧交互 更贴心 个性化服务 从准确指令 到自然对话 随口记事 适时提醒 全新小艺” 大有不同 好的．邦徐记佳了 场景化服务组合 每杂场景 轻松绾排 更能干 高效生产力 资讯内容 快速摘要 看图说话 服务直达 文家创作 内容辅助 照片趣玩 玩出创意 畅快玩 全新华为方舟引擎带来图形、多媒体、内存、调度、存储、低功耗 六大引擎技术，多设备也流畅 华为方舟引擎1。 图形 寧 多媒恷引擎 肉葆 𦘦 调度引禁 低功栝引禁 20 % 个 更流畅 30分钟个 更省电 超安心 风险更可视，用机更安心 全新应用管控中心。 更可视 更安心 新无风险 风險成用安装 拦截提醒 风险应用安装 自动管控。 风险应用安装 最小授予 凤險成用运行 主动抬截 应用跟踪管理。 是否允许应用跟踪你在其他应用和网站上的活动 由你说了算"""  # noqa: E501""

content3 = "Apple reported its third quarter results for fiscal 2023, ending July 1, 2023. The Company posted quarterly revenue of $81.8 billion, down 1 percent year over year, and quarterly earnings per diluted share of $1.26, up 5 percent year over year. Services revenue reached an all-time high, and the installed base of active devices set an all-time record. The board of directors declared a cash dividend of $0.24 per share of the Company’s common stock, payable on August 17, 2023. Apple also provided live streaming of its Q3 2023 financial results conference call. The Company is continuing to advance its values, while championing innovation that enriches the lives of customers and leaves the world better than it found it."  # noqa: E501

content4 = "AMD has announced the release of two new graphics cards, the Radeon Pro W7600 and W7500, as part of their mid-range professional video card lineup. These cards are based on AMD's RDNA 3 architecture and offer advanced features such as AV1 encoding support, improved compute and ray tracing throughput, and DisplayPort 2.1 outputs. The Radeon Pro W7600 is a full-height card with 32 compute units and a boost clock of 2.43GHz, while the W7500 is a sub-75W card that can be powered entirely by a PCIe slot. Both cards are expected to hit the market later this quarter."  # noqa: E501

content5 = "Nikon has issued an important notice to its users regarding a potential issue with the Z 8 digital mirrorless camera. Some users have reported that the camera strap lug may become loose or detach from the camera body under certain conditions. Nikon has identified the affected serial numbers and will provide free repairs and cover shipping costs, regardless of whether the camera is still under warranty. Users are advised to contact Nikon's customer support hotline or service centers for assistance. This issue also includes cameras that were previously identified for a different problem related to lens rotation."  # noqa: E501

content6 = " 一加手机岛 23-8-4 08:31 发布于广东 来自一加Ace2Pro 性能巅峰 淘汰 8GB， 12GB 起步， 16GB 普及， 24GB 引领，还有呢？ #一加Ace2Pro#，2023年8月，敬请期待。 关注＋转评，抽送一个一加真无线耳机。 8GB 淘汰 12GB 起步 116GB 普及 24GB 引领"  # noqa: E501

microsoft_content_input = """Hello Windows Insiders, today we are releasing Windows 11 Insider Preview Build 23521 to the Dev Channel.

What’s new in Build 23521
Changes and Improvements
[Windows 365]
Windows Insiders in the Dev and Beta Channels can participate in the public preview of Windows 365 Switch. Windows 365 Switch provides the ability to easily move between a Windows 365 Cloud PC and the local desktop using the same familiar keyboard commands, as well as a mouse-click or a swipe gesture through Task View on the Windows 11 taskbar. Please read this blog post for all the details on how to participate.
Easily switch between a Windows 365 Cloud PC and the local desktop via Task View.
Easily switch between a Windows 365 Cloud PC and the local desktop via Task View.
[Windows Copilot]
Windows Insiders in the Dev Channel who login and are managed by AAD (soon to be Microsoft Entra ID) will see Windows Copilot enabled for them again without the need to enable it via Group Policy Editor.
[Taskbar & System Tray]
To make it easier to enable never combined mode on the taskbar, we have updated the settings. You can turn never combined mode on by simply adjusting “Combine taskbar buttons and hide labels” to never. And we provide a separate setting for turning this on for other taskbars (multiple monitor scenarios for example).
Updated settings for never combined mode.
Updated settings for never combined mode.
[Dynamic Lighting]
You can now instantly sync your Windows accent color with the devices around you with the “Match my Windows accent color” toggle under “Effects” for Dynamic Lighting via Settings > Personalization > Dynamic Lighting. This improvement started rolling out in last week’s Dev Channel flight.
We have added the ability to choose a custom color to light up your devices with.
[Task Manager]
We’ve updated the Task Manager settings page to match the design principles of Windows 11. The design has a similar look and feel to the Settings in Windows 11 and provides a cleaner UI separating categories into different sections. Oh, and we updated some dialogs in Task Manager too.
Redesigned Task Manager settings.
Redesigned Task Manager settings.
[Windows Spotlight]
After doing an OS update, in certain cases such as using the default Windows 11 background or a solid color – Windows Spotlight may be enabled for you. If you decide you don’t want Windows Spotlight enabled, you can turn it off and in future OS updates it should not be enabled for you again unless you choose to re-enable the experience.
[Search on the Taskbar]
Windows Search now uses the Microsoft Bing Search app to return web content and search results. In the European Economic Area (EEA), you can enable installed Microsoft Store apps that implement a web search provider to return web content and search results in Windows Search through Settings.
[Settings]
The end task feature under System > For Developers no longer requires Developer Mode to be enabled first before it can be used.
[ADDED] Under Settings > Personalization > Device usage, we have added “Development” to the list and toggling it on will launch Dev Home. This matches what is shown in OOBE (“out of box experience”).
[Other]
In the European Economic Area (EEA), Windows will now require consent to share data between Windows and other signed-in Microsoft services. You will see some Windows features start to check for consent now, with more being added in future builds. Without consent to share data between Windows and other signed-in Microsoft services, some functionality in Windows features may be unavailable, for example certain types of file recommendations under “Recommended” on the Start menu.
Fixes
[File Explorer]
Fixed an issue where you couldn’t drag a file out of an archived folder to extract it with one of the newly supported archive formats.
Fix an issue where when extracting one of the newly supported archive formats using the Extract All option in the context menu, it wasn’t working unless Windows Explorer was set as the default for that file type.
When trying to extract one of the new archive formats and the file is password encrypted, it will now show a message saying this isn’t currently supported.
Fixed a bug where Insiders may have experienced a File Explorer crash when dragging the scroll bar or attempting to close the window during an extended file-loading process.
We fixed the following issues for Insiders who have the modernized File Explorer address bar that began rolling out with Build 23475:

Fixed an issue which was causing the search box in File Explorer to not work well with IMEs.
Fixed an issue where pasting using the context menu in the address bar wasn’t working (or other context menu actions in the address bar).
We fixed the following issues for Insiders who have the modernized File Explorer Home that began rolling out with Build 23475:

Fixed an issue where when trying to scroll with touch on Home might result in everything getting selected.
Fixed a white flash in dark theme when switching between Home and Gallery.
[Taskbar & System Tray]
Fixed an issue that removed the USB icon and its options from the system tray.
Fixed an issue where the titles were missing from taskbar previews when turning on tablet-optimized taskbar while using uncombined taskbar.
Fixed an issue where uncombined taskbar’s app indicators weren’t shown correctly after it showed something was being downloaded.
Fixed an explorer.exe crash impacting system tray reliability.
Fixed an issue where the End Task feature wasn’t working if you tried it when there were multiple windows open of that app.
Fixed an issue where using End Task on certain apps would cause other unrelated apps to close.
[HDR Backgrounds]
Fixed an issue where your HDR wallpaper might appear washed out although HDR was enabled.
Fixed an issue where it wasn’t possible to select .JXL files for your wallpaper slideshow.
[Other]
If Get Help isn’t installed, when opening one of the troubleshooters in Settings, it will now prompt you to install it, rather than showing an error about not having an app associated for the action.
NOTE: Some fixes noted here in Insider Preview builds from the Dev Channel may make their way into the servicing updates for the released version of Windows 11.

Known issues
[General]
We’re investigating reports that the taskbar isn’t loading for some Insiders when logging into their PC after installing this build. See this forum post for more details and workaround options.
We’re working on the fix for an issue causing explorer.exe to crash on the login screen when attempting to enter safe mode.
[Start menu]
Some apps under All apps on the Start menu, such as PWA apps installed via Microsoft Edge, may incorrectly be labeled as a system component.
[Search on the Taskbar]
[ADDED] Sometimes the tooltip when mousing over the search box does not match the current search highlight.
[Windows Copilot]
You can use Alt + Tab to switch out of Windows Copilot, but not back into it. Windows + C will move focus back to Windows Copilot
When first launching or after refreshing Copilot while using voice access you’ll need to use “Show grid” commands to click in the “Ask me anything” box for the first time.
[Input]
We’re investigating reports that typing with the Japanese and Chinese IMEs is not working correctly after the last flight.
Widgets Update: Pin widgets board open
We are beginning to roll out an update to Widgets for Windows Insiders in the Canary and Dev Channels that lets you pin the widgets board open, so your widgets board is always just a glance away. To pin the board open, simply click the pin icon in the top right corner of the board. Once your board is pinned open, the widgets board will no longer light dismiss.

New pin icon at the top of the widgets board to pin the widgets board open.
New pin icon at the top of the widgets board to pin the widgets board open.
While the board is pinned, you can still close it by:

Open the widgets board via the Widgets button on the taskbar.
Pressing the ESC key while Widgets is in the foreground.
Swiping on the left edge of the screen if you have a touch device.
To unpin the board, select the unpin icon in the top right corner of the board.

FEEDBACK: Please file feedback in Feedback Hub (WIN + F) under Desktop Environment > Widgets. 

Snipping Tool Update
We are beginning to roll out an update to Snipping Tool (version 11.2306.43.0 and higher) to Windows Insiders in the Canary and Dev Channels. This update introduces new buttons to edit in Paint for screenshots and edit in Clipchamp for screen recordings.

New buttons in Snipping Tool for editing screen clips in Paint and screen recordings in Clipchamp.
New buttons in Snipping Tool for editing screen clips in Paint and screen recordings in Clipchamp.
FEEDBACK: Please file feedback in Feedback Hub by clicking here.

For developers
You can download the latest Windows Insider SDK at aka.ms/windowsinsidersdk.

SDK NuGet packages are now also flighting at NuGet Gallery | WindowsSDK which include:

.NET TFM packages for use in .NET apps as described at aka.ms/windowsinsidersdk
C++ packages for Win32 headers and libs per architecture
BuildTools package when you just need tools like MakeAppx.exe, MakePri.exe, and SignTool.exe
These NuGet packages provide more granular access to the SDK and better integration in CI/CD pipelines.

SDK flights are now published for both the Canary and Dev Channels, so be sure to choose the right version for your Insider Channel.

Remember to use adaptive code when targeting new APIs to make sure your app runs on all customer machines, particularly when building against the Dev Channel SDK. Feature detection is recommended over OS version checks, as OS version checks are unreliable and will not work as expected in all cases."""  # noqa: E501


def loadWeb(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
# Format the website content as text
    formatted_text = soup.get_text()
    content = ''.join(formatted_text.split(" "))
    content = ''.join(formatted_text.split("\n"))
    ic(content)
    return content


def generate_news(url, content):
    prompt_template = """
    You are a good news writer. You can understand both English and Chinese very well. You are writing news in Chinese based on the content. Then content might be Chinese or English. You use markdown and add a link to the news. Excluding the url, ensure your answer is in 80-100 characters and contain details. Place one space between number and Chinese character. Ensure the accuracy and no misleading information. If the summary is not a news summary, return only NA.

URL: "https://9to5mac.com/2023/08/03/sources-iphone-15-event-september/",
summary: {content1},
answer:
[苹果](https://9to5mac.com/2023/08/03/sources-iphone-15-event-september/) 可能于 9 月 13 日召开秋季发布会，移动运营商强调员工不得在当天休假

URL:https://9to5mac.com/2023/08/03/sources-iphone-15-event-september/
summary: {content2}
answer:
[华为](https://weibo.com/3514064555/Nd1wycZaP) 发布鸿蒙 HarmonyOS 4 系统，新增实时活动，更新小艺 AI 并带来方舟引擎

URL:https://www.apple.com/newsroom/2023/08/apple-reports-third-quarter-results/
summary: {content3}
answer:
[苹果](https://www.apple.com/newsroom/2023/08/apple-reports-third-quarter-results/) 发布 2023 Q3 财报，营收 818 亿美元，同比下降 1%

URL:https://www.anandtech.com/show/19993/amd-announces-radeon-pro-w7600-w7500
summary: {content4}
answer:
[AMD](https://www.anandtech.com/show/19993/amd-announces-radeon-pro-w7600-w7500) 发布 Radeon PRO W7500 与 W7600 工作站显卡，售价为 $429 和 $599

URL:https://www.nikon.com.cn/sc_CN/service-and-support/service-advisory.tag/service_advisory_notices/service_advisory_for_z_8_1.dcr
summary: {content5}
answer:
[尼康](https://www.nikon.com.cn/sc_CN/service-and-support/service-advisory.tag/service_advisory_notices/service_advisory_for_z_8_1.dcr) 发布《致尊敬的尼康Z 8用户通知》，承认 Z 8 微单相机存在挂带孔松动问题，已启动召回程序


URL: https://weibo.com/3871046669/NcYF9pGgL
summary: {content6}
answer:
[一加](https://weibo.com/3871046669/NcYF9pGgL) 宣布为一加 Ace 2 Pro 手机推出 24GB 超大内存版本

URL: https://twitter.com/OpenAI/status/1687159114047291392
summary: Apologies, but I am unable to access the provided URL or any specific news content. However, I can still help you with any other questions or topics you may have.
answer:
NA

URL: {url7}
summary: {content7}
answer:

    """  # noqa: E501
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    return llm_chain({"content1": content1,
                      "content2": content2, "content3": content3, "content4": content4,
                      "content5": content5, "content6": content6, "content7": content,
                      "url7": url})


def summary(url):
    content = "url\n"+loadWeb(url)
    prompt_template = "You are a good news writer. Summarize the following news in 100 words. Ensure you get the details and accuacy. Avoid misleadings. If the news is Chinese, the summary should be Chinese News: {news}"  # noqa: E501
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    return llm_chain(content)


def generate_news_by_url(url, index=1):
    with st.spinner('生成新闻 '+str(index)+' 摘要中'):
        content = summary(url)
        st.subheader("新闻摘要 "+str(index))
        st.text(content["text"])
    with st.spinner('生成新闻 '+str(index)):
        news = generate_news(url, content)
        st.subheader("新闻 "+str(index))
        if news["text"] == "NA":
            st.write("无法抓取 URL 内容")
            st.write(url)
            sentry_sdk.capture_message({
                "message": "Can't retrieve URL content from "+url,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "context": {
                    "url": url,
                    "content": content
                }
            })
        else:
            st.write(news["text"])
    return news["text"]


async def send_telegram(message_text, user_id):
    bot = telegram.Bot(token=os.getenv("TELEGRAM_TOKEN"))
    await bot.sendMessage(chat_id=user_id, text=message_text,
                          parse_mode="MarkdownV2")


def main():
    st.title("早晚报生成器（Alpha）")
    selector = st.selectbox("选择你的输入", ["多个URL", "URL", "报道"])

    if selector == "多个URL":
        newsletterTypeSelector = st.selectbox("选择你的新闻类型", ["早报", "晚报"])
        url_input = st.text_area(
            label="输入多个 URL(Youtube, Weibo, Twitter,图片 等内容无法抓取)", value="https://twitter.com/OpenAI/status/1687159114047291392\nhttps://www.macrumors.com/2023/08/04/iphone-16-pro-stacked-camera-sensor/")
        tg_user_id = st.selectbox(
            "选择你的 TG 用户名", ["@glazecl", "@kris_michiel", "张启", "Aaron Lou", "@Newlearner365"])
        st.info("请确保至少和 @newlearner_news_bot 互动过一次，否则无法收到消息")
    elif selector == "报道":
        url_input = st.text_input(
            label="输入你的 URL", placeholder="https://blogs.windows.com/windows-insider/2023/08/10/announcing-windows-11-insider-preview-build-23521/")
        content_input = st.text_area(
            label="输入你的报道", placeholder=microsoft_content_input)
    else:
        url_input = st.text_input(f"输入你的 {selector}")
    sentry_sdk.set_context("character", {
        "url": url_input,
        "tg_user_id": tg_user_id
    })
    if st.button("生成并发送"):
        if selector == "URL":
            generate_news_by_url(url_input)
        if selector == "报道":
            with st.spinner('生成中'):
                news = generate_news(url_input, content_input)
                st.header("草稿")
                st.write(news["text"])
        elif selector == "多个URL":
            # remove empty lines in urls
            urls = [url for url in url_input.splitlines() if url != ""]
            ic(urls)
            news_content = []
            fail_url = []
            my_bar = st.progress(0, text="生成早晚报中")
            for i in range(len(urls)):
                news = generate_news_by_url(urls[i], i+1)
                if news != "NA":
                    news_content.append(news)
                else:
                    fail_url.append(urls[i])
                my_bar.progress(int((i+1)*100/len(urls)),
                                text="生成新闻 "+str(i+1))
            if len(fail_url) > 0:
                st.warning("无法获取以下 URL 的内容 "+', '.join(fail_url))
            st.subheader("草稿")
            modified_news_content = []
            for i, element in enumerate(news_content):
                if i+1 == 1:
                    modified_element = "1️⃣ "+element
                elif i+1 == 2:
                    modified_element = "2️⃣ "+element
                elif i+1 == 3:
                    modified_element = "3️⃣ "+element
                elif i+1 == 4:
                    modified_element = "4️⃣ "+element
                elif i+1 == 5:
                    modified_element = "5️⃣ "+element
                elif i+1 == 6:
                    modified_element = "6️⃣ "+element
                elif i+1 == 7:
                    modified_element = "7️⃣ "+element
                elif i+1 == 8:
                    modified_element = "8️⃣ "+element
                elif i+1 == 9:
                    modified_element = "9️⃣ "+element
                modified_news_content.append(modified_element)

            news_content = "\#News\n\n"
            if newsletterTypeSelector == "早报":
                news_content += "☀️ 自留地早报"
            else:
                news_content += "🌃 自留地晚报"
            news_content += "【" + \
                str(datetime.date.today().month) + "." + \
                str(datetime.date.today().day) + "】\n\n"
            news_content += "\n\n".join(modified_news_content)
            news_content += "\n\n频道：@ NewlearnerChannel"
            st.markdown(news_content)
            escaped_news_content = "\n\n".join(modified_news_content)
            with st.spinner('发送中'):
                escaped_news_content = '\.'.join(
                    escaped_news_content.split("."))
                escaped_news_content = '\!'.join(
                    escaped_news_content.split("!"))
                escaped_news_content = '\-'.join(
                    escaped_news_content.split("-"))
                # escaped_news_content = '\('.join(
                # escaped_news_content.split("("))
                # escaped_news_content = '\)'.join(
                # escaped_news_content.split(")"))
                escaped_news_content = '\='.join(
                    escaped_news_content.split("="))
                escaped_news_content = '\*'.join(
                    escaped_news_content.split("*"))
                escaped_news_content = '\_'.join(
                    escaped_news_content.split("_"))
                escaped_news_content = '\~'.join(
                    escaped_news_content.split("~"))
                news_content = "\#News\n\n"
                if newsletterTypeSelector == "早报":
                    news_content += "☀️ 自留地早报"
                else:
                    news_content += "🌃 自留地晚报"
                news_content += "【" + \
                    str(datetime.date.today().month) + "\." + \
                    str(datetime.date.today().day) + "】\n\n"
                news_content += escaped_news_content
                news_content += "\n\n频道：@ NewlearnerChannel"
                ic(news_content)
                ic(asyncio.run(send_telegram(
                    news_content,
                    tg_id_dict[tg_user_id])))
                st.success("发送成功")


if __name__ == "__main__":
    main()
