import tkinter as tk
import openai_audio  # 确保这个文件在同一目录下
import keyboard
import time
from tkinter import simpledialog
from tkinter.scrolledtext import ScrolledText
import logging
from get_api_key import get_api_key , save_config , set_api_data


# 创建一个日志处理器，它将日志消息发送到一个文本控件
class TextHandler(logging.Handler):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text.configure(state='normal')
            self.text.insert(tk.END, msg + '\n')
            self.text.configure(state='disabled')
            # 自动滚动到底部
            self.text.yview(tk.END)
        # 在主线程中更新文本控件
        self.text.after(0, append)


# 切换录音状态
def toggle_microphone():
    # 由于这个函数会更新GUI，我们使用root.after来安排更新
    root.after(0, _toggle_microphone)

def _toggle_microphone():
    # 这是实际更新GUI的函数
    if not openai_audio.is_recording:
        openai_audio.start_recording()
        microphone_button.config(text='停止录音')
    else:
        openai_audio.stop_recording()
        microphone_button.config(text='开始录音')

# 检测连续按两下CTRL的函数
def on_ctrl_press(event):
    global last_ctrl_press_time
    current_time = time.time()
    
    # 检查两次按键是否足够接近
    if (current_time - last_ctrl_press_time) <= double_press_threshold:
        toggle_microphone()  # 切换录音状态
    
    # 更新最后一次按键时间
    last_ctrl_press_time = current_time

# 设置API和URL的函数
def set_api_url():
    api_key = simpledialog.askstring("API设置", "请输入OpenAI API密钥:")
    url = simpledialog.askstring("API设置", "请输入API URL:")
    
    if api_key and url:
        set_api_data(api_key, url)
        save_config(api_key, url)  # 保存设置
    else:
        tk.messagebox.showerror("错误", "API密钥和URL不能为空")

# 全局变量来跟踪上一次CTRL键被按下的时间
last_ctrl_press_time = 0

# 定义连续按两下CTRL键的时间阈值（例如0.5秒）
double_press_threshold = 0.5

# 设置CTRL键监听器
hotkey_id = keyboard.on_press_key("ctrl", on_ctrl_press)

# 创建主窗口
root = tk.Tk()
root.title("语音转文字工具")

# 创建一个顶部框架来放置按钮
top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, fill=tk.X)

# 创建一个ScrolledText控件用于显示日志，并放置在主窗口
log_area = ScrolledText(root, state='disabled', height=10)
log_area.pack(side=tk.TOP, padx=0, pady=(0, 0), fill=tk.BOTH, expand=True)


# 配置日志的处理器为TextHandler
text_handler = TextHandler(log_area)
formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
text_handler.setFormatter(formatter)

# 获取日志器，并添加TextHandler作为处理器
logger = logging.getLogger()
logger.addHandler(text_handler)
logger.setLevel(logging.INFO)

# 创建设置按钮并放置在顶部框架
settings_button = tk.Button(top_frame, text="设置API和URL", command=set_api_url)
settings_button.pack(side=tk.LEFT, padx=0, pady=0)

# 设置窗口的置顶
root.wm_attributes("-topmost", 1)

# 设置窗口大小并禁止调整窗口大小
root.geometry('500x200')
root.resizable(False, False)


# 创建麦克风按钮并放置在顶部框架
microphone_button = tk.Button(top_frame, text="开始录音", command=toggle_microphone)
microphone_button.pack(side=tk.LEFT, padx=0, pady=(0, 0))

# 在应用程序关闭时调用的函数
def on_closing():
    # 注销所有热键
    keyboard.unhook(hotkey_id)
    # 销毁所有窗口
    root.destroy()

# 设置当窗口尝试关闭时的回调
root.protocol("WM_DELETE_WINDOW", on_closing)

# 运行主循环
root.mainloop()
