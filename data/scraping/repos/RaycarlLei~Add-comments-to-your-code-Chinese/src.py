import os
import time
import ctypes
import threading
import shutil
import datetime
import tkinter as tk
from tkinter import filedialog
import tkinter.scrolledtext as tkst
from tkinter import ttk
from tkinter import messagebox
import webbrowser
from tkinter import *
import subprocess

def install_package(package):
    subprocess.check_call(["pip", "install", package])

try:
    import openai
except ImportError:
    output_text.insert(tk.END, f'下载openai包失败\n')
    exit()

try:
    install_package("openai") 
except subprocess.CalledProcessError:
    output_text.insert(tk.END, f'安装openai包失败\n')
    exit()

def open_link(event, link):
    webbrowser.open(link) 

def update_time():
    global update
    if update:
        total_time = time.time() - start_time
        minutes = (int)(total_time // 60)
        seconds = (int)(total_time % 60)
        time_label.config(text=f"当前运行时间 {minutes}:{seconds}")
        total_time += 1
        time_label.after(1000, update_time)
    else:
        total_time = 0
        time_label.config(text="00:00")

        

def read_file(file_path, encoding):
    global code, output_text
    try:
        if not os.path.exists(file_path):
            output_text.insert(tk.END, f'读取文件地址{file_path}失败，请检测地址是否正确 \n')
            raise Exception("所读取的文件不存在File not exists")
        with open(file_path, 'r', encoding=encoding) as input_file:
            lines = input_file.readlines()
            for line in lines:
                if len(line) > 1000:
                    raise Exception("所读取文件中有超过1000字符长度的行。\nLine exceeds 1000 characters")
            code = ''.join(lines)
        output_text.insert(tk.END, f'文件使用{encoding}编码打开成功。\nfile opened with {encoding} successfully\n')
    except UnicodeDecodeError:
        output_text.insert(tk.END, f'文件使用{encoding}编码打开失败。\nfile can`t open with {encoding}\n')
        code = ''
    except Exception as e:
        output_text.insert(tk.END, f'错误 Error: {str(e)}\n')
        code = ''
    return code

def split_file(code):
    global parts, part
    parts = []
    part = ''

    if len(code) > 0:
        lines = code.split('\n')
        for line in lines:
            if len(part) + len(line) > 1000:
                parts.append(part)
                part = line + '\n'
            else:
                part += line + '\n'
        if part:
            parts.append(part)

    count = len(parts)
    output_text.insert(tk.END, f'已切割为{count}个文本块。\nfile splitted into {count} parts successfully\n')
    return parts, count

def write_api_key(event=None):
    api_key = api_key_entry.get()
    with open(api_file_path, "w") as file:
        file.write(api_key)

def write_file(formatted_now, annotated_code):
    global output_filename
    output_filename = f'code_out_{formatted_now}.txt'
    # 添加当前路径到文件名
    output_path = os.path.join(os.getcwd(), output_filename)
    
    # 检查要写入的内容是否为空
    if annotated_code:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(annotated_code)
        output_text.insert(tk.END, f'文件写出成功。file written successfully to {output_path}\n')
        return output_path
    else:
        output_text.insert(tk.END, '没有输出文件。Cannot write empty content to file.\n')
        return None


def save_text():
    custom_prompt_entry = entry.get()
    window.destroy()


def stop_annotation_code():
    global stop_thread
    stop_thread = True
    update_time()

def browse_file(file_path_entry):
    global filename
    filename = filedialog.askopenfilename()
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, filename)

def browse_folder(folder_path_entry):
    global foldername
    foldername = filedialog.askdirectory()
    folder_path_entry.delete(0, tk.END)
    folder_path_entry.insert(0, foldername)

def handle_selection(event):
    if prompt_combobox.get() == '自定义':
        global entry, window
        window = tk.Toplevel(root)
        window.title('自定义prompt')
        
        entry = tk.Entry(window, width=30)
        entry.pack()
        
        save_button = tk.Button(window, text='保存', command=save_text)
        save_button.pack()
        
        close_button = tk.Button(window, text='退出', command=lambda: window.destroy())
        close_button.pack()


def annotate_code(parts):
    
    global annotated_code, total_time, part_time, minutes, seconds, model_engine, stop_thread, start_time
    minutes = 0
    seconds = 0
    annotated_code = ''
    model_engine = 'text-davinci-003'
    output_text.insert(tk.END, f'模型引擎 {model_engine} 选择成功。\nmodel engine {model_engine} selected successfully\n')
    start_time = time.time()
    global update
    update = True
    update_time()

    progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=600, mode='determinate')
    progress_bar.pack()

    for i, part in enumerate(parts):
        if stop_thread:
            total_time = time.time() - start_time
            output_text.insert(tk.END, f'线程被终止\n总共花费时间：{total_time:.2f} 秒\nThread stopped\nTotal time taken: {total_time:.2f} seconds\n')
            stop_thread = False
            break
    
        if prompt_combobox_value == '添加中文代码注释':
            prompt = f'Add chinese code comments for the input. If you believe no comments are necessary, respond with the original input. You can only add comments to the input. You must not delete or modify any part of input. Your response should not include any explanations outside of code. If you believe no comments are necessary, respond with the original input. You can only add comments to the input. You can not delete or modify any part of input. Here is the input:""\n{part}\n""'
        elif prompt_combobox_value == '添加英文代码注释':
            prompt = f'Add English code comments for the input. If you believe no comments are necessary, respond with the original input. You can only add comments to the input. You must not delete or modify any part of input. Your response should not include any explanations outside of code. If you believe no comments are necessary, respond with the original input. You can only add comments to the input. You can not delete or modify any part of input. Here is the input:""\n{part}\n""'
        elif prompt_combobox_value == '自定义':
            prompt = custom_prompt_entry+f'""\n{part}\n""'
        part_time = time.time()
        output_text.insert(tk.END, f'Part {i+1}/{len(parts)} started at {time.strftime("%H:%M:%S", time.localtime())}\n')

        completions = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.5,
        )

        output_text.insert(tk.END, f'第 {i+1}/{len(parts)} 部分成功进行了注释。\nPart {i+1}/{len(parts)} annotated successfully\n')
        output_text.insert(tk.END, f'完成该部分所花费的时间为：{time.time() - part_time:.2f} 秒\nTime taken for this part: {time.time() - part_time:.2f} seconds\n\n\n')

        progress_bar['value'] = (i+1) / len(parts) * 100

        annotated_code += completions.choices[0].text

    total_time = time.time() - start_time
    minutes = total_time // 60
    seconds = total_time % 60
    update = False
    progress_bar.destroy()

    return annotated_code, total_time, part_time, minutes, seconds


def run_script():
    global file_path_entry, encoding_combobox, prompt_combobox_value, address_out, minutes, seconds, output_text, debug_mode, prompt_combobox

    prompt_combobox_value = prompt_combobox.get()
    file_path = file_path_entry.get()
    encoding = encoding_combobox.get()

    now = datetime.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")

    if debug_mode.get() == 1:
        folder_name = f"splitted_code_{formatted_now}"
        os.makedirs(folder_name, exist_ok=True) 

    code = read_file(file_path=file_path, encoding=encoding)
    parts, count = split_file(code=code)

    try:
        thread = threading.Thread(target=annotate_code, args=(parts,))
        thread.start()
        thread.join()
    except Exception as e:
        output_text.insert(tk.END, f'Error: {e}\n')
        return
    

    #如果没有响应内容，提前结束
    if annotate_code == None:
        return

    output_path = write_file(formatted_now, annotated_code)

    destination_path = os.path.join(address_out.get(), os.path.basename(output_path))
    if os.path.abspath(output_path) != os.path.abspath(destination_path):  # 检查文件路径是否相同
        shutil.copy(output_path, destination_path)

    os.startfile(destination_path)

    output_text.insert(tk.END, f'运行用时 {minutes:.0f} 分 {seconds:.2f} 秒\n一共切片{count}份，每次请求平均用时{(total_time/count):.2f}秒')

def run_script_thread():
    api_key = api_key_entry.get()
    try:
        openai.api_key = api_key
    # 测试API key可用性
        openai.Engine.list()
    except openai.error.AuthenticationError:
        output_text.insert(tk.END, "你输入的OpenAI API密钥错误或者失效了\n")
    

    global stop_thread
    stop_thread = False
    folder_path = os.getcwd()
    file_path = os.path.join(folder_path, 'api_key.txt')
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(api_key_entry.get())

    thread = threading.Thread(target=run_script)
    thread.start()
    

def run_main():
    global root
    root = tk.Tk()
    root.title("代码注释助手 Code Commentator")
    root.geometry("650x880")
    root.resizable(width=1, height=1)

    global file_path_entry, api_key_entry, encoding_combobox, output_text, address_out, debug_mode, folder_path, prompt_combobox, custom_prompt_entry
    global stop_thread
    stop_thread = False
    custom_prompt_entry = 'You msut response with the original input. Here is the original input:'
    

    blank_1 = tk.Label(root, text="\n")
    blank_1.pack()

    github_label = tk.Label(root, text="点我查看作者主页", fg="blue", cursor="hand2")
    github_label.bind("<Button-1>", lambda event: open_link(event, "https://github.com/RaycarlLei"))
    github_label.pack()

    blank_2 = tk.Label(root, text="\n")
    blank_2.pack()

    link_label = tk.Label(root, text="点我获取OpenAI API Key", fg="blue", cursor="hand2")
    link_label.bind("<Button-1>", lambda event: open_link(event, "https://beta.openai.com/"))
    link_label.pack()

    api_key_label = tk.Label(root, text="在下方输入你的OpenAI API Key。如果你还没有，请点上方链接获取。")
    api_key_label.pack()

    api_key_entry = tk.Entry(root, width=55)
    api_key_entry.pack()

    folder_path = os.getcwd()
    global api_file_path
    api_file_path = os.path.join(folder_path, "api_key.txt")

    if os.path.exists(api_file_path):
        with open(api_file_path, "r") as file:
            api_key = file.read().strip()
    else:
        api_key = ""
        with open(api_file_path, "w") as file:
            file.write(api_key)


    api_key_entry.insert(0, api_key)
    api_key_entry.bind("<KeyRelease>", write_api_key)

    address_out_label = tk.Label(root, text="\n输出文件路径：")
    address_out_label.pack()
    address_out = tk.Entry(root, width=80)
    address_out.pack()

    folder_path = os.path.dirname(os.path.abspath(__file__))
    address_out.insert(0, folder_path)

    browse_button_out = tk.Button(root, text="浏览", command=lambda: browse_folder(address_out))
    browse_button_out.pack()

    file_path_label = tk.Label(root, text="\n读取文件路径：")
    file_path_label.pack()
    file_path_entry = tk.Entry(root, width=80)
    file_path_entry.pack()

    browse_button_in = tk.Button(root, text="浏览", command=lambda: browse_file(file_path_entry))
    browse_button_in.pack()

    encoding_label = tk.Label(root, text="读取文件使用编码：")
    encoding_label.pack()

    encoding_combobox = ttk.Combobox(root, values=['utf-8', 'latin-1', 'cp1252'], width=7)
    encoding_combobox.set('utf-8')
    encoding_combobox.pack()

    prompt_combobox = ttk.Combobox(root, values=['添加中文代码注释', '添加英文代码注释','自定义'], width=14)
    prompt_combobox.set('添加中文代码注释')
    prompt_combobox.bind('<<ComboboxSelected>>', handle_selection)
    prompt_combobox.pack()



    debug_mode = tk.IntVar(value=0)
    debug_checkbox = tk.Checkbutton(root, text="**调试**将切片结果保存到文件夹中", variable=debug_mode)
    debug_checkbox.pack()

    run_button = tk.Button(root, text="运行脚本", command=run_script_thread)
    run_button.pack()

    global time_label
    time_label = tk.Label(root, text=" ")
    time_label.pack()

    stop_button = tk.Button(root, text="停止运行", command=stop_annotation_code)
    stop_button.pack()

    output_text = tk.Text(root, name="自动注释脚本", height=20, width=80)
    output_text.pack()

    root.mainloop()

run_main()
