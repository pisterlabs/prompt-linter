from tkinter.ttk import Combobox
from windows import *
import openai
import os
from Box_Message import Message_box
login_file_path = "../API_KEY/API_KEY"

def Latest_API_KEY():
    # 读取../API_KEY/API_KEY
    try:
        with open(login_file_path, "r", encoding="utf-8") as f:
            api_key = f.read()
    except:
        Message_box.config(state=tk.NORMAL)
        Message_box.delete(0.0, tk.END)
        Message_box.insert(tk.END, "API_KEY文件不存在\n")
        Message_box.config(state=tk.DISABLED)
        return
    # 读取最后一行数据（字典）
    api_key = api_key.split("\n")[-2]
    # 转换为字典
    api = eval(api_key)
    # 读取
    openai.api_key = api["API_KEY"]
    openai.api_base = api["API_BASE"]
    # 显示
    Message_box.config(state=tk.NORMAL)
    Message_box.delete(0.0, tk.END)
    Message_box.insert(tk.END, "已登陆最新API_KEY\n"+"API note: "+api["API_NAME"]+"\n")
    Message_box.config(state=tk.DISABLED)


def Reset_API_KEY():
    def login_api():
        key = api_key_entry.get()
        base = api_base_entry.get()
        note = api_name_entry.get()
        # 读取:如果是空的，或者都是空格，或者是None
        if base == "" or base.isspace() or base is None:
            base = "https://api.openai.com/v1"
        openai.api_key = key
        openai.api_base = base
        if note == "" or note.isspace() or note is None:
            # 赋值为当前时间
            import time
            note = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 初始写入
        with open(login_file_path, "r+", encoding="utf-8") as f:
            #滚到最后一行
            f.seek(0, 2) #第1个参数代表偏移量，第2个参数代表起始位置，0代表文件开头，1代表当前位置，2代表文件结尾
            f.write("\n")
            f.write(str({"API_KEY": key, "API_BASE": base, "API_NAME": note}))

        #读取，删除所有空行
        with open(login_file_path, "r", encoding="utf-8") as f:
            api_key = f.read()
            api_key = api_key.split("\n")
            # 删除所有空行
            while "" in api_key:
                api_key.remove("")

        # 重新写入
        with open(login_file_path, "w", encoding="utf-8") as f:
            for i in api_key:
                f.write(i+"\n")
        # 关闭
        login.destroy()
        # 显示
        Message_box.config(state=tk.NORMAL)
        Message_box.delete(0.0, tk.END)
        Message_box.insert(tk.END, "已登陆\n"+"API note: "+note+"\n")
        Message_box.config(state=tk.DISABLED)

    login = tk.Toplevel() # 创建一个子窗口
    login.geometry("300x200")
    login.title("Login")
    api_key_label = tk.Label(login, text="API_KEY(*):")
    api_key_entry = tk.Entry(login)
    api_base_label = tk.Label(login, text="API_BASE(option):") # 可选
    api_base_entry = tk.Entry(login)
    api_name_label = tk.Label(login, text="API_NOTE(option):")
    api_name_entry = tk.Entry(login)

    login_button = tk.Button(login, text="Login", command=lambda: login_api())

    api_key_label.pack()
    api_key_entry.pack()
    api_base_label.pack()
    api_base_entry.pack()
    api_name_label.pack()
    api_name_entry.pack()
    login_button.pack()


def choose_API_KEY():
    # 读取../API_KEY/API_KEY
    try:
        with open(login_file_path, "r", encoding="utf-8") as f:
            api_key_choose = f.read()
            api_key_choose = api_key_choose.split("\n")
            # 删除所有空行
            while "" in api_key_choose:
                api_key_choose.remove("")

        api_name_list_choose = []
        api_key_list_choose = []
        api_base_list_choose = []
        for i in api_key_choose:
            api_choose_list = eval(i)
            api_name_list_choose.append(api_choose_list["API_NAME"])
            api_key_list_choose.append(api_choose_list["API_KEY"])
            api_base_list_choose.append(api_choose_list["API_BASE"])
    except:
        Message_box.config(state=tk.NORMAL)
        Message_box.delete(0.0, tk.END)
        Message_box.insert(tk.END, "API_KEY文件错误\n")
        Message_box.config(state=tk.DISABLED)

    choosekey = tk.Toplevel()  # 创建一个子窗口
    choosekey.geometry("300x100")
    choosekey.title("选择 API_KEY")
    #创建标签
    Label_choose=tk.Label(choosekey,text="选择API_KEY")
    choosekey_var = tk.StringVar()
    # 下拉框
    api_name_option = Combobox(choosekey, values=api_name_list_choose, state="readonly",textvariable=choosekey_var)
    def confirm():
        openai.api_key = api_key_list_choose[api_name_list_choose.index(choosekey_var.get())]
        openai.api_base = api_base_list_choose[api_name_list_choose.index(choosekey_var.get())]
        # 显示
        Message_box.config(state=tk.NORMAL)
        Message_box.delete(0.0, tk.END)
        Message_box.insert(tk.END, "已登陆\n" + "API note: " + api_name_option.get() + "\n")
        Message_box.config(state=tk.DISABLED) \
        # 关闭
        choosekey.destroy()

    # 确认按钮
    confirm_button = tk.Button(choosekey, text="确认", command=lambda: confirm())
    # 下拉框变量
    Label_choose.pack()
    api_name_option.pack()
    confirm_button.pack()
    choosekey.mainloop()


def Delete_API_KEY():
    # 读取../API_KEY/API_KEY
    try:
        with open(login_file_path, "r", encoding="utf-8") as f:
            api_keys = f.read()
            api_keys = api_keys.split("\n")
            # 删除所有空行
            while "" in api_keys:
                api_keys.remove("")

        api_name_delete = []
        for i in api_keys:
            api_name_delete.append(eval(i)["API_NAME"])
    except:
        Message_box.config(state=tk.NORMAL)
        Message_box.delete(0.0, tk.END)
        Message_box.insert(tk.END, "API_KEY文件不存在\n")
        Message_box.config(state=tk.DISABLED)
    def delete():
        # 读取最新的API_KEY
        with open(login_file_path, "r", encoding="utf-8") as f:
            api_key_d = f.read()
        # 转换为列表
        api_key_d = api_key_d.split("\n")
        #删除所有空行
        while "" in api_key_d:
            api_key_d.remove("")
        delete_name=api_name_option.get()
        new_api_key_d=[]
        # 删除
        for i in api_key_d:
            if eval(i)["API_NAME"]!=delete_name:
                    new_api_key_d.append(i)
            if eval(i)["API_NAME"]==delete_name:
                    pass
        #重新写入
        with open(login_file_path, "w", encoding="utf-8") as f:
            for i in new_api_key_d:
                f.write(i+"\n")

        # 显示
        Message_box.config(state=tk.NORMAL)
        Message_box.delete(0.0, tk.END)
        Message_box.insert(tk.END, "已删除\n" + "API note: " + api_name_option.get() + "\n")
        Message_box.config(state=tk.DISABLED) \
        # 关闭窗口
        Deletekey.destroy()

    Deletekey = tk.Toplevel()  # 创建一个子窗口
    Deletekey.geometry("300x100")
    Deletekey.title("选择 API_KEY")
    # 创建标签
    Label_Delete = tk.Label(Deletekey, text="选择API_KEY")
    choosekey_var = tk.StringVar()
    # 下拉框
    api_name_option = Combobox(Deletekey, values=api_name_delete, state="readonly", textvariable=choosekey_var)
    # 确认按钮
    confirm_button = tk.Button(Deletekey, text="删除", command=lambda: delete())

    Label_Delete.pack()
    api_name_option.pack()
    confirm_button.pack()
    Deletekey.mainloop()



# -----------------------------------------------------------------------------------#
# 创建一个菜单
filemenu_login = tk.Menu(menubar, tearoff=0)
# 设置单选
filemenu_login.add_command(label="Latest API_KEY", command=lambda: Latest_API_KEY())
filemenu_login.add_command(label="Create API_KEY", command=lambda: Reset_API_KEY())
filemenu_login.add_command(label="Choose API_KEY", command=lambda: choose_API_KEY())
filemenu_login.add_separator()
filemenu_login.add_command(label="Delete API_KEY", command=lambda: Delete_API_KEY())
# -----------------------------------------------------------------------------------#
if os.path.exists(login_file_path):
    with open(login_file_path, "r", encoding="utf-8") as f:
        api_key = f.read()
        api_key = api_key.split("\n")
        # 删除所有空行
        while "" in api_key:
            api_key.remove("")
    # 读取每行数据中的API_NAME
    api_name = []
    api_key_list = []
    api_base_list = []
    for i in api_key:
        api_name.append(eval(i)["API_NAME"])
        api_key_list.append(eval(i)["API_KEY"])
        api_base_list.append(eval(i)["API_BASE"])
    api_key_init = api_key[0]
    # 转换为字典
    api = eval(api_key_init)
    # 读取
    openai.api_key = api["API_KEY"]
    openai.api_base = api["API_BASE"]
    # 显示
    Message_box.config(state=tk.NORMAL)
    Message_box.delete(0.0, tk.END)
    Message_box.insert(tk.END, "已登录 \nAPI note: " + api["API_NAME"] + "\n")
    Message_box.config(state=tk.DISABLED)
else:
    Message_box.config(state=tk.NORMAL)
    Message_box.delete(0.0, tk.END)
    Message_box.insert(tk.END, "未登录:\nAPI_KEY文件不存在\n")
    Message_box.config(state=tk.DISABLED)

if __name__=="__main__":
    pass