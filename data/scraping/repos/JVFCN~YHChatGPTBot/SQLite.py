import ast
import sqlite3
import threading

import dotenv

import OpenAI

# init
ThreadLocal = threading.local()
Connection = sqlite3.connect("data/Yunhu.db")
Cursor = Connection.cursor()
ChatInitContent = '[{\"role\": \"system\", \"content\": \"You are ChatGPT, a large language model trained by ' \
                  'OpenAI.Knowledge cutoff: 2021-09\"}]'
# 数据库初始化
Cursor.execute(
    "CREATE TABLE IF NOT EXISTS user_chat_info ("
    "userId INTEGER PRIMARY KEY,"
    "api_key TEXT NOT NULL DEFAULT 'defaultAPIKEY',"
    "admin BOOLEAN NOT NULL DEFAULT FALSE,"
    "chat TEXT NOT NULL DEFAULT '[{\"role\": \"system\", \"content\": \"You are ChatGPT, a large language model trained by OpenAI.Knowledge cutoff: 2021-09\"}]',"
    "model TEXT NOT NULL DEFAULT 'gpt-3.5-turbo',"
    "premium BOOLEAN NOT NULL DEFAULT FALSE,"
    "premium_expire INTEGER NOT NULL DEFAULT 0,"
    "free_times INTEGER NOT NULL DEFAULT 10"
    ")"
)  # 创建用户聊天信息表
Connection.commit()


# 获取用户的模型
def GetUserModel(UserId):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()

    Cursor_.execute("SELECT model FROM user_chat_info WHERE userId=?", (UserId,))  # 获取模型
    result = Cursor_.fetchone()
    return result[0]


# 用户是否为会员
def IsPremium(UserId):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()

    Cursor_.execute("SELECT premium FROM user_chat_info WHERE userId=?", (UserId,))
    result = Cursor_.fetchone()
    return bool(result[0])


# 获取用户会员到期时间
def GetPremiumExpire(UserId):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()

    Cursor_.execute("SELECT premium_expire FROM user_chat_info WHERE userId=?", (UserId,))
    result = Cursor_.fetchone()
    return result[0]


# 设置用户会员状态
def SetPremium(UserId, Premium, ExpireTime):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()
    """
    :param UserId: 用户ID
    :param Premium: 会员状态
    :param ExpireTime: 会员到期时间
    :return: None
    """
    Cursor_.execute(
        "UPDATE user_chat_info SET premium = ?, premium_expire = ? WHERE userId = ?",
        (Premium, ExpireTime, UserId)
    )  # 更新会员状态
    Connection_.commit()


# 更改用户的模型
def SetUserModel(UserId, Model):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()

    Cursor_.execute(
        "UPDATE user_chat_info SET model = ? WHERE userId=?", (Model, UserId,)
    )  # 更新模型
    Connection_.commit()


# 更新用户的ApiKey
def UpdateApiKey(UserId, NewApiKey):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()

    Cursor_.execute(
        "UPDATE user_chat_info SET api_key = ? WHERE userId = ?",
        (NewApiKey, UserId)
    )  # 更新ApiKey
    Connection_.commit()


# 更新用户的上下文
def UpdateUserChat(UserId, UpdatedChat):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()

    ChatString = str(UpdatedChat)  # 转换为字符串
    Cursor_.execute(
        "UPDATE user_chat_info SET chat = ? WHERE userId = ?",
        (ChatString, UserId)
    )  # 更新聊天记录
    Connection_.commit()


# 获取用户的上下文
def GetUserChat(UserId):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()
    Cursor_.execute("SELECT chat FROM user_chat_info WHERE userId=?", (UserId,))  # 获取聊天记录
    result = Cursor_.fetchone()
    ChatHistory = ast.literal_eval(result[0])
    if len(ChatHistory) > 6:  # 限制最大长度6
        ChatHistory.pop(1)  # 删除第一个元素
    print(ChatHistory)
    return ChatHistory  # 返回聊天记录


# 添加用户
def AddUser(UserId):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()
    Cursor_.execute(
        "INSERT OR IGNORE INTO user_chat_info (userId, api_key, admin, chat, model, premium, premium_expire, free_times) VALUES (?, ?, ?, ?, ?,?, ?,?)",
        (UserId, "defaultAPIKEY",False ,ChatInitContent, "gpt-3.5-turbo", False, 0, 10)
    )
    Connection_.commit()


# 获取用户的免费次数
def GetUserFreeTimes(UserId):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()

    Cursor_.execute("SELECT free_times FROM user_chat_info WHERE userId=?", (UserId,))
    result = Cursor_.fetchone()
    print(result)
    return result[0]


# 更改某用户的免费次数
def SetUserFreeTimes(UserId, FreeTimes):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()

    Cursor_.execute(
        "UPDATE user_chat_info SET free_times = ? WHERE userId = ?",
        (FreeTimes, UserId)
    )
    Connection_.commit()


# 更改所有用户的免费次数
def SetAllUserFreeTimes(FreeTimes):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()

    Cursor_.execute(
        "UPDATE user_chat_info SET free_times = ?",
        (FreeTimes,)
    )
    Connection_.commit()


# 重置所有用户的模型
def SetAllUserModel():
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()
    Cursor_.execute("SELECT userId FROM user_chat_info")
    UserIds = Cursor_.fetchall()

    for user_id in UserIds:
        Cursor_.execute(
            "UPDATE user_chat_info SET model = ? WHERE userId = ?",
            ("gpt-3.5-turbo", user_id[0])
        )

    Connection_.commit()


# 为用户设置admin权限
def SetUserPermission(UserId, IsAdmin):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()
    Cursor_.execute("UPDATE user_chat_info SET admin=? WHERE userId=?", (IsAdmin, UserId))
    Connection_.commit()


# 清除所有用户的上下文
def ClearAllUsersChat():
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()

    # 获取所有用户ID
    Cursor_.execute("SELECT userId FROM user_chat_info")
    UserIds = Cursor_.fetchall()

    # 遍历用户ID并清除聊天记录
    for user_id in UserIds:
        Cursor_.execute(
            "UPDATE user_chat_info SET chat = ? WHERE userId = ?",
            (ChatInitContent, user_id[0])
        )

    Connection_.commit()


# 清除用户的上下文(到默认状态)
def ClearUserChat(UserId):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()

    Cursor_.execute(
        "UPDATE user_chat_info SET chat = ? WHERE userId = ?",
        (ChatInitContent, UserId)
    )

    Connection_.commit()


# 检查用户是否有admin权限
def CheckUserPermission(UserId):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()
    Cursor_.execute("SELECT admin FROM user_chat_info WHERE userId=?", (UserId,))
    result = Cursor_.fetchone()
    if result is not None:
        return bool(result[0])
    else:
        return False


# 获取所有用户的id
def GetAllUserIds():
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()

    Cursor_.execute("SELECT userId FROM user_chat_info")

    # 将所有用户的id转为列表
    UserIds = [str(row[0]) for row in Cursor_.fetchall()]

    return UserIds


# 获取数据库连接
def GetDbConnection():
    if not hasattr(ThreadLocal, "connection"):
        ThreadLocal.connection = sqlite3.connect("data/Yunhu.db")
    return ThreadLocal.connection


# 获取用户的ApiKey
def GetApiKey(UserId):
    Connection_ = GetDbConnection()
    Cursor_ = Connection_.cursor()
    Cursor_.execute("SELECT api_key FROM user_chat_info WHERE userId = ?", (UserId,))
    result = Cursor_.fetchone()

    if result:
        return result[0]


# 设置所有用户的默认ApiKey
def SetDefaultApiKey(Key):
    dotenv.set_key("./data/.env", "DEFAULT_API", Key)
    OpenAI.DefaultApiKey = Key
    dotenv.load_dotenv()


# 关闭数据库连接
def CloseDbConnections():
    if hasattr(ThreadLocal, "connection"):
        ThreadLocal.connection.close()
