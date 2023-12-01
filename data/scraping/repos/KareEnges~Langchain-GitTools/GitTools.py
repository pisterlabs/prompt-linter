import subprocess

from langchain.tools import StructuredTool


def git_diff(file):
    """用git获取file对应的文件的修改记录，例如其为main.py时返回main.py的修改记录，如果文件不在根目录应该要写相对路径"""
    command = ["git", "diff", file]
    output = subprocess.check_output(command, universal_newlines=True, encoding="utf-8")
    return output


def git_init():
    """初始化一个Git仓库，直接调用即可在默认根目录创建一个仓库"""
    command = ["git", "init"]
    subprocess.check_call(command)


def git_add(file_path):
    """把file_path对应的文件纳入git跟踪,例如其为main.py时加入main.py做跟踪，如果文件不在根目录应该要写相对路径"""
    command = ["git", "add", file_path]
    subprocess.check_call(command)


def git_commit(message):
    """git提交代码，并且附上message作为更改的记录"""
    command = ["git", "commit", "-m", message]
    subprocess.check_call(command)


def git_log(file_path=None):
    """获取git仓库提交日志"""
    command = ["git", "log"]
    if file_path:
        command.append("--follow")
        command.append(file_path)
    output = subprocess.check_output(command, universal_newlines=True)
    return output


diff = StructuredTool.from_function(git_diff)
init = StructuredTool.from_function(git_init)
log = StructuredTool.from_function(git_log)
commit = StructuredTool.from_function(git_commit)
add = StructuredTool.from_function(git_add)


def gettools():
    return [diff, init, log, commit, add]


"""
if __name__ == "__main__":
    file_path = "main.py"  # 替换为你要跟踪的文件路径
    diff_output = git_diff(file_path)
    print(diff_output)
"""
