import os
import subprocess
from datetime import datetime

# Git 的绝对路径
GIT_PATH = r"C:\Program Files\Git\cmd\git.exe"


def run_git_backup():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"Auto-backup: {now}"

    try:
        # 确保在脚本所在目录运行
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        print(f"🚀 正在备份 our_version7...")

        # 1. 添加改动
        subprocess.run([GIT_PATH, "add", "."], check=True)

        # 2. 提交
        # 这里 check=False 是为了防止没有新文件时脚本报错中断
        subprocess.run([GIT_PATH, "commit", "-m", commit_message], check=False)

        # 3. 推送
        # 因为你现在本地和远程都叫 main，直接 push 即可
        print("📤 正在同步至 GitHub MLC 仓库...")
        subprocess.run([GIT_PATH, "push"], check=True)

        print(f"\n✅ 备份成功！时间：{now}")

    except Exception as e:
        print(f"\n❌ 备份过程中出错: {e}")


if __name__ == "__main__":
    run_git_backup()