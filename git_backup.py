import os
import subprocess
from datetime import datetime

# 推荐使用 cmd 目录下的 git.exe，兼容性最强
GIT_PATH = r"C:\Program Files\Git\cmd\git.exe"


def run_git_backup():
    # 获取当前时间，标注 SCMM 项目
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"Auto-backup at {now} (SCMM Project)"

    try:
        print(f"🚀 开始备份项目...")

        # 1. 确保在脚本所在的文件夹执行
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # 2. 执行 git add
        subprocess.run([GIT_PATH, "add", "."], check=True)
        print("✅ 已扫描并添加文件 (遵循 .gitignore)")

        # 3. 执行 git commit
        subprocess.run([GIT_PATH, "commit", "-m", commit_message], check=False)
        print(f"✅ 已存入本地仓库")

        # 4. 执行 git push
        print("📤 正在上传到 GitHub...")
        # 这里的 push 会自动推送到你之前配置好的 origin main
        subprocess.run([GIT_PATH, "push"], check=True)

        print(f"\n🎉 备份成功！")
        print(f"时间: {now}")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("提示: 确认你之前是否已经成功执行过 git push -u origin master:main --force")


if __name__ == "__main__":
    run_git_backup()