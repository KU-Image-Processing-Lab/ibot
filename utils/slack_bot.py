import argparse
import json
import os
import platform

import requests
from dotenv import load_dotenv

load_dotenv()


class Slack_bot:
    def __init__(self, icon=":gorilla:"):
        self.icon = icon

    def __call__(self, msg):
        WEB_HOOK_URL = os.environ.get("SLACK_WEB_HOOK_URL")

        requests.post(
            WEB_HOOK_URL,
            data=json.dumps(
                {
                    "text": msg,
                    "icon_emoji": self.icon,
                    "username": self.icon[1:-1],
                }
            ),
        )


if __name__ == "__main__":
    slack_bot = Slack_bot()
    default_text = f"""\
slack_bot.py Logs
-----------------------
OS: {platform.platform()}
USER: {os.environ.get('USER')}
Working Dir: {os.getcwd()}
"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=default_text)
    opt = parser.parse_args()
    slack_bot(msg=opt.text)
