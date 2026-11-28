"""Kaggle notebook chat UI backed by the compiled C++ generator."""
from __future__ import annotations

import subprocess
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display


def display_chat(binary, model, history):
    process = subprocess.Popen(
        [
            str(Path(binary)),
            "--model", str(Path(model)),
            "--history", str(Path(history)),
            "--max-new", "128",
            "--temperature", "0.82",
            "--top-p", "0.90",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    greeting = process.stdout.readline().rstrip("\n") if process.stdout else ""
    output = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #777",
            padding="12px",
            height="420px",
            overflow_y="auto",
            width="100%",
        )
    )
    message = widgets.Text(
        placeholder="Talk to MCR...",
        description="You:",
        continuous_update=False,
        layout=widgets.Layout(width="82%"),
    )
    send = widgets.Button(description="Send", button_style="primary")
    close = widgets.Button(description="Close")
    status = widgets.HTML(value="<b>MCR — natural-language dialogue</b>")

    with output:
        print(greeting or "MCR native process started.")

    def submit(user_text):
        user_text = str(user_text).strip()
        if not user_text:
            return
        with output:
            print(f"\nYou: {user_text}")
            if process.poll() is not None:
                print("MCR process is not running.")
                send.disabled = True
                return
            try:
                process.stdin.write(user_text + "\n")
                process.stdin.flush()
                reply = process.stdout.readline() if process.stdout else ""
                if reply:
                    print(reply.rstrip("\n"))
                else:
                    print("MCR closed its output stream.")
                    send.disabled = True
            except (BrokenPipeError, OSError) as error:
                print(f"Native chat error: {error}")
                send.disabled = True

    def on_send(_):
        user_text = message.value.strip()
        message.value = ""
        submit(user_text)

    def on_enter(change):
        if change.get("name") == "value" and change.get("new"):
            user_text = str(change["new"]).strip()
            message.value = ""
            submit(user_text)

    def stop(_):
        if process.poll() is None:
            try:
                process.stdin.write("/exit\n")
                process.stdin.flush()
                process.wait(timeout=3)
            except Exception:
                process.terminate()
        send.disabled = True
        message.disabled = True
        close.disabled = True
        status.value = "<b>MCR session closed</b>"

    send.on_click(on_send)
    message.observe(on_enter, names="value")
    close.on_click(stop)

    display(widgets.VBox([
        status,
        output,
        widgets.HBox([message, send, close]),
        widgets.HTML(
            "<small>Enter messages freely. MCR generates new text from the "
            "trained dialogue model and the recent conversation history. "
            "Commands: /reset, /help, /exit.</small>"
        ),
    ]))
