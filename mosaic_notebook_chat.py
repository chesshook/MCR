"""Notebook UI for the native MOSAIC executable (no Python model inference)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display


def display_chat(binary, model, memory):
    binary = str(Path(binary))
    model = str(Path(model))
    memory = str(Path(memory))

    process = subprocess.Popen(
        [binary, "--model", model, "--memory", memory],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    header = process.stdout.readline().rstrip("\n") if process.stdout else ""
    output = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #777",
            padding="10px",
            height="380px",
            overflow_y="auto",
            width="100%",
        )
    )
    message = widgets.Text(
        placeholder="Write a message to MOSAIC...",
        description="You:",
        layout=widgets.Layout(width="82%"),
    )
    send = widgets.Button(description="Send", button_style="primary")
    stop = widgets.Button(description="Close", button_style="")
    status = widgets.HTML(value="<b>MOSAIC native chat</b>")

    with output:
        print(header or "MOSAIC native chat started.")

    def submit(_):
        text = message.value.strip()
        if not text:
            return
        message.value = ""
        with output:
            print(f"\nYou: {text}")
            if process.poll() is not None:
                print("MOSAIC process has stopped.")
                send.disabled = True
                return
            try:
                process.stdin.write(text + "\n")
                process.stdin.flush()
                reply = process.stdout.readline() if process.stdout else ""
                if reply:
                    print(reply.rstrip("\n"))
                else:
                    print("No response: the native process closed its output.")
                    send.disabled = True
            except (BrokenPipeError, OSError) as error:
                print(f"Process error: {error}")
                send.disabled = True

    def close(_):
        if process.poll() is None:
            try:
                process.stdin.write("/exit\n")
                process.stdin.flush()
                process.wait(timeout=2)
            except Exception:
                process.terminate()
        send.disabled = True
        message.disabled = True
        stop.disabled = True
        status.value = "<b>MOSAIC chat closed</b>"

    send.on_click(submit)
    stop.on_click(close)
    display(widgets.VBox([
        status,
        output,
        widgets.HBox([message, send, stop]),
        widgets.HTML(
            "<small>Enter a greeting, tell it facts, correct them, ask about "
            "the conversation, or try arithmetic. Button sends each message.</small>"
        ),
    ]))
