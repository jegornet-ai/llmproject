import os
import asyncio
from dotenv import load_dotenv
import anthropic
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Label, Button, TextArea
from textual.containers import Vertical, Horizontal

load_dotenv()

MODEL_PARAMS = {
    "model": "claude-opus-4-5",
}

MODES = {
    "normal": {
        "label": "Обычный",
        "max_tokens": 8192,
        "prefix": "",
        "stop_sequences": [],
        "truncate_at": None,
        "system": "Ты полезный ассистент.",
    },
    "duck": {
        "label": "Утиный",
        "max_tokens": 512,
        "prefix": "Кря! Отвечаю:",
        "stop_sequences": ["."],
        "truncate_at": None,
        "system": "Ты утка. Всегда отвечай одним длинным подробным предложением заканчивающимся точкой. Начало ответа уже написано, продолжи его.",
    },
}


class ChatLog(TextArea):
    BINDINGS = [("ctrl+c", "copy_selection", "Копировать")]

    def action_copy_selection(self) -> None:
        selected = self.selected_text
        if selected:
            self.app.copy_to_clipboard(selected)


class ChatApp(App):
    CSS = """
    Vertical {
        padding: 1 2;
    }
    TextArea {
        height: 1fr;
        border: solid $primary;
    }
    #input-row {
        height: auto;
        margin-top: 1;
    }
    #input-row Input {
        width: 1fr;
    }
    #mode-btn {
        width: auto;
        min-width: 12;
        margin-left: 1;
    }
    #status {
        height: 1;
        color: $text-muted;
    }
    """

    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.mode = "normal"
        self.history: list[anthropic.types.MessageParam] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield ChatLog(id="log", read_only=True)
            with Horizontal(id="input-row"):
                yield Input(placeholder="Введите сообщение и нажмите Enter...")
                yield Button(MODES["normal"]["label"], id="mode-btn")
            yield Label("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        log = self.query_one(ChatLog)
        log.load_text("Claude TUI Chat\n\nВведите сообщение и нажмите Enter для отправки\n\n")

    def _log_append(self, text: str) -> None:
        log = self.query_one(ChatLog)
        log.load_text(log.text + text + "\n")
        log.scroll_end(animate=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mode-btn":
            self.mode = "duck" if self.mode == "normal" else "normal"
            event.button.label = MODES[self.mode]["label"]
            self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        if not message:
            return

        input_widget = self.query_one(Input)
        input_widget.value = ""

        if message == "/quit" or message == "/q":
            self.exit()
            return

        if message == "/clear":
            self.history.clear()
            log = self.query_one(ChatLog)
            log.load_text("Claude TUI Chat\n\nИстория очищена\n\n")
            return

        self._log_append(f"Вы: {message}")
        input_widget.disabled = True

        self.run_worker(self.send_to_claude(message), exclusive=True)

    async def _spinner(self, stop_event: asyncio.Event) -> None:
        frames = ["-", "\\", "|", "/"]
        i = 0
        status = self.query_one("#status", Label)
        while not stop_event.is_set():
            status.update(f"Ожидание ответа от Claude... {frames[i % len(frames)]}")
            i += 1
            await asyncio.sleep(0.1)

    async def send_to_claude(self, message: str) -> None:
        input_widget = self.query_one(Input)
        mode = MODES[self.mode]

        self.history.append(anthropic.types.MessageParam(role="user", content=message))

        def do_request() -> str:
            messages = list(self.history)
            system = mode["system"]
            kwargs = {
                **MODEL_PARAMS,
                "max_tokens": mode["max_tokens"],
                "system": system,
                "messages": messages,
            }
            if mode["stop_sequences"]:
                kwargs["stop_sequences"] = mode["stop_sequences"]
            if mode["prefix"]:
                kwargs["messages"] = messages + [
                    anthropic.types.MessageParam(role="assistant", content=mode["prefix"])
                ]
            with self.client.messages.stream(**kwargs) as stream:
                final = stream.get_final_message()
            continuation = final.content[0].text if final.content else ""
            text = (mode["prefix"] + continuation) if mode["prefix"] else continuation
            if mode["stop_sequences"] and final.stop_reason == "stop_sequence":
                text = text + final.stop_sequence
            return text

        stop_event = asyncio.Event()
        spinner_task = asyncio.create_task(self._spinner(stop_event))
        try:
            response = await asyncio.to_thread(do_request)
            self.history.append(anthropic.types.MessageParam(role="assistant", content=response))
            self._log_append(f"Claude: {response}")
        except Exception as e:
            self.history.pop()
            self._log_append(f"Ошибка: {e}")
        finally:
            stop_event.set()
            await spinner_task
            input_widget.disabled = False
            self.query_one("#status", Label).update("")
            input_widget.focus()


if __name__ == "__main__":
    app = ChatApp()
    app.run()
