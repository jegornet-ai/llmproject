import os
import asyncio
from dotenv import load_dotenv
import anthropic
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, Label
from textual.containers import Vertical

load_dotenv()


class ChatApp(App):
    CSS = """
    Vertical {
        padding: 1 2;
    }
    RichLog {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    Input {
        margin-top: 1;
    }
    #status {
        height: 1;
        color: $text-muted;
    }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield RichLog(markup=True, wrap=True, id="log")
            yield Input(placeholder="Введите сообщение и нажмите Enter...")
            yield Label("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        log = self.query_one(RichLog)
        log.write("[bold cyan]Claude TUI Chat[/bold cyan]")
        log.write("[dim]Введите сообщение и нажмите Enter для отправки[/dim]\n")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        if not message:
            return

        log = self.query_one(RichLog)
        input_widget = self.query_one(Input)

        log.write(f"[bold green]Вы:[/bold green] {message}")
        input_widget.value = ""
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
        log = self.query_one(RichLog)
        input_widget = self.query_one(Input)

        def do_request() -> str:
            with self.client.messages.stream(
                model="claude-opus-4-5",
                max_tokens=1024,
                messages=[anthropic.types.MessageParam(role="user", content=message)],
            ) as stream:
                return stream.get_final_text()

        stop_event = asyncio.Event()
        spinner_task = asyncio.create_task(self._spinner(stop_event))
        try:
            response = await asyncio.to_thread(do_request)
            log.write(f"[bold blue]Claude:[/bold blue] {response}")
        except Exception as e:
            log.write(f"[bold red]Ошибка:[/bold red] {e}")
        finally:
            stop_event.set()
            await spinner_task
            input_widget.disabled = False
            self.query_one("#status", Label).update("")
            input_widget.focus()


if __name__ == "__main__":
    app = ChatApp()
    app.run()
