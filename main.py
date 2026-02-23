import os
import threading
import requests
from dotenv import load_dotenv
import tkinter as tk
from tkinter import scrolledtext, messagebox

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8192
TEMPERATURE = 0
SYSTEM = "Ты полезный ассистент."
API_URL = "https://api.anthropic.com/v1/messages"


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Chat")
        self.root.geometry("1024x768")

        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.history: list[dict] = []

        self.setup_ui()

    def setup_ui(self):
        # Область чата
        self.chat_log = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            state='disabled',
            font=('Arial', 14),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        self.chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Настройка тегов для цветов
        self.chat_log.tag_config('user', foreground='#a0a0a0')
        self.chat_log.tag_config('assistant', foreground='#ffffff')
        self.chat_log.tag_config('system', foreground='#888888')

        # Фрейм для ввода
        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        # Поле ввода
        self.input_field = tk.Entry(
            input_frame,
            font=('Arial', 12)
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_field.bind('<Return>', self.on_enter_pressed)
        self.input_field.focus()

        # Кнопка отправки
        self.send_button = tk.Button(
            input_frame,
            text="Отправить",
            command=self.send_message,
            font=('Arial', 14)
        )
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))

        # Статус бар
        self.status_label = tk.Label(
            self.root,
            text="",
            font=('Arial', 10),
            fg='#666666',
            anchor='w'
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=(0, 5))

        # Начальное сообщение
        self.append_to_chat("Введите сообщение и нажмите Enter для отправки\n\n", 'system')

    def append_to_chat(self, text: str, tag: str = None):
        self.chat_log.config(state='normal')
        if tag:
            self.chat_log.insert(tk.END, text + "\n", tag)
        else:
            self.chat_log.insert(tk.END, text + "\n")
        self.chat_log.config(state='disabled')
        self.chat_log.see(tk.END)

    def on_enter_pressed(self, event):
        self.send_message()
        return 'break'

    def send_message(self):
        message = self.input_field.get().strip()
        if not message:
            return

        self.input_field.delete(0, tk.END)

        # Команды
        if message == "/quit" or message == "/q":
            self.root.quit()
            return

        if message == "/clear":
            self.history.clear()
            self.chat_log.config(state='normal')
            self.chat_log.delete(1.0, tk.END)
            self.chat_log.config(state='disabled')
            self.append_to_chat("История очищена\n\n", 'system')
            return

        self.append_to_chat(f"💬 {message}", 'user')

        # Блокируем ввод
        self.input_field.config(state='disabled')
        self.send_button.config(state='disabled')
        self.status_label.config(text="Ожидание ответа от ИИ...")

        # Отправляем запрос в отдельном потоке
        thread = threading.Thread(target=self.send_to_ai, args=(message,))
        thread.daemon = True
        thread.start()

    def send_to_ai(self, message: str):
        self.history.append({"role": "user", "content": message})

        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            payload = {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "system": SYSTEM,
                "messages": self.history
            }

            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()
            assistant_message = data["content"][0]["text"] if data.get("content") else ""

            self.history.append({"role": "assistant", "content": assistant_message})

            # Обновляем GUI в основном потоке
            self.root.after(0, lambda: self.append_to_chat(f"🤖 {assistant_message}", 'assistant'))

        except Exception as e:
            self.history.pop()
            self.root.after(0, lambda: self.append_to_chat(f"Ошибка: {e}", 'system'))

        finally:
            # Разблокируем ввод в основном потоке
            self.root.after(0, self._enable_input)

    def _enable_input(self):
        self.input_field.config(state='normal')
        self.send_button.config(state='normal')
        self.status_label.config(text="")
        self.input_field.focus()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
