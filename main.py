import os
import json
import threading
import requests
from dotenv import load_dotenv
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import tiktoken

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8192
TEMPERATURE = 0
SYSTEM = "Я полезный ассистент."
API_URL = "https://api.anthropic.com/v1/messages"
HISTORY_FILE = "chat_history.json"

# Настройки сжатия истории
KEEP_LAST_PAIRS = 2  # Сколько последних пар запрос-ответ хранить "как есть"
COMPRESS_INTERVAL = 4  # Каждый N-ый запрос выполнять сжатие

# Лимиты токенов для различных моделей (стандартный контекст)
# Источник: https://platform.claude.com/docs/en/build-with-claude/context-windows
MODEL_TOKEN_LIMITS = {
    # Claude 4.x модели (стандартный контекст: 200K, бета 1M с специальным заголовком)
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-4-5": 200000,
    "claude-sonnet-4": 200000,
    "claude-haiku-4-5": 200000,

    # Claude 3.x модели (стандартный контекст: 200K)
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
}


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Chat")
        self.root.geometry("1024x768")

        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        # Используем cl100k_base encoding (GPT-4/Claude совместимый)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"Ошибка инициализации токенизатора: {e}")
            self.tokenizer = None

        self.history: list[dict] = []
        self.conversation_summary: str = ""  # Summary предыдущих сообщений
        self.request_count: int = 0  # Счётчик запросов для сжатия
        self.attached_file_path: str | None = None
        self.attached_file_content: str | None = None

        self.setup_ui()
        self.load_history()
        self.show_tokens_in_status_bar()

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
        self.chat_log.tag_config('tokens', foreground='#666666', font=('Arial', 10, 'italic'))

        # Фрейм для прикрепления файла
        file_frame = tk.Frame(self.root)
        file_frame.pack(padx=10, pady=(0, 5), fill=tk.X)

        # Кнопка прикрепления файла
        self.attach_button = tk.Button(
            file_frame,
            text="📎 Прикрепить файл",
            command=self.attach_file,
            font=('Arial', 10)
        )
        self.attach_button.pack(side=tk.LEFT)

        # Метка с именем прикреплённого файла
        self.file_label = tk.Label(
            file_frame,
            text="",
            font=('Arial', 10),
            fg='#666666',
            anchor='w'
        )
        self.file_label.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)

        # Кнопка очистки прикреплённого файла
        self.clear_file_button = tk.Button(
            file_frame,
            text="✕",
            command=self.clear_attached_file,
            font=('Arial', 10),
            state='disabled'
        )
        self.clear_file_button.pack(side=tk.RIGHT)

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

    def count_tokens(self, text: str) -> int:
        """Подсчитывает количество токенов в тексте"""
        if not text:
            return 0

        try:
            if self.tokenizer:
                # Используем tiktoken для точного подсчёта
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            else:
                # Если tokenizer не инициализирован, используем приблизительную оценку
                # Для Claude ~3.5-4 символа на токен (примерно)
                return max(1, len(text) // 4)
        except Exception as e:
            print(f"Ошибка подсчёта токенов: {e}")
            # Fallback на приблизительную оценку
            return max(1, len(text) // 4)

    def count_message_tokens(self, messages: list[dict]) -> int:
        """Подсчитывает общее количество токенов в списке сообщений"""
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.get('content', ''))
        # Добавляем токены системного промпта
        total += self.count_tokens(SYSTEM)
        return total

    def get_total_history_tokens(self) -> int:
        """Возвращает общее количество токенов в истории диалога"""
        return self.count_message_tokens(self.history)

    def create_summary(self, messages: list[dict]) -> str:
        """Создаёт краткое резюме диалога через LLM"""
        if not messages:
            return ""

        try:
            # Формируем текст диалога для создания summary
            conversation_text = ""
            for msg in messages:
                role = "Пользователь" if msg["role"] == "user" else "Ассистент"
                conversation_text += f"{role}: {msg['content']}\n\n"

            # Запрашиваем summary у LLM
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            summary_prompt = f"""Создай краткое резюме следующего диалога.
Резюме должно включать ключевые темы, решённые задачи и важные детали.
Резюме должно быть на русском языке и занимать не более 3-4 предложений.

Диалог:
{conversation_text}"""

            payload = {
                "model": MODEL,
                "max_tokens": 500,
                "temperature": 0,
                "system": "Ты помощник, который создаёт краткие резюме диалогов.",
                "messages": [
                    {"role": "user", "content": summary_prompt}
                ]
            }

            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()
            summary = data["content"][0]["text"] if data.get("content") else ""

            return summary.strip()

        except Exception as e:
            print(f"Ошибка создания summary: {e}")
            return ""

    def compress_history(self):
        """Сжимает историю диалога, оставляя только последние N пар и создавая summary для остальных"""
        # Проверяем, нужно ли сжимать
        # История должна содержать больше, чем KEEP_LAST_PAIRS пар (т.е. минимум KEEP_LAST_PAIRS * 2 + 2 сообщений)
        if len(self.history) <= KEEP_LAST_PAIRS * 2:
            return

        # Вычисляем количество сообщений, которые нужно оставить
        messages_to_keep = KEEP_LAST_PAIRS * 2

        # Разделяем историю на части: для сжатия и для сохранения
        messages_to_compress = self.history[:-messages_to_keep]
        messages_to_keep_list = self.history[-messages_to_keep:]

        # Создаём summary для старых сообщений
        self.root.after(0, lambda: self.append_to_chat("Сжатие истории...", 'system'))

        new_summary = self.create_summary(messages_to_compress)

        if new_summary:
            # Добавляем новый summary к существующему
            if self.conversation_summary:
                self.conversation_summary += f"\n\n{new_summary}"
            else:
                self.conversation_summary = new_summary

            # Заменяем историю на последние сообщения
            self.history = messages_to_keep_list

            # Сохраняем обновлённую историю
            self.save_history()

            self.root.after(0, lambda: self.append_to_chat(f"История сжата. Summary обновлён.\n", 'system'))
            self.root.after(0, self.show_tokens_in_status_bar)

    def get_current_system_prompt(self) -> str:
        """Возвращает текущий системный промпт с учётом summary"""
        if self.conversation_summary:
            return f"{SYSTEM}\n\nКонтекст предыдущих диалогов:\n{self.conversation_summary}"
        return SYSTEM

    def get_model_token_limit(self) -> int:
        """Возвращает лимит токенов для текущей модели"""
        return MODEL_TOKEN_LIMITS.get(MODEL, 200000)

    def show_tokens_in_status_bar(self):
        """Обновляет статус бар с информацией о токенах"""
        total_tokens = self.get_total_history_tokens()
        token_limit = self.get_model_token_limit()
        percentage = (total_tokens / token_limit * 100) if token_limit > 0 else 0

        status_text = f"Токенов в истории: {total_tokens:,} / {token_limit:,} ({percentage:.1f}%) | Модель: {MODEL}"
        self.status_label.config(text=status_text)

    def load_history(self):
        """Загружает историю диалога из JSON файла"""
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Проверяем формат данных
                if isinstance(data, dict):
                    # Новый формат с метаданными
                    self.history = data.get('history', [])
                    self.conversation_summary = data.get('summary', '')
                    self.request_count = data.get('request_count', 0)
                else:
                    # Старый формат - только список сообщений
                    self.history = data
                    self.conversation_summary = ""
                    self.request_count = 0

                # Восстанавливаем сообщения в GUI
                for msg in self.history:
                    if msg['role'] == 'user':
                        self.append_to_chat(f"💬 {msg['content']}", 'user')
                        # Показываем токены для запроса пользователя
                        if 'tokens' in msg:
                            self.append_to_chat(f"   [{msg['tokens']} токенов]", 'tokens')
                    elif msg['role'] == 'assistant':
                        self.append_to_chat(f"🤖 {msg['content']}", 'assistant')
                        # Показываем токены для ответа ассистента
                        if 'tokens' in msg:
                            self.append_to_chat(f"   [{msg['tokens']} токенов]", 'tokens')
        except Exception as e:
            self.append_to_chat(f"Ошибка загрузки истории: {e}", 'system')

    def save_history(self):
        """Сохраняет историю диалога в JSON файл"""
        try:
            # Сохраняем в новом формате с метаданными
            data = {
                'history': self.history,
                'summary': self.conversation_summary,
                'request_count': self.request_count
            }
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения истории: {e}")

    def clear_history(self):
        self.history.clear()
        self.conversation_summary = ""
        self.request_count = 0
        self.chat_log.config(state='normal')
        self.chat_log.delete(1.0, tk.END)
        self.chat_log.config(state='disabled')
        # Удаляем файл истории
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        self.append_to_chat("История очищена\n\n", 'system')
        self.show_tokens_in_status_bar()

    def attach_file(self):
        """Открывает диалог выбора файла и читает его содержимое"""
        file_path = filedialog.askopenfilename(
            title="Выберите текстовый файл",
            filetypes=[
                ("Текстовые файлы", "*.txt"),
                ("Все файлы", "*.*")
            ]
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                self.attached_file_path = file_path
                self.attached_file_content = content

                # Обновляем UI
                filename = os.path.basename(file_path)
                self.file_label.config(text=f"Прикреплён: {filename}")
                self.clear_file_button.config(state='normal')

            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось прочитать файл: {e}")

    def clear_attached_file(self):
        """Очищает прикреплённый файл"""
        self.attached_file_path = None
        self.attached_file_content = None
        self.file_label.config(text="")
        self.clear_file_button.config(state='disabled')

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
            self.clear_history()
            return

        if message == "/system" or message == "/s":
            current_prompt = self.get_current_system_prompt()
            self.append_to_chat("Текущий системный промпт:", 'system')
            self.append_to_chat(f"{current_prompt}\n", 'system')
            return

        # Формируем полное сообщение с учётом прикреплённого файла
        full_message = message
        if self.attached_file_content:
            full_message = f"{message}\n\n{self.attached_file_content}"

        self.append_to_chat(f"💬 {message}", 'user')

        # Если есть прикреплённый файл, показываем это
        if self.attached_file_path:
            filename = os.path.basename(self.attached_file_path)
            self.append_to_chat(f"   [📎 {filename}]", 'system')

        # Блокируем ввод
        self._set_input_waiting()

        # Отправляем запрос в отдельном потоке
        thread = threading.Thread(target=self.send_to_ai, args=(full_message,))
        thread.daemon = True
        thread.start()

    def send_to_ai(self, message: str):
        # Подсчитываем токены для запроса пользователя
        user_tokens = self.count_tokens(message)

        self.history.append({
            "role": "user",
            "content": message,
            "tokens": user_tokens
        })

        # Отображаем токены запроса
        self.root.after(0, lambda: self.append_to_chat(f"   [{user_tokens} токенов]", 'tokens'))

        # Увеличиваем счётчик запросов
        self.request_count += 1

        # Проверяем, нужно ли выполнить сжатие
        if self.request_count % COMPRESS_INTERVAL == 0:
            self.compress_history()

        success = False
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            # Создаём копию истории без поля tokens для API
            messages_for_api = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.history
            ]

            # Используем системный промпт с учётом summary
            current_system_prompt = self.get_current_system_prompt()

            payload = {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "system": current_system_prompt,
                "messages": messages_for_api
            }

            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()
            assistant_message = data["content"][0]["text"] if data.get("content") else ""

            # Получаем информацию об использовании токенов из ответа API
            usage = data.get("usage", {})
            assistant_tokens = usage.get("output_tokens", self.count_tokens(assistant_message))

            self.history.append({
                "role": "assistant",
                "content": assistant_message,
                "tokens": assistant_tokens
            })
            self.save_history()

            # Обновляем GUI в основном потоке
            self.root.after(0, lambda: self.append_to_chat(f"🤖 {assistant_message}", 'assistant'))
            self.root.after(0, lambda: self.append_to_chat(f"   [{assistant_tokens} токенов]", 'tokens'))
            self.root.after(0, self.show_tokens_in_status_bar)

            success = True

        except Exception as e:
            self.history.pop()
            self.root.after(0, lambda: self.append_to_chat(f"Ошибка: {e}", 'system'))

        finally:
            # Разблокируем ввод в основном потоке
            self.root.after(0, self._enable_input)

            # Очищаем прикреплённый файл после успешной отправки
            if success:
                self.root.after(0, self.clear_attached_file)

    def _enable_input(self):
        self.input_field.config(state='normal')
        self.send_button.config(state='normal')
        self.status_label.config(text="")
        self.show_tokens_in_status_bar()

    def _set_input_waiting(self):
        self.input_field.config(state='disabled')
        self.send_button.config(state='disabled')
        self.status_label.config(text="Ожидание ответа от ИИ...")


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
