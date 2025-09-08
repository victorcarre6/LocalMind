import threading
import platform
import tkinter as tk
import time
from tkinter import ttk, scrolledtext
import webbrowser

from llm_executor import generate_response
import main
from main import on_ask

# === Fonctions principale ===

root = tk.Tk()

conversation_counter = 0
keyword_count_var = tk.IntVar(value=5)
context_count_var = tk.IntVar(value=3)
checkbox_thinking = tk.BooleanVar(value=False)
checkbox_show_thinking = tk.BooleanVar(value=False)
checkbox_memory_recall = tk.BooleanVar(value=True)

main.set_gui_vars(keyword_count_var, context_count_var)


def preload_models_and_update_status():
    label_status.config(text="Loading models...", foreground="#FFD700")
    try:
        main.load_models_in_background()
        label_status.config(text="Ready", foreground="white")
    except Exception as e:
        label_status.config(text=f"Error loading models: {e}", foreground="#ff6b6b")

def on_generate(event=None):
    global conversation_counter

    start_on_generate = time.time()

    user_input = entry_question.get("1.0", tk.END).strip()
    history_limit = min(conversation_counter, 3)
    context_count = context_count_var.get()
    keyword_count = keyword_count_var.get()
    memory_recall = checkbox_memory_recall.get()

    if not user_input:
        update_status("Please enter a question.", error=True)
        return

    running_prompt = True

    def update_chrono_prompt():
        nonlocal running_prompt
        if not running_prompt:
            return
        elapsed = time.time() - start_on_generate
        update_status(f"Processing prompt... ({elapsed:.1f}s)")
        root.after(100, update_chrono_prompt)

    update_chrono_prompt()

    root.update()

    try:
        start_on_ask = time.time()
        final_prompt = on_ask(user_input, context_limit=context_count, keyword_count=keyword_count, recall=memory_recall, history_limit=history_limit)
        running_prompt = False
        end_on_ask = time.time()

        print("--INFO-- final_prompt :")
        print(final_prompt)

        print("==== MÉTRIQUES DE LA CONVERSATION ====")
        print(f"[TIMER on_generate (on_ask total)] Durée de processing du prompt : {end_on_ask - start_on_ask:.2f} s")

        running_response = True

        def update_chrono_response():
            nonlocal running_response
            if not running_response:
                return
            elapsed = time.time() - end_on_ask
            update_status(f"Generating response... ({elapsed:.1f}s)")
            root.after(100, update_chrono_response)

        update_chrono_response()

        response = generate_response(
            user_input,
            final_prompt,
            enable_thinking=checkbox_thinking.get(),
            show_thinking=checkbox_show_thinking.get()
        )
        end_generate = time.time()
        running_response = False
        print(f"[TIMER on_generate] Durée totale de l'échange : {end_generate - start_on_generate:.2f} s")
        conversation_counter += 1
        print(f"--INFO-- conversation_counter: {conversation_counter}")
        # Insert user message in chat history
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, "You: " + user_input + "\n", "user")
        chat_history.insert(tk.END, "Assistant: " + response + "\n\n", "assistant")
        chat_history.config(state=tk.DISABLED)

        entry_question.delete("1.0", tk.END)
        update_status(f"Response generated successfully ({end_generate - start_on_generate:.1f} s).", success=True)
        
    except Exception as e:
        running_prompt = False
        update_status(f"Error: {e}", error=True)



# === Fonctions d'affichage ===

def update_status(message, error=False, success=False):
    label_status.config(text=message)
    if error:
        label_status.config(foreground='#ff6b6b')
    elif success:
        label_status.config(foreground='#599258')
    else:
        label_status.config(foreground='white')

def open_github(event):
    webbrowser.open_new("https://github.com/victorcarre6")

def show_help():
    help_window = tk.Toplevel(root)
    help_window.title("Help")
    help_window.geometry("600x350")
    help_window.configure(bg="#323232")
    help_window.resizable(False, False)

    frame = tk.Frame(help_window, bg="#323232")
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    title_label = tk.Label(
        frame,
        text="LLM Assistant — Don't panic !",
        font=("Segoe UI", 12, "bold"),
        bg="#323232",
        fg="white",
        justify=tk.CENTER
    )
    title_label.pack(fill=tk.X, pady=(0, 2))

    help_text = (
        "• XXXXXXXXXXXX.\n\n"
        "• Generate answer: Extracts the keywords from your question, searches for similar past conversations in your SQL database, and summarizes the relevant content using a local LLM.\n"
        "   The final prompt is displayed and automatically copied to your clipboard!\n\n"
        "• Settings: XXXXX\n\n"
        "• Memory: Opens an advanced statistics panel, including:\n"
        "   Visualization of keywords extracted from your question, from the transitory generated prompt.\n"
        "   Correlation graphs between the keywords.\n"
        "   Database insights: number and frequency of keywords, and used LLM models.\n\n"
        "• Workflow: XXXXX\n\n"
        "• Tools: XXXXX\n\n"
        "To learn more, troubleshoot potential script issues, or get in touch, visit:\n"
        "github.com/victorcarre6/llm-memorization."
    )
    label = tk.Label(
        frame,
        text=help_text,
        font=("Segoe UI", 12),
        bg="#323232",
        fg="white",
        justify=tk.LEFT,
        wraplength=550
    )
    label.pack(fill=tk.BOTH, expand=True)

def bring_to_front():
    root.update()
    root.deiconify()
    root.lift()
    root.attributes('-topmost', True)
    root.after(200, lambda: root.attributes('-topmost', False))

def open_settings():
    settings_window = tk.Toplevel(root)
    settings_window.title("Settings")
    settings_window.geometry("200x250")
    settings_window.configure(bg="#323232")
    settings_window.resizable(False, False)

    settings_frame = ttk.Frame(settings_window, padding=10, style='TFrame')
    settings_frame.pack(fill=tk.BOTH, expand=True)

    # Checkboxes
    checkbox_frame = ttk.Frame(settings_frame, style='TFrame')
    checkbox_frame.pack(anchor='w')

    chk_enable_thinking = ttk.Checkbutton(
        checkbox_frame,
        text="Enable Thinking",
        variable=checkbox_thinking,
        style='Custom.TCheckbutton'
    )
    chk_enable_thinking.pack(anchor='w', pady=2)

    chk_show_thinking = ttk.Checkbutton(
        checkbox_frame,
        text="Show Thinking",
        variable=checkbox_show_thinking,
        style='Custom.TCheckbutton'
    )
    chk_show_thinking.pack(anchor='w', pady=2)

    chk_memory_recall = ttk.Checkbutton(
        checkbox_frame,
        text="Memory Recall",
        variable=checkbox_memory_recall,
        style='Custom.TCheckbutton'
    )
    chk_memory_recall.pack(anchor='w', pady=2)


    # Sliders
    slider_keywords_frame = ttk.Frame(settings_frame, style='TFrame')
    slider_keywords_frame.pack(anchor='w', pady=(0,10))

    label_keyword_count = ttk.Label(slider_keywords_frame, text=f"Number of keywords: {keyword_count_var.get()}", style='TLabel')
    label_keyword_count.pack(anchor='w')

    slider_keywords = ttk.Scale(
        slider_keywords_frame,
        from_=1,
        to=15,
        orient="horizontal",
        variable=keyword_count_var,
        length=150,
        command=lambda val: label_keyword_count.config(text=f"Number of keywords: {int(float(val))}")
    )
    slider_keywords.pack(anchor='w')

    slider_context_frame = ttk.Frame(settings_frame, style='TFrame')
    slider_context_frame.pack(anchor='w', pady=(0,10))

    label_contexts_count = ttk.Label(slider_context_frame, text=f"Number of contexts: {context_count_var.get()}", style='TLabel')
    label_contexts_count.pack(anchor='w')

    slider_contexts = ttk.Scale(
        slider_context_frame,
        from_=1,
        to=10,
        orient=tk.HORIZONTAL,
        variable=context_count_var,
        length=150,
        command=lambda val: label_contexts_count.config(text=f"Number of contexts: {int(float(val))}")
    )
    slider_contexts.pack(anchor='w')

# === CONFIGURATION DE L'INTERFACE ===
root.title("LLM Assistant")
root.geometry("800x500")
root.configure(bg="#323232")

# Style global unique
style = ttk.Style(root)
style.theme_use('clam')

# Configuration du style
style_config = {
    'Green.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 13),
        'padding': 2
    },
    'Bottom.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 11),
        'padding': 2
    },
    'Blue.TLabel': {
        'background': '#323232',
        'foreground': '#599258',
        'font': ('Segoe UI', 10, 'italic underline'),
        'padding': 0
    },
    'TLabel': {
        'background': '#323232',
        'foreground': 'white',
        'font': ('Segoe UI', 13)
    },
    'TEntry': {
        'fieldbackground': '#FDF6EE',
        'foreground': 'black',
        'font': ('Segoe UI', 13)
    },
    'TFrame': {
        'background': '#323232'
    },
    'Status.TLabel': {
        'background': '#323232',
        'font': ('Segoe UI', 13)
    },
    'TNotebook': {
        'background': '#323232',
        'borderwidth': 0
    },
    'TNotebook.Tab': {
        'background': '#2a2a2a',
        'foreground': 'white',
        'padding': (10, 4)
    },
    'Custom.Treeview': {
        'background': '#2a2a2a',
        'foreground': 'white',
        'fieldbackground': '#2a2a2a',
        'font': ('Segoe UI', 12),
        'bordercolor': '#323232',
        'borderwidth': 0,
    },
    'Custom.Treeview.Heading': {
        'background': '#323232',
        'foreground': '#599258',
        'font': ('Segoe UI', 13, 'bold'),
        'relief': 'flat'
    },
    'TCheckbutton': {
        'background': '#323232',
        'foreground': 'white',
        'font': ('Segoe UI', 12),
        'focuscolor': '#323232',
        'indicatorcolor': '#599258'
    }
}

for style_name, app_config in style_config.items():
    style.configure(style_name, **app_config)

style.map('Green.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])

style.map("TNotebook.Tab",
          background=[("selected", "#323232"), ("active", "#2a2a2a")],
          foreground=[("selected", "white"), ("active", "white")])

style.map('Bottom.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])

style.map('TCheckbutton',
          background=[('active', '#323232'), ('pressed', '#323232')],
          foreground=[('active', 'white'), ('pressed', 'white')])

# === WIDGETS PRINCIPAUX ===
main_frame = ttk.Frame(root, padding=10, style='TFrame')
main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Chat history area at the top
chat_history = scrolledtext.ScrolledText(
    main_frame,
    width=100,
    height=20,
    font=('Segoe UI', 13),
    wrap=tk.WORD,
    bg="#1E1E1E",
    fg="black",
    insertbackground="black",
    state=tk.DISABLED
)
chat_history.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

# Define tags for styling user and assistant messages
chat_history.tag_configure("user", foreground="#599258", font=('Segoe UI', 13, 'bold'))
chat_history.tag_configure("assistant", foreground="#CECABF", font=('Segoe UI', 13))

# Input frame at the bottom
input_frame = tk.Frame(main_frame, bg="#323232")
input_frame.pack(fill=tk.X, expand=False)

# Reduce input width from 80 to 60
entry_question = tk.Text(input_frame, height=4, width=60, wrap="word", font=('Segoe UI', 13))
entry_question.pack(side="left", fill="both", expand=True)

# Scrollbar personnalisée for input text
style.configure("Vertical.TScrollbar",
                troughcolor='#FDF6EE',
                background='#C0C0C0',
                darkcolor='#C0C0C0',
                lightcolor='#C0C0C0',
                bordercolor='#FDF6EE',
                arrowcolor='black',
                relief='flat')

# Place scrollbar to the left of the send button, but right of input
scrollbar = ttk.Scrollbar(
    input_frame,
    orient="vertical",
    command=entry_question.yview,
    style="Vertical.TScrollbar"
)
scrollbar.pack(side="left", fill="y")
entry_question.config(yscrollcommand=scrollbar.set)

entry_question.bind("<Return>", lambda event: (on_generate(), "break"))

# Send button with arrow "▲" to the right of scrollbar
btn_ask = ttk.Button(
    input_frame,
    text="▲",
    command=on_generate,
    style='Green.TButton',
    width=3
)
btn_ask.pack(side="right", padx=(5, 0), pady=(0, 0))


# === MENU CONTEXTE (clic droit) ===

# Détection de l'OS
if platform.system() == "Darwin":
    right_click_event = "<Button-2>"
else:
    right_click_event = "<Button-3>"

# Context menu for chat_history
chat_context_menu = tk.Menu(chat_history, tearoff=0)
chat_context_menu.add_command(label="Copier", command=lambda: chat_history.event_generate("<<Copy>>"))
chat_context_menu.add_command(label="Coller", command=lambda: chat_history.event_generate("<<Paste>>"))
chat_context_menu.add_command(label="Tout sélectionner", command=lambda: chat_history.tag_add("sel", "1.0", "end"))

def show_chat_context_menu(event):
    try:
        chat_context_menu.tk_popup(event.x_root, event.y_root)
    finally:
        chat_context_menu.grab_release()

chat_history.bind(right_click_event, show_chat_context_menu)

# Menu contextuel pour entry_question (zone de question)
question_context_menu = tk.Menu(entry_question, tearoff=0)
question_context_menu.add_command(label="Copier", command=lambda: entry_question.event_generate("<<Copy>>"))
question_context_menu.add_command(label="Coller", command=lambda: entry_question.event_generate("<<Paste>>"))
question_context_menu.add_command(label="Tout sélectionner", command=lambda: entry_question.tag_add("sel", "1.0", "end"))

def show_question_context_menu(event):
    try:
        question_context_menu.tk_popup(event.x_root, event.y_root)
    finally:
        question_context_menu.grab_release()

entry_question.bind(right_click_event, show_question_context_menu)

# Ajouter aussi aux frames si nécessaire
input_frame.bind(right_click_event, show_question_context_menu)
main_frame.bind(right_click_event, show_chat_context_menu)


# === BARRE DE STATUT ET BOUTONS ===
status_buttons_frame = ttk.Frame(main_frame, style='TFrame')
status_buttons_frame.pack(fill=tk.X, pady=(5, 2))

label_status = ttk.Label(
    status_buttons_frame,
    text="Ready",
    style='Status.TLabel',
    foreground='white',
    anchor='w'
)
label_status.pack(side=tk.LEFT, anchor='w')

right_buttons = ttk.Frame(status_buttons_frame, style='TFrame')
right_buttons.pack(side=tk.RIGHT, anchor='e')

btn_settings = ttk.Button(right_buttons, text="Settings", command=open_settings, style='Bottom.TButton', width=8)
btn_settings.pack(side=tk.TOP, pady=(0, 3))

btn_help = ttk.Button(right_buttons, text="Help", style='Bottom.TButton', command=show_help, width=8)
btn_help.pack(side=tk.TOP)


# === FOOTER ===
footer_frame = ttk.Frame(root, style='TFrame')
footer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))

dev_label = ttk.Label(footer_frame, text="Developped by Victor Carré —", style='TLabel', font=('Segoe UI', 10))
dev_label.pack(side=tk.LEFT)

github_link = ttk.Label(footer_frame, text="GitHub", style='Blue.TLabel', cursor="hand2")
github_link.pack(side=tk.LEFT)
github_link.bind("<Button-1>", open_github)

bring_to_front()

# Lancer le thread de préchargement des modèles AVANT l'ouverture de la fenêtre principale
preload_thread = threading.Thread(target=preload_models_and_update_status, daemon=True)
preload_thread.start()

root.mainloop()