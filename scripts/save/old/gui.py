import platform
import tkinter as tk
import time
from tkinter import ttk, scrolledtext
import webbrowser

from llm_executor import generate_response
import memorization
from memorization import on_ask

# === Fonctions utilitaires ===

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

def on_generate():
    user_input = entry_question.get("1.0", tk.END).strip()
    context_count = context_count_var.get()
    keyword_count = keyword_count_var.get()
    if not user_input:
        update_status("Please enter a question.", error=True)
        return
    update_status("Generating response...")
    root.update()

    try:

        start_on_ask = time.time()
        final_prompt = on_ask(user_input, context_limit=context_count, keyword_count=keyword_count)
        end_on_ask = time.time()
        print(f"Durée de processing du prompt : {end_on_ask - start_on_ask:.2f} s")

        start_generate = time.time()
        response = generate_response(
            user_input,
            final_prompt,
            enable_thinking=checkbox_thinking.get(),
            show_thinking=checkbox_show_thinking.get()
        )
        end_generate = time.time()
        print(f"Durée de génération de la réponse : {end_generate - start_generate:.2f} s")
        print(f"Durée totale de l'échange : {end_generate - start_on_ask:.2f} s")
        text_output.config(state=tk.NORMAL)
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, response)
        text_output.config(state=tk.DISABLED)
        update_status("Response generated successfully.", success=True)
    except Exception as e:
        update_status(f"Error: {e}", error=True)

# === CONFIGURATION DE L'INTERFACE ===
root = tk.Tk()

keyword_count_var = tk.IntVar(value=5)
context_count_var = tk.IntVar(value=3)

memorization.set_gui_vars(keyword_count_var, context_count_var)

root.title("LLM Assistant")
root.geometry("800x300")
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

# Section question - Centrée
question_header = ttk.Frame(main_frame, style='TFrame')
question_header.pack(fill='x', pady=(0, 1))
ttk.Label(question_header, text="Ask a question :").pack(expand=True)

question_frame = tk.Frame(main_frame, bg="#323232")
question_frame.pack(pady=(0, 5), fill='x', expand=True)

entry_question = tk.Text(question_frame, height=4, width=80, wrap="word", font=('Segoe UI', 13))
entry_question.pack(side="left", fill="both", expand=True)

# Scrollbar personnalisée
style.configure("Vertical.TScrollbar",
                troughcolor='#FDF6EE',
                background='#C0C0C0',
                darkcolor='#C0C0C0',
                lightcolor='#C0C0C0',
                bordercolor='#FDF6EE',
                arrowcolor='black',
                relief='flat')

scrollbar = ttk.Scrollbar(
    question_frame,
    orient="vertical",
    command=entry_question.yview,
    style="Vertical.TScrollbar"
)
scrollbar.pack(side="right", fill="y")
entry_question.config(yscrollcommand=scrollbar.set)

entry_question.bind("<Return>", lambda event: on_generate())



# === CONTROLS & SLIDERS ===

control_frame = ttk.Frame(main_frame, style='TFrame')
control_frame.pack(fill='x', pady=(0, 10), padx=5)

slider_keywords_frame = ttk.Frame(control_frame, style='TFrame')
slider_keywords_frame.grid(row=0, column=0, sticky='w')

label_keyword_count = ttk.Label(slider_keywords_frame, text=f"Number of keywords: {keyword_count_var.get()}", style='TLabel')
label_keyword_count.pack(anchor='w')

slider_keywords = ttk.Scale(
    slider_keywords_frame,
    from_=1,
    to=15,
    orient="horizontal",
    variable=keyword_count_var,
    length=140,
    command=lambda val: label_keyword_count.config(text=f"Number of keywords: {int(float(val))}")
)
slider_keywords.pack(anchor='w')

slider_context_frame = ttk.Frame(control_frame, style='TFrame')
slider_context_frame.grid(row=0, column=1, padx=20, sticky='w')

label_contexts_count = ttk.Label(slider_context_frame, text=f"Number of contexts: {context_count_var.get()}", style='TLabel')
label_contexts_count.pack(anchor='w')

slider_contexts = ttk.Scale(
    slider_context_frame,
    from_=1,
    to=10,
    orient=tk.HORIZONTAL,
    variable=context_count_var,
    length=140,
    command=lambda val: label_contexts_count.config(text=f"Number of contexts: {int(float(val))}")
)
slider_contexts.pack(anchor='w')

checkbox_frame = ttk.Frame(control_frame, style='TFrame')
checkbox_frame.grid(row=0, column=2, padx=20, sticky='w')

checkbox_thinking = tk.BooleanVar(value=False)
chk_enable_thinking = ttk.Checkbutton(
    checkbox_frame,
    text="Enable Thinking",
    variable=checkbox_thinking,
    style='Custom.TCheckbutton'
)
chk_enable_thinking.pack(side='left', padx=(0, 10))

checkbox_show_thinking = tk.BooleanVar(value=False)
chk_show_thinking = ttk.Checkbutton(
    checkbox_frame,
    text="Show Thinking",
    variable=checkbox_show_thinking,
    style='Custom.TCheckbutton'
)
chk_show_thinking.pack(side='left')

button_frame = ttk.Frame(control_frame, style='TFrame')
button_frame.grid(row=0, column=2, sticky='e')

btn_ask = ttk.Button(button_frame, text="Generate answer", command=on_generate, style='Green.TButton')
btn_ask.pack(side='left', padx=5)
control_frame.grid_columnconfigure(2, weight=1)

# === ZONE DE SORTIE ÉTENDABLE ===
output_expanded = tk.BooleanVar(value=False)

def toggle_output():
    if output_expanded.get():
        text_output.pack_forget()
        toggle_btn.config(text="▼ Show result")
        output_expanded.set(False)
        root.geometry("800x300")
    else:
        text_output.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        toggle_btn.config(text="▲ Hide result")
        output_expanded.set(True)
        root.geometry("800x700")

output_frame = ttk.Frame(main_frame, style='TFrame')
output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

toggle_btn = ttk.Button(
    output_frame,
    text="▼ Show result",
    command=toggle_output,
    style='Green.TButton'
)
toggle_btn.pack(fill=tk.X, pady=(0, 5))

text_output = scrolledtext.ScrolledText(
    output_frame,
    width=100,
    height=20,
    font=('Segoe UI', 13),
    wrap=tk.WORD,
    bg="#FDF6EE",
    fg="black",
    insertbackground="black"
)

# === MENU CONTEXTE (clic droit) ===

# Détection de l'OS
if platform.system() == "Darwin":
    right_click_event = "<Button-2>"
else:
    right_click_event = "<Button-3>"

output_context_menu = tk.Menu(text_output, tearoff=0)
output_context_menu.add_command(label="Copier", command=lambda: text_output.event_generate("<<Copy>>"))
output_context_menu.add_command(label="Coller", command=lambda: text_output.event_generate("<<Paste>>"))
output_context_menu.add_command(label="Tout sélectionner", command=lambda: text_output.tag_add("sel", "1.0", "end"))

def show_output_context_menu(event):
    try:
        output_context_menu.tk_popup(event.x_root, event.y_root)
    finally:
        output_context_menu.grab_release()

text_output.bind(right_click_event, show_output_context_menu)

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
question_frame.bind(right_click_event, show_question_context_menu)
output_frame.bind(right_click_event, show_output_context_menu)


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

#btn_info = ttk.Button(right_buttons, text="More", style='Bottom.TButton', command=show_infos, width=8)
#btn_info.pack(side=tk.TOP, pady=(0, 3))

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

root.mainloop()