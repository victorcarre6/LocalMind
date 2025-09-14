import threading
import platform
import tkinter as tk
import time
import json
import os
from pathlib import Path
from tkinter import ttk, scrolledtext, messagebox
import webbrowser
from llm_executor import generate_response
import main
from main import on_ask, show_infos

# --- Config loading (robust, same as main.py) ---
PROJECT_ROOT = Path(__file__).parent.parent

def expand_path(value):
    if isinstance(value, Path):
        return str(value.resolve())
    expanded = Path(os.path.expanduser(value))
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return str(expanded.resolve())

def load_config(config_path):
    config_path = expand_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    config = raw_config
    return config

CONFIG_PATH = PROJECT_ROOT / "resources" / "config.json"
config = load_config(CONFIG_PATH)
main.init_db_connection(main.db_path)

# === Récupération des variables ===

root = tk.Tk()

conversation_counter = 0
memory_conf = config.get("memory_parameters", {})
llm_conf = config.get("models", {}).get("llm", {})

keyword_count_var = tk.IntVar(value=memory_conf.get("keyword_count", 10))
context_count_var = tk.IntVar(value=memory_conf.get("context_count", 5))
instant_memory_count_var = tk.IntVar(value=memory_conf.get("instant_memory_count", 3))
checkbox_thinking = tk.BooleanVar(value=llm_conf.get("enable_thinking", False))
checkbox_show_thinking = tk.BooleanVar(value=llm_conf.get("show_thinking", False))
checkbox_memory_recall = tk.BooleanVar(value=memory_conf.get("memory_recall", True))
checkbox_instant_memory = tk.BooleanVar(value=memory_conf.get("instant_memory", True))
threshold_count_var = tk.DoubleVar(value=memory_conf.get("similarity_score_threshold", 0.6))
temp_var = tk.DoubleVar(value=memory_conf.get("sampler_params", {}).get("temp", 0.75))
system_prompt_var = tk.StringVar(value=llm_conf.get("system_prompt", ""))
checkbox_ephemeral_mode = tk.BooleanVar(value=False)

def reset_to_defaults(settings_window=None):
    global memory_conf, llm_conf

    # Valeurs par défaut des variables
    DEFAULTS = {
        "keyword_count": 5,
        "context_count": 3,
        "memory_recall": True,
        "instant_memory": True,
        "instant_memory_count": 3,
        "similarity_score_threshold": 0.6,
        "enable_thinking": False,
        "show_thinking": False,
        "temp": 0.75,
    }

    # Appliquer les valeurs par défaut aux variables dans le config.json
    keyword_count_var.set(DEFAULTS["keyword_count"])
    context_count_var.set(DEFAULTS["context_count"])
    instant_memory_count_var.set(DEFAULTS["instant_memory_count"])
    checkbox_memory_recall.set(DEFAULTS["memory_recall"])
    checkbox_instant_memory.set(DEFAULTS["instant_memory"])
    threshold_count_var.set(DEFAULTS["similarity_score_threshold"])
    checkbox_thinking.set(DEFAULTS["enable_thinking"])
    checkbox_show_thinking.set(DEFAULTS["show_thinking"])
    temp_var.set(DEFAULTS["temp"])

    update_status("Settings reset to defaults.", success=True)

    if settings_window is not None:
        settings_window.destroy()

def save_gui_config(*args):
    config["memory_parameters"]["keyword_count"] = keyword_count_var.get()
    config["memory_parameters"]["context_count"] = context_count_var.get()
    config["memory_parameters"]["instant_memory_count"] = instant_memory_count_var.get()
    config["memory_parameters"]["memory_recall"] = checkbox_memory_recall.get()
    config["memory_parameters"]["instant_memory"] = checkbox_instant_memory.get()
    config["memory_parameters"]["similarity_score_threshold"] = round(threshold_count_var.get(), 2)
    config["models"]["llm"]["sampler_params"]["temp"] = round(temp_var.get(), 2)
    config["models"]["llm"]["enable_thinking"] = checkbox_thinking.get()
    config["models"]["llm"]["show_thinking"] = checkbox_show_thinking.get()
    with open(expand_path(CONFIG_PATH), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# --- Auto-save changes to config.json when variables change ---
keyword_count_var.trace_add("write", save_gui_config)
context_count_var.trace_add("write", save_gui_config)
instant_memory_count_var.trace_add("write", save_gui_config)
checkbox_thinking.trace_add("write", save_gui_config)
checkbox_show_thinking.trace_add("write", save_gui_config)
checkbox_memory_recall.trace_add("write", save_gui_config)
checkbox_instant_memory.trace_add("write", save_gui_config)
threshold_count_var.trace_add("write", save_gui_config)
temp_var.trace_add("write", save_gui_config)
label_temperature = None


main.set_gui_vars(keyword_count_var, context_count_var)

# === FONCTIONS PRINCIPALES ===

def preload_models_and_update_status():
    label_status.config(text="Loading models...", foreground="#FFD700")
    try:
        main.load_models_in_background()
        label_status.config(text="Ready", foreground="white")
    except Exception as e:
        label_status.config(text=f"Error loading models: {e}", foreground="#ff6b6b")

def on_generate():
    global conversation_counter

    start_on_generate = time.time()

    user_input = entry_question.get("1.0", tk.END).strip()
    instant_memory_count = instant_memory_count_var.get()
    context_count = context_count_var.get()
    keyword_count = keyword_count_var.get()
    memory_recall = checkbox_memory_recall.get()
    instant_memory = checkbox_instant_memory.get()
    similarity_threshold = threshold_count_var.get()
    instant_memory_count = instant_memory_count_var.get()
    system_prompt = system_prompt_var.get()
    history_limit = min(conversation_counter, instant_memory_count)

    if not user_input:
        update_status("Please enter a question.", error=True)
        return

    update_status("Processing prompt...")

    root.update()

    try:
        start_on_ask = time.time()
        final_prompt = on_ask(
            user_input,
            context_limit=context_count,
            keyword_count=keyword_count,
            recall=memory_recall,
            history_limit=history_limit,
            instant_memory=instant_memory,
            similarity_threshold=similarity_threshold,
            system_prompt=system_prompt
        )
        end_on_ask = time.time()

        print("")
        print("==== PROMPT GÉNÉRÉ ====")
        print(final_prompt)
        print("")
        print("==== MÉTRIQUES DE LA CONVERSATION ====")
        print(f"Durée de processing du prompt : {end_on_ask - start_on_ask:.2f} s")

        update_status("Generating response...")
        root.update()

        response = generate_response(
            user_input,
            final_prompt,
            enable_thinking=checkbox_thinking.get(),
            show_thinking=checkbox_show_thinking.get(),
            ephemeral_mode=checkbox_ephemeral_mode.get()
        )
        end_generate = time.time()
        print(f"Durée totale de l'échange : {end_generate - start_on_generate:.2f} s")
        
        if not checkbox_ephemeral_mode.get():
            conversation_counter += 1

        # Insert user message in chat history
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, "You: " + user_input + "\n", "user")
        chat_history.insert(tk.END, "Assistant: " + response + "\n\n", "assistant")
        chat_history.config(state=tk.DISABLED)

        entry_question.delete("1.0", tk.END)
        update_status(f"Response generated successfully ({end_generate - start_on_generate:.1f} s).", success=True)
        
    except Exception as e:
        update_status(f"Error: {e}", error=True)

# === Fonctions de paramétrage ===

def erase_short_term_memory():
    global conversation_counter
    conversation_counter = 0
    try:
        label_status.config(text="Short-term memory erased!", foreground="#ff6b6b")
    except Exception:
        pass

def update_temp(val):
    temp_val = round(float(val), 2)
    if label_temperature is not None:
        label_temperature.config(text=f"Temperature: {temp_val:.2f}")
    config['models']['llm']['sampler_params']['temp'] = temp_val
    # Use robust path for config
    config_path = expand_path(CONFIG_PATH)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

def open_system_prompt():
    sp_window = tk.Toplevel(root)
    sp_window.title("System Prompt")
    sp_window.geometry("450x350")
    sp_window.configure(bg="#323232")
    sp_window.resizable(False, False)

    sp_frame = ttk.Frame(sp_window, padding=10, style="TFrame")
    sp_frame.pack(fill=tk.BOTH, expand=False)

    text_widget = tk.Text(
        sp_frame,
        wrap="word",
        font=("Segoe UI", 12),
        bg="#1E1E1E",
        fg="white",
        insertbackground="white",
        height=18
    )
    text_widget.pack(fill=tk.BOTH, expand=True)
    # Insert current system_prompt from config
    text_widget.insert("1.0", config['models']['llm'].get('system_prompt', ''))

    def save_and_close():
        new_prompt = text_widget.get("1.0", tk.END).strip()
        update_system_prompt(new_prompt)
        sp_window.destroy()

    save_button = ttk.Button(
        sp_frame,
        text="Save",
        command=save_and_close,
        style="Green.TButton"
    )
    save_button.pack(pady=(5, 0))


def update_system_prompt(new_prompt: str):
    config['models']['llm']['system_prompt'] = new_prompt
    system_prompt_var.set(new_prompt)
    config_path = expand_path(CONFIG_PATH)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

# === PROFILS : Gestion des profils utilisateur ===
def open_profiles_menu():
    profiles_window = tk.Toplevel(root)
    profiles_window.title("Workspaces")
    profiles_window.geometry("225x175")
    profiles_window.configure(bg="#323232")
    profiles_window.resizable(False, False)

    # Frame principal
    main_frame = ttk.Frame(profiles_window, padding=10, style="TFrame")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Listbox pour les profils
    listbox_frame = tk.Frame(main_frame, bg="#323232")
    listbox_frame.pack(fill=tk.X, expand=False, pady=(5, 0))

    profiles_listbox = tk.Listbox(listbox_frame, font=('Segoe UI', 13), selectmode=tk.SINGLE, bg="#1E1E1E", fg="white", bd=0, highlightthickness=1, selectbackground="#599258", selectforeground="white", height=6)
    profiles_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(listbox_frame, orient="vertical", command=profiles_listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    profiles_listbox.config(yscrollcommand=scrollbar.set)

    # Charger les profils et peupler la Listbox
    def refresh_profiles_listbox(selected_name=None):
        """Remplit la Listbox avec la liste complète des profils."""
        profiles_listbox.delete(0, tk.END)
        all_profiles = main.get_all_profiles()
        for name in all_profiles:
            profiles_listbox.insert(tk.END, name)
        # Sélectionner le profil actif si possible
        current = selected_name if selected_name is not None else getattr(main, "active_profile_name", "Default")
        try:
            idx = all_profiles.index(current)
            profiles_listbox.selection_set(idx)
            profiles_listbox.see(idx)
        except ValueError:
            profiles_listbox.selection_set(0)

    refresh_profiles_listbox()

    # --- Boutons ---
    btns_frame = ttk.Frame(main_frame, style="TFrame")
    btns_frame.pack(fill=tk.X, pady=(10, 0))

    def get_selected_profile():
        try:
            idx = profiles_listbox.curselection()
            if not idx:
                return None
            return profiles_listbox.get(idx[0])
        except Exception:
            return None

    def on_add():
        # Custom Toplevel for adding a profile
        add_win = tk.Toplevel(profiles_window)
        add_win.title("Add Profile")
        add_win.configure(bg="#323232")
        add_win.resizable(False, False)
        add_win.transient(profiles_window)
        add_win.grab_set()

        add_frame = ttk.Frame(add_win, padding=15, style="TFrame")
        add_frame.pack(fill=tk.BOTH, expand=True)

        label = ttk.Label(add_frame, text="Enter new profile name:", style="TLabel")
        label.pack(anchor="center", pady=(0, 8))

        entry_var = tk.StringVar()
        entry = tk.Entry(add_frame, textvariable=entry_var, font=('Segoe UI', 13), width=24, bg="#323232", fg="white", insertbackground="white")
        entry.pack(fill=tk.X, pady=(0, 8))
        entry.focus_set()

        btns = ttk.Frame(add_frame, style="TFrame")
        btns.pack(fill=tk.X, pady=(5, 0))

        def do_add():
            name = entry_var.get().strip()
            if not name:
                return
            if name in ("Default", "All"):
                messagebox.showerror("Error", "Profile name already used.", parent=add_win)
                return
            all_names = [profiles_listbox.get(i) for i in range(profiles_listbox.size())]
            if name in all_names:
                messagebox.showerror("Error", "Profile already exists.", parent=add_win)
                return
            main.add_profile(name)
            refresh_profiles_listbox(selected_name=name)
            add_win.destroy()

        def do_cancel():
            add_win.destroy()

        btn_add_profile = ttk.Button(btns, text="Add", command=do_add, style="Bottom.TButton", width=8)
        btn_add_profile.pack(side=tk.LEFT, padx=(0, 5))
        btn_cancel = ttk.Button(btns, text="Cancel", command=do_cancel, style="Bottom.TButton", width=8)
        btn_cancel.pack(side=tk.LEFT)

        add_win.bind("<Return>", lambda e: do_add())
        add_win.bind("<Escape>", lambda e: do_cancel())

    def on_edit():
        sel = get_selected_profile()
        if sel in ("Default", "All") or sel is None:
            messagebox.showinfo("Edit", "Cannot edit reserved profile.")
            return
        # Custom Toplevel for editing a profile (styled like on_add)
        edit_win = tk.Toplevel(profiles_window)
        edit_win.title("Edit Profile")
        edit_win.configure(bg="#323232")
        edit_win.resizable(False, False)
        edit_win.transient(profiles_window)
        edit_win.grab_set()

        edit_frame = ttk.Frame(edit_win, padding=15, style="TFrame")
        edit_frame.pack(fill=tk.BOTH, expand=True)

        label = ttk.Label(edit_frame, text="Edit profile name:", style="TLabel")
        label.pack(anchor="center", pady=(0, 8))

        entry_var = tk.StringVar(value=sel)
        entry = tk.Entry(edit_frame, textvariable=entry_var, font=('Segoe UI', 13), width=24, bg="#323232", fg="white", insertbackground="white")
        entry.pack(fill=tk.X, pady=(0, 8))
        entry.focus_set()

        btns = ttk.Frame(edit_frame, style="TFrame")
        btns.pack(fill=tk.X, pady=(5, 0))

        def do_save():
            new_name = entry_var.get().strip()
            if not new_name:
                return
            if new_name in ("Default", "All"):
                messagebox.showerror("Error", "Invalid profile name.", parent=edit_win)
                return
            all_names = [profiles_listbox.get(i) for i in range(profiles_listbox.size())]
            if new_name in all_names and new_name != sel:
                messagebox.showerror("Error", "Profile already exists.", parent=edit_win)
                return
            main.edit_profile(sel, new_name)
            refresh_profiles_listbox(selected_name=new_name)
            edit_win.destroy()

        def do_cancel():
            edit_win.destroy()

        btn_save = ttk.Button(btns, text="Save", command=do_save, style="Bottom.TButton", width=8)
        btn_save.pack(side=tk.LEFT, padx=(0, 5))
        btn_cancel = ttk.Button(btns, text="Cancel", command=do_cancel, style="Bottom.TButton", width=8)
        btn_cancel.pack(side=tk.LEFT)

        edit_win.bind("<Return>", lambda e: do_save())
        edit_win.bind("<Escape>", lambda e: do_cancel())

    def on_delete():
        sel = get_selected_profile()
        if sel is None:
            return
        if sel in ("Default", "All"):
            confirm = messagebox.askyesno(
                "Delete Conversations",
                f"Are you sure you want to delete '{sel}' and all its conversations? This cannot be undone. Note that '{sel}' is a placeholder profile and cannot be deleted."
            )
            if confirm:
                main.delete_profile(sel)
                refresh_profiles_listbox(selected_name="Default")
            return
        confirm = messagebox.askyesno("Delete Profile", f"Are you sure you want to delete '{sel}' and all its conversations? This cannot be undone.")
        if confirm:
            main.delete_profile(sel)
            refresh_profiles_listbox(selected_name="Default")

    def on_ok():
        sel = get_selected_profile()
        if sel:
            main.active_profile_name = sel
        profiles_window.destroy()

    btn_add = ttk.Button(btns_frame, text="Add", command=on_add, style="Bottom.TButton", width=3)
    btn_add.pack(side=tk.LEFT, padx=2)
    btn_edit = ttk.Button(btns_frame, text="Edit", command=on_edit, style="Bottom.TButton", width=3)
    btn_edit.pack(side=tk.LEFT, padx=2)
    btn_delete = ttk.Button(btns_frame, text="Delete", command=on_delete, style="ResetGrey.TButton", width=5)
    btn_delete.pack(side=tk.LEFT, padx=2)
    btn_ok = ttk.Button(btns_frame, text="Select", command=on_ok, style="Bottom.TButton", width=5)
    btn_ok.pack(side=tk.RIGHT, padx=2)

    # Double-clic sélectionne et ferme
    def on_double_click(event):
        on_ok()
    profiles_listbox.bind("<Double-1>", on_double_click)

    # Touche entrée = Ok
    profiles_window.bind("<Return>", lambda e: on_ok())
    # Echap = fermer sans changer
    profiles_window.bind("<Escape>", lambda e: profiles_window.destroy())

    # Focus sur la liste
    profiles_listbox.focus_set()

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
    help_window.geometry("500x500")
    help_window.configure(bg="#323232")
    help_window.resizable(False, False)

    frame = tk.Frame(help_window, bg="#323232")
    frame.place(x=0, y=0, width=960, height=450)

    title_label = tk.Label(
        frame,
        text="LocalMind — Don't panic !",
        font=("Segoe UI", 12, "bold"),
        bg="#323232",
        fg="white",
        justify=tk.CENTER
    )
    title_label.pack(fill=tk.X, pady=(0, 2))

    help_text = (
            "• Generate (▲): Ask a question and get an answer using the memory system.\n\n"
            "• Workspaces:\n"
            "   - Default: General-purpose workspace, used if none is selected.\n"
            "   - All: Uses the entire database memory.\n"
            "   - Add: Create a new workspace to compartmentalize memory.\n"
            "   - Edit / Delete: Manage workspaces. Deleting one permanently erases its memory!\n\n"
            "• Settings:\n"
            "   - Prompt processing:\n"
            "      • Long-term memory: Enable, adjust keyword count, depth, and similarity threshold.\n"
            "      • Ephemeral mode: Private mode, prevents adding new entries to the database.\n"
            "      • Short-term memory: Enable, set depth, or reset.\n"
            "   - Model tuning:\n"
            "      • Reasoning mode: Enable model’s internal reasoning (if supported).\n"
            "      • Show reasoning: Display reasoning steps in chat when enabled.\n"
            "      • Temperature: Controls creativity/randomness.\n"
            "      • Edit system prompt: Modify global instructions (format, persona, etc).\n\n"
            "• More: Advanced statistics panel with:\n"
            "   - Keywords: Visualize extracted keywords from questions and prompts.\n"
            "   - Contexts: Show retrieved Q&A pairs with relevance scores.\n"
            "   - Heatmap correlation: Display correlations between keywords.\n"
            "   - Database: General information about your memory database.\n\n"
            "To learn more, troubleshoot issues, or get in touch, visit:\n"
            "github.com/victorcarre6/llm-memorization."
        )
    
    label = tk.Label(
        frame,
        text=help_text,
        font=("Segoe UI", 12),
        bg="#323232",
        fg="white",
        justify=tk.LEFT
    )
    label.pack(fill=tk.BOTH, expand=True)

def bring_to_front():
    root.update()
    root.deiconify()
    root.lift()
    root.attributes('-topmost', True)
    root.after(200, lambda: root.attributes('-topmost', False))

def open_settings():
    global temp_var, label_temperature
    settings_window = tk.Toplevel(root)
    settings_window.title("Settings")
    settings_window.geometry("250x610")
    settings_window.configure(bg="#323232")
    settings_window.resizable(False, False)

    settings_frame = ttk.Frame(settings_window, padding=10, style='TFrame')
    settings_frame.pack(fill=tk.BOTH, expand=True)

    # --- Prompt Processing Section ---
    lbl_prompt_header = ttk.Label(settings_frame, text="-- Prompt Processing --", style='TLabel', font=('Segoe UI', 14))
    lbl_prompt_header.pack(anchor='center', pady=(0,5))

    chk_memory_recall = ttk.Checkbutton(
        settings_frame,
        text="Long-term memory",
        variable=checkbox_memory_recall,
        style='Custom.TCheckbutton'
    )
    chk_memory_recall.pack(anchor='center', pady=2)

    label_keyword_count = ttk.Label(settings_frame, text=f"Number of keywords: {keyword_count_var.get()}", style='TLabel')
    label_keyword_count.pack(anchor='center')

    slider_keywords = ttk.Scale(
        settings_frame,
        from_=1,
        to=15,
        orient="horizontal",
        variable=keyword_count_var,
        length=150,
        command=lambda val: label_keyword_count.config(text=f"Number of keywords: {int(float(val))}")
    )
    slider_keywords.pack(anchor='center', pady=(0,5))

    label_contexts_count = ttk.Label(settings_frame, text=f"Long-term memory depth: {context_count_var.get()}", style='TLabel')
    label_contexts_count.pack(anchor='center')

    slider_contexts = ttk.Scale(
        settings_frame,
        from_=1,
        to=10,
        orient=tk.HORIZONTAL,
        variable=context_count_var,
        length=150,
        command=lambda val: label_contexts_count.config(text=f"Long-term memory depth: {int(float(val))}")
    )
    slider_contexts.pack(anchor='center', pady=(0,10))

    label_threshold_count = ttk.Label(settings_frame, text=f"Similarity threshold: {threshold_count_var.get():1}", style='TLabel')
    label_threshold_count.pack(anchor='center')

    slider_threshold = ttk.Scale(
        settings_frame,
        from_=0.0,
        to=2.0,
        orient=tk.HORIZONTAL,
        variable=threshold_count_var,
        length=150,
        command=lambda val: label_threshold_count.config(text=f"Similarity threshold: {float(val):.1f}")
    )
    slider_threshold.pack(anchor='center', pady=(0,5))

    chk_ephemeral_mode = ttk.Checkbutton(
        settings_frame,
        text="Ephemeral mode",
        variable=checkbox_ephemeral_mode,
        style='Custom.TCheckbutton'
    )
    chk_ephemeral_mode.pack(anchor='center', pady=(5,5))

    # Vertical spacing
    spacer = ttk.Label(settings_frame, text="")
    spacer.pack(pady=(6,0))

    chk_instant_memory = ttk.Checkbutton(
        settings_frame,
        text="Short-term memory",
        variable=checkbox_instant_memory,
        style='Custom.TCheckbutton'
    )
    chk_instant_memory.pack(anchor='center', pady=2)

    label_instant_memory_count = ttk.Label(settings_frame, text=f"Short-term memory depth: {instant_memory_count_var.get()}", style='TLabel')
    label_instant_memory_count.pack(anchor='center')

    slider_instant_memory = ttk.Scale(
        settings_frame,
        from_=1,
        to=10,
        orient=tk.HORIZONTAL,
        variable=instant_memory_count_var,
        length=150,
        command=lambda val: label_instant_memory_count.config(text=f"Short-term memory depth: {int(float(val))}")
    )
    slider_instant_memory.pack(anchor='center', pady=(0,10))

    erase_btn = ttk.Button(
        settings_frame,
        text="Reset short-term memory",
        command=erase_short_term_memory,
        style="Reset.TButton",
        cursor="hand2"
    )
    erase_btn.pack(anchor='center', pady=(4,8))

    # Vertical spacing
    spacer = ttk.Label(settings_frame, text="")
    spacer.pack(pady=(10,0))

    # --- Model Tuning Section ---
    lbl_model_header = ttk.Label(settings_frame, text="-- Model tuning --", style='TLabel', font=('Segoe UI', 14))
    lbl_model_header.pack(anchor='center', pady=(0,5))

    chk_enable_thinking = ttk.Checkbutton(
        settings_frame,
        text="Enable Thinking",
        variable=checkbox_thinking,
        style='Custom.TCheckbutton'
    )
    chk_enable_thinking.pack(anchor='center', pady=2)

    chk_show_thinking = ttk.Checkbutton(
        settings_frame,
        text="Show Thinking",
        variable=checkbox_show_thinking,
        style='Custom.TCheckbutton'
    )
    chk_show_thinking.pack(anchor='center', pady=2)

    label_temperature = ttk.Label(settings_frame, text=f"Temperature: {temp_var.get():.2f}", style='TLabel')
    label_temperature.pack(anchor='center')

    slider_temperature = ttk.Scale(
        settings_frame,
        from_=0.0,
        to=1.0,
        orient=tk.HORIZONTAL,
        variable=temp_var,
        length=150,
        command=lambda val: update_temp(val)
    )
    slider_temperature.pack(anchor='center', pady=(0,10))

    system_prompt_edit_btn = ttk.Button(
        settings_frame,
        text="Edit system prompt",
        command=open_system_prompt,
        style="SPrompt.TButton",
        cursor="hand2"
    )
    system_prompt_edit_btn.pack(anchor='center', pady=(4,15))

    reset_btn = ttk.Button(
        settings_frame,
        text="Reset settings to defaults",
        command=lambda: reset_to_defaults(settings_window),
        style="ResetGrey.TButton",
        cursor="hand2"
    )
    reset_btn.pack(anchor='center', pady=(15,8))

# === CONFIGURATION DE L'INTERFACE ===
root.title("LocalMind")
root.geometry("550x600")
root.configure(bg="#323232")

# Style global unique
style = ttk.Style(root)
style.theme_use('clam')

# Configuration du style
style_config = {
    'Green.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 12),
        'padding': 2
    },
    'Bottom.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 12),
        'padding': 2
    },
    'Reset.TButton': {
        'background': '#A52A2A',      
        'foreground': 'white',
        'font': ('Segoe UI', 12, 'bold'),
        'padding': 2
    },
    'ResetGrey.TButton': {
        'background': "#A0A0A0",      
        'foreground': 'white',
        'font': ('Segoe UI', 12, 'bold'),
        'padding': 2
    },
    'SPrompt.TButton': {
        'background': '#599258',      
        'foreground': 'white',
        'font': ('Segoe UI', 12, 'bold'),
        'padding': 2
    },
    'Blue.TLabel': {
        'background': '#323232',
        'foreground': '#599258',
        'font': ('Segoe UI', 11, 'italic underline'),
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
        'font': ('Segoe UI', 13),
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

style.map('SPrompt.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])

style.map('Bottom.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])

style.map('TCheckbutton',
          background=[('active', '#323232'), ('pressed', '#323232')],
          foreground=[('active', 'white'), ('pressed', 'white')])

style.map("Reset.TButton",
          background=[('active', '#5C0000'), ('pressed', '#3E0000')],
          foreground=[('disabled', '#d9d9d9')])

style.map("ResetGrey.TButton",
          background=[('active', '#5C0000'), ('pressed', '#3E0000')],
          foreground=[('disabled', '#d9d9d9')])

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
entry_question = tk.Text(input_frame, height=4, width=20, wrap="word", font=('Segoe UI', 13))
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



entry_question.bind("<Return>", lambda event: (on_generate(), "break"))


input_button_frame = tk.Frame(input_frame, bg="#323232")
input_button_frame.pack(side="right", fill=tk.Y)
btn_ask = ttk.Button(
    input_button_frame,
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


#
# === BARRE DE STATUT ET BOUTONS ===
# Place buttons directly under input area, with Profiles/Settings left, More/Help right, and status below.
status_buttons_frame = ttk.Frame(main_frame, style='TFrame')
status_buttons_frame.pack(fill=tk.X, pady=(5, 2))

# Left and right button frames
left_buttons = ttk.Frame(status_buttons_frame, style='TFrame')
left_buttons.pack(side=tk.LEFT, anchor='w')
right_buttons = ttk.Frame(status_buttons_frame, style='TFrame')
right_buttons.pack(side=tk.RIGHT, anchor='e')

# Left: Profiles and Settings
btn_profiles = ttk.Button(left_buttons, text="Workspaces", command=open_profiles_menu, style='Bottom.TButton', width=10)
btn_profiles.pack(side=tk.LEFT, padx=(0, 5))
btn_settings = ttk.Button(left_buttons, text="Settings", command=open_settings, style='Bottom.TButton', width=8)
btn_settings.pack(side=tk.LEFT, padx=(0, 5))

# Right: More and Help
btn_infos = ttk.Button(right_buttons, text="More", command=show_infos, style='Bottom.TButton', width=8)
btn_infos.pack(side=tk.LEFT, padx=(0, 5))
btn_help = ttk.Button(right_buttons, text="Help", style='Bottom.TButton', command=show_help, width=8)
btn_help.pack(side=tk.LEFT, padx=(0, 0))

# Status label below the buttons, spanning the full width
label_status = ttk.Label(
    main_frame,
    text="Ready",
    style='Status.TLabel',
    foreground='white',
    anchor='w'
)
label_status.pack(fill=tk.X, anchor='w', pady=(0, 2))


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