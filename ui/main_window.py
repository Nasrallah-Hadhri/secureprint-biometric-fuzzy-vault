"""
Interface Utilisateur CustomTkinter — SecurePrint
Three tabs: Enrôlement / Authentification / Gestion (RGPD)
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Modules SecurePrint
from modules.encryption import generate_key, encrypt_template
from modules.storage import init_db, enroll_user, get_all_users, get_template, delete_user, get_user_by_name
from modules.matcher import full_pipeline, authenticate

# ──────────────────────────────────────────────
# Thème
# ──────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT   = "#2563EB"   # Bleu vif
SUCCESS  = "#16A34A"   # Vert
DANGER   = "#DC2626"   # Rouge
WARN     = "#D97706"   # Orange
BG_CARD  = "#1E293B"   # Carte sombre
BG_ROOT  = "#0F172A"   # Fond général


def _cv2_to_photoimage(img_bgr: np.ndarray, max_w: int = 280, max_h: int = 220) -> ImageTk.PhotoImage:
    """Convertit une image OpenCV BGR en PhotoImage Tkinter (redimensionnée)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if len(img_bgr.shape) == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Enrôlement
# ──────────────────────────────────────────────────────────────────────────────
class EnrollTab(ctk.CTkFrame):
    def __init__(self, master, refresh_callback=None, **kwargs):
        super().__init__(master, fg_color=BG_ROOT, **kwargs)
        self._refresh_cb = refresh_callback
        self._image_path: Optional[str] = None
        self._photo = None
        self._build_ui()

    def _build_ui(self):
        # ── Titre ──
        ctk.CTkLabel(self, text="🔐  Enrôlement d'Empreinte",
                     font=ctk.CTkFont("Segoe UI", 22, "bold"),
                     text_color="white").pack(pady=(28, 4))
        ctk.CTkLabel(self, text="Associez une image d'empreinte digitale à un identifiant.",
                     font=ctk.CTkFont("Segoe UI", 13),
                     text_color="#94A3B8").pack(pady=(0, 18))

        # ── Carte principale ──
        card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=16)
        card.pack(padx=40, pady=8, fill="both", expand=True)

        # Colonne gauche : sélection image
        left = ctk.CTkFrame(card, fg_color="transparent")
        left.grid(row=0, column=0, padx=24, pady=24, sticky="n")

        self._preview_label = ctk.CTkLabel(left,
            text="Aucune image sélectionnée",
            width=280, height=220,
            fg_color="#0F172A",
            corner_radius=12,
            text_color="#475569",
            font=ctk.CTkFont("Segoe UI", 11))
        self._preview_label.pack()

        ctk.CTkButton(left, text="📂  Choisir une image",
                      command=self._pick_image,
                      fg_color=ACCENT, hover_color="#1D4ED8",
                      width=200, height=40,
                      font=ctk.CTkFont("Segoe UI", 13, "bold"),
                      corner_radius=10).pack(pady=(12, 0))

        # Colonne droite : formulaire
        right = ctk.CTkFrame(card, fg_color="transparent")
        right.grid(row=0, column=1, padx=24, pady=24, sticky="ns")
        card.columnconfigure(1, weight=1)

        ctk.CTkLabel(right, text="Nom de l'utilisateur",
                     font=ctk.CTkFont("Segoe UI", 13, "bold"),
                     text_color="white").pack(anchor="w", pady=(0, 4))

        self._name_entry = ctk.CTkEntry(right, placeholder_text="Ex : Alice Dupont",
                                        width=300, height=42,
                                        font=ctk.CTkFont("Segoe UI", 13),
                                        corner_radius=10)
        self._name_entry.pack(pady=(0, 12))

        # Rôle
        ctk.CTkLabel(right, text="Rôle",
                     font=ctk.CTkFont("Segoe UI", 13, "bold"),
                     text_color="white").pack(anchor="w", pady=(0, 4))
        self._role_var = tk.StringVar(value="user")
        role_frame = ctk.CTkFrame(right, fg_color="transparent")
        role_frame.pack(anchor="w", pady=(0, 16))
        ctk.CTkRadioButton(role_frame, text="👤  Utilisateur",
                           variable=self._role_var, value="user",
                           font=ctk.CTkFont("Segoe UI", 13),
                           text_color="#94A3B8",
                           fg_color=ACCENT).pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(role_frame, text="🛡️  Administrateur",
                           variable=self._role_var, value="admin",
                           font=ctk.CTkFont("Segoe UI", 13),
                           text_color="#FCD34D",
                           fg_color="#D97706").pack(side="left")

        ctk.CTkButton(right, text="✅  Enrôler",
                      command=self._enroll,
                      fg_color=SUCCESS, hover_color="#15803D",
                      width=300, height=48,
                      font=ctk.CTkFont("Segoe UI", 14, "bold"),
                      corner_radius=10).pack()

        # Status
        self._status_label = ctk.CTkLabel(right, text="",
                                          font=ctk.CTkFont("Segoe UI", 12),
                                          wraplength=300,
                                          text_color="#94A3B8")
        self._status_label.pack(pady=(16, 0))

        # Info minutiae
        self._info_frame = ctk.CTkFrame(right, fg_color="#0F172A", corner_radius=10)
        self._info_frame.pack(fill="x", pady=(12, 0))
        self._info_label = ctk.CTkLabel(self._info_frame, text="",
                                         font=ctk.CTkFont("Segoe UI", 11),
                                         text_color="#64748B", wraplength=280)
        self._info_label.pack(padx=12, pady=10)

    def _pick_image(self):
        path = filedialog.askopenfilename(
            title="Sélectionner une image d'empreinte",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Tous", "*.*")]
        )
        if not path:
            return
        self._image_path = path
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Fichier illisible.")
            photo = _cv2_to_photoimage(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            self._photo = photo
            self._preview_label.configure(image=photo, text="")
        except Exception as e:
            self._status_label.configure(text=f"⚠️ {e}", text_color=WARN)

    def _enroll(self):
        name = self._name_entry.get().strip()
        role = self._role_var.get()  # 'admin' or 'user'
        if not name:
            self._status_label.configure(text="⚠️ Veuillez entrer un nom.", text_color=WARN)
            return
        if not self._image_path:
            self._status_label.configure(text="⚠️ Veuillez choisir une image.", text_color=WARN)
            return

        self._status_label.configure(text="⏳ Traitement en cours...", text_color="#94A3B8")
        self.update()

        def _run():
            try:
                template_bytes, minutiae = full_pipeline(self._image_path)
                encrypted = encrypt_template(template_bytes)
                enroll_user(name, encrypted, role=role)
                role_label = "🛡️ Administrateur" if role == "admin" else "👤 Utilisateur"
                msg = (f"✅ Enrôlement réussi !\n"
                       f"   • Nom : {name}  |  Rôle : {role_label}\n"
                       f"   • Minutiae détectées : {len(minutiae)}\n"
                       f"   • Template chiffré (AES) en base SQLite.")
                self.after(0, lambda: self._on_success(msg, len(minutiae)))
                if self._refresh_cb:
                    self.after(100, self._refresh_cb)
            except ValueError as e:
                self.after(0, lambda: self._status_label.configure(
                    text=f"❌ {e}", text_color=DANGER))
            except Exception as e:
                self.after(0, lambda: self._status_label.configure(
                    text=f"❌ Erreur : {e}", text_color=DANGER))

        threading.Thread(target=_run, daemon=True).start()

    def _on_success(self, msg: str, n_minutiae: int):
        self._status_label.configure(text=msg, text_color=SUCCESS)
        self._info_label.configure(
            text=f"🔒 Seul le template mathématique est stocké ({n_minutiae} minutiae).\n"
                 "L'image originale n'est jamais sauvegardée (RGPD).",
            text_color="#64748B")
        self._name_entry.delete(0, "end")
        self._role_var.set("user")  # reset role to default


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Authentification
# ──────────────────────────────────────────────────────────────────────────────
class AuthTab(ctk.CTkFrame):
    def __init__(self, master, on_admin_unlock=None, **kwargs):
        super().__init__(master, fg_color=BG_ROOT, **kwargs)
        self._image_path: Optional[str] = None
        self._photo = None
        self._on_admin_unlock = on_admin_unlock
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="🔍  Authentification",
                     font=ctk.CTkFont("Segoe UI", 22, "bold"),
                     text_color="white").pack(pady=(28, 4))
        ctk.CTkLabel(self, text="Vérifiez une empreinte contre les profils enregistrés.",
                     font=ctk.CTkFont("Segoe UI", 13),
                     text_color="#94A3B8").pack(pady=(0, 18))

        card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=16)
        card.pack(padx=40, pady=8, fill="both", expand=True)

        # Gauche : image
        left = ctk.CTkFrame(card, fg_color="transparent")
        left.grid(row=0, column=0, padx=24, pady=24, sticky="n")

        self._preview_label = ctk.CTkLabel(left,
            text="Aucune image sélectionnée",
            width=280, height=220, fg_color="#0F172A",
            corner_radius=12, text_color="#475569",
            font=ctk.CTkFont("Segoe UI", 11))
        self._preview_label.pack()

        ctk.CTkButton(left, text="📂  Choisir une image",
                      command=self._pick_image,
                      fg_color=ACCENT, hover_color="#1D4ED8",
                      width=200, height=40,
                      font=ctk.CTkFont("Segoe UI", 13, "bold"),
                      corner_radius=10).pack(pady=(12, 0))

        # Droite : formulaire
        right = ctk.CTkFrame(card, fg_color="transparent")
        right.grid(row=0, column=1, padx=24, pady=24, sticky="ns")
        card.columnconfigure(1, weight=1)

        ctk.CTkLabel(right, text="Utilisateur cible",
                     font=ctk.CTkFont("Segoe UI", 13, "bold"),
                     text_color="white").pack(anchor="w", pady=(0, 4))

        self._user_var = tk.StringVar(value="— Sélectionner —")
        self._user_menu = ctk.CTkOptionMenu(right, variable=self._user_var,
                                             values=["— Sélectionner —"],
                                             width=300, height=42,
                                             font=ctk.CTkFont("Segoe UI", 13),
                                             corner_radius=10,
                                             fg_color="#1E3A5F",
                                             button_color=ACCENT,
                                             button_hover_color="#1D4ED8")
        self._user_menu.pack(pady=(0, 8))

        ctk.CTkButton(right, text="🔄  Rafraîchir la liste",
                      command=self._refresh_users,
                      fg_color="#334155", hover_color="#475569",
                      width=300, height=36,
                      font=ctk.CTkFont("Segoe UI", 12),
                      corner_radius=8).pack(pady=(0, 16))

        # Seuil
        ctk.CTkLabel(right, text="Seuil de similarité",
                     font=ctk.CTkFont("Segoe UI", 12, "bold"),
                     text_color="white").pack(anchor="w")
        self._thresh_var = tk.DoubleVar(value=0.40)
        self._thresh_slider = ctk.CTkSlider(right, from_=0.01, to=0.99,
                                             variable=self._thresh_var,
                                             width=300,
                                             command=self._update_thresh_label)
        self._thresh_slider.pack(pady=(4, 2))
        self._thresh_label = ctk.CTkLabel(right, text="0.40",
                                           font=ctk.CTkFont("Segoe UI", 12),
                                           text_color="#94A3B8")
        self._thresh_label.pack(anchor="w", pady=(0, 16))

        ctk.CTkButton(right, text="🔎  Authentifier",
                      command=self._authenticate,
                      fg_color=ACCENT, hover_color="#1D4ED8",
                      width=300, height=48,
                      font=ctk.CTkFont("Segoe UI", 14, "bold"),
                      corner_radius=10).pack()

        # Résultat
        self._result_frame = ctk.CTkFrame(right, fg_color="#0F172A", corner_radius=14)
        self._result_frame.pack(fill="x", pady=(18, 0))
        self._result_label = ctk.CTkLabel(self._result_frame, text="En attente...",
                                           font=ctk.CTkFont("Segoe UI", 18, "bold"),
                                           text_color="#475569")
        self._result_label.pack(pady=14)
        self._score_label = ctk.CTkLabel(self._result_frame, text="",
                                          font=ctk.CTkFont("Segoe UI", 12),
                                          text_color="#64748B")
        self._score_label.pack(pady=(0, 10))

        self._refresh_users()

    def _update_thresh_label(self, val=None):
        self._thresh_label.configure(text=f"{self._thresh_var.get():.2f}")

    def _pick_image(self):
        path = filedialog.askopenfilename(
            title="Sélectionner une image d'empreinte",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Tous", "*.*")]
        )
        if not path:
            return
        self._image_path = path
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            photo = _cv2_to_photoimage(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            self._photo = photo
            self._preview_label.configure(image=photo, text="")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def _refresh_users(self):
        users = get_all_users()
        names = [u[1] for u in users] if users else []
        values = names if names else ["—Aucun utilisateur enregistré—"]
        self._user_menu.configure(values=values)
        if names:
            self._user_var.set(names[0])
        else:
            self._user_var.set(values[0])

    def _authenticate(self):
        name = self._user_var.get()
        if not self._image_path:
            messagebox.showwarning("SecurePrint", "Veuillez choisir une image.")
            return
        if name.startswith("—"):
            messagebox.showwarning("SecurePrint", "Veuillez sélectionner un utilisateur.")
            return

        self._result_label.configure(text="⏳ Analyse...", text_color="#94A3B8")
        self._score_label.configure(text="")
        self.update()

        _name = name

        def _run():
            try:
                user = get_user_by_name(_name)
                if not user:
                    self.after(0, lambda: self._result_label.configure(
                        text="❌ Utilisateur introuvable", text_color=DANGER))
                    return

                uid, uname, urole = user  # unpack role too
                blob = get_template(uid)
                if not blob:
                    self.after(0, lambda: self._result_label.configure(
                        text="❌ Template introuvable", text_color=DANGER))
                    return

                threshold = round(self._thresh_var.get(), 2)
                score, is_match = authenticate(self._image_path, blob, threshold)
                self.after(0, lambda: self._show_result(score, is_match, threshold, _name, urole))
            except Exception as e:
                self.after(0, lambda: self._result_label.configure(
                    text=f"❌ Erreur : {e}", text_color=DANGER))

        threading.Thread(target=_run, daemon=True).start()

    def _show_result(self, score: float, is_match: bool, threshold: float,
                     name: str = "", role: str = "user"):
        if is_match:
            self._result_label.configure(
                text="✅  ACCÈS AUTORISÉ", text_color=SUCCESS)
            # A user with role 'admin' unlocks admin mode. Optionally restrict it to a
            # single account name with the SECUREPRINT_ADMIN_NAME environment variable.
            admin_name = os.environ.get("SECUREPRINT_ADMIN_NAME", "").strip()
            if role == "admin" and (not admin_name or name == admin_name) and self._on_admin_unlock:
                self._on_admin_unlock()
        else:
            self._result_label.configure(
                text="🚫  ACCÈS REFUSÉ", text_color=DANGER)

        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        self._score_label.configure(
            text=f"Score : {score:.4f}  |  Seuil : {threshold:.2f}\n[{bar}]",
            text_color="#94A3B8")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Gestion RGPD
# ──────────────────────────────────────────────────────────────────────────────
class ManageTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=BG_ROOT, **kwargs)
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="👥  Gestion des Utilisateurs (RGPD)",
                     font=ctk.CTkFont("Segoe UI", 22, "bold"),
                     text_color="white").pack(pady=(28, 4))
        ctk.CTkLabel(self, text="Droit à l'oubli : suppression définitive de l'identité et du template.",
                     font=ctk.CTkFont("Segoe UI", 13),
                     text_color="#94A3B8").pack(pady=(0, 18))

        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(padx=40, fill="x")

        ctk.CTkButton(toolbar, text="🔄  Actualiser",
                      command=self._refresh,
                      fg_color="#334155", hover_color="#475569",
                      width=140, height=38,
                      font=ctk.CTkFont("Segoe UI", 12),
                      corner_radius=8).pack(side="left")

        self._count_label = ctk.CTkLabel(toolbar, text="",
                                          font=ctk.CTkFont("Segoe UI", 12),
                                          text_color="#94A3B8")
        self._count_label.pack(side="left", padx=16)

        self._scroll = ctk.CTkScrollableFrame(self, fg_color=BG_CARD,
                                               corner_radius=14, height=340)
        self._scroll.pack(padx=40, pady=12, fill="both", expand=True)

        self._refresh()

    def _refresh(self):
        for w in self._scroll.winfo_children():
            w.destroy()

        users = get_all_users()
        self._count_label.configure(text=f"{len(users)} utilisateur(s) enregistré(s)")

        if not users:
            ctk.CTkLabel(self._scroll, text="Aucun utilisateur enregistré.",
                         font=ctk.CTkFont("Segoe UI", 13),
                         text_color="#475569").pack(pady=30)
            return

        for uid, name, role in users:
            row = ctk.CTkFrame(self._scroll, fg_color="#0F172A", corner_radius=10)
            row.pack(fill="x", pady=4, padx=4)

            # Rôle badge
            if role == "admin":
                badge_text = "🛡️ Admin"
                badge_color = "#78350F"
                badge_fg = "#FCD34D"
            else:
                badge_text = "👤 User"
                badge_color = "#1E3A5F"
                badge_fg = "#93C5FD"

            ctk.CTkLabel(row, text=f"  {name}",
                         font=ctk.CTkFont("Segoe UI", 14),
                         text_color="white").pack(side="left", padx=(12, 4), pady=12)
            ctk.CTkLabel(row, text=badge_text,
                         font=ctk.CTkFont("Segoe UI", 11, "bold"),
                         text_color=badge_fg,
                         fg_color=badge_color,
                         corner_radius=6,
                         padx=8, pady=2).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=f"ID #{uid}",
                         font=ctk.CTkFont("Segoe UI", 11),
                         text_color="#475569").pack(side="left", padx=8)

            uid_copy = uid
            name_copy = name
            ctk.CTkButton(row, text="🗑  Supprimer",
                           command=lambda u=uid_copy, n=name_copy: self._delete(u, n),
                           fg_color=DANGER, hover_color="#991B1B",
                           width=130, height=34,
                           font=ctk.CTkFont("Segoe UI", 12),
                           corner_radius=8).pack(side="right", padx=16, pady=8)

    def _delete(self, user_id: int, name: str):
        if not messagebox.askyesno(
            "Confirmation suppression",
            f"Supprimer définitivement '{name}' et toutes ses données ?\n"
            "Cette action est irréversible (droit à l'oubli RGPD).",
            icon="warning"
        ):
            return
        ok = delete_user(user_id)
        if ok:
            messagebox.showinfo("SecurePrint", f"✅ '{name}' supprimé de la base.")
        else:
            messagebox.showerror("SecurePrint", "Erreur lors de la suppression.")
        self._refresh()


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Évaluation FVC2000
# ──────────────────────────────────────────────────────────────────────────────
class FVCEvalTab(ctk.CTkFrame):
    """Onglet d'évaluation biométrique FVC2000 (FAR, FRR, EER)."""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=BG_ROOT, **kwargs)
        self._dataset_dir = tk.StringVar()
        self._running = False
        self._canvas_widget = None
        self._result_data = None
        self._build_ui()

    # ── Construction UI ────────────────────────────────────────────────────
    def _build_ui(self):
        # Titre
        ctk.CTkLabel(self, text="📊  Évaluation FVC2000",
                     font=ctk.CTkFont("Segoe UI", 22, "bold"),
                     text_color="white").pack(pady=(24, 2))
        ctk.CTkLabel(self,
                     text="Protocole standard FVC : paires genuines / imposteurs → FAR, FRR, EER",
                     font=ctk.CTkFont("Segoe UI", 12),
                     text_color="#94A3B8").pack(pady=(0, 12))

        # ── Carte de configuration ──────────────────────────────────────────
        cfg = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=14)
        cfg.pack(padx=40, pady=(0, 8), fill="x")

        # Ligne 1 : sélection du dossier
        row1 = ctk.CTkFrame(cfg, fg_color="transparent")
        row1.pack(fill="x", padx=20, pady=(16, 6))
        ctk.CTkLabel(row1, text="Dossier dataset FVC :",
                     font=ctk.CTkFont("Segoe UI", 13, "bold"),
                     text_color="white", width=200, anchor="w").pack(side="left")
        self._dir_entry = ctk.CTkEntry(row1, textvariable=self._dataset_dir,
                                       placeholder_text="ex: data/fvc2000/DB1_B",
                                       width=340, height=38,
                                       font=ctk.CTkFont("Segoe UI", 12),
                                       corner_radius=8)
        self._dir_entry.pack(side="left", padx=(0, 8))
        ctk.CTkButton(row1, text="📁 Parcourir", command=self._browse,
                      fg_color="#334155", hover_color="#475569",
                      width=110, height=38,
                      font=ctk.CTkFont("Segoe UI", 12),
                      corner_radius=8).pack(side="left")

        # Ligne 2 : options
        row2 = ctk.CTkFrame(cfg, fg_color="transparent")
        row2.pack(fill="x", padx=20, pady=(0, 6))
        self._enroll_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(row2, text="Enrôler dans la base SQLite SecurePrint",
                        variable=self._enroll_var,
                        font=ctk.CTkFont("Segoe UI", 12),
                        text_color="#94A3B8",
                        checkmark_color="white",
                        fg_color=ACCENT).pack(side="left", padx=(0, 24))
        ctk.CTkLabel(row2, text="Seuil système :",
                     font=ctk.CTkFont("Segoe UI", 12),
                     text_color="#94A3B8").pack(side="left")
        self._thresh_var = tk.DoubleVar(value=0.60)
        ctk.CTkSlider(row2, from_=0.1, to=0.99, variable=self._thresh_var,
                      width=120).pack(side="left", padx=6)
        self._thresh_lbl = ctk.CTkLabel(row2, text="0.60",
                                         font=ctk.CTkFont("Segoe UI", 12),
                                         text_color="white", width=36)
        self._thresh_lbl.pack(side="left")
        self._thresh_var.trace_add("write", lambda *_: self._thresh_lbl.configure(
            text=f"{self._thresh_var.get():.2f}"))

        # Ligne 3 : bouton lancer + progress
        row3 = ctk.CTkFrame(cfg, fg_color="transparent")
        row3.pack(fill="x", padx=20, pady=(4, 16))
        self._run_btn = ctk.CTkButton(row3, text="▶  Lancer l'évaluation",
                                       command=self._run,
                                       fg_color=ACCENT, hover_color="#1D4ED8",
                                       width=220, height=42,
                                       font=ctk.CTkFont("Segoe UI", 13, "bold"),
                                       corner_radius=10)
        self._run_btn.pack(side="left")
        self._prog_lbl = ctk.CTkLabel(row3, text="En attente…",
                                       font=ctk.CTkFont("Segoe UI", 11),
                                       text_color="#475569")
        self._prog_lbl.pack(side="left", padx=14)

        self._progress = ctk.CTkProgressBar(cfg, width=560, height=10,
                                             corner_radius=5,
                                             fg_color="#0F172A",
                                             progress_color=ACCENT)
        self._progress.pack(padx=20, pady=(0, 14))
        self._progress.set(0)

        # ── Zone de résultats (métriques + graphique) ───────────────────────
        self._bottom = ctk.CTkFrame(self, fg_color="transparent")
        self._bottom.pack(padx=40, pady=4, fill="both", expand=True)

        # Colonne gauche : métriques
        self._metrics_frame = ctk.CTkFrame(self._bottom,
                                            fg_color=BG_CARD, corner_radius=14,
                                            width=260)
        self._metrics_frame.pack(side="left", fill="y", padx=(0, 8))
        self._metrics_frame.pack_propagate(False)
        self._build_metrics_placeholder()

        # Colonne droite : graphique matplotlib
        self._plot_frame = ctk.CTkFrame(self._bottom,
                                         fg_color=BG_CARD, corner_radius=14)
        self._plot_frame.pack(side="left", fill="both", expand=True)
        self._build_plot_placeholder()

    def _build_metrics_placeholder(self):
        ctk.CTkLabel(self._metrics_frame,
                     text="Métriques",
                     font=ctk.CTkFont("Segoe UI", 14, "bold"),
                     text_color="white").pack(pady=(18, 4))
        ctk.CTkLabel(self._metrics_frame,
                     text="Lancez l'évaluation\npour afficher les résultats.",
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color="#475569",
                     justify="center").pack(pady=10)

    def _build_plot_placeholder(self):
        ctk.CTkLabel(self._plot_frame,
                     text="Le graphique FAR/FRR et les distributions\ns'afficheront ici après l'évaluation.",
                     font=ctk.CTkFont("Segoe UI", 12),
                     text_color="#475569",
                     justify="center").pack(expand=True)

    # ── Actions ────────────────────────────────────────────────────────────
    def _browse(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(
            title="Sélectionner le dossier FVC2000 (ex: DB1_B)")
        if d:
            self._dataset_dir.set(d)

    def _run(self):
        if self._running:
            return
        d = self._dataset_dir.get().strip()
        if not d:
            self._prog_lbl.configure(text="⚠ Veuillez choisir un dossier.",
                                      text_color=WARN)
            return
        if not os.path.isdir(d):
            self._prog_lbl.configure(text="⚠ Dossier introuvable.",
                                      text_color=WARN)
            return

        self._running = True
        self._run_btn.configure(state="disabled")
        self._progress.set(0)
        self._prog_lbl.configure(text="Démarrage…", text_color="#94A3B8")

        import threading
        threading.Thread(target=self._eval_thread, args=(d,), daemon=True).start()

    def _eval_thread(self, dataset_dir: str):
        try:
            from modules.evaluator import run_fvc_evaluation
            results = run_fvc_evaluation(
                dataset_dir,
                n_thresholds=150,
                progress_callback=self._on_progress,
            )
            self.after(0, lambda: self._on_done(results))
        except Exception as e:
            self.after(0, lambda: self._on_error(str(e)))

    def _on_progress(self, current: int, total: int, msg: str):
        pct = current / max(total, 1)
        self.after(0, lambda: self._progress.set(pct))
        self.after(0, lambda: self._prog_lbl.configure(
            text=msg[:55], text_color="#94A3B8"))

    def _on_error(self, msg: str):
        self._running = False
        self._run_btn.configure(state="normal")
        self._prog_lbl.configure(text=f"❌ {msg[:60]}", text_color=DANGER)
        self._progress.set(0)

    def _on_done(self, results: dict):
        self._running = False
        self._result_data = results
        self._progress.set(1.0)
        eer_pct = results["eer"] * 100
        self._prog_lbl.configure(
            text=f"✅ Terminé — EER = {eer_pct:.2f}%", text_color=SUCCESS)
        self._run_btn.configure(state="normal")
        self._update_metrics(results)
        self._update_plot(results)

    # ── Métriques ──────────────────────────────────────────────────────────
    def _update_metrics(self, r: dict):
        for w in self._metrics_frame.winfo_children():
            w.destroy()

        eer_pct = r["eer"] * 100
        eer_color = SUCCESS if eer_pct < 10 else (WARN if eer_pct < 25 else DANGER)

        ctk.CTkLabel(self._metrics_frame, text="Métriques FVC",
                     font=ctk.CTkFont("Segoe UI", 14, "bold"),
                     text_color="white").pack(pady=(18, 8))

        # EER grand
        ctk.CTkFrame(self._metrics_frame, fg_color="#0F172A",
                     corner_radius=10, height=80).pack(fill="x", padx=14, pady=4)
        eer_card = self._metrics_frame.winfo_children()[-1]
        ctk.CTkLabel(eer_card, text="EER",
                     font=ctk.CTkFont("Segoe UI", 11), text_color="#94A3B8").pack(pady=(10, 0))
        ctk.CTkLabel(eer_card, text=f"{eer_pct:.2f}%",
                     font=ctk.CTkFont("Segoe UI", 26, "bold"),
                     text_color=eer_color).pack()
        ctk.CTkLabel(eer_card, text=f"seuil = {r['eer_threshold']:.4f}",
                     font=ctk.CTkFont("Segoe UI", 10), text_color="#475569").pack(pady=(0, 8))

        def _row(label, value, vcolor="white"):
            f = ctk.CTkFrame(self._metrics_frame, fg_color="#0F172A", corner_radius=8)
            f.pack(fill="x", padx=14, pady=3)
            ctk.CTkLabel(f, text=label, font=ctk.CTkFont("Segoe UI", 11),
                         text_color="#64748B", anchor="w").pack(side="left", padx=10, pady=7)
            ctk.CTkLabel(f, text=value, font=ctk.CTkFont("Segoe UI", 11, "bold"),
                         text_color=vcolor, anchor="e").pack(side="right", padx=10)

        _row("Doigts",          str(r["n_fingers"]))
        _row("Paires genuines", str(r["n_genuine_pairs"]), "#16A34A")
        _row("Paires impostor", str(r["n_impostor_pairs"]), "#DC2626")
        _row("Score moy. gen.", f"{r['genuine_mean']:.4f}", "#16A34A")
        _row("Score moy. imp.", f"{r['impostor_mean']:.4f}", "#DC2626")
        _row("σ genuins",       f"{r['genuine_std']:.4f}")
        _row("σ imposteurs",    f"{r['impostor_std']:.4f}")

    # ── Graphique ──────────────────────────────────────────────────────────
    def _update_plot(self, r: dict):
        for w in self._plot_frame.winfo_children():
            w.destroy()
        self._canvas_widget = None

        if not HAS_MATPLOTLIB:
            ctk.CTkLabel(self._plot_frame,
                         text="matplotlib non installé.\nExécutez : pip install matplotlib",
                         font=ctk.CTkFont("Segoe UI", 12),
                         text_color=WARN).pack(expand=True)
            return

        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import numpy as np

        gen  = r["genuine_scores"]
        imp  = r["impostor_scores"]
        far  = np.array(r["far"])
        frr  = np.array(r["frr"])
        thr  = np.array(r["thresholds"])
        eer  = r["eer"]
        eer_t = r["eer_threshold"]

        fig = Figure(figsize=(7, 4), facecolor="#1E293B")
        fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.13, wspace=0.35)

        ax1 = fig.add_subplot(121, facecolor="#0F172A")
        ax2 = fig.add_subplot(122, facecolor="#0F172A")

        for ax in (ax1, ax2):
            ax.tick_params(colors="#64748B", labelsize=7)
            for s in ax.spines.values():
                s.set_edgecolor("#334155")

        bins = np.linspace(0, 1, 35)
        ax1.hist(gen, bins=bins, alpha=0.75, color="#16A34A",
                 label=f"Genuins ({len(gen)})")
        ax1.hist(imp, bins=bins, alpha=0.75, color="#DC2626",
                 label=f"Imposteurs ({len(imp)})")
        ax1.axvline(eer_t, color="#FBBF24", linestyle="--", linewidth=1.2)
        ax1.set_xlabel("Score", color="#94A3B8", fontsize=8)
        ax1.set_ylabel("Paires", color="#94A3B8", fontsize=8)
        ax1.set_title("Distributions", color="white", fontsize=9)
        leg1 = ax1.legend(fontsize=7, facecolor="#0F172A",
                          edgecolor="#334155", labelcolor="white")

        ax2.plot(thr, far * 100, color="#DC2626", linewidth=1.8, label="FAR (%)")
        ax2.plot(thr, frr * 100, color="#2563EB", linewidth=1.8, label="FRR (%)")
        ax2.axvline(eer_t, color="#FBBF24", linestyle="--", linewidth=1.2,
                    label=f"EER={eer*100:.1f}%")
        ax2.scatter([eer_t], [eer * 100], color="#FBBF24", s=50, zorder=5)
        ax2.set_xlabel("Seuil", color="#94A3B8", fontsize=8)
        ax2.set_ylabel("Taux (%)", color="#94A3B8", fontsize=8)
        ax2.set_title("FAR / FRR", color="white", fontsize=9)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 100)
        leg2 = ax2.legend(fontsize=7, facecolor="#0F172A",
                          edgecolor="#334155", labelcolor="white")

        fig.suptitle(
            f"FVC Eval — {r['n_fingers']} doigts   EER = {eer*100:.2f}%",
            color="white", fontsize=10
        )

        canvas = FigureCanvasTkAgg(fig, master=self._plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._canvas_widget = canvas


# ──────────────────────────────────────────────────────────────────────────────
# Fenêtre principale
# ──────────────────────────────────────────────────────────────────────────────
class SecurePrintApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SecurePrint — Authentification Biométrique Chiffrée")
        self.geometry("950x680")
        self.minsize(900, 600)
        self.configure(fg_color=BG_ROOT)
        self._is_admin = False

        # ── Header ──────────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="#1E293B", height=64, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(header,
                     text="🔐  SecurePrint",
                     font=ctk.CTkFont("Segoe UI", 20, "bold"),
                     text_color="white").pack(side="left", padx=24, pady=16)
        ctk.CTkLabel(header,
                     text="Authentification Biométrique Chiffrée  ·  AES-256  ·  RGPD",
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color="#64748B").pack(side="left", padx=8)

        # Admin badge + exit button (hidden until admin logs in)
        self._exit_admin_btn = ctk.CTkButton(
            header, text="🚪  Quitter Admin",
            command=self._exit_admin,
            fg_color=DANGER, hover_color="#991B1B",
            width=148, height=36,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            corner_radius=8)
        # not packed yet

        self._admin_badge = ctk.CTkLabel(
            header, text="🛡️  Mode Administrateur",
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            text_color="#FCD34D",
            fg_color="#78350F",
            corner_radius=8,
            padx=12, pady=4)
        # not packed yet

        # ── Tabview ─────────────────────────────────────────────────────────
        self._tabview = ctk.CTkTabview(
            self, fg_color=BG_ROOT,
            segmented_button_fg_color="#1E293B",
            segmented_button_selected_color=ACCENT,
            segmented_button_unselected_color="#1E293B",
            segmented_button_selected_hover_color="#1D4ED8",
            text_color="white")
        self._tabview.pack(fill="both", expand=True, padx=0, pady=0)

        # ── Decide initial tab layout ────────────────────────────────────────
        users = get_all_users()
        if not users:
            # First run / empty DB: show all tabs so admin can enroll
            self._build_all_tabs(first_run=True)
        else:
            # Normal mode: auth tab only
            self._build_auth_only()

    # ── Tab builders ────────────────────────────────────────────────────────

    def _build_auth_only(self):
        """Show only the Authentification tab (locked/normal mode)."""
        self._tabview.add("  Authentification  ")
        self._auth_tab = AuthTab(
            self._tabview.tab("  Authentification  "),
            on_admin_unlock=self._on_admin_unlock)
        self._auth_tab.pack(fill="both", expand=True)

    def _build_all_tabs(self, first_run: bool = False):
        """Add Enrôlement, Gestion, Évaluation FVC tabs (admin-only)."""
        # Auth tab may already exist
        existing = [self._tabview.tab(t) for t in self._tabview._tab_dict] if hasattr(self._tabview, '_tab_dict') else []
        _ = existing  # unused, just guard

        if "  Authentification  " not in (self._tabview._tab_dict or {}):
            self._tabview.add("  Authentification  ")
            self._auth_tab = AuthTab(
                self._tabview.tab("  Authentification  "),
                on_admin_unlock=self._on_admin_unlock)
            self._auth_tab.pack(fill="both", expand=True)

        for tab_name in ["  Enrôlement  ", "  Gestion  ", "  Évaluation FVC  "]:
            try:
                self._tabview.add(tab_name)
            except Exception:
                pass  # already exists

        self._manage_tab = ManageTab(self._tabview.tab("  Gestion  "))
        self._manage_tab.pack(fill="both", expand=True)

        self._enroll_tab = EnrollTab(
            self._tabview.tab("  Enrôlement  "),
            refresh_callback=self._manage_tab._refresh)
        self._enroll_tab.pack(fill="both", expand=True)

        self._fvc_tab = FVCEvalTab(self._tabview.tab("  Évaluation FVC  "))
        self._fvc_tab.pack(fill="both", expand=True)

        if first_run:
            # Navigate to enrolment tab so admin can register first
            self._tabview.set("  Enrôlement  ")

    # ── Admin mode callbacks ─────────────────────────────────────────────────

    def _on_admin_unlock(self):
        """Called when the admin fingerprint is verified successfully."""
        if self._is_admin:
            return
        self._is_admin = True

        # Show admin badge and exit button in header
        self._admin_badge.pack(side="right", padx=(0, 16), pady=14)
        self._exit_admin_btn.pack(side="right", padx=(0, 6), pady=14)

        # Unlock all tabs
        self._build_all_tabs(first_run=False)
        messagebox.showinfo(
            "SecurePrint — Mode Admin",
            "🛡️  Accès administrateur accordé.\n"
            "Tous les modules sont maintenant disponibles.")

    def _exit_admin(self):
        """Revoke admin mode and collapse back to auth-only."""
        if not messagebox.askyesno(
            "Quitter Mode Admin",
            "Voulez-vous quitter le mode administrateur ?\n"
            "Les onglets Enrôlement, Gestion et Évaluation FVC seront masqués.",
            icon="warning"
        ):
            return

        self._is_admin = False

        # Hide admin UI
        self._admin_badge.pack_forget()
        self._exit_admin_btn.pack_forget()

        # Remove admin-only tabs
        for tab in ["  Enrôlement  ", "  Gestion  ", "  Évaluation FVC  "]:
            try:
                self._tabview.delete(tab)
            except Exception:
                pass

        # Return to auth tab
        self._tabview.set("  Authentification  ")


def launch():
    """Initialise la base de données, la clé de chiffrement, puis lance l'UI."""
    init_db()
    generate_key()
    app = SecurePrintApp()
    app.mainloop()
