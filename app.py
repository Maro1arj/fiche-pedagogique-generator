import os
import io
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Générateur de Fiche Pédagogique",
    page_icon="📘",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
#  SIDEBAR — API + Model
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "Clé API Gemini",
    type="password",
    value=os.getenv("GEMINI_API_KEY", ""),
    help="Obtenez-en une gratuitement sur https://aistudio.google.com/apikey",
)

model_name = st.sidebar.selectbox(
    "Modèle",
    ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
    index=0,
)

output_format = st.sidebar.radio(
    "Format de sortie",
    ["Markdown", "LaTeX (.tex)", "Les deux"],
    index=2,
)

# ─────────────────────────────────────────────────────────────
#  FILE EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────
def extract_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def extract_docx(file) -> str:
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def extract_any(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        return extract_pdf(file)
    if name.endswith(".docx"):
        return extract_docx(file)
    if name.endswith(".txt"):
        return extract_txt(file)
    raise ValueError(f"Format non supporté : {file.name}")

# ─────────────────────────────────────────────────────────────
#  THE PROMPT — encodes the layout from your PDF
# ─────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTIONS = """
Tu es un enseignant-chercheur en didactique des mathématiques au Maroc.
Tu maîtrises parfaitement les orientations pédagogiques officielles,
l'approche par compétences, le socioconstructivisme, la pédagogie de l'erreur
et l'évaluation diagnostique / formative / sommative.

On te fournit le CONTENU d'une leçon de mathématiques.
Tu dois produire une FICHE PÉDAGOGIQUE complète et un SCÉNARIO PÉDAGOGIQUE
INTÉGRANT LES TICE, en respectant STRICTEMENT la structure ci-dessous.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARTIE I — FICHE PÉDAGOGIQUE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 1. En-tête
| Champ | Valeur |
|---|---|
| Établissement | ... |
| Fiche pédagogique du cours N° | ... |
| Matière | Mathématiques |
| Réalisé(e) par | ... |
| Niveau scolaire | ... |
| Semestre | ... |
| Année scolaire | ... |
| Durée | ... |
| Les outils didactiques | ... |
| Les orientations pédagogiques | (résumé des instructions officielles relatives à la leçon) |

## 2. Structure de la leçon et capacités visées
- **Prérequis** : ...
- **Les compétences attendues** : ...
- **Les extensions** : ...

| Séance | Durée | Capacité attendue |
|---|---|---|
| 1 | ... | ... |

## 3. Scénario de la fiche pédagogique

| Séance | Étapes | Activités / contenu / exercices | Rôle de l'apprenant | Rôle de l'enseignant | Durée & méthode | Outils didactiques | Erreurs / difficultés | Recommandations |
|---|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

Les étapes doivent couvrir :
- L'activité diagnostique (5–10 min)
- L'activité constructiviste décomposée en 5 phases :
  phase exploratoire, phase d'action, phase de formulation,
  phase de validation, phase d'institutionnalisation (résumé)
- Application / évaluation (10–15 min)
- Exercices de soutien et de renforcement

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARTIE II — SCÉNARIO PÉDAGOGIQUE INTÉGRANT LES TICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 1. Carte d'identité du scénario
| Champ | Valeur |
|---|---|
| L'enseignant | ... |
| Titre | ... |
| Niveau | ... |
| L'activité | ... |
| Lieu et délais | ... |
| Compétences transversales | ... |
| Compétences disciplinaires | ... |
| Les prérequis | ... |
| Les outils technologiques | ... |
| Le programme utilisé | ... |
| La valeur ajoutée | ... |
| Résumé | ... |

## 2. Le déroulement

| Questions | Rôle du professeur | Rôle de l'apprenant | Remarques |
|---|---|---|---|
| ... | ... | ... | ... |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLES DE STYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Langue : FRANÇAIS.
- Style : moderne, clair, professionnel, pédagogique.
- Ton : confiant, didactique, orienté vers l'action de l'enseignant.
- Contenu concret : situations problèmes, exemples numériques, QCM,
  exercices d'application et de soutien — adaptés au niveau indiqué.
- Utilise des tableaux Markdown.
- Ne rien inventer qui contredit les orientations pédagogiques marocaines.
- Si une information manque, indique « ... » (à compléter par l'enseignant).
"""

USER_TEMPLATE = """
Voici le CONTENU de la leçon à transformer en fiche pédagogique :

──────────────────────────────
{lesson}
──────────────────────────────

Informations complémentaires fournies par l'enseignant :
- Établissement : {etab}
- Niveau scolaire : {niveau}
- Nom de l'enseignant : {prof}
- Année scolaire : {annee}
- Durée : {duree}

Génère maintenant la fiche pédagogique + le scénario TICE complets,
en respectant scrupuleusement la structure et le style demandés.
"""

# ─────────────────────────────────────────────────────────────
#  GENERATION
# ─────────────────────────────────────────────────────────────
def generate_markdown(lesson_text, meta):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = SYSTEM_INSTRUCTIONS + "\n\n" + USER_TEMPLATE.format(
        lesson=lesson_text, **meta
    )
    response = model.generate_content(prompt)
    return response.text

def markdown_to_latex(markdown_text):
    """Second call : convertit le markdown en LaTeX propre."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"""
Convertis le document Markdown suivant en un document LaTeX COMPLET
et COMPILABLE avec pdflatex (classe article, encodage utf8, langue française
via \\usepackage[utf8]{{inputenc}} \\usepackage[T1]{{fontenc}} \\usepackage[french]{{babel}}).

Contraintes :
- Utilise \\begin{{tabular}}{{|p{{...}}|...}} pour les tableaux, avec \\hline.
- Utilise \\section*{{}}, \\subsection*{{}}, \\textbf{{}}.
- Ne mets AUCUN texte en dehors du document LaTeX.
- Renvoie UNIQUEMENT le code LaTeX, sans commentaire.

Document Markdown :

{markdown_text}
"""
    response = model.generate_content(prompt)
    latex = response.text.strip()
    # nettoyage des éventuels ```latex ... ```
    if latex.startswith("```"):
        latex = latex.split("\n", 1)[1]
        if latex.endswith("```"):
            latex = latex[:-3]
    return latex

# ─────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────
st.title("📘 Générateur de Fiche Pédagogique — Mathématiques")
st.caption("Collez votre leçon ou importez un fichier. L'IA produit la fiche + le scénario TICE.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("1. Contenu de la leçon")
    uploaded = st.file_uploader(
        "Importer un fichier (.pdf, .docx, .txt)",
        type=["pdf", "docx", "txt"],
    )
    pasted = st.text_area(
        "...ou collez le texte de la leçon ici",
        height=280,
        placeholder="Ex : La proportionnalité — tableau, coefficient, pourcentages, échelle...",
    )

with col2:
    st.subheader("2. Informations")
    etab   = st.text_input("Établissement", "")
    niveau = st.text_input("Niveau scolaire", "1AC")
    prof   = st.text_input("Enseignant", "")
    annee  = st.text_input("Année scolaire", "2025-2026")
    duree  = st.text_input("Durée", "2h")

st.divider()

if st.button("🚀 Générer la fiche pédagogique", type="primary", use_container_width=True):
    if not api_key:
        st.error("Veuillez saisir votre clé API Gemini dans la barre latérale.")
        st.stop()

    # Récupération du contenu
    lesson_text = ""
    if uploaded is not None:
        try:
            lesson_text = extract_any(uploaded)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            st.stop()
    if pasted.strip():
        lesson_text += "\n\n" + pasted

    if not lesson_text.strip():
        st.warning("Veuillez fournir le contenu de la leçon.")
        st.stop()

    meta = {
        "etab": etab or "...",
        "niveau": niveau or "...",
        "prof": prof or "...",
        "annee": annee or "...",
        "duree": duree or "...",
    }

    with st.spinner("Génération en cours... (cela peut prendre 20–60 secondes)"):
        try:
            md = generate_markdown(lesson_text, meta)
        except Exception as e:
            st.error(f"Erreur Gemini : {e}")
            st.stop()

    st.success("✅ Fiche générée !")

    # Aperçu Markdown
    if output_format in ("Markdown", "Les deux"):
        st.subheader("📄 Aperçu (Markdown)")
        st.markdown(md)
        st.download_button(
            "⬇️ Télécharger .md",
            data=md.encode("utf-8"),
            file_name="fiche_pedagogique.md",
            mime="text/markdown",
        )

    # Génération LaTeX
    if output_format in ("LaTeX (.tex)", "Les deux"):
        with st.spinner("Conversion en LaTeX..."):
            try:
                tex = markdown_to_latex(md)
                st.subheader("📐 Code LaTeX")
                st.code(tex, language="latex")
                st.download_button(
                    "⬇️ Télécharger .tex",
                    data=tex.encode("utf-8"),
                    file_name="fiche_pedagogique.tex",
                    mime="application/x-tex",
                )
            except Exception as e:
                st.error(f"Erreur lors de la conversion LaTeX : {e}")

st.divider()
st.caption("Fait avec ❤️ — Python · Streamlit · Gemini")
