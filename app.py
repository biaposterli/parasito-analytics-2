"""
LaPaHV — Laboratório de Parasitologia Humana e Veterinária
Análise epidemiológica de levantamentos de parasitoses intestinais em pré-escolares.

Rodar localmente:
    pip install -r requirements.txt
    streamlit run app.py
"""
import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis_engine import (
    METHOD_COLUMNS,
    PATOGENICOS,
    COMENSAIS,
    build_per_child,
    compute_metrics,
    normalize_columns,
    validate_columns,
)
from report_pdf import build_pdf_report

APP_DIR = Path(__file__).parent
LOGO_PATH = APP_DIR / "logo.png"

# ----------------------------------------------------------------
# Paleta extraída da logo do LaPaHV
# ----------------------------------------------------------------
BG = "#F5F0EA"
SURFACE = "#FFFFFF"
INK = "#11483D"
INK_SOFT = "#3E5F55"
INK_FAINT = "#7C8B81"
TEAL = "#328567"
TEAL_DARK = "#11483D"
TEAL_TINT = "#E2F0E7"
BRICK = "#9C4A2E"
BRICK_TINT = "#F1E2D8"
AMBER = "#5F8A4E"
SAGE = "#7DAE84"
LINE = "#DAE1D5"

st.set_page_config(
    page_title="LaPaHV — Análise de Parasitoses",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------
# CSS de marca
# ----------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif;
        color: {INK};
    }}
    .stApp {{
        background-color: {BG};
    }}
    h1, h2, h3 {{
        font-family: 'Fraunces', serif !important;
        color: {TEAL_DARK} !important;
    }}
    /* ----- eyebrow / section labels ----- */
    .lapahv-eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12.5px;
        letter-spacing: .06em;
        text-transform: uppercase;
        color: {TEAL};
    }}
    .lapahv-eyebrow.on-dark {{ color: {SAGE}; }}

    /* ----- top brand bar ----- */
    .lapahv-topbar {{
        display:flex; align-items:center; gap:14px;
        padding: 4px 0 18px 0;
        border-bottom: 1px solid {LINE};
        margin-bottom: 22px;
    }}
    .lapahv-topbar .name {{
        font-family:'Fraunces', serif; font-weight:600; font-size: 19px; color:{TEAL_DARK};
        line-height:1.15;
    }}
    .lapahv-topbar .sub {{
        font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:.04em;
        color:{INK_FAINT}; text-transform:uppercase;
    }}

    /* ----- hero ----- */
    .lapahv-hero {{
        background: {SURFACE};
        border: 1px solid {LINE};
        border-radius: 4px;
        padding: 30px 34px;
        margin-bottom: 22px;
    }}
    .lapahv-hero h2 {{
        margin: 4px 0 10px 0 !important;
        font-size: 30px !important;
    }}
    .lapahv-hero p {{
        color: {INK_SOFT}; font-size: 15px; max-width: 720px; margin-bottom: 0;
    }}

    /* ----- generic section card ----- */
    .lapahv-card {{
        background: {SURFACE};
        border: 1px solid {LINE};
        border-radius: 4px;
        padding: 18px 20px;
        height: 100%;
    }}
    .lapahv-card .kicker {{
        font-family:'IBM Plex Mono', monospace; font-size: 11.5px; letter-spacing:.05em;
        text-transform:uppercase; color:{INK_FAINT}; margin-bottom: 8px; display:block;
    }}

    /* ----- step header (numbered badge + title) ----- */
    .lapahv-step {{
        display:flex; align-items:center; gap:12px; margin: 6px 0 2px 0;
    }}
    .lapahv-step .badge {{
        flex: 0 0 auto;
        width: 34px; height: 34px; border-radius: 50%;
        background: {TEAL_DARK}; color: white;
        font-family:'IBM Plex Mono', monospace; font-weight:600; font-size: 14px;
        display:flex; align-items:center; justify-content:center;
    }}
    .lapahv-step .step-title {{
        font-family:'Fraunces', serif; font-weight:600; color:{TEAL_DARK}; font-size: 21px;
    }}
    .lapahv-step-wrap {{
        background:{SURFACE}; border:1px solid {LINE}; border-radius:4px;
        padding: 22px 24px 24px 24px; margin-bottom: 18px;
    }}

    /* ----- metrics ----- */
    div[data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {LINE};
        padding: 16px 18px;
        border-radius: 4px;
    }}
    div[data-testid="stMetric"] label {{
        font-family: 'IBM Plex Mono', monospace !important;
        text-transform: uppercase;
        font-size: 11px !important;
        letter-spacing: .04em;
        color: {INK_FAINT} !important;
    }}

    /* ----- buttons ----- */
    .stDownloadButton button, .stButton button {{
        background-color: {TEAL_DARK};
        color: white;
        border: 1px solid {TEAL_DARK};
        font-family: 'IBM Plex Mono', monospace;
        border-radius: 3px;
        font-size: 13px;
    }}
    .stDownloadButton button:hover, .stButton button:hover {{
        background-color: {TEAL};
        border-color: {TEAL};
        color: white;
    }}

    /* ----- tags ----- */
    .lapahv-tag {{
        display:inline-block; font-family:'IBM Plex Mono', monospace; font-size:12px;
        padding:3px 9px; margin:2px; border-radius:3px; border:1px solid;
    }}

    /* ----- notes ----- */
    .lapahv-note {{
        background: {BRICK_TINT}; border-left: 3px solid {BRICK};
        padding: 12px 16px; font-size: 14px; color: {INK_SOFT}; border-radius: 3px;
    }}

    /* ----- section title inside report ----- */
    .lapahv-section-title {{
        font-family:'Fraunces', serif; font-weight:600; color:{TEAL_DARK};
        font-size: 19px; margin: 2px 0 2px 0;
    }}
    .lapahv-section-subtitle {{
        font-family:'Fraunces', serif; font-weight:600; color:{TEAL_DARK};
        font-size: 15.5px; margin: 2px 0 2px 0;
    }}
    .lapahv-section-caption {{
        color:{INK_FAINT}; font-size: 12.5px; margin-bottom: 10px;
    }}

    /* ----- tabs (sectorized report) ----- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid {LINE};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: .03em;
        color: {INK_FAINT};
        padding: 10px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        color: {TEAL_DARK} !important;
        border-bottom: 2px solid {TEAL_DARK} !important;
        font-weight: 600;
    }}

    /* ----- status pill (sidebar) ----- */
    .lapahv-pill {{
        display:inline-flex; align-items:center; gap:6px;
        font-family:'IBM Plex Mono', monospace; font-size: 11.5px;
        padding: 5px 10px; border-radius: 20px; border: 1px solid {LINE};
        color: {INK_SOFT}; background: {SURFACE};
    }}
    .lapahv-pill .dot {{
        width:7px; height:7px; border-radius:50%; background:{TEAL};
    }}

    hr {{ border-color: {LINE}; }}
    section[data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {LINE};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

CHART_COLORWAY = [TEAL, BRICK, AMBER, SAGE, TEAL_DARK, "#84978D"]
PLOTLY_LAYOUT = dict(
    font_family="Inter, sans-serif",
    plot_bgcolor="white",
    paper_bgcolor="white",
    colorway=CHART_COLORWAY,
    margin=dict(t=20, b=20, l=10, r=10),
)


def section_title(text, caption=None):
    st.markdown(f'<div class="lapahv-section-title">{text}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="lapahv-section-caption">{caption}</div>', unsafe_allow_html=True)


def subsection_title(text, caption=None):
    st.markdown(f'<div class="lapahv-section-subtitle">{text}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="lapahv-section-caption">{caption}</div>', unsafe_allow_html=True)


def step_header(number, title):
    st.markdown(
        f"""<div class="lapahv-step">
        <div class="badge">{number}</div>
        <div class="step-title">{title}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def format_ic(inf, sup):
    """Formata um par (limite_inferior, limite_superior) de IC95% para exibição.
    Retorna um texto neutro quando o intervalo não pôde ser calculado (n=0)."""
    if inf is None or sup is None:
        return "IC95% não calculável (n=0)"
    return f"IC95%: {inf:.1f}–{sup:.1f}%"


def with_ic_column(df: pd.DataFrame, prev_col="prevalencia", inf_col="ic95_inf", sup_col="ic95_sup",
                    label="IC 95%") -> pd.DataFrame:
    """Devolve uma cópia do DataFrame com uma coluna extra combinando o IC95% num
    único texto legível ('11.0 – 42.1%'), para exibir ao lado da prevalência sem
    poluir a tabela com duas colunas numéricas soltas."""
    out = df.copy()
    if inf_col in out.columns and sup_col in out.columns:
        out[label] = out.apply(
            lambda r: f"{r[inf_col]:.1f} – {r[sup_col]:.1f}%" if pd.notna(r[inf_col]) and pd.notna(r[sup_col]) else "—",
            axis=1,
        )
        out = out.drop(columns=[inf_col, sup_col])
    return out


# ----------------------------------------------------------------
# Modelo de planilha (bytes) — usado no Passo 01 e na sidebar
# ----------------------------------------------------------------
def generate_template_bytes() -> bytes:
    headers = [
        "id_paciente", "coleta", "nome_crianca", "nome_responsavel",
        "status_amostra", "status_lamina",
        "metodo_graham", "metodo_baermann_picanco", "metodo_hpj", "metodo_willis",
        "observacoes",
    ]
    example = pd.DataFrame(
        [
            ["F-001", "P1", "Exemplo Da Silva", "Nome Do Responsável", "Entregue", "Entregue", "-", "-", "E. nana", "-", ""],
            ["F-001", "P2", "Exemplo Da Silva", "Nome Do Responsável", "Entregue", "Entregue", "Enterobius vermicularis", "-", "-", "-", ""],
            ["F-001", "P3", "Exemplo Da Silva", "Nome Do Responsável", "Não entregue", "Entregue", "-", "", "", "", "amostra fecal não coletada"],
        ],
        columns=headers,
    )
    legenda = pd.DataFrame(
        [
            ["id_paciente", "Código único da criança (repete nas linhas de P1/P2/P3)", "texto livre, ex.: F-001"],
            ["coleta", "Qual das até 3 coletas essa linha representa", "P1, P2 ou P3"],
            ["nome_crianca", "Nome da criança", "texto livre"],
            ["nome_responsavel", "Nome do responsável (opcional)", "texto livre ou vazio"],
            ["status_amostra", "Status de entrega do POTE DE FEZES — usado por HPJ, Willis e Baermann-Picanço", "Entregue / Não entregue"],
            ["status_lamina", "Status de entrega da LÂMINA — usado exclusivamente pelo método Graham", "Entregue / Não entregue"],
            ["metodo_graham", "Resultado do Graham (só se status_lamina = Entregue)", "'-' (negativo), 'Amostra insuficiente', ou espécie(s) separadas por ' + '"],
            ["metodo_baermann_picanco", "Resultado do Baermann-Picanço (só se status_amostra = Entregue)", "idem acima"],
            ["metodo_hpj", "Resultado do HPJ (só se status_amostra = Entregue)", "idem acima"],
            ["metodo_willis", "Resultado do Willis (só se status_amostra = Entregue)", "idem acima"],
            ["observacoes", "Observações livres (opcional)", "texto livre ou vazio"],
        ],
        columns=["Coluna", "O que é", "Valores aceitos"],
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        example.to_excel(writer, sheet_name="Dados", index=False)
        legenda.to_excel(writer, sheet_name="Legenda", index=False)
    return buf.getvalue()


TEMPLATE_BYTES = generate_template_bytes()

# ==================================================================
# SIDEBAR — marca, fluxo de trabalho e status
# ==================================================================
with st.sidebar:
    col_l, col_t = st.columns([1, 3])
    with col_l:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=48)
    with col_t:
        st.markdown(
            """<div style="line-height:1.2;">
            <div style="font-family:'Fraunces',serif; font-weight:600; font-size:16px; color:#11483D;">LaPaHV</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#7C8B81; letter-spacing:.03em;">ANÁLISE DE PARASITOSES</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="lapahv-eyebrow">Fluxo de trabalho</span>', unsafe_allow_html=True)
    st.markdown(
        """
- **01 · Baixe** o modelo de planilha
- **02 · Preencha** com os dados da coleta
- **03 · Envie** o arquivo e receba o relatório
        """
    )

    st.divider()
    st.download_button(
        "⬇ Modelo (.xlsx)",
        data=TEMPLATE_BYTES,
        file_name="Modelo_Levantamento_Parasitoses.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

    st.divider()
    st.markdown('<span class="lapahv-eyebrow">Sobre as amostras</span>', unsafe_allow_html=True)
    st.caption(
        "**Pote de fezes** → HPJ, Willis, Baermann-Picanço.\n\n"
        "**Lâmina (swab)** → exclusivamente Graham."
    )

    st.divider()
    st.markdown('<span class="lapahv-eyebrow">Classificação clínica</span>', unsafe_allow_html=True)
    st.caption(
        "*Entamoeba histolytica/dispar* é tratada como **patogênica**: a diferenciação "
        "morfológica entre as duas formas não é possível no laboratório, então todo achado "
        "do complexo é reportado como potencialmente patogênico."
    )

# ==================================================================
# TOPO — marca + título
# ==================================================================
st.markdown(
    f"""<div class="lapahv-topbar">
        <div>
            <div class="name">Laboratório de Parasitologia Humana e Veterinária</div>
            <div class="sub">Painel de análise epidemiológica</div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# ==================================================================
# HERO
# ==================================================================
st.markdown(
    """<div class="lapahv-hero">
    <span class="lapahv-eyebrow">Duas amostras, um relatório correto</span>
    <h2>Da planilha de campo ao relatório epidemiológico.</h2>
    <p>Todo exame coproparasitológico infantil coleta dois materiais separados — e cada um segue
    seu próprio caminho até o diagnóstico. Baixe o modelo, preencha os dados da sua pesquisa e
    envie abaixo para receber a análise por criança, já respeitando qual método pôde de fato ser
    feito em cada uma.</p>
    </div>""",
    unsafe_allow_html=True,
)

diag1, diag2 = st.columns(2)
with diag1:
    st.markdown(
        f"""<div class="lapahv-card">
        <span class="kicker">Pote de fezes</span>
        <span class="lapahv-tag" style="border-color:{LINE}; color:{INK_SOFT};">HPJ</span>
        <span class="lapahv-tag" style="border-color:{LINE}; color:{INK_SOFT};">Willis</span>
        <span class="lapahv-tag" style="border-color:{LINE}; color:{INK_SOFT};">Baermann-Picanço</span>
        </div>""",
        unsafe_allow_html=True,
    )
with diag2:
    st.markdown(
        f"""<div class="lapahv-card">
        <span class="kicker">Lâmina (swab)</span>
        <span class="lapahv-tag" style="border-color:{LINE}; color:{INK_SOFT};">Graham</span>
        </div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ==================================================================
# PASSO 01 — Baixar modelo
# ==================================================================
with st.container():
    st.markdown('<div class="lapahv-step-wrap">', unsafe_allow_html=True)
    step_header(1, "Baixe o modelo de planilha")
    st.write(
        "Um arquivo .xlsx com as colunas certas, os valores aceitos em cada uma e uma linha de "
        "exemplo — para preencher com os dados da sua coleta."
    )
    st.download_button(
        "⬇ Baixar modelo (.xlsx)",
        data=TEMPLATE_BYTES,
        file_name="Modelo_Levantamento_Parasitoses.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ==================================================================
# PASSO 02 — Enviar planilha
# ==================================================================
with st.container():
    st.markdown('<div class="lapahv-step-wrap">', unsafe_allow_html=True)
    step_header(2, "Envie a planilha preenchida")
    st.write(
        "Aceita o modelo baixado acima, preenchido com uma linha por coleta (P1/P2/P3) de cada "
        "criança."
    )
    uploaded_file = st.file_uploader("Escolha o arquivo .xlsx", type=["xlsx", "xls"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ==================================================================
# PASSO 03 — Relatório (sectorizado em abas)
# ==================================================================
if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = next((s for s in xls.sheet_names if s.strip().lower() == "dados"), xls.sheet_names[0])
        df_raw = pd.read_excel(xls, sheet_name=sheet_name)
        df = normalize_columns(df_raw)
        errors = validate_columns(df)
    except Exception as exc:  # noqa: BLE001
        errors = [f"Não consegui ler esse arquivo. Confira se é um .xlsx válido, exportado a "
                  f"partir do modelo. ({exc})"]
        df = None

    if errors:
        for e in errors:
            st.error(e)
    else:
        metrics = compute_metrics(df)

        if metrics["total"] == 0:
            st.error("Nenhuma criança identificada. Confira se a coluna **id_paciente** está preenchida.")
        else:
            st.success(
                f"Planilha processada: {metrics['total']} crianças, {len(df)} coletas. "
                "Relatório gerado abaixo."
            )

            with st.container():
                st.markdown('<div class="lapahv-step-wrap">', unsafe_allow_html=True)
                step_header(3, "Relatório da análise")
                st.caption(
                    f"{metrics['total']} crianças cadastradas · {len(metrics['fecal'])} com amostra "
                    f"fecal entregue · {len(metrics['apenas_lamina'])} só com lâmina."
                )

                n_inconclusivas = (
                    len(metrics["fecal_inconclusivo"])
                    + len(metrics["lamina_only_inconclusivo"])
                )

                tab_geral, tab_especies, tab_metodos, tab_base, tab_export = st.tabs(
                    ["📊  Visão geral", "🦠  Espécies & parasitos", "🔬  Métodos & amostragem", "📋  Base por criança", "⬇  Exportar"]
                )

                # ---------------------------------------------------------
                # ABA 1 — VISÃO GERAL
                # ---------------------------------------------------------
                with tab_geral:
                    section_title("Resumo executivo")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Prevalência — amostra fecal", f"{metrics['prev_fecal']:.1f}%",
                                   help="Base: crianças com resultado CONCLUSIVO em pelo menos um método "
                                        "fecal (HPJ, Willis ou Baermann-Picanço). Achados exclusivos do "
                                        "Graham (lâmina) não entram aqui — veja 'Espécies & parasitos' "
                                        "para a prevalência de Enterobius. Crianças com todos os resultados "
                                        "fecais marcados como 'Amostra insuficiente' são excluídas do "
                                        "denominador (não contam como negativas). IC95% calculado pelo "
                                        "método de Wilson.")
                        st.caption(format_ic(metrics["prev_fecal_ic95_inf"], metrics["prev_fecal_ic95_sup"]))
                    with c2:
                        st.metric("Prevalência — só lâmina", f"{metrics['prev_lamina']:.1f}%",
                                   help="Base: crianças que só entregaram lâmina (nunca o pote de fezes) e "
                                        "tiveram resultado conclusivo no Graham. Reflete apenas Enterobius "
                                        "vermicularis — único parasita pesquisável só com a lâmina. IC95% "
                                        "calculado pelo método de Wilson (preferível ao normal para n "
                                        "pequeno, como costuma ser o caso deste subgrupo).")
                        st.caption(format_ic(metrics["prev_lamina_ic95_inf"], metrics["prev_lamina_ic95_sup"]))
                    with c3:
                        st.metric("Prevalência combinada", f"{metrics['prev_combinada']:.1f}%",
                                   help="Todas as crianças com pelo menos um resultado conclusivo, em "
                                        "qualquer domínio (fezes e/ou lâmina) — use com cautela, mistura "
                                        "profundidades diagnósticas diferentes. IC95% calculado pelo "
                                        "método de Wilson.")
                        st.caption(format_ic(metrics["prev_combinada_ic95_inf"], metrics["prev_combinada_ic95_sup"]))

                    if n_inconclusivas > 0:
                        st.markdown(
                            f"""<div class="lapahv-note"><strong>Amostras inconclusivas:</strong>
                            {len(metrics['fecal_inconclusivo'])} criança(s) entregaram pote de fezes mas
                            tiveram <em>todos</em> os métodos fecais marcados como "Amostra insuficiente"
                            (ou sem resultado registrado){', ' + str(len(metrics['lamina_only_inconclusivo'])) + ' criança(s) na mesma situação só com lâmina' if len(metrics['lamina_only_inconclusivo']) else ''}.
                            Essas crianças foram excluídas dos denominadores de prevalência acima — elas
                            <u>não</u> contam como negativas, pois não houve diagnóstico conclusivo.</div>""",
                            unsafe_allow_html=True,
                        )

                    if len(metrics["apenas_lamina"]) > 0:
                        st.markdown(
                            f"""<div class="lapahv-note" style="margin-top:8px;"><strong>Atenção:</strong> {len(metrics['apenas_lamina'])}
                            criança(s) só entregaram a lâmina, nunca o pote de fezes — para elas, apenas
                            <em>Enterobius vermicularis</em> pôde ser pesquisado. Por isso a prevalência
                            principal do estudo considera só quem teve amostra fecal analisada; o subgrupo
                            de só-lâmina é reportado à parte.</div>""",
                            unsafe_allow_html=True,
                        )

                    st.write("")
                    section_title("Crianças por profundidade de amostragem")
                    cat_df = metrics["cat_counts"].rename("n_criancas").to_frame()
                    cat_df["%"] = (100 * cat_df["n_criancas"] / metrics["total"]).round(1)
                    st.dataframe(cat_df, width='stretch')

                # ---------------------------------------------------------
                # ABA 2 — ESPÉCIES & PARASITOS
                # ---------------------------------------------------------
                with tab_especies:
                    section_title(
                        "Prevalência de todos os parasitos",
                        "Reúne, num só gráfico, as espécies encontradas por métodos fecais (HPJ, "
                        "Willis, Baermann-Picanço) e por Graham/lâmina. As bases de cálculo diferem "
                        "por domínio — a tabela ao lado do gráfico mostra o denominador (Base N) e "
                        "o(s) método(s) que detectou(aram) cada espécie. Quando a mesma espécie foi "
                        "encontrada em métodos de domínios diferentes (ex.: Enterobius vermicularis, "
                        "tipicamente pelo Graham, mas ocasionalmente também visível num método "
                        "fecal), ela aparece numa única linha \"Fecal + Lâmina\" — o cálculo é feito "
                        "por criança, então quem foi detectado por mais de um método conta uma vez "
                        "só, não duas.",
                    )
                    colT, colU = st.columns([3, 2])
                    with colT:
                        if not metrics["todos_parasitos_resumo"].empty:
                            fig_all = px.bar(
                                metrics["todos_parasitos_resumo"].sort_values("prevalencia"),
                                x="prevalencia", y="especie", orientation="h",
                                color="categoria",
                                color_discrete_map={"Patogênico": BRICK, "Comensal": AMBER, "Não classificado": SAGE},
                                pattern_shape="dominio",
                                labels={"prevalencia": "Prevalência (%)", "especie": "", "dominio": "Amostra"},
                                hover_data={"metodos": True, "base_n": True, "n": True},
                            )
                            fig_all.update_layout(**PLOTLY_LAYOUT, showlegend=True, legend_title="")
                            st.plotly_chart(fig_all, width='stretch')
                        else:
                            st.info("Nenhum parasito detectado nesta base.")
                    with colU:
                        todos_display = with_ic_column(metrics["todos_parasitos_resumo"]).rename(columns={
                            "especie": "Espécie", "categoria": "Categoria", "dominio": "Amostra",
                            "n": "N", "prevalencia": "Prevalência %", "base_n": "Base N", "metodos": "Método(s)",
                        })
                        st.dataframe(todos_display, width='stretch', hide_index=True)
                        st.caption("IC95% pelo método de Wilson, calculado sobre o denominador (Base N) de cada espécie.")

                    st.write("")
                    subsection_title(
                        "Prevalência por método diagnóstico e espécie",
                        "Cada célula mostra a prevalência (%) daquela espécie especificamente pelo "
                        "método indicado, com denominador = crianças com resultado conclusivo NAQUELE "
                        "método. Uma mesma espécie pode aparecer em mais de um método fecal (ex.: um "
                        "ovo de helminto pode ser visto tanto no HPJ quanto no Willis).",
                    )
                    if not metrics["metodo_especie_resumo"].empty:
                        pivot = metrics["metodo_especie_resumo"].pivot_table(
                            index="especie", columns="metodo", values="prevalencia", aggfunc="first",
                        ).reindex(columns=["Graham", "HPJ", "Willis", "Baermann-Picanço"])
                        st.dataframe(pivot, width='stretch')
                        with st.expander("Ver com intervalos de confiança (IC95%, Wilson)"):
                            me_display = with_ic_column(metrics["metodo_especie_resumo"]).rename(columns={
                                "metodo": "Método", "especie": "Espécie", "n": "N",
                                "prevalencia": "Prevalência %", "categoria": "Categoria",
                            })
                            st.dataframe(me_display, width='stretch', hide_index=True)
                    else:
                        st.info("Nenhum dado suficiente para o cruzamento método x espécie.")

                    st.write("")
                    section_title(
                        "Prevalência por espécie — métodos fecais",
                        "Base: fezes com resultado conclusivo. Mostra cada espécie encontrada por "
                        "HPJ, Willis ou Baermann-Picanço especificamente — inclusive Enterobius "
                        "vermicularis, se algum caso tiver sido identificado incidentalmente num "
                        "método fecal (achado válido, não é erro). A prevalência combinada dessa "
                        "espécie com o Graham, sem contar a mesma criança duas vezes, está no "
                        "gráfico unificado acima.",
                    )
                    colA, colB = st.columns([3, 2])
                    with colA:
                        if not metrics["especies_resumo"].empty:
                            fig = px.bar(
                                metrics["especies_resumo"].sort_values("prevalencia"),
                                x="prevalencia", y="especie", orientation="h",
                                color="categoria",
                                color_discrete_map={"Patogênico": BRICK, "Comensal": AMBER, "Não classificado": SAGE},
                                labels={"prevalencia": "Prevalência (%)", "especie": ""},
                            )
                            fig.update_layout(**PLOTLY_LAYOUT, showlegend=True, legend_title="")
                            st.plotly_chart(fig, width='stretch')
                        else:
                            st.info("Nenhuma espécie fecal detectada nesta base.")
                    with colB:
                        esp_display = with_ic_column(metrics["especies_resumo"]).rename(columns={
                            "especie": "Espécie", "n": "N", "prevalencia": "Prevalência %", "categoria": "Categoria",
                        })
                        st.dataframe(esp_display, width='stretch', hide_index=True)
                        st.caption("IC95% pelo método de Wilson (base: fezes conclusivas).")

                    st.write("")
                    section_title("Mono x poliparasitismo", "Base: espécies de origem fecal.")
                    colC, colD = st.columns([2, 3])
                    with colC:
                        fig2 = go.Figure(
                            data=[go.Pie(
                                labels=["Negativo", "Monoparasitismo", "Poliparasitismo"],
                                values=[metrics["neg"], metrics["mono"], metrics["poli"]],
                                marker_colors=[SAGE, TEAL, BRICK],
                                hole=0.45,
                            )]
                        )
                        fig2.update_layout(**PLOTLY_LAYOUT)
                        st.plotly_chart(fig2, width='stretch')
                    with colD:
                        st.markdown("**Combinações mais frequentes**")
                        if metrics["combos_resumo"].empty:
                            st.info("Nenhuma coinfecção registrada.")
                        else:
                            st.dataframe(metrics["combos_resumo"], width='stretch', hide_index=True)

                # ---------------------------------------------------------
                # ABA 3 — MÉTODOS & AMOSTRAGEM
                # ---------------------------------------------------------
                with tab_metodos:
                    section_title(
                        "Comparação entre métodos diagnósticos",
                        "Denominador = crianças com resultado conclusivo naquele método específico "
                        "(exclui quem teve só 'Amostra insuficiente' nesse método).",
                    )
                    colE, colF = st.columns([3, 2])
                    with colE:
                        fig3 = px.bar(
                            metrics["metodos_resumo"], x="metodo", y="prevalencia",
                            labels={"prevalencia": "Prevalência (%)", "metodo": ""},
                        )
                        fig3.update_traces(marker_color=TEAL)
                        fig3.update_layout(**PLOTLY_LAYOUT)
                        st.plotly_chart(fig3, width='stretch')
                    with colF:
                        met_display = with_ic_column(metrics["metodos_resumo"]).rename(columns={
                            "metodo": "Método", "amostra_biologica": "Amostra",
                            "n_criancas_testaveis": "Testáveis", "n_criancas_positivas": "Positivas",
                            "n_criancas_inconclusivas": "Inconclusivas", "prevalencia": "Prevalência %",
                        })
                        st.dataframe(met_display, width='stretch', hide_index=True)
                        st.caption("IC95% pelo método de Wilson.")

                    st.write("")
                    subsection_title(
                        "HPJ x Willis — teste de McNemar",
                        "Compara os dois métodos aplicados à MESMA amostra de fezes da mesma criança "
                        "(dados pareados) — testa se um método detecta mais positivos que o outro, "
                        "usando só as crianças em que os dois métodos discordaram entre si.",
                    )
                    mc = metrics["mcnemar_hpj_willis"]
                    if mc["n_pareado"] == 0 or mc["tabela"] is None:
                        st.info("Sem crianças com resultado conclusivo em HPJ e Willis simultaneamente — "
                                "teste não calculado.")
                    else:
                        tb = mc["tabela"]
                        mc_col1, mc_col2 = st.columns([2, 3])
                        with mc_col1:
                            mc_table = pd.DataFrame(
                                [[tb["pp"], tb["pn"]], [tb["np"], tb["nn"]]],
                                index=["HPJ +", "HPJ −"], columns=["Willis +", "Willis −"],
                            )
                            st.dataframe(mc_table, width='stretch')
                        with mc_col2:
                            metodo_label = {
                                "exato": "teste exato (< 25 discordâncias)",
                                "chi2_corrigido": "qui-quadrado com correção de continuidade",
                                "sem_discordancia": "sem discordâncias — nada a testar",
                            }.get(mc["metodo"], mc["metodo"])
                            st.metric("p-valor (McNemar)", f"{mc['p_valor']:.4f}" if mc["p_valor"] is not None else "—")
                            st.caption(
                                f"n pareado = {mc['n_pareado']} · {tb['pn'] + tb['np']} discordância(s) "
                                f"({tb['pn']} HPJ+/Willis−, {tb['np']} HPJ−/Willis+) · {metodo_label}."
                            )
                            if mc["p_valor"] is not None and mc["p_valor"] < 0.05:
                                st.caption("p < 0,05 — diferença estatisticamente significativa entre os métodos nesta amostra.")
                            elif mc["p_valor"] is not None:
                                st.caption("p ≥ 0,05 — sem evidência estatística de diferença entre os métodos nesta amostra.")

                    st.write("")
                    section_title(
                        "Efeito do número de potes de fezes entregues",
                        "Usa somente positividade fecal (HPJ/Willis/Baermann-Picanço). Compara "
                        "SUBGRUPOS diferentes de crianças (quem entregou 1, 2 ou 3 potes), o que pode "
                        "ter viés de seleção — veja a curva cumulativa abaixo, calculada no mesmo grupo "
                        "de crianças, para uma estimativa sem esse viés.",
                    )
                    if not metrics["efeito_n_coletas"].empty:
                        fig4 = px.bar(
                            metrics["efeito_n_coletas"], x="n_potes_entregues", y="prevalencia",
                            labels={"prevalencia": "Prevalência (%)", "n_potes_entregues": "Nº de potes entregues"},
                            text="n_criancas",
                        )
                        fig4.update_traces(marker_color=TEAL_DARK, texttemplate="n=%{text}", textposition="outside")
                        fig4.update_layout(**PLOTLY_LAYOUT)
                        st.plotly_chart(fig4, width='stretch')
                    else:
                        st.info("Dados insuficientes para este gráfico.")

                    ca = metrics["cochran_armitage_efeito_coletas"]
                    if ca["p_valor"] is not None:
                        st.caption(
                            f"Teste de tendência de Cochran-Armitage: Z = {ca['estatistica_z']:.3f}, "
                            f"p = {ca['p_valor']:.4f} ({ca['n_grupos']} grupos). {ca['aviso']}"
                        )
                    elif ca["n_grupos"] and ca["n_grupos"] >= 2:
                        st.caption(f"Teste de tendência de Cochran-Armitage não calculável nesta base. {ca['aviso']}")

                    st.write("")
                    section_title("Ganho marginal por amostra — curva cumulativa")
                    if not metrics["fecal_cumulativa"].empty:
                        n_grupo = int(metrics["fecal_cumulativa"]["n_criancas"].iloc[0])
                        k_max = int(metrics["fecal_cumulativa"]["k"].max())
                        st.caption(
                            f"Mesmo grupo de {n_grupo} criança(s) que entregou o número máximo de potes "
                            f"observado no estudo ({k_max}), medindo quantas ficariam positivas se o "
                            "laboratório parasse na 1ª, 2ª ... até a última coleta. Como é a mesma "
                            "criança sendo acompanhada (medida repetida), este gráfico não sofre o viés "
                            "de comparar subgrupos diferentes de crianças."
                        )
                        fig5 = px.line(
                            metrics["fecal_cumulativa"], x="k", y="prevalencia_cumulativa", markers=True,
                            labels={"k": "Nº de potes considerados (cumulativo)", "prevalencia_cumulativa": "Prevalência cumulativa (%)"},
                        )
                        fig5.update_traces(line_color=TEAL_DARK, marker_color=BRICK)
                        fig5.update_layout(**PLOTLY_LAYOUT)
                        st.plotly_chart(fig5, width='stretch')
                    else:
                        st.info("Dados insuficientes para a curva cumulativa (nenhuma criança com coletas identificadas por P1/P2/P3).")

                # ---------------------------------------------------------
                # ABA 4 — BASE POR CRIANÇA
                # ---------------------------------------------------------
                with tab_base:
                    section_title("Base completa por criança")
                    display_cols = [
                        "id_paciente", "nome_crianca", "categoria_amostragem",
                        "n_coletas_pote_entregue", "n_coletas_lamina_entregue",
                        "fecal_status", "lamina_status",
                        "positivo_algum_metodo", "especies_str",
                    ]
                    st.dataframe(
                        metrics["por_crianca"][display_cols].sort_values("id_paciente"),
                        width='stretch', hide_index=True, height=460,
                    )

                # ---------------------------------------------------------
                # ABA 5 — EXPORTAR
                # ---------------------------------------------------------
                def generate_report_excel_bytes(m: dict) -> bytes:
                    buf = io.BytesIO()
                    list_cols = ["especies", "especies_fecais", "especies_lamina",
                                 "especies_Graham", "especies_Baermann-Picanço", "especies_HPJ", "especies_Willis"]
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        m["por_crianca"].drop(columns=[c for c in list_cols if c in m["por_crianca"].columns]).to_excel(
                            writer, sheet_name="Base_por_Crianca", index=False
                        )
                        m["cat_counts"].rename("n").to_frame().to_excel(writer, sheet_name="Categoria_Amostragem")
                        pd.DataFrame([
                            {"metrica": "Prevalência — amostra fecal (conclusiva)", "valor_pct": m["prev_fecal"],
                             "ic95_inf": m["prev_fecal_ic95_inf"], "ic95_sup": m["prev_fecal_ic95_sup"],
                             "n_criancas": len(m["fecal_conclusivo"])},
                            {"metrica": "Prevalência — só lâmina (conclusiva)", "valor_pct": m["prev_lamina"],
                             "ic95_inf": m["prev_lamina_ic95_inf"], "ic95_sup": m["prev_lamina_ic95_sup"],
                             "n_criancas": len(m["lamina_only_conclusivo"])},
                            {"metrica": "Prevalência combinada (conclusiva)", "valor_pct": m["prev_combinada"],
                             "ic95_inf": m["prev_combinada_ic95_inf"], "ic95_sup": m["prev_combinada_ic95_sup"],
                             "n_criancas": len(m["combinada_base"])},
                            {"metrica": "Inconclusivas — fezes (amostra insuficiente em tudo)", "valor_pct": None,
                             "ic95_inf": None, "ic95_sup": None, "n_criancas": len(m["fecal_inconclusivo"])},
                            {"metrica": "Inconclusivas — só lâmina", "valor_pct": None,
                             "ic95_inf": None, "ic95_sup": None, "n_criancas": len(m["lamina_only_inconclusivo"])},
                        ]).to_excel(writer, sheet_name="Prevalencia_Geral", index=False)
                        m["todos_parasitos_resumo"].to_excel(writer, sheet_name="Todos_os_Parasitos", index=False)
                        m["especies_resumo"].to_excel(writer, sheet_name="Prevalencia_por_Especie", index=False)
                        m["metodo_especie_resumo"].to_excel(writer, sheet_name="Prevalencia_Metodo_x_Especie", index=False)
                        pd.DataFrame([
                            {"categoria": "Negativo", "n": m["neg"]},
                            {"categoria": "Monoparasitismo", "n": m["mono"]},
                            {"categoria": "Poliparasitismo", "n": m["poli"]},
                        ]).to_excel(writer, sheet_name="Poliparasitismo", index=False)
                        m["combos_resumo"].to_excel(writer, sheet_name="Combinacoes", index=False)
                        m["metodos_resumo"].to_excel(writer, sheet_name="Comparacao_Metodos", index=False)
                        m["efeito_n_coletas"].to_excel(writer, sheet_name="Prevalencia_x_NColetas", index=False)
                        m["fecal_cumulativa"].to_excel(writer, sheet_name="Curva_Cumulativa_Fecal", index=False)

                        mc = m["mcnemar_hpj_willis"]
                        tb = mc["tabela"] or {}
                        pd.DataFrame([{
                            "n_pareado": mc["n_pareado"],
                            "HPJ+ / Willis+": tb.get("pp"),
                            "HPJ+ / Willis-": tb.get("pn"),
                            "HPJ- / Willis+": tb.get("np"),
                            "HPJ- / Willis-": tb.get("nn"),
                            "estatistica": mc["estatistica"],
                            "p_valor": mc["p_valor"],
                            "metodo": mc["metodo"],
                        }]).to_excel(writer, sheet_name="McNemar_HPJ_x_Willis", index=False)

                        ca = m["cochran_armitage_efeito_coletas"]
                        pd.DataFrame([{
                            "n_grupos": ca["n_grupos"],
                            "estatistica_z": ca["estatistica_z"],
                            "p_valor": ca["p_valor"],
                            "aviso": ca["aviso"],
                        }]).to_excel(writer, sheet_name="CochranArmitage_NPotes", index=False)
                    return buf.getvalue()

                with tab_export:
                    section_title("Baixe os relatórios completos")
                    st.write(
                        "O Excel traz todas as tabelas em abas separadas, prontas para uso em outras "
                        "análises. O PDF traz um relatório formatado, pronto para impressão ou envio."
                    )
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            "⬇ Baixar relatório em Excel",
                            data=generate_report_excel_bytes(metrics),
                            file_name="Relatorio_Analise_Epidemiologica.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width="stretch",
                        )
                    with col_dl2:
                        st.download_button(
                            "⬇ Baixar relatório em PDF",
                            data=build_pdf_report(metrics, logo_path=str(LOGO_PATH) if LOGO_PATH.exists() else None),
                            file_name="Relatorio_Analise_Epidemiologica.pdf",
                            mime="application/pdf",
                            width="stretch",
                        )

                st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption(
    "Nota metodológica: a prevalência é calculada por criança, não por exame — uma criança conta "
    "como positiva se qualquer uma de suas coletas (P1/P2/P3) revelou o parasita. O pote de fezes "
    "alimenta os métodos HPJ, Willis e Baermann-Picanço; a lâmina alimenta exclusivamente o "
    "método de Graham. Crianças cujos únicos resultados foram 'Amostra insuficiente' são "
    "reportadas à parte como inconclusivas, e não entram nos denominadores de prevalência."
)
