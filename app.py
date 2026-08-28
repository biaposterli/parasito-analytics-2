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
INK = "#11483D"
INK_SOFT = "#3E5F55"
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
)

# ----------------------------------------------------------------
# CSS de marca
# ----------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

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
    .lapahv-eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12.5px;
        letter-spacing: .05em;
        text-transform: uppercase;
        color: {TEAL};
    }}
    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid {LINE};
        padding: 16px 18px;
        border-radius: 2px;
    }}
    div[data-testid="stMetric"] label {{
        font-family: 'IBM Plex Mono', monospace !important;
        text-transform: uppercase;
        font-size: 11px !important;
        letter-spacing: .04em;
        color: #7C8B81 !important;
    }}
    .stDownloadButton button, .stButton button {{
        background-color: {TEAL_DARK};
        color: white;
        border: 1px solid {TEAL_DARK};
        font-family: 'IBM Plex Mono', monospace;
        border-radius: 2px;
    }}
    .stDownloadButton button:hover, .stButton button:hover {{
        background-color: {TEAL};
        border-color: {TEAL};
        color: white;
    }}
    .lapahv-tag {{
        display:inline-block; font-family:'IBM Plex Mono', monospace; font-size:12px;
        padding:3px 9px; margin:2px; border-radius:2px; border:1px solid;
    }}
    .lapahv-note {{
        background: {BRICK_TINT}; border-left: 3px solid {BRICK};
        padding: 12px 16px; font-size: 14px; color: {INK_SOFT}; border-radius: 2px;
    }}
    hr {{ border-color: {LINE}; }}
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

# ----------------------------------------------------------------
# Cabeçalho
# ----------------------------------------------------------------
col_logo, col_title = st.columns([1, 8])
with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=76)
with col_title:
    st.markdown('<div class="lapahv-eyebrow">Laboratório de Parasitologia Humana e Veterinária</div>', unsafe_allow_html=True)
    st.markdown("## Duas amostras, *um* relatório correto.")

st.write(
    "Todo exame coproparasitológico infantil coleta dois materiais separados — e cada um segue "
    "seu próprio caminho até o diagnóstico. Baixe o modelo, preencha os dados da sua pesquisa e "
    "envie abaixo para receber a análise por criança, já respeitando qual método pôde de fato ser "
    "feito em cada uma."
)

diag1, diag2 = st.columns(2)
with diag1:
    st.markdown(
        f"""<div style="border:1px solid {LINE}; padding:14px 18px; background:white;">
        <span class="lapahv-eyebrow">Pote de fezes</span><br>
        <span class="lapahv-tag" style="border-color:{LINE}; color:{INK_SOFT};">HPJ</span>
        <span class="lapahv-tag" style="border-color:{LINE}; color:{INK_SOFT};">Willis</span>
        <span class="lapahv-tag" style="border-color:{LINE}; color:{INK_SOFT};">Baermann-Picanço</span>
        </div>""",
        unsafe_allow_html=True,
    )
with diag2:
    st.markdown(
        f"""<div style="border:1px solid {LINE}; padding:14px 18px; background:white;">
        <span class="lapahv-eyebrow">Lâmina (swab)</span><br>
        <span class="lapahv-tag" style="border-color:{LINE}; color:{INK_SOFT};">Graham</span>
        </div>""",
        unsafe_allow_html=True,
    )

st.divider()

# ----------------------------------------------------------------
# PASSO 1 — Baixar modelo
# ----------------------------------------------------------------
st.markdown("##### Passo 01")
st.markdown("### Baixe o modelo de planilha")
st.write(
    "Um arquivo .xlsx com as colunas certas, os valores aceitos em cada uma e uma linha de "
    "exemplo — para preencher com os dados da sua coleta."
)


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


st.download_button(
    "⬇ Baixar modelo (.xlsx)",
    data=generate_template_bytes(),
    file_name="Modelo_Levantamento_Parasitoses.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.divider()

# ----------------------------------------------------------------
# PASSO 2 — Enviar planilha
# ----------------------------------------------------------------
st.markdown("##### Passo 02")
st.markdown("### Envie a planilha preenchida")
st.write(
    "Aceita o modelo baixado acima, preenchido com uma linha por coleta (P1/P2/P3) de cada "
    "criança."
)

uploaded_file = st.file_uploader("Escolha o arquivo .xlsx", type=["xlsx", "xls"])

st.divider()

# ----------------------------------------------------------------
# PASSO 3 — Relatório
# ----------------------------------------------------------------
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

            st.markdown("##### Passo 03")
            st.markdown("### Relatório da análise")
            st.caption(
                f"{metrics['total']} crianças cadastradas · {len(metrics['fecal'])} com amostra "
                f"fecal analisada · {len(metrics['apenas_lamina'])} só com lâmina."
            )

            # ---- resumo executivo ----
            c1, c2, c3 = st.columns(3)
            c1.metric("Prevalência — amostra fecal", f"{metrics['prev_fecal']:.1f}%",
                       help="Métrica principal do estudo: painel diagnóstico completo "
                            "(Graham + HPJ + Willis + Baermann-Picanço).")
            c2.metric("Prevalência — só lâmina", f"{metrics['prev_lamina']:.1f}%",
                       help="Reflete apenas Enterobius vermicularis — único parasita "
                            "pesquisável só com a lâmina.")
            c3.metric("Prevalência combinada", f"{metrics['prev_combinada']:.1f}%",
                       help="Todas as crianças com algum dado — use com cautela, mistura "
                            "profundidades diagnósticas diferentes.")

            if len(metrics["apenas_lamina"]) > 0:
                st.markdown(
                    f"""<div class="lapahv-note"><strong>Atenção:</strong> {len(metrics['apenas_lamina'])}
                    criança(s) só entregaram a lâmina, nunca o pote de fezes — para elas, apenas
                    <em>Enterobius vermicularis</em> pôde ser pesquisado. Por isso a prevalência
                    principal do estudo considera só quem teve amostra fecal analisada; o subgrupo
                    de só-lâmina é reportado à parte.</div>""",
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown("#### Crianças por profundidade de amostragem")
            cat_df = metrics["cat_counts"].rename("n_criancas").to_frame()
            cat_df["%"] = (100 * cat_df["n_criancas"] / metrics["total"]).round(1)
            st.dataframe(cat_df, width='stretch')

            # ---- prevalência por espécie ----
            st.write("")
            st.markdown("#### Prevalência por espécie (base: fezes analisadas)")
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
                    st.info("Nenhuma espécie detectada nesta base.")
            with colB:
                st.dataframe(metrics["especies_resumo"], width='stretch', hide_index=True)

            # ---- poliparasitismo ----
            st.write("")
            st.markdown("#### Mono x poliparasitismo")
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

            # ---- comparação de métodos ----
            st.write("")
            st.markdown("#### Comparação entre métodos diagnósticos")
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
                st.dataframe(metrics["metodos_resumo"], width='stretch', hide_index=True)

            # ---- efeito do numero de coletas ----
            st.write("")
            st.markdown("#### Efeito do número de potes de fezes entregues")
            if not metrics["efeito_n_coletas"].empty:
                fig4 = px.bar(
                    metrics["efeito_n_coletas"], x="n_potes_entregues", y="prevalencia",
                    labels={"prevalencia": "Prevalência (%)", "n_potes_entregues": "Nº de potes entregues"},
                    text="n_criancas",
                )
                fig4.update_traces(marker_color=TEAL_DARK, texttemplate="n=%{text}", textposition="outside")
                fig4.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig4, width='stretch')

            # ---- tabela por crianca ----
            st.write("")
            st.markdown("#### Base completa por criança")
            display_cols = [
                "id_paciente", "nome_crianca", "categoria_amostragem",
                "n_coletas_pote_entregue", "n_coletas_lamina_entregue",
                "positivo_algum_metodo", "especies_str",
            ]
            st.dataframe(
                metrics["por_crianca"][display_cols].sort_values("id_paciente"),
                width='stretch', hide_index=True, height=380,
            )

            # ---- exportar excel ----
            def generate_report_excel_bytes(m: dict) -> bytes:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    m["por_crianca"].drop(columns=["especies"]).to_excel(writer, sheet_name="Base_por_Crianca", index=False)
                    m["cat_counts"].rename("n").to_frame().to_excel(writer, sheet_name="Categoria_Amostragem")
                    pd.DataFrame([
                        {"metrica": "Prevalência — amostra fecal analisada", "valor_pct": m["prev_fecal"], "n_criancas": len(m["fecal"])},
                        {"metrica": "Prevalência — só lâmina", "valor_pct": m["prev_lamina"], "n_criancas": len(m["apenas_lamina"])},
                        {"metrica": "Prevalência combinada", "valor_pct": m["prev_combinada"], "n_criancas": len(m["analisavel"])},
                    ]).to_excel(writer, sheet_name="Prevalencia_Geral", index=False)
                    m["especies_resumo"].to_excel(writer, sheet_name="Prevalencia_por_Especie", index=False)
                    pd.DataFrame([
                        {"categoria": "Negativo", "n": m["neg"]},
                        {"categoria": "Monoparasitismo", "n": m["mono"]},
                        {"categoria": "Poliparasitismo", "n": m["poli"]},
                    ]).to_excel(writer, sheet_name="Poliparasitismo", index=False)
                    m["combos_resumo"].to_excel(writer, sheet_name="Combinacoes", index=False)
                    m["metodos_resumo"].to_excel(writer, sheet_name="Comparacao_Metodos", index=False)
                    m["efeito_n_coletas"].to_excel(writer, sheet_name="Prevalencia_x_NColetas", index=False)
                return buf.getvalue()

            st.write("")
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

st.divider()
st.caption(
    "Nota metodológica: a prevalência é calculada por criança, não por exame — uma criança conta "
    "como positiva se qualquer uma de suas coletas (P1/P2/P3) revelou o parasita. O pote de fezes "
    "alimenta os métodos HPJ, Willis e Baermann-Picanço; a lâmina alimenta exclusivamente o "
    "método de Graham."
)
