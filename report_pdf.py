"""
Geração do relatório em PDF — LaPaHV
Usa reportlab (layout) + matplotlib (gráficos estáticos), ambas bibliotecas puras em Python,
sem dependências de sistema — rodam sem problemas no Streamlit Community Cloud.
"""
import io
from datetime import datetime

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable,
)

# ---------------------------------------------------------------- paleta
INK = colors.HexColor("#11483D")
INK_SOFT = colors.HexColor("#3E5F55")
TEAL = colors.HexColor("#328567")
TEAL_DARK = colors.HexColor("#11483D")
TEAL_TINT = colors.HexColor("#E2F0E7")
BRICK = colors.HexColor("#9C4A2E")
BRICK_TINT = colors.HexColor("#F1E2D8")
AMBER = colors.HexColor("#5F8A4E")
SAGE = colors.HexColor("#7DAE84")
LINE = colors.HexColor("#DAE1D5")
BG = colors.HexColor("#F5F0EA")

MPL_TEAL = "#328567"
MPL_TEAL_DARK = "#11483D"
MPL_BRICK = "#9C4A2E"
MPL_AMBER = "#5F8A4E"
MPL_SAGE = "#7DAE84"
MPL_LINE = "#DAE1D5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": MPL_LINE,
    "axes.labelcolor": "#3E5F55",
    "text.color": "#11483D",
    "xtick.color": "#3E5F55",
    "ytick.color": "#3E5F55",
    "axes.grid": True,
    "grid.color": MPL_LINE,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _fig_to_image(fig, width_mm=160):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w = width_mm * mm
    from PIL import Image as PILImage
    pil = PILImage.open(buf)
    aspect = pil.height / pil.width
    buf.seek(0)
    return Image(buf, width=w, height=w * aspect)


def _chart_especies(especies_df):
    if especies_df.empty:
        return None
    df = especies_df.sort_values("prevalencia")
    colors_map = {"Patogênico": MPL_BRICK, "Comensal": MPL_AMBER, "Não classificado": MPL_SAGE}
    bar_colors = [colors_map.get(c, MPL_SAGE) for c in df["categoria"]]
    fig, ax = plt.subplots(figsize=(6.2, max(1.6, 0.4 * len(df))))
    ax.barh(df["especie"], df["prevalencia"], color=bar_colors)
    ax.set_xlabel("Prevalência (%)")
    for i, v in enumerate(df["prevalencia"]):
        ax.text(v + 0.5, i, f"{v}%", va="center", fontsize=8)
    fig.tight_layout()
    return _fig_to_image(fig)


def _chart_todos_parasitos(todos_df):
    """Gráfico unificado de prevalência por espécie (fecal + Graham/lâmina), com
    hachura indicando o domínio da amostra (fezes vs. lâmina) e cor indicando a
    categoria (patogênico/comensal)."""
    if todos_df.empty:
        return None
    df = todos_df.sort_values("prevalencia")
    colors_map = {"Patogênico": MPL_BRICK, "Comensal": MPL_AMBER, "Não classificado": MPL_SAGE}
    hatch_map = {"Fecal": "", "Lâmina (Graham)": "///", "Fecal + Lâmina": "xx"}
    bar_colors = [colors_map.get(c, MPL_SAGE) for c in df["categoria"]]
    hatches = [hatch_map.get(d, "") for d in df["dominio"]]
    fig, ax = plt.subplots(figsize=(6.2, max(1.6, 0.42 * len(df))))
    bars = ax.barh(df["especie"], df["prevalencia"], color=bar_colors)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
        bar.set_edgecolor("white")
    ax.set_xlabel("Prevalência (%)")
    for i, (v, met) in enumerate(zip(df["prevalencia"], df["metodos"])):
        ax.text(v + 0.5, i, f"{v}% · {met}", va="center", fontsize=7)
    fig.tight_layout()
    return _fig_to_image(fig)


def _chart_poli(neg, mono, poli):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    vals = [neg, mono, poli]
    labels = ["Negativo", "Monoparasitismo", "Poliparasitismo"]
    cols = [MPL_SAGE, MPL_TEAL, MPL_BRICK]
    if sum(vals) == 0:
        plt.close(fig)
        return None
    ax.pie(vals, labels=labels, autopct=lambda p: f"{p:.0f}%" if p > 0 else "", colors=cols,
           wedgeprops={"linewidth": 1, "edgecolor": "white"}, textprops={"fontsize": 8})
    fig.tight_layout()
    return _fig_to_image(fig, width_mm=90)


def _chart_metodos(metodos_df):
    fig, ax = plt.subplots(figsize=(6.2, 3))
    ax.bar(metodos_df["metodo"], metodos_df["prevalencia"], color=MPL_TEAL)
    ax.set_ylabel("Prevalência (%)")
    # eixo fixo em 0-105%: evita que o rótulo de texto (v + 0.5) fique fora da
    # área visível quando TODAS as prevalências são 0% (domínio all-negative),
    # o que antes inflava desproporcionalmente a imagem via bbox_inches="tight"
    ax.set_ylim(0, 105)
    for i, v in enumerate(metodos_df["prevalencia"]):
        ax.text(i, v + 2, f"{v}%", ha="center", fontsize=8)
    fig.tight_layout()
    return _fig_to_image(fig)


def _chart_ncoletas(efeito_df):
    if efeito_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3))
    labels = [f"{int(r.n_potes_entregues)} pote(s)\nn={int(r.n_criancas)}" for r in efeito_df.itertuples()]
    ax.bar(labels, efeito_df["prevalencia"], color=MPL_TEAL_DARK)
    ax.set_ylabel("Prevalência (%)")
    # mesmo racional do _chart_metodos: eixo fixo evita estouro do bbox quando
    # todos os subgrupos têm 0% de prevalência.
    ax.set_ylim(0, 105)
    for i, v in enumerate(efeito_df["prevalencia"]):
        ax.text(i, v + 2, f"{v}%", ha="center", fontsize=8)
    fig.tight_layout()
    return _fig_to_image(fig)


def _chart_cumulativa(cum_df):
    if cum_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3))
    ax.plot(cum_df["k"], cum_df["prevalencia_cumulativa"], marker="o", color=MPL_TEAL_DARK)
    ax.set_xlabel("Nº de potes considerados (cumulativo)")
    ax.set_ylabel("Prevalência cumulativa (%)")
    ax.set_xticks(cum_df["k"])
    # eixo fixo em 0-105%, mesmo racional das outras funções _chart_*: evita que
    # o rótulo de texto acima do ponto estoure o bbox quando a prevalência
    # cumulativa é 0% (ou quando há um único ponto, k=1).
    ax.set_ylim(0, 105)
    for x, v in zip(cum_df["k"], cum_df["prevalencia_cumulativa"]):
        ax.text(x, v + 3, f"{v}%", ha="center", fontsize=8)
    fig.tight_layout()
    return _fig_to_image(fig)


def _styles():
    ss = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("lp_title", parent=ss["Title"], fontName="Helvetica-Bold",
                                 fontSize=20, textColor=TEAL_DARK, alignment=TA_LEFT, spaceAfter=2),
        "eyebrow": ParagraphStyle("lp_eyebrow", parent=ss["Normal"], fontName="Helvetica-Bold",
                                   fontSize=9, textColor=TEAL, spaceAfter=10, tracking=0.5),
        "h2": ParagraphStyle("lp_h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                              fontSize=13.5, textColor=TEAL_DARK, spaceBefore=16, spaceAfter=6),
        "body": ParagraphStyle("lp_body", parent=ss["Normal"], fontName="Helvetica",
                                fontSize=9.5, textColor=INK_SOFT, leading=13.5),
        "note": ParagraphStyle("lp_note", parent=ss["Normal"], fontName="Helvetica",
                                fontSize=9, textColor=INK_SOFT, leading=13, backColor=BRICK_TINT,
                                borderPadding=8, leftIndent=4),
        "small": ParagraphStyle("lp_small", parent=ss["Normal"], fontName="Helvetica",
                                 fontSize=8, textColor=colors.HexColor("#7C8B81")),
    }
    return styles


def _stat_table(metrics):
    data = [
        ["PREVALÊNCIA — AMOSTRA FECAL", "PREVALÊNCIA — SÓ LÂMINA", "PREVALÊNCIA COMBINADA"],
        [f"{metrics['prev_fecal']:.1f}%", f"{metrics['prev_lamina']:.1f}%", f"{metrics['prev_combinada']:.1f}%"],
        [
            f"IC95% {_ic_texto(metrics['prev_fecal_ic95_inf'], metrics['prev_fecal_ic95_sup'])}",
            f"IC95% {_ic_texto(metrics['prev_lamina_ic95_inf'], metrics['prev_lamina_ic95_sup'])}",
            f"IC95% {_ic_texto(metrics['prev_combinada_ic95_inf'], metrics['prev_combinada_ic95_sup'])}",
        ],
        [
            f"{int(metrics['fecal_conclusivo']['positivo_fecal'].sum())} de {len(metrics['fecal_conclusivo'])} crianças (conclusivas)",
            f"{int(metrics['lamina_only_conclusivo']['positivo_lamina'].sum())} de {len(metrics['lamina_only_conclusivo'])} crianças (conclusivas)",
            f"{int(metrics['combinada_base']['positivo_algum_metodo'].sum())} de {len(metrics['combinada_base'])} crianças (conclusivas)",
        ],
    ]
    t = Table(data, colWidths=[56 * mm, 56 * mm, 56 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#7C8B81")),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 20),
        ("TEXTCOLOR", (0, 1), (-1, 1), TEAL_DARK),
        ("FONTNAME", (0, 2), (-1, 2), "Helvetica"),
        ("FONTSIZE", (0, 2), (-1, 2), 8),
        ("TEXTCOLOR", (0, 2), (-1, 2), TEAL),
        ("FONTNAME", (0, 3), (-1, 3), "Helvetica"),
        ("FONTSIZE", (0, 3), (-1, 3), 8),
        ("TEXTCOLOR", (0, 3), (-1, 3), colors.HexColor("#7C8B81")),
        ("BOX", (0, 0), (0, -1), 0.7, LINE),
        ("BOX", (1, 0), (1, -1), 0.7, LINE),
        ("BOX", (2, 0), (2, -1), 0.7, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _ic_texto(inf, sup):
    """Formata um par (limite_inferior, limite_superior) de IC95% como texto curto
    para caber em tabela do PDF. Retorna travessão quando não calculável (n=0)."""
    if inf is None or sup is None or pd.isna(inf) or pd.isna(sup):
        return "—"
    return f"{inf:.1f}–{sup:.1f}%"


def _with_ic_column(df, prev_col="prevalencia", inf_col="ic95_inf", sup_col="ic95_sup", label="ic95"):
    """Devolve cópia do DataFrame com uma coluna de texto 'IC 95%' combinando os
    limites de Wilson, no lugar das duas colunas numéricas ic95_inf/ic95_sup —
    mantém a tabela do PDF compacta o bastante para caber na largura da página."""
    out = df.copy()
    if inf_col in out.columns and sup_col in out.columns:
        out[label] = [
            _ic_texto(i, s) for i, s in zip(out[inf_col], out[sup_col])
        ]
        out = out.drop(columns=[inf_col, sup_col])
    return out


def _df_table(df, col_labels=None, col_widths=None, max_rows=None):
    styles = _styles()
    cell_style = ParagraphStyle("lp_cell", parent=styles["body"], fontSize=8, leading=10.5, textColor=INK)
    header_style = ParagraphStyle("lp_cell_header", parent=styles["body"], fontSize=8,
                                   fontName="Helvetica-Bold", textColor=colors.HexColor("#7C8B81"))
    if df.empty:
        return Paragraph("Sem dados.", styles["body"])
    if max_rows:
        df = df.head(max_rows)
    headers = col_labels or list(df.columns)
    header_row = [Paragraph(str(h), header_style) for h in headers]
    body_rows = [[Paragraph(str(v), cell_style) for v in row] for row in df.astype(str).values.tolist()]
    data = [header_row] + body_rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, colors.HexColor("#AFC6B4")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def build_pdf_report(metrics: dict, logo_path: str | None = None) -> bytes:
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=18 * mm, bottomMargin=16 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
        title="Relatório de Análise Epidemiológica — LaPaHV",
    )
    story = []

    # ---- cabeçalho ----
    header_cells = []
    if logo_path:
        try:
            header_cells.append(Image(logo_path, width=16 * mm, height=20 * mm))
        except Exception:
            header_cells.append("")
    title_block = [
        Paragraph("LABORATÓRIO DE PARASITOLOGIA HUMANA E VETERINÁRIA", styles["eyebrow"]),
        Paragraph("Relatório de análise epidemiológica", styles["title"]),
        Paragraph(f"Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}", styles["small"]),
    ]
    if header_cells:
        t = Table([[header_cells[0], title_block]], colWidths=[20 * mm, 150 * mm])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (1, 0), (1, 0), 8)]))
        story.append(t)
    else:
        story.extend(title_block)

    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=LINE))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(
        f"{metrics['total']} crianças cadastradas &middot; {len(metrics['fecal'])} com amostra fecal "
        f"entregue &middot; {len(metrics['apenas_lamina'])} só com lâmina.",
        styles["body"],
    ))
    story.append(Spacer(1, 4 * mm))

    # ---- resumo executivo ----
    story.append(_stat_table(metrics))
    story.append(Spacer(1, 3 * mm))

    n_inconclusivas = len(metrics["fecal_inconclusivo"]) + len(metrics["lamina_only_inconclusivo"])
    if n_inconclusivas > 0:
        note_inc = (
            f"<b>Amostras inconclusivas:</b> {len(metrics['fecal_inconclusivo'])} criança(s) com "
            "pote de fezes entregue tiveram todos os métodos fecais marcados como \"Amostra "
            "insuficiente\" (ou sem resultado registrado)"
            + (f"; {len(metrics['lamina_only_inconclusivo'])} criança(s) na mesma situação só com "
               "lâmina" if len(metrics['lamina_only_inconclusivo']) else "")
            + ". Essas crianças foram excluídas dos denominadores de prevalência — não contam "
              "como negativas."
        )
        story.append(Paragraph(note_inc, styles["note"]))
        story.append(Spacer(1, 2 * mm))

    if len(metrics["apenas_lamina"]) > 0:
        note = (
            f"<b>Atenção:</b> {len(metrics['apenas_lamina'])} criança(s) só entregaram a lâmina, nunca "
            "o pote de fezes — para elas, apenas <i>Enterobius vermicularis</i> pôde ser pesquisado. "
            "A prevalência principal do estudo considera só quem teve amostra fecal analisada; o "
            "subgrupo de só-lâmina é reportado à parte."
        )
        story.append(Paragraph(note, styles["note"]))

    # ---- profundidade de amostragem ----
    story.append(Paragraph("Crianças por profundidade de amostragem", styles["h2"]))
    cat_df = metrics["cat_counts"].rename("n_criancas").to_frame()
    cat_df["%"] = (100 * cat_df["n_criancas"] / metrics["total"]).round(1)
    cat_df = cat_df.reset_index().rename(columns={"index": "categoria_amostragem", "categoria_amostragem": "Categoria"})
    cat_df.columns = ["Categoria", "Nº crianças", "%"]
    story.append(_df_table(cat_df, col_widths=[80 * mm, 35 * mm, 25 * mm]))

    # ---- prevalência de todos os parasitos (fecal + Graham/lâmina, unificado) ----
    story.append(Paragraph("Prevalência de todos os parasitos", styles["h2"]))
    story.append(Paragraph(
        "Espécies fecais (HPJ/Willis/Baermann-Picanço) e achados de Graham/lâmina, reunidos num "
        "só gráfico. As bases de cálculo diferem por domínio (ver coluna \"Base N\" na tabela "
        "abaixo). Quando a mesma espécie foi encontrada em métodos de domínios diferentes (ex.: "
        "Enterobius vermicularis, tipicamente pelo Graham, mas ocasionalmente também visível num "
        "método fecal), ela aparece numa única linha \"Fecal + Lâmina\", com denominador e "
        "numerador calculados por criança — quem foi detectado em mais de um método conta uma "
        "vez só, não duas.",
        styles["small"],
    ))
    story.append(Spacer(1, 1.5 * mm))
    todos_img = _chart_todos_parasitos(metrics["todos_parasitos_resumo"])
    if todos_img:
        story.append(todos_img)
        story.append(Spacer(1, 2 * mm))
    todos_table = _with_ic_column(metrics["todos_parasitos_resumo"]).rename(columns={
        "especie": "Espécie", "categoria": "Categoria", "dominio": "Amostra", "n": "N",
        "prevalencia": "Prevalência %", "base_n": "Base N", "metodos": "Método(s)", "ic95": "IC 95%",
    })
    story.append(_df_table(todos_table, col_widths=[32 * mm, 18 * mm, 20 * mm, 10 * mm, 18 * mm, 14 * mm, 28 * mm, 20 * mm]))

    # ---- prevalência por método e espécie ----
    story.append(Paragraph("Prevalência por método diagnóstico e espécie", styles["h2"]))
    story.append(Paragraph(
        "Cada valor é a prevalência (%) daquela espécie especificamente pelo método indicado; "
        "denominador = crianças com resultado conclusivo naquele método. Uma mesma espécie pode "
        "aparecer em mais de um método fecal.",
        styles["small"],
    ))
    story.append(Spacer(1, 1.5 * mm))
    if not metrics["metodo_especie_resumo"].empty:
        pivot = metrics["metodo_especie_resumo"].pivot_table(
            index="especie", columns="metodo", values="prevalencia", aggfunc="first",
        ).reindex(columns=["Graham", "HPJ", "Willis", "Baermann-Picanço"])
        pivot_df = pivot.reset_index().rename(columns={"especie": "Espécie"})
        pivot_df = pivot_df.fillna("—")
        story.append(_df_table(pivot_df, col_widths=[50 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm]))
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(
            "Intervalos de confiança de 95% (Wilson) correspondentes a cada célula acima:",
            styles["small"],
        ))
        story.append(Spacer(1, 1 * mm))
        me_table = _with_ic_column(metrics["metodo_especie_resumo"]).rename(columns={
            "metodo": "Método", "especie": "Espécie", "n": "N",
            "prevalencia": "Prevalência %", "categoria": "Categoria", "ic95": "IC 95%",
        })
        story.append(_df_table(me_table, col_widths=[25 * mm, 40 * mm, 12 * mm, 22 * mm, 25 * mm, 26 * mm]))
    else:
        story.append(Paragraph("Sem dados suficientes para este cruzamento.", styles["body"]))

    # ---- prevalência por espécie ----
    story.append(Paragraph("Prevalência por espécie — métodos fecais (base: fezes com resultado conclusivo)", styles["h2"]))
    story.append(Paragraph(
        "Cada espécie encontrada por HPJ, Willis ou Baermann-Picanço, especificamente — inclusive "
        "Enterobius vermicularis, se algum caso tiver sido identificado incidentalmente num método "
        "fecal (achado válido, não erro de digitação). A prevalência combinada dessa espécie com o "
        "Graham, sem contar a mesma criança duas vezes, está no gráfico unificado acima.",
        styles["small"],
    ))
    story.append(Spacer(1, 1.5 * mm))
    esp_img = _chart_especies(metrics["especies_resumo"])
    if esp_img:
        story.append(esp_img)
        story.append(Spacer(1, 2 * mm))
    esp_table = _with_ic_column(metrics["especies_resumo"]).rename(
        columns={"especie": "Espécie", "n": "N", "prevalencia": "Prevalência %", "categoria": "Categoria", "ic95": "IC 95%"}
    )
    story.append(_df_table(esp_table, col_widths=[52 * mm, 16 * mm, 24 * mm, 24 * mm, 24 * mm]))

    # ---- poliparasitismo ----
    story.append(Paragraph("Mono x poliparasitismo (base: espécies de origem fecal)", styles["h2"]))
    poli_img = _chart_poli(metrics["neg"], metrics["mono"], metrics["poli"])
    if poli_img:
        story.append(poli_img)
        story.append(Spacer(1, 2 * mm))
    if not metrics["combos_resumo"].empty:
        story.append(Paragraph("Combinações mais frequentes:", styles["body"]))
        combo_table = metrics["combos_resumo"].rename(columns={"combinacao": "Combinação", "n": "N"})
        story.append(_df_table(combo_table, col_widths=[110 * mm, 20 * mm]))

    # ---- comparação de métodos ----
    story.append(Paragraph("Comparação entre métodos diagnósticos", styles["h2"]))
    story.append(Paragraph(
        "Denominador = crianças com resultado conclusivo naquele método específico (exclui "
        "\"Amostra insuficiente\").",
        styles["small"],
    ))
    story.append(Spacer(1, 1.5 * mm))
    met_img = _chart_metodos(metrics["metodos_resumo"])
    if met_img:
        story.append(met_img)
        story.append(Spacer(1, 2 * mm))
    met_table = _with_ic_column(metrics["metodos_resumo"]).rename(columns={
        "metodo": "Método", "amostra_biologica": "Amostra", "n_criancas_testaveis": "Testáveis",
        "n_criancas_positivas": "Positivas", "n_criancas_inconclusivas": "Inconclusivas",
        "prevalencia": "Prevalência %", "ic95": "IC 95%",
    })
    story.append(_df_table(met_table, col_widths=[24 * mm, 20 * mm, 16 * mm, 16 * mm, 18 * mm, 20 * mm, 22 * mm]))

    story.append(Spacer(1, 3 * mm))
    subsection_style = ParagraphStyle(
        "lp_h3_inline", parent=styles["h2"], fontSize=11.5, spaceBefore=6, spaceAfter=4,
    )
    story.append(Paragraph("HPJ x Willis — teste de McNemar", subsection_style))
    mc = metrics["mcnemar_hpj_willis"]
    if mc["n_pareado"] == 0 or mc["tabela"] is None:
        story.append(Paragraph(
            "Sem crianças com resultado conclusivo em HPJ e Willis simultaneamente — teste não calculado.",
            styles["body"],
        ))
    else:
        tb = mc["tabela"]
        story.append(Paragraph(
            "Compara os dois métodos aplicados à mesma amostra de fezes da mesma criança (dados "
            "pareados), usando só as crianças em que os dois métodos discordaram entre si.",
            styles["small"],
        ))
        story.append(Spacer(1, 1.5 * mm))
        mc_table_data = pd.DataFrame(
            [["HPJ +", tb["pp"], tb["pn"]], ["HPJ −", tb["np"], tb["nn"]]],
            columns=["", "Willis +", "Willis −"],
        )
        story.append(_df_table(mc_table_data, col_widths=[30 * mm, 30 * mm, 30 * mm]))
        story.append(Spacer(1, 2 * mm))
        metodo_label = {
            "exato": "teste exato (< 25 discordâncias)",
            "chi2_corrigido": "qui-quadrado com correção de continuidade",
            "sem_discordancia": "sem discordâncias — nada a testar",
        }.get(mc["metodo"], mc["metodo"])
        story.append(Paragraph(
            f"n pareado = {mc['n_pareado']} &middot; {tb['pn'] + tb['np']} discordância(s) "
            f"({tb['pn']} HPJ+/Willis−, {tb['np']} HPJ−/Willis+) &middot; {metodo_label} "
            f"&middot; <b>p-valor = {mc['p_valor']:.4f}</b>.",
            styles["body"],
        ))

    # ---- efeito n coletas ----
    story.append(Paragraph("Efeito do número de potes de fezes entregues", styles["h2"]))
    story.append(Paragraph(
        "Baseado apenas em positividade fecal. Compara subgrupos diferentes de crianças — pode "
        "ter viés de seleção; ver curva cumulativa abaixo para uma leitura sem esse viés.",
        styles["small"],
    ))
    story.append(Spacer(1, 1.5 * mm))
    nc_img = _chart_ncoletas(metrics["efeito_n_coletas"])
    if nc_img:
        story.append(nc_img)

    ca = metrics["cochran_armitage_efeito_coletas"]
    if ca["p_valor"] is not None:
        story.append(Spacer(1, 1.5 * mm))
        story.append(Paragraph(
            f"Teste de tendência de Cochran-Armitage: Z = {ca['estatistica_z']:.3f}, "
            f"p-valor = {ca['p_valor']:.4f} ({ca['n_grupos']} grupos). {ca['aviso']}",
            styles["small"],
        ))

    # ---- curva cumulativa ----
    if not metrics["fecal_cumulativa"].empty:
        story.append(Paragraph("Ganho marginal por amostra — curva cumulativa", styles["h2"]))
        n_grupo = int(metrics["fecal_cumulativa"]["n_criancas"].iloc[0])
        story.append(Paragraph(
            f"Mesmo grupo de {n_grupo} criança(s) que entregou o número máximo de potes "
            "observado no estudo, medida repetida (1ª, 1ª+2ª ... coletas). Sem viés de comparar "
            "subgrupos diferentes de crianças.",
            styles["small"],
        ))
        story.append(Spacer(1, 1.5 * mm))
        cum_img = _chart_cumulativa(metrics["fecal_cumulativa"])
        if cum_img:
            story.append(cum_img)

    # ---- base por criança ----
    story.append(Paragraph("Base por criança", styles["h2"]))
    story.append(Paragraph(
        "Tabela completa disponível no arquivo Excel exportado junto com este PDF; abaixo, uma "
        "amostra das primeiras linhas.",
        styles["body"],
    ))
    story.append(Spacer(1, 2 * mm))
    child_cols = ["id_paciente", "nome_crianca", "categoria_amostragem", "especies_str"]
    child_df = metrics["por_crianca"][child_cols].sort_values("id_paciente").rename(columns={
        "id_paciente": "ID", "nome_crianca": "Nome", "categoria_amostragem": "Amostragem", "especies_str": "Espécies",
    })
    story.append(_df_table(child_df, col_widths=[22 * mm, 42 * mm, 42 * mm, 54 * mm], max_rows=30))
    if len(child_df) > 30:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(
            f"… e mais {len(child_df) - 30} crianças. Veja a lista completa no Excel exportado.",
            styles["small"],
        ))

    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=LINE))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Nota metodológica: a prevalência é calculada por criança, não por exame — uma criança conta "
        "como positiva se qualquer uma de suas coletas (P1/P2/P3) revelou o parasita. O pote de fezes "
        "alimenta os métodos HPJ, Willis e Baermann-Picanço; a lâmina alimenta exclusivamente o "
        "método de Graham. Crianças cujos únicos resultados foram \"Amostra insuficiente\" são "
        "reportadas à parte como inconclusivas e não entram nos denominadores de prevalência.",
        styles["small"],
    ))

    doc.build(story)
    return buf.getvalue()
