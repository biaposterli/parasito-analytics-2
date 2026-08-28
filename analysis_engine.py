"""Motor de análise epidemiológica — LaPaHV
Porte direto da lógica já validada no notebook / site (mesma matemática, mesmos resultados).
"""
import re
import unicodedata
import pandas as pd
import numpy as np

PARASITE_MAP = {
    "e. vermiculares": "Enterobius vermicularis",
    "e. vermiculares +++": "Enterobius vermicularis",
    "e. vermicularis": "Enterobius vermicularis",
    "enterobius vermicularis": "Enterobius vermicularis",
    "e. nana": "Endolimax nana",
    "e.nana": "Endolimax nana",
    "e. nana +++": "Endolimax nana",
    "endolimax nana": "Endolimax nana",
    "g. lamblia": "Giardia lamblia",
    "giardia lamblia": "Giardia lamblia",
    "b. coli": "Balantidium coli",
    "balantidium coli": "Balantidium coli",
    "e. histolytica/dispar": "Entamoeba histolytica/dispar",
    "entamoeba histolytica/dispar": "Entamoeba histolytica/dispar",
    "i. butschlii": "Iodamoeba butschlii",
    "iodamoeba butschlii": "Iodamoeba butschlii",
    "iodamoeba": "Iodamoeba butschlii",
}
PATOGENICOS = {"Enterobius vermicularis", "Giardia lamblia", "Balantidium coli"}
COMENSAIS = {"Endolimax nana", "Entamoeba histolytica/dispar", "Iodamoeba butschlii"}
NEGATIVE_TOKENS = {"-", "negativo", "neg"}
INSUFFICIENT_TOKENS = {"amostra insuficiente", "insuficiente"}

METHOD_COLUMNS = [
    ("metodo_graham", "Graham", "status_lamina"),
    ("metodo_baermann_picanco", "Baermann-Picanço", "status_amostra"),
    ("metodo_hpj", "HPJ", "status_amostra"),
    ("metodo_willis", "Willis", "status_amostra"),
]

REQUIRED_COLUMNS = [
    "id_paciente", "coleta", "nome_crianca", "status_amostra", "status_lamina",
    "metodo_graham", "metodo_baermann_picanco", "metodo_hpj", "metodo_willis",
]


def norm_text(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if not isinstance(x, str):
        return str(x)
    t = " ".join(x.strip().split())
    return t if t else None


def std_status(x):
    t = norm_text(x)
    if t is None:
        return None
    up = "".join(c for c in unicodedata.normalize("NFD", t.upper()) if unicodedata.category(c) != "Mn")
    if "NAO" in up:
        return "Não entregue"
    if "ENTREGUE" in up:
        return "Entregue"
    return t


def parse_result_cell(raw):
    t = norm_text(raw)
    if t is None:
        return (None, [], False)
    low = t.lower()
    cleaned = low.replace("+++", "").strip()
    if cleaned in NEGATIVE_TOKENS or low == "-":
        return ("Negativo", [], False)
    if cleaned in INSUFFICIENT_TOKENS:
        return ("Amostra insuficiente", [], False)
    parts = re.split(r"\s*\+\s*|\s*,\s*", cleaned)
    species = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        std = PARASITE_MAP.get(p, p.capitalize())
        if std not in species:
            species.append(std)
    if not species:
        return (None, [], False)
    return (" + ".join(species), species, True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"Colunas ausentes: {', '.join(missing)}. Baixe o modelo novamente e confira os cabeçalhos."]
    return []


def build_per_child(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, nome, status_key in METHOD_COLUMNS:
        parsed = df[col].apply(parse_result_cell) if col in df.columns else None

    rows = []
    for id_paciente, g in df.groupby(df["id_paciente"].apply(norm_text), dropna=True):
        if id_paciente is None:
            continue
        nome = norm_text(g["nome_crianca"].iloc[0]) or ""
        n_pote = 0
        n_lamina = 0
        especies_set = set()
        positivo_metodo = {"Graham": False, "Baermann-Picanço": False, "HPJ": False, "Willis": False}

        for _, row in g.iterrows():
            status_amostra = std_status(row.get("status_amostra"))
            status_lamina = std_status(row.get("status_lamina"))
            if status_amostra == "Entregue":
                n_pote += 1
            if status_lamina == "Entregue":
                n_lamina += 1
            for col, nome_m, status_key in METHOD_COLUMNS:
                status = status_amostra if status_key == "status_amostra" else status_lamina
                if status != "Entregue":
                    continue
                label, species, positive = parse_result_cell(row.get(col))
                if positive:
                    positivo_metodo[nome_m] = True
                    especies_set.update(species)

        if n_pote > 0 and n_lamina > 0:
            categoria = "Fezes e lâmina"
        elif n_pote > 0:
            categoria = "Apenas fezes (sem lâmina)"
        elif n_lamina > 0:
            categoria = "Apenas lâmina (sem fezes)"
        else:
            categoria = "Nenhum material"

        especies = sorted(especies_set)
        rows.append({
            "id_paciente": id_paciente,
            "nome_crianca": nome,
            "n_coletas_registradas": len(g),
            "n_coletas_pote_entregue": n_pote,
            "n_coletas_lamina_entregue": n_lamina,
            "categoria_amostragem": categoria,
            "participou_estudo": n_pote > 0 or n_lamina > 0,
            "positivo_algum_metodo": len(especies) > 0,
            "n_especies_distintas": len(especies),
            "poliparasitado": len(especies) > 1,
            "especies": especies,
            "especies_str": "; ".join(especies),
            "positivo_Graham": positivo_metodo["Graham"],
            "positivo_Baermann-Picanço": positivo_metodo["Baermann-Picanço"],
            "positivo_HPJ": positivo_metodo["HPJ"],
            "positivo_Willis": positivo_metodo["Willis"],
            "tem_patogenico": any(e in PATOGENICOS for e in especies),
            "tem_comensal": any(e in COMENSAIS for e in especies),
        })
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> dict:
    por_crianca = build_per_child(df)
    total = len(por_crianca)
    analisavel = por_crianca[por_crianca["participou_estudo"]]
    fecal = analisavel[analisavel["n_coletas_pote_entregue"] > 0]
    apenas_lamina = analisavel[analisavel["n_coletas_pote_entregue"] == 0]

    def pct(num, den):
        return round(100 * num / den, 1) if den else 0.0

    cat_counts = por_crianca["categoria_amostragem"].value_counts().reindex(
        ["Fezes e lâmina", "Apenas fezes (sem lâmina)", "Apenas lâmina (sem fezes)", "Nenhum material"],
        fill_value=0,
    )

    prev_fecal = pct(fecal["positivo_algum_metodo"].sum(), len(fecal))
    prev_lamina = pct(apenas_lamina["positivo_algum_metodo"].sum(), len(apenas_lamina))
    prev_combinada = pct(analisavel["positivo_algum_metodo"].sum(), len(analisavel))

    especie_count = {}
    for especies in fecal["especies"]:
        for e in especies:
            especie_count[e] = especie_count.get(e, 0) + 1
    especies_resumo = pd.DataFrame([
        {
            "especie": e, "n": n, "prevalencia": pct(n, len(fecal)),
            "categoria": "Patogênico" if e in PATOGENICOS else ("Comensal" if e in COMENSAIS else "Não classificado"),
        }
        for e, n in especie_count.items()
    ]).sort_values("n", ascending=False).reset_index(drop=True)

    poli = int((fecal["n_especies_distintas"] > 1).sum())
    mono = int((fecal["n_especies_distintas"] == 1).sum())
    neg = int((fecal["n_especies_distintas"] == 0).sum())

    combos_count = {}
    for especies in fecal.loc[fecal["poliparasitado"], "especies"]:
        key = " + ".join(sorted(especies))
        combos_count[key] = combos_count.get(key, 0) + 1
    combos_resumo = pd.DataFrame(
        [{"combinacao": k, "n": v} for k, v in combos_count.items()]
    ).sort_values("n", ascending=False).reset_index(drop=True) if combos_count else pd.DataFrame(columns=["combinacao", "n"])

    metodos_rows = []
    for col, nome_m, status_key in METHOD_COLUMNS:
        n_col = "n_coletas_lamina_entregue" if col == "metodo_graham" else "n_coletas_pote_entregue"
        testaveis = analisavel[analisavel[n_col] > 0]
        positivos = int(testaveis[f"positivo_{nome_m}"].sum())
        metodos_rows.append({
            "metodo": nome_m,
            "amostra_biologica": "Lâmina" if col == "metodo_graham" else "Pote de fezes",
            "n_criancas_testaveis": len(testaveis),
            "n_criancas_positivas": positivos,
            "prevalencia": pct(positivos, len(testaveis)),
        })
    metodos_resumo = pd.DataFrame(metodos_rows)

    efeito_rows = []
    for n_pote, g in fecal.groupby("n_coletas_pote_entregue"):
        efeito_rows.append({
            "n_potes_entregues": int(n_pote),
            "n_criancas": len(g),
            "prevalencia": pct(g["positivo_algum_metodo"].sum(), len(g)),
        })
    efeito_n_coletas = pd.DataFrame(efeito_rows).sort_values("n_potes_entregues").reset_index(drop=True)

    return {
        "por_crianca": por_crianca,
        "total": total,
        "analisavel": analisavel,
        "fecal": fecal,
        "apenas_lamina": apenas_lamina,
        "cat_counts": cat_counts,
        "prev_fecal": prev_fecal,
        "prev_lamina": prev_lamina,
        "prev_combinada": prev_combinada,
        "especies_resumo": especies_resumo,
        "poli": poli, "mono": mono, "neg": neg,
        "combos_resumo": combos_resumo,
        "metodos_resumo": metodos_resumo,
        "efeito_n_coletas": efeito_n_coletas,
    }
