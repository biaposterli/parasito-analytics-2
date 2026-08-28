"""Motor de análise epidemiológica — LaPaHV (v2)

Correções em relação à v1:
  1) prev_fecal deixa de contar como "positiva" uma criança cujo único achado veio da
     lâmina (Graham) — agora usa exclusivamente os métodos fecais (HPJ, Willis,
     Baermann-Picanço).
  2) "Amostra insuficiente" passa a ser tratada como resultado INCONCLUSIVO, não mais
     como negativo silencioso — tanto no domínio fecal quanto no domínio lâmina, e
     também método a método (metodos_resumo).
  3) A tabela de prevalência por espécie ("base: fezes analisadas") passa a listar
     apenas espécies detectadas por métodos fecais; achados exclusivos de Graham
     (essencialmente Enterobius vermicularis) já aparecem, com o denominador correto,
     na tabela de comparação de métodos.
  4) Mono/poliparasitismo e combinações passam a ser calculados só com espécies de
     origem fecal (consistente com a definição clássica de coinfecção em
     coproparasitológico), evitando misturar achado de swab perianal com achado de
     fezes na mesma contagem.
  5) "Efeito do número de potes entregues" deixa de usar positividade combinada
     (incluindo lâmina) como desfecho — agora usa só positividade fecal. Além disso,
     é acrescentada uma curva de positividade CUMULATIVA (mesma coorte de crianças que
     entregaram o número máximo de potes, medida repetida), que é o desenho
     correto para estimar o ganho marginal de cada amostra adicional sem o viés de
     seleção de comparar subgrupos diferentes de crianças.
  6) Entamoeba histolytica/dispar passa a ser classificada como PATOGÊNICA (não mais
     comensal): como a diferenciação morfológica entre E. histolytica (patogênica) e
     E. dispar (comensal) não é possível no laboratório, todo achado desse complexo
     precisa ser tratado clinicamente como potencialmente patogênico.
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
PATOGENICOS = {
    "Enterobius vermicularis",
    "Giardia lamblia",
    "Balantidium coli",
    "Entamoeba histolytica/dispar",
}
COMENSAIS = {"Endolimax nana", "Iodamoeba butschlii"}
NEGATIVE_TOKENS = {"-", "negativo", "neg"}
INSUFFICIENT_TOKENS = {"amostra insuficiente", "insuficiente"}

# (coluna, nome do método, coluna de status que rege esse método, domínio biológico)
METHOD_COLUMNS = [
    ("metodo_graham", "Graham", "status_lamina", "lamina"),
    ("metodo_baermann_picanco", "Baermann-Picanço", "status_amostra", "fecal"),
    ("metodo_hpj", "HPJ", "status_amostra", "fecal"),
    ("metodo_willis", "Willis", "status_amostra", "fecal"),
]
FECAL_METHODS = [m for m in METHOD_COLUMNS if m[3] == "fecal"]
LAMINA_METHODS = [m for m in METHOD_COLUMNS if m[3] == "lamina"]

REQUIRED_COLUMNS = [
    "id_paciente", "coleta", "nome_crianca", "status_amostra", "status_lamina",
    "metodo_graham", "metodo_baermann_picanco", "metodo_hpj", "metodo_willis",
]

ORDEM_COLETA = {"P1": 1, "P2": 2, "P3": 3}


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
    """Retorna (label, especies, positivo)."""
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


def _classify_instance(label, positive):
    """Classifica uma tentativa (uma célula de resultado, já com status 'Entregue') em
    'positivo' / 'negativo' / 'inconclusivo'. 'Amostra insuficiente' e células vazias
    (apesar de material entregue) contam como inconclusivo — nunca como negativo."""
    if positive:
        return "positivo"
    if label == "Negativo":
        return "negativo"
    # "Amostra insuficiente" (label específico) ou célula vazia/sem_resultado (label None)
    return "inconclusivo"


def _reduce_status(instance_list):
    """Reduz uma lista de status de tentativas (de um mesmo domínio/método, para uma
    mesma criança) a um único status: positivo > negativo > inconclusivo.
    Retorna None se não houve nenhuma tentativa (material não entregue)."""
    if not instance_list:
        return None
    if "positivo" in instance_list:
        return "positivo"
    if "negativo" in instance_list:
        return "negativo"
    return "inconclusivo"


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
    rows = []
    for id_paciente, g in df.groupby(df["id_paciente"].apply(norm_text), dropna=True):
        if id_paciente is None:
            continue
        nome = norm_text(g["nome_crianca"].iloc[0]) or ""
        n_pote = 0
        n_lamina = 0

        especies_fecais_set = set()
        especies_lamina_set = set()

        # tentativas por domínio (para status fecal_status / lamina_status)
        fecal_instances = []
        lamina_instances = []

        # tentativas por método individual (para metodos_resumo, sem viés de inconclusivo)
        instances_por_metodo = {nome_m: [] for _, nome_m, _, _ in METHOD_COLUMNS}
        positivo_metodo = {nome_m: False for _, nome_m, _, _ in METHOD_COLUMNS}

        for _, row in g.iterrows():
            status_amostra = std_status(row.get("status_amostra"))
            status_lamina = std_status(row.get("status_lamina"))
            if status_amostra == "Entregue":
                n_pote += 1
            if status_lamina == "Entregue":
                n_lamina += 1

            for col, nome_m, status_key, dominio in METHOD_COLUMNS:
                status = status_amostra if status_key == "status_amostra" else status_lamina
                if status != "Entregue":
                    continue  # método não tentado nesta coleta
                label, species, positive = parse_result_cell(row.get(col))
                inst = _classify_instance(label, positive)
                instances_por_metodo[nome_m].append(inst)
                if dominio == "fecal":
                    fecal_instances.append(inst)
                else:
                    lamina_instances.append(inst)
                if positive:
                    positivo_metodo[nome_m] = True
                    if dominio == "fecal":
                        especies_fecais_set.update(species)
                    else:
                        especies_lamina_set.update(species)

        fecal_status = _reduce_status(fecal_instances)
        lamina_status = _reduce_status(lamina_instances)
        status_metodo = {nome_m: _reduce_status(v) for nome_m, v in instances_por_metodo.items()}

        if n_pote > 0 and n_lamina > 0:
            categoria = "Fezes e lâmina"
        elif n_pote > 0:
            categoria = "Apenas fezes (sem lâmina)"
        elif n_lamina > 0:
            categoria = "Apenas lâmina (sem fezes)"
        else:
            categoria = "Nenhum material"

        especies_fecais = sorted(especies_fecais_set)
        especies_lamina = sorted(especies_lamina_set)
        especies_total = sorted(especies_fecais_set | especies_lamina_set)

        positivo_fecal = fecal_status == "positivo"
        positivo_lamina = lamina_status == "positivo"

        rows.append({
            "id_paciente": id_paciente,
            "nome_crianca": nome,
            "n_coletas_registradas": len(g),
            "n_coletas_pote_entregue": n_pote,
            "n_coletas_lamina_entregue": n_lamina,
            "categoria_amostragem": categoria,
            "participou_estudo": n_pote > 0 or n_lamina > 0,

            # status por domínio: 'positivo' / 'negativo' / 'inconclusivo' / None (sem material)
            "fecal_status": fecal_status,
            "lamina_status": lamina_status,
            "positivo_fecal": positivo_fecal,
            "positivo_lamina": positivo_lamina,
            # combinado (qualquer domínio conclusivo positivo) — usado só na prevalência combinada
            "positivo_algum_metodo": bool(positivo_fecal or positivo_lamina),

            "especies_fecais": especies_fecais,
            "especies_fecais_str": "; ".join(especies_fecais),
            "especies_lamina": especies_lamina,
            "especies_lamina_str": "; ".join(especies_lamina),
            "especies": especies_total,
            "especies_str": "; ".join(especies_total),

            "n_especies_distintas_fecais": len(especies_fecais),
            "poliparasitado_fecal": len(especies_fecais) > 1,

            "positivo_Graham": positivo_metodo["Graham"],
            "positivo_Baermann-Picanço": positivo_metodo["Baermann-Picanço"],
            "positivo_HPJ": positivo_metodo["HPJ"],
            "positivo_Willis": positivo_metodo["Willis"],
            "status_Graham": status_metodo["Graham"],
            "status_Baermann-Picanço": status_metodo["Baermann-Picanço"],
            "status_HPJ": status_metodo["HPJ"],
            "status_Willis": status_metodo["Willis"],

            "tem_patogenico": any(e in PATOGENICOS for e in especies_total),
            "tem_comensal": any(e in COMENSAIS for e in especies_total),
        })
    if not rows:
        # planilha sem nenhuma linha de dado válido (ou sem id_paciente preenchido):
        # devolve DataFrame vazio mas com as colunas certas, para não quebrar o
        # restante do pipeline (compute_metrics acessa essas colunas diretamente).
        return pd.DataFrame(columns=[
            "id_paciente", "nome_crianca", "n_coletas_registradas",
            "n_coletas_pote_entregue", "n_coletas_lamina_entregue", "categoria_amostragem",
            "participou_estudo", "fecal_status", "lamina_status", "positivo_fecal",
            "positivo_lamina", "positivo_algum_metodo", "especies_fecais", "especies_fecais_str",
            "especies_lamina", "especies_lamina_str", "especies", "especies_str",
            "n_especies_distintas_fecais", "poliparasitado_fecal",
            "positivo_Graham", "positivo_Baermann-Picanço", "positivo_HPJ", "positivo_Willis",
            "status_Graham", "status_Baermann-Picanço", "status_HPJ", "status_Willis",
            "tem_patogenico", "tem_comensal",
        ])
    return pd.DataFrame(rows)


def build_fecal_cumulative_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Curva de positividade cumulativa por nº de potes considerados, medida na MESMA
    coorte de crianças (as que entregaram o número máximo de potes observado no
    estudo) — evita o viés de selecionar subgrupos diferentes de crianças por nº de
    potes entregues (quem entrega mais pode diferir sistematicamente de quem entrega
    menos)."""
    df = df.copy()
    records = []  # (id_paciente, rank_coleta, positivo_na_coleta)
    for id_paciente, g in df.groupby(df["id_paciente"].apply(norm_text), dropna=True):
        if id_paciente is None:
            continue
        for _, row in g.iterrows():
            status_amostra = std_status(row.get("status_amostra"))
            if status_amostra != "Entregue":
                continue
            coleta = norm_text(row.get("coleta"))
            rank = ORDEM_COLETA.get(coleta)
            if rank is None:
                continue
            positivo_instance = False
            for col, nome_m, status_key, dominio in FECAL_METHODS:
                _, _, positive = parse_result_cell(row.get(col))
                if positive:
                    positivo_instance = True
            records.append((id_paciente, rank, positivo_instance))

    cols = ["k", "n_criancas", "prevalencia_cumulativa"]
    if not records:
        return pd.DataFrame(columns=cols)

    rec_df = pd.DataFrame(records, columns=["id_paciente", "rank", "positivo"])
    n_by_child = rec_df.groupby("id_paciente")["rank"].nunique()
    if n_by_child.empty:
        return pd.DataFrame(columns=cols)
    max_n = int(n_by_child.mode().iloc[0])
    cohort_ids = n_by_child[n_by_child == max_n].index
    cohort = rec_df[rec_df["id_paciente"].isin(cohort_ids)]

    out_rows = []
    n = len(cohort_ids)
    for k in range(1, max_n + 1):
        sub = cohort[cohort["rank"] <= k]
        cum_pos = sub.groupby("id_paciente")["positivo"].any().reindex(cohort_ids, fill_value=False)
        pos = int(cum_pos.sum())
        out_rows.append({"k": k, "n_criancas": n, "prevalencia_cumulativa": round(100 * pos / n, 1) if n else 0.0})
    return pd.DataFrame(out_rows)


def _empty_metrics(por_crianca: pd.DataFrame) -> dict:
    """Estrutura de retorno usada quando não há nenhuma criança identificada
    (planilha vazia ou sem id_paciente preenchido). Evita quebrar o pipeline em
    DataFrames com 0 linhas, onde o pandas cria colunas com dtype 'object' e a
    indexação booleana perde as colunas."""
    empty_cat = pd.Series([0, 0, 0, 0], index=[
        "Fezes e lâmina", "Apenas fezes (sem lâmina)", "Apenas lâmina (sem fezes)", "Nenhum material",
    ])
    empty_child = por_crianca  # já vem com as colunas certas, só sem linhas
    empty_especies = pd.DataFrame(columns=["especie", "n", "prevalencia", "categoria"])
    empty_combos = pd.DataFrame(columns=["combinacao", "n"])
    empty_metodos = pd.DataFrame(columns=[
        "metodo", "amostra_biologica", "n_criancas_testaveis", "n_criancas_positivas",
        "n_criancas_inconclusivas", "prevalencia",
    ])
    empty_efeito = pd.DataFrame(columns=["n_potes_entregues", "n_criancas", "prevalencia"])
    empty_cumulativa = pd.DataFrame(columns=["k", "n_criancas", "prevalencia_cumulativa"])
    return {
        "por_crianca": empty_child,
        "total": 0,
        "analisavel": empty_child,
        "fecal": empty_child,
        "apenas_lamina": empty_child,
        "fecal_conclusivo": empty_child,
        "fecal_inconclusivo": empty_child,
        "lamina_only_conclusivo": empty_child,
        "lamina_only_inconclusivo": empty_child,
        "combinada_base": empty_child,
        "combinada_inconclusiva": empty_child,
        "cat_counts": empty_cat,
        "prev_fecal": 0.0,
        "prev_lamina": 0.0,
        "prev_combinada": 0.0,
        "especies_resumo": empty_especies,
        "poli": 0, "mono": 0, "neg": 0,
        "combos_resumo": empty_combos,
        "metodos_resumo": empty_metodos,
        "efeito_n_coletas": empty_efeito,
        "fecal_cumulativa": empty_cumulativa,
    }


def compute_metrics(df: pd.DataFrame) -> dict:
    por_crianca = build_per_child(df)
    total = len(por_crianca)
    if total == 0:
        return _empty_metrics(por_crianca)

    analisavel = por_crianca[por_crianca["participou_estudo"].astype(bool)]
    fecal = analisavel[analisavel["n_coletas_pote_entregue"] > 0]
    apenas_lamina = analisavel[analisavel["n_coletas_pote_entregue"] == 0]

    def pct(num, den):
        return round(100 * num / den, 1) if den else 0.0

    cat_counts = por_crianca["categoria_amostragem"].value_counts().reindex(
        ["Fezes e lâmina", "Apenas fezes (sem lâmina)", "Apenas lâmina (sem fezes)", "Nenhum material"],
        fill_value=0,
    )

    # ---- prevalências: só entram no denominador crianças com resultado CONCLUSIVO
    # (positivo ou negativo) no domínio relevante — "inconclusivo" (amostra
    # insuficiente / sem resultado) é reportado à parte, não vira negativo.
    fecal_conclusivo = fecal[fecal["fecal_status"].isin(["positivo", "negativo"])]
    fecal_inconclusivo = fecal[fecal["fecal_status"] == "inconclusivo"]

    lamina_only_conclusivo = apenas_lamina[apenas_lamina["lamina_status"].isin(["positivo", "negativo"])]
    lamina_only_inconclusivo = apenas_lamina[apenas_lamina["lamina_status"] == "inconclusivo"]

    tem_dominio_conclusivo = analisavel["fecal_status"].isin(["positivo", "negativo"]) | \
        analisavel["lamina_status"].isin(["positivo", "negativo"])
    combinada_base = analisavel[tem_dominio_conclusivo]
    combinada_inconclusiva = analisavel[~tem_dominio_conclusivo]

    prev_fecal = pct(fecal_conclusivo["positivo_fecal"].sum(), len(fecal_conclusivo))
    prev_lamina = pct(lamina_only_conclusivo["positivo_lamina"].sum(), len(lamina_only_conclusivo))
    prev_combinada = pct(combinada_base["positivo_algum_metodo"].sum(), len(combinada_base))

    # ---- prevalência por espécie — só espécies de origem FECAL (HPJ/Willis/BP).
    # Enterobius (Graham/lâmina) já é reportado, com denominador próprio e correto,
    # em metodos_resumo.
    especie_count = {}
    for especies in fecal_conclusivo["especies_fecais"]:
        for e in especies:
            especie_count[e] = especie_count.get(e, 0) + 1
    if especie_count:
        especies_resumo = pd.DataFrame([
            {
                "especie": e, "n": n, "prevalencia": pct(n, len(fecal_conclusivo)),
                "categoria": "Patogênico" if e in PATOGENICOS else ("Comensal" if e in COMENSAIS else "Não classificado"),
            }
            for e, n in especie_count.items()
        ]).sort_values("n", ascending=False).reset_index(drop=True)
    else:
        especies_resumo = pd.DataFrame(columns=["especie", "n", "prevalencia", "categoria"])

    # ---- mono/poliparasitismo — restrito a espécies fecais, base = fecal_conclusivo
    poli = int((fecal_conclusivo["n_especies_distintas_fecais"] > 1).sum())
    mono = int((fecal_conclusivo["n_especies_distintas_fecais"] == 1).sum())
    neg = int((fecal_conclusivo["n_especies_distintas_fecais"] == 0).sum())

    combos_count = {}
    for especies in fecal_conclusivo.loc[fecal_conclusivo["poliparasitado_fecal"], "especies_fecais"]:
        key = " + ".join(sorted(especies))
        combos_count[key] = combos_count.get(key, 0) + 1
    combos_resumo = pd.DataFrame(
        [{"combinacao": k, "n": v} for k, v in combos_count.items()]
    ).sort_values("n", ascending=False).reset_index(drop=True) if combos_count else pd.DataFrame(columns=["combinacao", "n"])

    # ---- comparação de métodos — denominador = crianças com status CONCLUSIVO
    # naquele método específico (exclui quem só teve "amostra insuficiente" nesse
    # método, mesmo que tenha entregue material).
    metodos_rows = []
    for col, nome_m, status_key, dominio in METHOD_COLUMNS:
        status_col = f"status_{nome_m}"
        conclusivos = analisavel[analisavel[status_col].isin(["positivo", "negativo"])]
        inconclusivos = analisavel[analisavel[status_col] == "inconclusivo"]
        positivos = int((conclusivos[status_col] == "positivo").sum())
        metodos_rows.append({
            "metodo": nome_m,
            "amostra_biologica": "Lâmina" if dominio == "lamina" else "Pote de fezes",
            "n_criancas_testaveis": len(conclusivos),
            "n_criancas_positivas": positivos,
            "n_criancas_inconclusivas": len(inconclusivos),
            "prevalencia": pct(positivos, len(conclusivos)),
        })
    metodos_resumo = pd.DataFrame(metodos_rows)

    # ---- efeito do nº de potes — corrigido para usar só positividade FECAL
    # (antes misturava achado de lâmina) e restrito a quem teve resultado fecal
    # conclusivo. Mantém como leitura descritiva rápida; ver também
    # fecal_cumulativa para a versão sem viés de seleção entre subgrupos.
    efeito_rows = []
    for n_pote, g in fecal_conclusivo.groupby("n_coletas_pote_entregue"):
        efeito_rows.append({
            "n_potes_entregues": int(n_pote),
            "n_criancas": len(g),
            "prevalencia": pct(g["positivo_fecal"].sum(), len(g)),
        })
    efeito_n_coletas = pd.DataFrame(efeito_rows).sort_values("n_potes_entregues").reset_index(drop=True) \
        if efeito_rows else pd.DataFrame(columns=["n_potes_entregues", "n_criancas", "prevalencia"])

    fecal_cumulativa = build_fecal_cumulative_curve(df)

    return {
        "por_crianca": por_crianca,
        "total": total,
        "analisavel": analisavel,
        "fecal": fecal,
        "apenas_lamina": apenas_lamina,
        "fecal_conclusivo": fecal_conclusivo,
        "fecal_inconclusivo": fecal_inconclusivo,
        "lamina_only_conclusivo": lamina_only_conclusivo,
        "lamina_only_inconclusivo": lamina_only_inconclusivo,
        "combinada_base": combinada_base,
        "combinada_inconclusiva": combinada_inconclusiva,
        "cat_counts": cat_counts,
        "prev_fecal": prev_fecal,
        "prev_lamina": prev_lamina,
        "prev_combinada": prev_combinada,
        "especies_resumo": especies_resumo,
        "poli": poli, "mono": mono, "neg": neg,
        "combos_resumo": combos_resumo,
        "metodos_resumo": metodos_resumo,
        "efeito_n_coletas": efeito_n_coletas,
        "fecal_cumulativa": fecal_cumulativa,
    }
