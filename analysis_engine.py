"""Motor de análise epidemiológica — LaPaHV (v3)

Alterações da v2 -> v3:
  7) Acrescentadas medidas de incerteza e testes de hipótese, que faltavam
     completamente na v2 (só havia estatística descritiva — frequências e
     prevalências em %):
       - Intervalo de Confiança de 95% (método de Wilson) para TODAS as
         prevalências reportadas (prev_fecal, prev_lamina, prev_combinada,
         especies_resumo, metodos_resumo, metodo_especie_resumo,
         todos_parasitos_resumo).
       - Teste de McNemar (dados pareados) comparando HPJ x Willis.
       - Teste de tendência de Cochran-Armitage para a série "efeito do nº
         de potes entregues" (com aviso explícito de que essa comparação é
         entre SUBGRUPOS diferentes de pacientes, sujeita a viés de seleção —
         a leitura sem esse viés continua sendo a curva cumulativa).
     Todos os testes são implementados "do zero", sem depender de scipy ou
     statsmodels (que não são dependências atuais do projeto — ver
     requirements.txt), usando apenas math/numpy.

Correções mantidas da v1 -> v2:
  1) prev_fecal deixa de contar como "positiva" um paciente cujo único achado veio da
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
     é acrescentada uma curva de positividade CUMULATIVA (mesmo grupo de crianças que
     entregaram o número máximo de potes, medida repetida), que é o desenho
     correto para estimar o ganho marginal de cada amostra adicional sem o viés de
     seleção de comparar subgrupos diferentes de crianças.
  6) Entamoeba histolytica/dispar passa a ser classificada como PATOGÊNICA (não mais
     comensal): como a diferenciação morfológica entre E. histolytica (patogênica) e
     E. dispar (comensal) não é possível no laboratório, todo achado desse complexo
     precisa ser tratado clinicamente como potencialmente patogênico.

Alterações da v3 -> v4:
  8) Catálogo de métodos ampliado e DESACOPLADO da planilha: METHOD_CATALOG agora
     lista os métodos coproparasitológicos reconhecidos pelo sistema (HPJ, Willis,
     Baermann-Picanço, Faust, Kato-Katz, MIFC/Blagg, Ritchie — domínio fecal — e
     Graham — domínio lâmina). A cada upload, get_active_methods(df) filtra esse
     catálogo pelas colunas metodo_* que realmente existem na planilha enviada — a
     planilha não precisa mais trazer todos os métodos, nem precisa trazer só os
     quatro originais. Todo o pipeline (build_per_child, compute_metrics,
     metodos_resumo, metodo_especie_resumo, todos_parasitos_resumo, McNemar) passou
     a operar sobre essa lista ativa, não mais sobre uma constante fixa.
"""
import math
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

# ----------------------------------------------------------------------
# Catálogo de métodos reconhecidos pelo sistema (coluna, nome de exibição,
# coluna de status que rege esse método, domínio biológico). Este é o
# universo POSSÍVEL de métodos — não significa que uma planilha específica
# traga todos eles. Use get_active_methods(df) para saber quais desses estão
# de fato presentes numa planilha enviada.
#
# Domínio "fecal"  -> depende de status_amostra (pote de fezes)
# Domínio "lamina" -> depende de status_lamina (fita/swab)
# ----------------------------------------------------------------------
METHOD_CATALOG = [
    ("metodo_graham", "Graham", "status_lamina", "lamina"),
    ("metodo_hpj", "HPJ", "status_amostra", "fecal"),
    ("metodo_willis", "Willis", "status_amostra", "fecal"),
    ("metodo_baermann_picanco", "Baermann-Picanço", "status_amostra", "fecal"),
    ("metodo_faust", "Faust", "status_amostra", "fecal"),
    ("metodo_kato_katz", "Kato-Katz", "status_amostra", "fecal"),
    ("metodo_mifc", "MIFC (Blagg)", "status_amostra", "fecal"),
    ("metodo_ritchie", "Ritchie (formol-éter)", "status_amostra", "fecal"),
]

# Mantido por compatibilidade com código/planilhas antigas que só conheciam
# estes quatro métodos — não use para decidir o que processar; use
# get_active_methods(df) para isso.
METHOD_COLUMNS = METHOD_CATALOG

REQUIRED_COLUMNS = ["id_paciente", "coleta", "nome_paciente"]

ORDEM_COLETA = {"P1": 1, "P2": 2, "P3": 3}


def get_active_methods(df: pd.DataFrame):
    """Filtra METHOD_CATALOG pelas colunas metodo_* que existem de fato na
    planilha enviada (após normalize_columns). A ordem do catálogo é
    preservada, então a ordem de exibição (gráficos, tabelas, pivots) fica
    estável entre planilhas diferentes.

    Isso é o que permite que uma planilha traga só HPJ+Willis+Graham, outra
    traga HPJ+Willis+Faust+Kato-Katz, e ambas sejam lidas corretamente —
    sem exigir todas as colunas do catálogo nem travar quando alguma falta.
    """
    return [m for m in METHOD_CATALOG if m[0] in df.columns]


def active_fecal_methods(active_methods):
    return [m for m in active_methods if m[3] == "fecal"]


def active_lamina_methods(active_methods):
    return [m for m in active_methods if m[3] == "lamina"]


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
    # Compatibilidade com planilhas antigas: o sistema deixou de assumir que toda
    # análise é sobre crianças (nome_crianca -> nome_paciente, termo genérico), mas
    # planilhas já preenchidas com o cabeçalho antigo continuam sendo lidas.
    if "nome_paciente" not in df.columns and "nome_crianca" in df.columns:
        df = df.rename(columns={"nome_crianca": "nome_paciente"})
    return df


def validate_columns(df: pd.DataFrame):
    """Valida a planilha enviada. Diferente da v3, NÃO exige mais que todas as
    colunas de método do catálogo estejam presentes — só exige:
      1) as colunas-base (id_paciente, coleta, nome_paciente);
      2) pelo menos UM método reconhecido (metodo_*) na planilha;
      3) a coluna de status do domínio correspondente a cada método ativo
         (status_amostra se algum método fecal está presente; status_lamina
         se o Graham, ou outro método de lâmina, está presente).
    """
    errors = []
    missing_base = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_base:
        errors.append(
            f"Colunas ausentes: {', '.join(missing_base)}. Baixe o modelo novamente e confira os cabeçalhos."
        )

    active = get_active_methods(df)
    if not active:
        known = ", ".join(nome for _, nome, _, _ in METHOD_CATALOG)
        errors.append(
            "Nenhuma coluna de método reconhecida foi encontrada na planilha "
            f"(procurei por: {known}). Confira se os cabeçalhos de método seguem o "
            "padrão 'metodo_<nome>' do modelo."
        )
    else:
        domains_present = {m[3] for m in active}
        if "fecal" in domains_present and "status_amostra" not in df.columns:
            errors.append(
                "A planilha tem método(s) de amostra fecal, mas falta a coluna "
                "'status_amostra' (status de entrega do pote de fezes)."
            )
        if "lamina" in domains_present and "status_lamina" not in df.columns:
            errors.append(
                "A planilha tem método(s) de lâmina/swab, mas falta a coluna "
                "'status_lamina' (status de entrega da lâmina)."
            )
    return errors


# ======================================================================
# ESTATÍSTICA INFERENCIAL — IC95% (Wilson) e testes de hipótese
# ======================================================================
#
# Por que Wilson e não a aproximação normal (Wald)?
# A aproximação normal (p ± 1.96*sqrt(p(1-p)/n)) fica ruim (limites fora de
# [0,100], cobertura real abaixo do nominal) exatamente nos cenários mais
# comuns aqui: n pequeno (subgrupos de "só lâmina", ou "n de discordância" do
# McNemar) e proporções perto de 0% ou 100% (várias espécies têm prevalência
# baixa). O Wilson score interval não tem esses dois problemas e é o método
# recomendado por padrão para proporções binomiais em epidemiologia.

_Z95 = 1.959963984540054  # quantil normal padrão para IC de 95% (bicaudal)


def wilson_ci(positivos, n, z=_Z95):
    """Intervalo de Confiança de Wilson para uma proporção binomial.

    Fórmula fechada (Wilson, 1927):
        centro = (p + z²/2n) / (1 + z²/n)
        margem = z * sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)
        IC = centro ± margem

    Onde p = positivos/n. Diferente do intervalo de Wald (p ± z*erro-padrão),
    o Wilson não assume simetria em torno de p e nunca extrapola [0, 1] —
    por isso é preferível quando n é pequeno ou p está perto de 0 ou 1,
    situação comum neste estudo (espécies raras, subgrupos pequenos).

    Retorna uma tupla (limite_inferior_%, limite_superior_%) já em
    percentual e arredondada a 1 casa decimal, ou (None, None) se n=0
    (denominador vazio — sem base para estimar incerteza).
    """
    if n is None or n == 0 or pd.isna(n):
        return (None, None)
    n = int(n)
    positivos = int(positivos)
    p = positivos / n
    denom = 1 + (z ** 2) / n
    centro = p + (z ** 2) / (2 * n)
    margem = z * math.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2))
    lo = (centro - margem) / denom
    hi = (centro + margem) / denom
    lo = max(0.0, lo)
    hi = min(1.0, hi)
    return (round(100 * lo, 1), round(100 * hi, 1))


def _wilson_ci_pairs(numeradores, denominadores):
    """Aplica wilson_ci em paralelo a duas sequências (numerador, denominador) e
    devolve duas listas prontas para virar colunas 'ic95_inf' / 'ic95_sup'."""
    los, his = [], []
    for num, den in zip(numeradores, denominadores):
        lo, hi = wilson_ci(num, den)
        los.append(lo)
        his.append(hi)
    return los, his


def mcnemar_hpj_willis(por_paciente: pd.DataFrame) -> dict:
    """Teste de McNemar comparando os métodos diagnósticos HPJ e Willis.

    Racional: HPJ e Willis são aplicados à MESMA amostra de fezes da mesma
    criança — são medidas PAREADAS, não duas amostras independentes. Um
    qui-quadrado de independência (ou teste de proporções para amostras
    independentes) estaria errado aqui, pois ignoraria o pareamento e
    infla o erro tipo I. O McNemar usa só as células DISCORDANTES da tabela
    2x2 (criança positiva num método e negativa no outro) para testar se as
    duas taxas de detecção diferem.

    Base: crianças com resultado CONCLUSIVO (positivo ou negativo, nunca
    "amostra insuficiente"/inconclusivo) em AMBOS os métodos — interseção,
    não união, para manter o pareamento válido.

    Só é calculado se a planilha enviada tinha os dois métodos (HPJ e
    Willis) — caso contrário as colunas status_HPJ/status_Willis nem
    existem em por_paciente, e a função retorna n_pareado=0 abaixo.

    Tabela 2x2 (contagens de crianças):
                         Willis +      Willis -
        HPJ +               pp            pn
        HPJ -               np            nn

    Estatística:
      - Se nº de discordâncias (pn + np) < 25: teste EXATO binomial (mais
        robusto para amostras pequenas — comum neste tipo de estudo).
            p = 2 * P(X <= min(pn, np))   com X ~ Binomial(pn+np, 0.5)
      - Caso contrário: aproximação qui-quadrado com correção de
        continuidade de Yates:
            χ² = (|pn - np| - 1)² / (pn + np),  1 grau de liberdade
        Como χ²(1 g.l.) é o quadrado de uma Normal(0,1), o p-valor é obtido
        via função erro complementar: p = erfc(sqrt(χ²/2)).

    Retorna None nos campos numéricos se não houver crianças pareadas
    suficientes (n_pareado = 0), para não quebrar o pipeline.
    """
    if por_paciente.empty or "status_HPJ" not in por_paciente.columns or "status_Willis" not in por_paciente.columns:
        return {"n_pareado": 0, "tabela": None, "estatistica": None, "p_valor": None, "metodo": None}

    base = por_paciente[
        por_paciente["status_HPJ"].isin(["positivo", "negativo"])
        & por_paciente["status_Willis"].isin(["positivo", "negativo"])
    ]
    n = len(base)
    if n == 0:
        return {"n_pareado": 0, "tabela": None, "estatistica": None, "p_valor": None, "metodo": None}

    hpj_pos = base["status_HPJ"] == "positivo"
    willis_pos = base["status_Willis"] == "positivo"

    pp = int((hpj_pos & willis_pos).sum())
    pn = int((hpj_pos & ~willis_pos).sum())
    np_ = int((~hpj_pos & willis_pos).sum())
    nn = int((~hpj_pos & ~willis_pos).sum())

    n_disc = pn + np_  # pacientes em que os dois métodos discordam

    if n_disc == 0:
        # Concordância perfeita entre os dois métodos nesta amostra — nada a testar,
        # não há evidência de diferença (nem poderia haver).
        return {
            "n_pareado": n,
            "tabela": {"pp": pp, "pn": pn, "np": np_, "nn": nn},
            "estatistica": 0.0,
            "p_valor": 1.0,
            "metodo": "sem_discordancia",
        }

    if n_disc < 25:
        k = min(pn, np_)
        p_valor = 2 * sum(math.comb(n_disc, i) for i in range(0, k + 1)) * (0.5 ** n_disc)
        p_valor = min(1.0, p_valor)
        estatistica = float(k)
        metodo = "exato"
    else:
        estatistica = ((abs(pn - np_) - 1) ** 2) / n_disc
        p_valor = math.erfc(math.sqrt(estatistica / 2))
        metodo = "chi2_corrigido"

    return {
        "n_pareado": n,
        "tabela": {"pp": pp, "pn": pn, "np": np_, "nn": nn},
        "estatistica": round(estatistica, 4),
        "p_valor": round(p_valor, 4),
        "metodo": metodo,
    }


def cochran_armitage_trend(grupos: pd.DataFrame) -> dict:
    """Teste de tendência de Cochran-Armitage.

    Testa se a proporção de positivos cresce (ou decresce) linearmente
    conforme uma variável ordinal aumenta — aqui, o número de potes de
    fezes entregues (1, 2, 3...). É o teste correto para "existe
    tendência?" em dados de proporção por grupos ordenados; um
    qui-quadrado de independência comum testaria só "as proporções são
    diferentes?", sem usar a ordem dos grupos (perde poder estatístico
    quando a hipótese de interesse é especificamente uma tendência).

    Espera um DataFrame com colunas 'n_potes_entregues' (score/nível
    ordinal, usado como t_i), 'n_pacientes' (n_i) e 'n_positivos' (x_i) —
    contagens EXATAS, não prevalência em % já arredondada.

    Estatística (aproximação normal):
        p̄ = ΣX / ΣN
        t̄ = Σ(n_i·t_i) / ΣN
        T  = Σ x_i·(t_i - t̄)
        Var(T) = p̄(1-p̄) · Σ n_i·(t_i - t̄)²
        Z = T / sqrt(Var(T))          ~ N(0,1) sob H0 (sem tendência)
        p-valor bicaudal = erfc(|Z| / sqrt(2))

    IMPORTANTE — limite deste teste (ver docstring de
    build_fecal_cumulative_curve): os grupos aqui são SUBGRUPOS diferentes
    de crianças (quem entregou 1, 2 ou 3 potes), não medidas repetidas na
    mesma criança. Uma tendência estatisticamente significativa pode
    refletir viés de seleção (ex.: pacientes que entregam mais potes podem
    ser sistematicamente diferentes) tanto quanto um efeito real do número
    de coletas. A leitura sem esse viés é a curva cumulativa
    (fecal_cumulativa), que dispensa este teste por ser medida repetida no
    mesmo grupo.
    """
    aviso = (
        "Compara SUBGRUPOS diferentes de pacientes (quem entregou 1, 2 ou 3 potes) — "
        "sujeito a viés de seleção. A leitura mais confiável do ganho por coleta "
        "adicional é a curva cumulativa (fecal_cumulativa), medida repetida no mesmo "
        "grupo de pacientes, que não depende deste teste."
    )
    if grupos.empty or len(grupos) < 2:
        return {
            "estatistica_z": None, "p_valor": None, "n_grupos": len(grupos),
            "aviso": aviso,
        }

    t = grupos["n_potes_entregues"].astype(float).to_numpy()
    n_i = grupos["n_pacientes"].astype(float).to_numpy()
    x_i = grupos["n_positivos"].astype(float).to_numpy()

    N = n_i.sum()
    X = x_i.sum()
    if N == 0 or X == 0 or X == N:
        # sem variabilidade (0% ou 100% positivos no total) — tendência não
        # calculável de forma estável.
        return {
            "estatistica_z": None, "p_valor": None, "n_grupos": len(grupos),
            "aviso": aviso,
        }

    p_bar = X / N
    t_bar = float((n_i * t).sum() / N)
    T = float((x_i * (t - t_bar)).sum())
    var_T = p_bar * (1 - p_bar) * float((n_i * (t - t_bar) ** 2).sum())
    if var_T <= 0:
        return {
            "estatistica_z": None, "p_valor": None, "n_grupos": len(grupos),
            "aviso": aviso,
        }

    z = T / math.sqrt(var_T)
    p_valor = math.erfc(abs(z) / math.sqrt(2))

    return {
        "estatistica_z": round(z, 4),
        "p_valor": round(p_valor, 4),
        "n_grupos": len(grupos),
        "aviso": aviso,
    }


def build_per_child(df: pd.DataFrame, active_methods=None) -> pd.DataFrame:
    """Constrói a base por criança.

    active_methods: lista de tuplas (col, nome, status_key, dominio) — os
    métodos que de fato existem nesta planilha, tipicamente
    get_active_methods(df). Se None, é recalculada a partir de df (aceita
    tanto planilhas com todos os métodos do catálogo quanto planilhas com
    só um subconjunto).
    """
    if active_methods is None:
        active_methods = get_active_methods(df)
    fecal_methods = active_fecal_methods(active_methods)

    df = df.copy()
    rows = []
    for id_paciente, g in df.groupby(df["id_paciente"].apply(norm_text), dropna=True):
        if id_paciente is None:
            continue
        nome = norm_text(g["nome_paciente"].iloc[0]) or ""
        n_pote = 0
        n_lamina = 0

        especies_fecais_set = set()
        especies_lamina_set = set()

        # tentativas por domínio (para status fecal_status / lamina_status)
        fecal_instances = []
        lamina_instances = []

        # tentativas por método individual (para metodos_resumo, sem viés de inconclusivo)
        instances_por_metodo = {nome_m: [] for _, nome_m, _, _ in active_methods}
        positivo_metodo = {nome_m: False for _, nome_m, _, _ in active_methods}
        # espécies encontradas especificamente por cada método (para o cruzamento
        # espécie x método na tabela "todos os parasitos")
        especies_por_metodo = {nome_m: set() for _, nome_m, _, _ in active_methods}

        for _, row in g.iterrows():
            status_amostra = std_status(row.get("status_amostra"))
            status_lamina = std_status(row.get("status_lamina"))
            if status_amostra == "Entregue":
                n_pote += 1
            if status_lamina == "Entregue":
                n_lamina += 1

            for col, nome_m, status_key, dominio in active_methods:
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
                    especies_por_metodo[nome_m].update(species)
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

        row_out = {
            "id_paciente": id_paciente,
            "nome_paciente": nome,
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

            "tem_patogenico": any(e in PATOGENICOS for e in especies_total),
            "tem_comensal": any(e in COMENSAIS for e in especies_total),
        }

        # Colunas por-método, geradas dinamicamente para cada método ATIVO nesta
        # planilha (em vez de nomes fixos tipo "status_HPJ" hardcoded) — é isso
        # que permite ler planilhas com qualquer subconjunto do catálogo de
        # métodos, incluindo métodos novos (Faust, Kato-Katz, MIFC, Ritchie...)
        # sem precisar tocar neste código de novo.
        for _, nome_m, _, _ in active_methods:
            row_out[f"positivo_{nome_m}"] = positivo_metodo[nome_m]
            row_out[f"status_{nome_m}"] = status_metodo[nome_m]
            row_out[f"especies_{nome_m}"] = sorted(especies_por_metodo[nome_m])

        rows.append(row_out)

    base_cols = [
        "id_paciente", "nome_paciente", "n_coletas_registradas",
        "n_coletas_pote_entregue", "n_coletas_lamina_entregue", "categoria_amostragem",
        "participou_estudo", "fecal_status", "lamina_status", "positivo_fecal",
        "positivo_lamina", "positivo_algum_metodo", "especies_fecais", "especies_fecais_str",
        "especies_lamina", "especies_lamina_str", "especies", "especies_str",
        "n_especies_distintas_fecais", "poliparasitado_fecal",
        "tem_patogenico", "tem_comensal",
    ]
    metodo_cols = []
    for _, nome_m, _, _ in active_methods:
        metodo_cols += [f"positivo_{nome_m}", f"status_{nome_m}", f"especies_{nome_m}"]

    if not rows:
        # planilha sem nenhuma linha de dado válido (ou sem id_paciente preenchido):
        # devolve DataFrame vazio mas com as colunas certas (incluindo as colunas
        # por-método dos métodos ativos), para não quebrar o restante do pipeline
        # (compute_metrics acessa essas colunas diretamente).
        return pd.DataFrame(columns=base_cols + metodo_cols)
    return pd.DataFrame(rows)


def build_fecal_cumulative_curve(df: pd.DataFrame, fecal_methods=None) -> pd.DataFrame:
    """Curva de positividade cumulativa por nº de potes considerados, medida na MESMA
    grupo de crianças (as que entregaram o número máximo de potes observado no
    estudo) — evita o viés de selecionar subgrupos diferentes de crianças por nº de
    potes entregues (quem entrega mais pode diferir sistematicamente de quem entrega
    menos).

    fecal_methods: lista de tuplas (col, nome, status_key, dominio) restrita ao
    domínio fecal — tipicamente active_fecal_methods(get_active_methods(df)).
    Qualquer método fecal presente na planilha (HPJ, Willis, Baermann-Picanço,
    Faust, Kato-Katz, MIFC, Ritchie...) conta para "positivo_instance" abaixo.
    """
    if fecal_methods is None:
        fecal_methods = active_fecal_methods(get_active_methods(df))

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
            for col, nome_m, status_key, dominio in fecal_methods:
                _, _, positive = parse_result_cell(row.get(col))
                if positive:
                    positivo_instance = True
            records.append((id_paciente, rank, positivo_instance))

    cols = ["k", "n_pacientes", "prevalencia_cumulativa"]
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
        out_rows.append({"k": k, "n_pacientes": n, "prevalencia_cumulativa": round(100 * pos / n, 1) if n else 0.0})
    return pd.DataFrame(out_rows)


def _empty_metrics(por_paciente: pd.DataFrame, active_methods=None) -> dict:
    """Estrutura de retorno usada quando não há nenhuma criança identificada
    (planilha vazia ou sem id_paciente preenchido). Evita quebrar o pipeline em
    DataFrames com 0 linhas, onde o pandas cria colunas com dtype 'object' e a
    indexação booleana perde as colunas."""
    active_methods = active_methods or []
    empty_cat = pd.Series([0, 0, 0, 0], index=[
        "Fezes e lâmina", "Apenas fezes (sem lâmina)", "Apenas lâmina (sem fezes)", "Nenhum material",
    ])
    empty_child = por_paciente  # já vem com as colunas certas, só sem linhas
    empty_especies = pd.DataFrame(columns=["especie", "n", "prevalencia", "categoria", "ic95_inf", "ic95_sup"])
    empty_combos = pd.DataFrame(columns=["combinacao", "n"])
    empty_metodos = pd.DataFrame(columns=[
        "metodo", "amostra_biologica", "n_pacientes_testaveis", "n_pacientes_positivas",
        "n_pacientes_inconclusivas", "prevalencia", "ic95_inf", "ic95_sup",
    ])
    empty_efeito = pd.DataFrame(columns=["n_potes_entregues", "n_pacientes", "n_positivos", "prevalencia"])
    empty_cumulativa = pd.DataFrame(columns=["k", "n_pacientes", "prevalencia_cumulativa"])
    empty_metodo_especie = pd.DataFrame(columns=["metodo", "especie", "n", "prevalencia", "categoria", "ic95_inf", "ic95_sup"])
    empty_todos_parasitos = pd.DataFrame(columns=[
        "especie", "categoria", "dominio", "n", "prevalencia", "base_n", "metodos", "ic95_inf", "ic95_sup",
    ])
    return {
        "por_paciente": empty_child,
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
        "prev_fecal_ic95_inf": None,
        "prev_fecal_ic95_sup": None,
        "prev_lamina": 0.0,
        "prev_lamina_ic95_inf": None,
        "prev_lamina_ic95_sup": None,
        "prev_combinada": 0.0,
        "prev_combinada_ic95_inf": None,
        "prev_combinada_ic95_sup": None,
        "especies_resumo": empty_especies,
        "poli": 0, "mono": 0, "neg": 0,
        "combos_resumo": empty_combos,
        "metodos_resumo": empty_metodos,
        "efeito_n_coletas": empty_efeito,
        "fecal_cumulativa": empty_cumulativa,
        "metodo_especie_resumo": empty_metodo_especie,
        "todos_parasitos_resumo": empty_todos_parasitos,
        "mcnemar_hpj_willis": mcnemar_hpj_willis(empty_child),
        "cochran_armitage_efeito_coletas": cochran_armitage_trend(empty_efeito),
        "metodos_ativos": active_methods,
        "metodos_ativos_nomes": [nome for _, nome, _, _ in active_methods],
    }


def compute_metrics(df: pd.DataFrame) -> dict:
    active_methods = get_active_methods(df)
    fecal_methods_ativos = active_fecal_methods(active_methods)

    por_paciente = build_per_child(df, active_methods)
    total = len(por_paciente)
    if total == 0:
        return _empty_metrics(por_paciente, active_methods)

    analisavel = por_paciente[por_paciente["participou_estudo"].astype(bool)]
    fecal = analisavel[analisavel["n_coletas_pote_entregue"] > 0]
    apenas_lamina = analisavel[analisavel["n_coletas_pote_entregue"] == 0]

    def pct(num, den):
        return round(100 * num / den, 1) if den else 0.0

    cat_counts = por_paciente["categoria_amostragem"].value_counts().reindex(
        ["Fezes e lâmina", "Apenas fezes (sem lâmina)", "Apenas lâmina (sem fezes)", "Nenhum material"],
        fill_value=0,
    )

    # ---- prevalências: só entram no denominador pacientes com resultado CONCLUSIVO
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

    # ---- IC95% (Wilson) das três prevalências principais ----
    prev_fecal_ic95_inf, prev_fecal_ic95_sup = wilson_ci(
        fecal_conclusivo["positivo_fecal"].sum(), len(fecal_conclusivo)
    )
    prev_lamina_ic95_inf, prev_lamina_ic95_sup = wilson_ci(
        lamina_only_conclusivo["positivo_lamina"].sum(), len(lamina_only_conclusivo)
    )
    prev_combinada_ic95_inf, prev_combinada_ic95_sup = wilson_ci(
        combinada_base["positivo_algum_metodo"].sum(), len(combinada_base)
    )

    # ---- prevalência por espécie — só espécies de origem FECAL (métodos fecais
    # ativos nesta planilha). Enterobius (Graham/lâmina) já é reportado, com
    # denominador próprio e correto, em metodos_resumo.
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
        ic_inf, ic_sup = _wilson_ci_pairs(especies_resumo["n"], [len(fecal_conclusivo)] * len(especies_resumo))
        especies_resumo["ic95_inf"] = ic_inf
        especies_resumo["ic95_sup"] = ic_sup
    else:
        especies_resumo = pd.DataFrame(columns=["especie", "n", "prevalencia", "categoria", "ic95_inf", "ic95_sup"])

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
    # Nota: sem teste de hipótese aqui de propósito — os n das combinações (1-6,
    # tipicamente) são baixos demais para qualquer teste ter poder estatístico
    # relevante. Mantido como contagem bruta descritiva.

    # ---- comparação de métodos — denominador = pacientes com status CONCLUSIVO
    # naquele método específico (exclui quem só teve "amostra insuficiente" nesse
    # método, mesmo que tenha entregue material). Itera só sobre os métodos
    # ATIVOS nesta planilha.
    metodos_rows = []
    for col, nome_m, status_key, dominio in active_methods:
        status_col = f"status_{nome_m}"
        conclusivos = analisavel[analisavel[status_col].isin(["positivo", "negativo"])]
        inconclusivos = analisavel[analisavel[status_col] == "inconclusivo"]
        positivos = int((conclusivos[status_col] == "positivo").sum())
        metodos_rows.append({
            "metodo": nome_m,
            "amostra_biologica": "Lâmina" if dominio == "lamina" else "Pote de fezes",
            "n_pacientes_testaveis": len(conclusivos),
            "n_pacientes_positivas": positivos,
            "n_pacientes_inconclusivas": len(inconclusivos),
            "prevalencia": pct(positivos, len(conclusivos)),
        })
    metodos_resumo = pd.DataFrame(metodos_rows) if metodos_rows else pd.DataFrame(columns=[
        "metodo", "amostra_biologica", "n_pacientes_testaveis", "n_pacientes_positivas",
        "n_pacientes_inconclusivas", "prevalencia",
    ])
    if not metodos_resumo.empty:
        ic_inf, ic_sup = _wilson_ci_pairs(metodos_resumo["n_pacientes_positivas"], metodos_resumo["n_pacientes_testaveis"])
        metodos_resumo["ic95_inf"] = ic_inf
        metodos_resumo["ic95_sup"] = ic_sup
    else:
        metodos_resumo["ic95_inf"] = pd.Series(dtype=float)
        metodos_resumo["ic95_sup"] = pd.Series(dtype=float)

    # ---- prevalência por espécie x método — para cada método ATIVO, denominador =
    # crianças com resultado conclusivo NAQUELE método (mesma base de
    # metodos_resumo), contando quantas foram positivas para cada espécie
    # especificamente através desse método.
    metodo_especie_rows = []
    for col, nome_m, status_key, dominio in active_methods:
        status_col = f"status_{nome_m}"
        especies_col = f"especies_{nome_m}"
        conclusivos = analisavel[analisavel[status_col].isin(["positivo", "negativo"])]
        especie_count_m = {}
        for especies in conclusivos[especies_col]:
            for e in especies:
                especie_count_m[e] = especie_count_m.get(e, 0) + 1
        for e, n in especie_count_m.items():
            metodo_especie_rows.append({
                "metodo": nome_m,
                "especie": e,
                "n": n,
                "prevalencia": pct(n, len(conclusivos)),
                "categoria": "Patogênico" if e in PATOGENICOS else ("Comensal" if e in COMENSAIS else "Não classificado"),
            })
    metodo_especie_resumo = pd.DataFrame(metodo_especie_rows) if metodo_especie_rows else \
        pd.DataFrame(columns=["metodo", "especie", "n", "prevalencia", "categoria"])
    if not metodo_especie_resumo.empty:
        den_by_metodo = dict(zip(metodos_resumo["metodo"], metodos_resumo["n_pacientes_testaveis"]))
        dens = [den_by_metodo.get(m, 0) for m in metodo_especie_resumo["metodo"]]
        ic_inf, ic_sup = _wilson_ci_pairs(metodo_especie_resumo["n"], dens)
        metodo_especie_resumo["ic95_inf"] = ic_inf
        metodo_especie_resumo["ic95_sup"] = ic_sup
    else:
        metodo_especie_resumo["ic95_inf"] = pd.Series(dtype=float)
        metodo_especie_resumo["ic95_sup"] = pd.Series(dtype=float)

    # ---- prevalência de TODOS os parasitos (fecais + lâmina), unificada numa só
    # tabela, indicando por qual(is) método(s) cada espécie foi detectada.
    # Espécies fecais usam o denominador combinado (fecal_conclusivo, já que todos
    # os métodos fecais ativos alimentam a mesma prevalência por criança);
    # Enterobius vermicularis, tipicamente diagnosticado por um método de lâmina
    # (Graham) mas ocasionalmente também visualizado num método fecal (achado
    # incidental, igualmente válido — não é erro de digitação), usa o
    # denominador próprio do(s) método(s) que a encontraram.
    #
    # IMPORTANTE: quando uma mesma espécie é encontrada em métodos de domínios
    # diferentes, apresentar uma linha por domínio faria o mesmo paciente ser
    # contado separadamente em cada linha, inflando a impressão de nº de casos.
    # Por isso, espécies que aparecem em mais de um domínio são reportadas numa
    # ÚNICA linha "Fecal + Lâmina", com denominador e numerador calculados por
    # PACIENTE (união dos métodos relevantes) — um paciente positivo em qualquer
    # um desses métodos conta uma vez só, mesmo que tenha sido detectada em mais
    # de um.
    #
    # Se esta planilha não tem NENHUM método de domínio lâmina ativo (ex.: só
    # métodos fecais foram enviados), este bloco simplesmente não encontra
    # nenhuma espécie de lâmina e todas as linhas saem como "Fecal" — sem
    # necessidade de tratamento especial.
    metodos_by_especie = {}
    if not metodo_especie_resumo.empty:
        metodos_by_especie = (
            metodo_especie_resumo.groupby("especie")["metodo"]
            .apply(lambda s: " + ".join(sorted(set(s))))
            .to_dict()
        )

    lamina_methods_ativos = active_lamina_methods(active_methods)
    lamina_nomes_ativos = {nome for _, nome, _, _ in lamina_methods_ativos}

    especies_fecais_todas = set(especies_resumo["especie"]) if not especies_resumo.empty else set()
    lamina_especies_df = metodo_especie_resumo[metodo_especie_resumo["metodo"].isin(lamina_nomes_ativos)] \
        if (not metodo_especie_resumo.empty and lamina_nomes_ativos) else pd.DataFrame(columns=metodo_especie_resumo.columns)
    especies_lamina_todas = set(lamina_especies_df["especie"]) if not lamina_especies_df.empty else set()
    especies_multi_dominio = especies_fecais_todas & especies_lamina_todas

    # base_n de lâmina: se houver mais de um método de lâmina ativo (raro hoje,
    # mas o catálogo permite), usa o maior denominador entre eles como
    # referência de exibição; a prevalência por espécie já vem de
    # metodo_especie_resumo, calculada com o denominador correto do método que
    # de fato a detectou.
    lamina_conclusivos_n = int(
        metodos_resumo.loc[metodos_resumo["metodo"].isin(lamina_nomes_ativos), "n_pacientes_testaveis"].max()
    ) if (not metodos_resumo.empty and lamina_nomes_ativos and
          metodos_resumo["metodo"].isin(lamina_nomes_ativos).any()) else 0

    lamina_label = " / ".join(sorted(lamina_nomes_ativos)) if lamina_nomes_ativos else "Lâmina"

    todos_rows = []
    for _, r in especies_resumo.iterrows():
        if r["especie"] in especies_multi_dominio:
            continue  # tratada de forma combinada abaixo, para não contar a criança duas vezes
        todos_rows.append({
            "especie": r["especie"],
            "categoria": r["categoria"],
            "dominio": "Fecal",
            "n": r["n"],
            "prevalencia": r["prevalencia"],
            "base_n": len(fecal_conclusivo),
            "metodos": metodos_by_especie.get(r["especie"], ""),
        })
    for _, r in lamina_especies_df.iterrows():
        if r["especie"] in especies_multi_dominio:
            continue
        todos_rows.append({
            "especie": r["especie"],
            "categoria": r["categoria"],
            "dominio": f"Lâmina ({lamina_label})",
            "n": r["n"],
            "prevalencia": r["prevalencia"],
            "base_n": lamina_conclusivos_n,
            "metodos": r["metodo"],
        })

    for especie in especies_multi_dominio:
        metodos_desta_especie = [
            nome_m for _, nome_m, _, _ in active_methods
            if nome_m in metodos_by_especie.get(especie, "").split(" + ")
        ]
        status_cols = [f"status_{m}" for m in metodos_desta_especie if f"status_{m}" in por_paciente.columns]
        especies_cols = [f"especies_{m}" for m in metodos_desta_especie if f"especies_{m}" in por_paciente.columns]
        conclusivo_mask = pd.Series(False, index=por_paciente.index)
        positivo_mask = pd.Series(False, index=por_paciente.index)
        for sc in status_cols:
            conclusivo_mask = conclusivo_mask | por_paciente[sc].isin(["positivo", "negativo"])
        for ec in especies_cols:
            positivo_mask = positivo_mask | por_paciente[ec].apply(lambda lst: especie in lst)
        base_n = int(conclusivo_mask.sum())
        n = int(positivo_mask.sum())
        categoria = "Patogênico" if especie in PATOGENICOS else ("Comensal" if especie in COMENSAIS else "Não classificado")
        todos_rows.append({
            "especie": especie,
            "categoria": categoria,
            "dominio": "Fecal + Lâmina",
            "n": n,
            "prevalencia": pct(n, base_n),
            "base_n": base_n,
            "metodos": metodos_by_especie.get(especie, ""),
        })

    todos_parasitos_resumo = pd.DataFrame(todos_rows).sort_values("prevalencia", ascending=False).reset_index(drop=True) \
        if todos_rows else pd.DataFrame(columns=["especie", "categoria", "dominio", "n", "prevalencia", "base_n", "metodos"])
    if not todos_parasitos_resumo.empty:
        ic_inf, ic_sup = _wilson_ci_pairs(todos_parasitos_resumo["n"], todos_parasitos_resumo["base_n"])
        todos_parasitos_resumo["ic95_inf"] = ic_inf
        todos_parasitos_resumo["ic95_sup"] = ic_sup
    else:
        todos_parasitos_resumo["ic95_inf"] = pd.Series(dtype=float)
        todos_parasitos_resumo["ic95_sup"] = pd.Series(dtype=float)

    # ---- efeito do nº de potes — usa só positividade FECAL (qualquer método
    # fecal ativo) e restrito a quem teve resultado fecal conclusivo. Mantém
    # como leitura descritiva rápida; ver também fecal_cumulativa para a versão
    # sem viés de seleção entre subgrupos.
    efeito_rows = []
    for n_pote, g in fecal_conclusivo.groupby("n_coletas_pote_entregue"):
        n_pos = int(g["positivo_fecal"].sum())
        efeito_rows.append({
            "n_potes_entregues": int(n_pote),
            "n_pacientes": len(g),
            "n_positivos": n_pos,
            "prevalencia": pct(n_pos, len(g)),
        })
    efeito_n_coletas = pd.DataFrame(efeito_rows).sort_values("n_potes_entregues").reset_index(drop=True) \
        if efeito_rows else pd.DataFrame(columns=["n_potes_entregues", "n_pacientes", "n_positivos", "prevalencia"])

    fecal_cumulativa = build_fecal_cumulative_curve(df, fecal_methods_ativos)

    # ---- testes de hipótese ----
    mcnemar_hpj_willis_res = mcnemar_hpj_willis(por_paciente)
    cochran_armitage_res = cochran_armitage_trend(efeito_n_coletas)

    return {
        "por_paciente": por_paciente,
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
        "prev_fecal_ic95_inf": prev_fecal_ic95_inf,
        "prev_fecal_ic95_sup": prev_fecal_ic95_sup,
        "prev_lamina": prev_lamina,
        "prev_lamina_ic95_inf": prev_lamina_ic95_inf,
        "prev_lamina_ic95_sup": prev_lamina_ic95_sup,
        "prev_combinada": prev_combinada,
        "prev_combinada_ic95_inf": prev_combinada_ic95_inf,
        "prev_combinada_ic95_sup": prev_combinada_ic95_sup,
        "especies_resumo": especies_resumo,
        "poli": poli, "mono": mono, "neg": neg,
        "combos_resumo": combos_resumo,
        "metodos_resumo": metodos_resumo,
        "efeito_n_coletas": efeito_n_coletas,
        "fecal_cumulativa": fecal_cumulativa,
        "metodo_especie_resumo": metodo_especie_resumo,
        "todos_parasitos_resumo": todos_parasitos_resumo,
        "mcnemar_hpj_willis": mcnemar_hpj_willis_res,
        "cochran_armitage_efeito_coletas": cochran_armitage_res,
        "metodos_ativos": active_methods,
        "metodos_ativos_nomes": [nome for _, nome, _, _ in active_methods],
    }
