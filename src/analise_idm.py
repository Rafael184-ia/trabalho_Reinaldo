# %% [markdown]
# # Projeto Final - Big Data e Data Science (FATESG 2026/01)
# **Base:** Índice de Desempenho dos Municípios (IDM) - IMB/Goiás, via BDE-Goiás
#
# Este arquivo funciona como script (`python src/analise_idm.py`) e como notebook
# (cada bloco `# %%` vira uma célula). Rode sempre a partir da raiz do repositório.

# %%
import base64
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:  # rodando como notebook
    ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent

DATA = ROOT / "data"
OUT_FIG = ROOT / "dashboard"
OUT_RES = ROOT / "results"
for p in (DATA, OUT_FIG, OUT_RES):
    p.mkdir(exist_ok=True)

SEED = 42
ANOS = [2012, 2014, 2016, 2018, 2020, 2024]
DIMS = ["IDM Economia", "IDM Educação", "IDM Infraestrutura", "IDM Saúde", "IDM Segurança", "IDM Trabalho"]
FEATS = [d for d in DIMS if d != "IDM Economia"]
TARGET = "IDM Economia"

# paleta (azul / laranja / cinza neutro) - cores de significado fixo em todos os gráficos
AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#8a8984"
TEXTO, TEXTO_SEC, GRADE = "#1c1c1b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRADE, "axes.labelcolor": TEXTO_SEC, "xtick.color": TEXTO_SEC,
    "ytick.color": TEXTO_SEC, "text.color": TEXTO, "axes.titlecolor": TEXTO,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8,
    "font.size": 10, "legend.frameon": False,
})

# %% [markdown]
# ## R1 - Coleta e tratamento dos dados
# O arquivo exportado do BDE tem uma linha por (município, variável) e uma coluna por edição do IDM.
# Passos: ler (cp1252, separador `;`), limpar nomes, converter vírgula decimal, remover duplicatas
# (os títulos de dimensão aparecem repetidos), reorganizar para 1 linha por (município, ano).

# %%
bruto = pd.read_csv(DATA / "consulta_idm_bde.csv", sep=";", encoding="cp1252", dtype=str)
print("Formato bruto:", bruto.shape)

bruto["Variável"] = bruto["Variável"].str.replace(r"\s*\(\)$", "", regex=True).str.strip()
n_dup = bruto.duplicated(["Localidade", "Variável"]).sum()
bruto = bruto.drop_duplicates(["Localidade", "Variável"])
print(f"Linhas duplicadas removidas: {n_dup}")

for a in map(str, ANOS):
    bruto[a] = pd.to_numeric(bruto[a].str.replace(",", ".", regex=False), errors="coerce")

longo = bruto.melt(id_vars=["Localidade", "Variável"], value_vars=list(map(str, ANOS)),
                   var_name="Ano", value_name="valor")
longo["Ano"] = longo["Ano"].astype(int)
w = (longo.pivot_table(index=["Localidade", "Ano"], columns="Variável", values="valor")
          .reset_index().rename_axis(columns=None))
w.to_csv(DATA / "idm_municipios_processado.csv", index=False, encoding="utf-8")
print("Base processada:", w.shape, "|", w.Localidade.nunique(), "municípios |", w.Ano.nunique(), "edições")

# %% [markdown]
# ### Valores nulos

# %%
nulos = w.drop(columns=["Localidade", "Ano"]).isna().sum()
print("Nulos nas 6 dimensões do IDM:\n", nulos[DIMS].to_string())
print(w.loc[w[DIMS].isna().any(axis=1), ["Localidade", "Ano"] + DIMS].to_string(index=False))

# Indicadores que não existem em determinadas edições (ausência estrutural, não erro de coleta)
estrutural = (w.groupby("Ano").apply(lambda g: g.isna().mean()).T.drop(["Localidade", "Ano"], errors="ignore"))
estrutural = estrutural[(estrutural >= 0.99).any(axis=1)]
print("\nIndicadores sem nenhum valor em alguma edição (fração de nulos por ano):")
print(estrutural.round(2).to_string())

# %% [markdown]
# ### Dicionário de colunas (R1)

# %%
GRUPOS = {
    "Economia": ["Equilíbrio orçamentário do município", "Evolução do PIB", "PIB per capita",
                 "Recursos próprios", "Setor agropecuário", "Setor industrial", "Setor de Serviços"],
    "Educação": ["Adequação dos professores do ensino infantil", "Adequação dos professores do ensino fundamental",
                 "Adequação dos professores do ensino médio", "Atendimento educacional da população de 4 a 5 anos",
                 "Atendimento educacional da população de 6 a 14 anos", "Atendimento educacional da população de 15 a 17 anos",
                 "IDEB 5º ano", "IDEB 9º ano"],
    "Infraestrutura": ["Infraestrutura dos prédios das escolas públicas urbanas", "Cobertura da rede de água tratada",
                       "Cobertura da rede de energia elétrica", "Cobertura da rede de esgoto",
                       "Cobertura da rede de telefonia fixa e internet"],
    "Saúde": ["Cobertura ESF", "Cobertura Vacinal tetravalente", "Cobertura ESF - Saúde Bucal", "Leitos SUS por 1.000 hab.",
              "Médicos SUS por 1.000 hab.", "Mortalidade Infantil", "Morte por causas externas (violentas)",
              "Prenatal c/ 7 consultas"],
    "Segurança": ["Crimes contra a dignidade sexual", "Crimes contra a pessoa", "Crimes contra o patrimônio",
                  "Contravenções penais", "Tráfico de drogas"],
    "Trabalho": ["Empregos formais", "Nível de escolaridade dos trabalhadores", "Remuneração mediana",
                 "Variação do número de empregos formais"],
}
linhas = [
    ("Localidade", "-", "Texto", "Nome do município goiano (246 municípios)", "IMB / BDE-Goiás", "todas"),
    ("Ano", "-", "Inteiro", "Edição do IDM (bianual: 2012, 2014, 2016, 2018, 2020, 2024)", "IMB / BDE-Goiás", "todas"),
    ("IDM Geral", "Geral", "Pontuação 0-10", "Índice sintético geral; é a média das 6 dimensões (NÃO usado como alvo)", "IMB / BDE-Goiás", "todas"),
]
for dim in DIMS:
    linhas.append((dim, dim.replace("IDM ", ""), "Pontuação 0-10", f"Índice sintético da dimensão {dim.replace('IDM ', '')}", "IMB / BDE-Goiás", "todas"))
for dim, vars_ in GRUPOS.items():
    for v in vars_:
        linhas.append((v, dim, "Pontuação 0-10", f"Indicador componente da dimensão {dim}, em pontuação padronizada", "IMB / BDE-Goiás", ""))
dic = pd.DataFrame(linhas, columns=["coluna", "dimensao", "unidade", "descricao", "fonte", "edicoes_com_dados"])
for i, r in dic.iterrows():
    if r.coluna in w.columns and r.coluna not in ("Localidade", "Ano"):
        ok = w.groupby("Ano")[r.coluna].apply(lambda s: s.notna().any())
        dic.loc[i, "edicoes_com_dados"] = "/".join(str(a) for a in ok[ok].index)
        dic.loc[i, "min"] = round(w[r.coluna].min(), 2)
        dic.loc[i, "max"] = round(w[r.coluna].max(), 2)
        dic.loc[i, "media"] = round(w[r.coluna].mean(), 2)
        dic.loc[i, "pct_nulos"] = round(100 * w[r.coluna].isna().mean(), 1)
dic.to_csv(DATA / "dicionario_colunas.csv", index=False, encoding="utf-8")
print(dic.to_string(index=False, max_colwidth=45))

# %% [markdown]
# ### Normalização e verificação de vazamento
# Todas as colunas já estão em pontuação 0-10. A padronização (z-score) é feita **dentro do pipeline**,
# ajustada só no treino. O IDM Geral é a média das 6 dimensões, então é excluído das variáveis do modelo.

# %%
media_dims = w[DIMS].mean(axis=1)
corr_geral = np.corrcoef(media_dims.fillna(media_dims.mean()), w["IDM Geral"])[0, 1]
print(f"Correlação IDM Geral x média das 6 dimensões: {corr_geral:.3f}  -> IDM Geral não pode ser alvo/feature")

estat = w.groupby("Ano")[DIMS + ["IDM Geral"]].mean().round(2)
print("\nMédia de cada dimensão por edição:\n", estat.to_string())

# %% [markdown]
# ## R2 - Dashboard (gráficos)

# %%
def salvar(fig, nome):
    fig.savefig(OUT_FIG / nome, dpi=150, bbox_inches="tight")


def g_ranking(ax, ano=2024):
    d = w[w.Ano == ano].set_index("Localidade")["IDM Geral"].sort_values()
    base = pd.concat([d.head(10), d.tail(10)])
    cores = [LARANJA] * 10 + [AZUL] * 10
    ax.barh(base.index, base.values, color=cores, height=0.72)
    for y, v in enumerate(base.values):
        ax.text(v + 0.03, y, f"{v:.2f}".replace(".", ","), va="center", fontsize=8, color=TEXTO_SEC)
    ax.set_xlim(0, base.max() * 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"10 melhores e 10 piores municípios - IDM Geral {ano}", loc="left")
    ax.set_xlabel("IDM Geral (0-10)")
    ax.tick_params(axis="y", labelsize=8)
    ax.text(0.98, 0.12, "● Melhores 10", color=AZUL, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")
    ax.text(0.98, 0.06, "● Piores 10", color=LARANJA, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")


def g_heatmap(ax):
    cmap = LinearSegmentedColormap.from_list("div", [LARANJA, "#f3f2ee", AZUL])
    c = w[DIMS].corr()
    nomes = [d.replace("IDM ", "") for d in DIMS]
    sns.heatmap(c, vmin=-1, vmax=1, cmap=cmap, annot=True, fmt=".2f", linewidths=2, linecolor="white",
                xticklabels=nomes, yticklabels=nomes, ax=ax, cbar_kws={"shrink": 0.8, "label": "Correlação de Pearson"},
                annot_kws={"fontsize": 9})
    ax.set_title("Correlação entre as dimensões do IDM (todas as edições)", loc="left")
    ax.grid(False)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)


def g_evolucao(fig, subspec):
    gs = subspec.subgridspec(2, 3, hspace=0.55, wspace=0.35)
    m = w.groupby("Ano")[DIMS].mean()
    eixo_x = np.arange(len(ANOS))
    lim = (0, 10)
    for i, dim in enumerate(DIMS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.plot(eixo_x[:5], m[dim].values[:5], color=AZUL, lw=2, marker="o", ms=4)
        ax.plot(eixo_x[4:], m[dim].values[4:], color=AZUL, lw=2, ls=(0, (3, 2)), marker="o", ms=4)
        ax.set_title(dim.replace("IDM ", ""), fontsize=9, loc="left")
        ax.set_ylim(*lim)
        ax.set_xticks(eixo_x)
        ax.set_xticklabels([str(a)[2:] for a in ANOS], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", visible=False)
        ax.annotate(f"{m[dim].values[-1]:.1f}".replace(".", ","), (eixo_x[-1], m[dim].values[-1]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color=TEXTO_SEC)
    fig.text(subspec.get_position(fig).x0, subspec.get_position(fig).y1 + 0.035,
             "Evolução da média municipal por dimensão (edições 2012-2024)", fontsize=12, fontweight="bold", ha="left")
    fig.text(subspec.get_position(fig).x0, subspec.get_position(fig).y0 - 0.025,
             "Tracejado: 2020→2024 (edição com mudança brusca de nível em Saúde, Educação e Segurança; ver README).",
             fontsize=7.5, color=TEXTO_SEC, ha="left")


def g_boxplot(ax):
    long_box = w.melt(id_vars=["Localidade", "Ano"], value_vars=DIMS, var_name="Dimensão", value_name="Pontuação")
    long_box["Dimensão"] = long_box["Dimensão"].str.replace("IDM ", "")
    ordem = long_box.groupby("Dimensão")["Pontuação"].median().sort_values().index
    sns.boxplot(data=long_box, x="Dimensão", y="Pontuação", order=ordem, color=AZUL, width=0.55,
                linewidth=1.2, fliersize=2.5, flierprops={"markerfacecolor": CINZA, "markeredgecolor": CINZA},
                boxprops={"alpha": 0.85}, medianprops={"color": "white", "linewidth": 2}, ax=ax)
    ax.set_title("Distribuição das dimensões entre municípios (todas as edições)", loc="left")
    ax.set_xlabel(""); ax.set_ylabel("Pontuação (0-10)")
    ax.set_ylim(0, 10.3)
    ax.grid(axis="x", visible=False)


# gráficos individuais
for nome, fn, size in [("01_ranking.png", g_ranking, (7, 7)), ("02_correlacao.png", g_heatmap, (7, 5.6)),
                       ("04_boxplot.png", g_boxplot, (8, 5))]:
    f, a = plt.subplots(figsize=size); fn(a); salvar(f, nome); plt.close(f)
f = plt.figure(figsize=(9, 5.2)); g_evolucao(f, f.add_gridspec(1, 1, left=0.07, right=0.97, top=0.88, bottom=0.12)[0]); salvar(f, "03_evolucao.png"); plt.close(f)

# painel único (dashboard)
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.92, bottom=0.07, hspace=0.38, wspace=0.28)
g_ranking(fig.add_subplot(gs[0, 0]))
g_heatmap(fig.add_subplot(gs[0, 1]))
g_evolucao(fig, gs[1, 0])
g_boxplot(fig.add_subplot(gs[1, 1]))
fig.suptitle("IDM dos municípios goianos - IMB/BDE-Goiás", fontsize=18, fontweight="bold", x=0.07, ha="left", y=0.975)
salvar(fig, "dashboard.png"); plt.close(fig)
print("Gráficos salvos em", OUT_FIG)

# %% [markdown]
# ## R3 - Modelagem preditiva
# **Problema:** prever o **IDM Economia** (valor numérico contínuo → **regressão**) a partir das outras 5 dimensões.
# **Algoritmos:** Regressão Linear (modelo simples, interpretável, serve de referência) e Random Forest
# (captura relações não lineares e interações).
# **Validação:** como cada município aparece em 6 edições, a divisão é feita **por município**
# (um município nunca está no treino e no teste ao mesmo tempo), evitando resultado inflado.

# %%
y = w[TARGET]
grupos = w["Localidade"]
ESPECS = {"A: 5 dimensões": FEATS, "B: 5 dimensões + edição (Ano)": FEATS + ["Ano"]}


def preproc(cols):
    num = [c for c in cols if c != "Ano"]
    trans = [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num)]
    if "Ano" in cols:
        trans.append(("ano", OneHotEncoder(handle_unknown="ignore"), ["Ano"]))
    return ColumnTransformer(trans)


def modelo(tipo, cols, **kw):
    reg = {"Regressão Linear": LinearRegression(),
           "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, **kw),
           "Baseline (média)": DummyRegressor()}[tipo]
    return Pipeline([("pre", preproc(cols)), ("reg", reg)])


def metricas(yt, yp):
    return {"R2": r2_score(yt, yp), "MAE": mean_absolute_error(yt, yp), "RMSE": mean_squared_error(yt, yp) ** 0.5}


idx_tr, idx_te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(w, y, grupos))
print(f"Treino: {len(idx_tr)} linhas ({grupos.iloc[idx_tr].nunique()} municípios) | "
      f"Teste: {len(idx_te)} linhas ({grupos.iloc[idx_te].nunique()} municípios)")

# %%
resultados, melhores, ajustados = [], {}, {}
for espec, cols in ESPECS.items():
    X = w[cols]
    for tipo in ["Baseline (média)", "Regressão Linear", "Random Forest"]:
        if tipo == "Random Forest":
            gs_cv = GridSearchCV(modelo(tipo, cols), {"reg__max_depth": [4, 8, None], "reg__min_samples_leaf": [1, 3, 5]},
                                 cv=GroupKFold(3), scoring="r2", n_jobs=-1)
            gs_cv.fit(X.iloc[idx_tr], y.iloc[idx_tr], groups=grupos.iloc[idx_tr])
            m = gs_cv.best_estimator_
            melhores[espec] = {k.replace("reg__", ""): v for k, v in gs_cv.best_params_.items()}
        else:
            m = modelo(tipo, cols).fit(X.iloc[idx_tr], y.iloc[idx_tr])
        pred = m.predict(X.iloc[idx_te])
        ajustados[(espec, tipo)] = (m, pred)
        r = metricas(y.iloc[idx_te], pred)
        # validação cruzada 5-fold por município (hiperparâmetros fixos para o RF)
        m_cv = modelo(tipo, cols, **({"min_samples_leaf": 3} if tipo == "Random Forest" else {}))
        cv = cross_validate(m_cv, X, y, groups=grupos, cv=GroupKFold(5),
                            scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"])
        resultados.append({"Especificação": espec, "Modelo": tipo, **{f"{k} (teste)": v for k, v in r.items()},
                           "R2 (CV 5-fold)": cv["test_r2"].mean(), "R2 CV desvio": cv["test_r2"].std(),
                           "MAE (CV 5-fold)": -cv["test_neg_mean_absolute_error"].mean(),
                           "RMSE (CV 5-fold)": -cv["test_neg_root_mean_squared_error"].mean()})

res = pd.DataFrame(resultados)
res.to_csv(OUT_RES / "metricas_modelos.csv", index=False, encoding="utf-8")
json.dump({"melhores_hiperparametros_rf": melhores, "metricas": res.round(4).to_dict(orient="records")},
          open(OUT_RES / "metricas_modelos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(res.round(3).to_string(index=False))
print("\nMelhores hiperparâmetros do Random Forest:", melhores)

# %% [markdown]
# ### Gráficos do modelo (valores previstos x reais, importância das variáveis)

# %%
ESP_FINAL = "B: 5 dimensões + edição (Ano)"
m_rf, pred_rf = ajustados[(ESP_FINAL, "Random Forest")]
_, pred_lr = ajustados[(ESP_FINAL, "Regressão Linear")]
yt = y.iloc[idx_te]

fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
for ax, p, nome, cor in [(axs[0], pred_lr, "Regressão Linear", LARANJA), (axs[1], pred_rf, "Random Forest", AZUL)]:
    ax.scatter(yt, p, s=14, color=cor, alpha=0.55, edgecolor="none")
    lim = [0, max(yt.max(), p.max()) + 0.3]
    ax.plot(lim, lim, color=CINZA, lw=1.2, ls="--")
    r = metricas(yt, p)
    ax.set_title(f"{nome}  (R² = {r['R2']:.2f}".replace(".", ",") + f"; MAE = {r['MAE']:.2f})".replace(".", ","), loc="left", fontsize=11)
    ax.set_xlabel("IDM Economia real")
axs[0].set_ylabel("IDM Economia previsto")
fig.suptitle("Conjunto de teste (municípios não vistos no treino) - linha tracejada = previsão perfeita", fontsize=10, color=TEXTO_SEC, y=1.0)
salvar(fig, "05_previsto_vs_real.png"); plt.close(fig)

imp = permutation_importance(m_rf, w[ESCOLHA := ESPECS[ESP_FINAL]].iloc[idx_te], yt, n_repeats=30, random_state=SEED, scoring="r2")
imp_s = pd.Series(imp.importances_mean, index=[c.replace("IDM ", "") for c in ESCOLHA]).sort_values()
imp_s.to_csv(OUT_RES / "importancia_variaveis_rf.csv", header=["queda_R2_ao_embaralhar"], encoding="utf-8")
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(imp_s.index, imp_s.values, color=AZUL, height=0.6)
for yy, v in enumerate(imp_s.values):
    ax.text(v + 0.003, yy, f"{v:.3f}".replace(".", ","), va="center", fontsize=9, color=TEXTO_SEC)
ax.set_title("Importância das variáveis (Random Forest, teste)", loc="left")
ax.set_xlabel("Queda do R² ao embaralhar a variável")
ax.grid(axis="y", visible=False)
salvar(fig, "06_importancia.png"); plt.close(fig)

coef = pd.Series(m_rf.named_steps["pre"].get_feature_names_out()).tolist()
lr_m = ajustados[(ESP_FINAL, "Regressão Linear")][0]
coefs = pd.Series(lr_m.named_steps["reg"].coef_, index=lr_m.named_steps["pre"].get_feature_names_out()).round(3)
print("Coeficientes da Regressão Linear (variáveis padronizadas):\n", coefs.to_string())
print("\nImportância (Random Forest):\n", imp_s.sort_values(ascending=False).round(3).to_string())

# %% [markdown]
# ### Dashboard HTML (autossuficiente)

# %%
def b64(nome):
    return base64.b64encode((OUT_FIG / nome).read_bytes()).decode()


rf_b = res[(res["Especificação"] == ESP_FINAL) & (res.Modelo == "Random Forest")].iloc[0]
lr_b = res[(res["Especificação"] == ESP_FINAL) & (res.Modelo == "Regressão Linear")].iloc[0]
top = w[w.Ano == 2024].sort_values("IDM Geral", ascending=False).iloc[0]
pior = w[w.Ano == 2024].sort_values("IDM Geral").iloc[0]
fmt = lambda v: f"{v:.2f}".replace(".", ",")
tabela = res[["Especificação", "Modelo", "R2 (teste)", "MAE (teste)", "RMSE (teste)", "R2 (CV 5-fold)"]].round(3)
tab_html = "".join(
    "<tr>" + "".join(f"<td>{str(v).replace('.', ',') if isinstance(v, float) else v}</td>" for v in row) + "</tr>"
    for row in tabela.itertuples(index=False))
cab = "".join(f"<th>{c}</th>" for c in tabela.columns)

html = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dashboard IDM Goiás</title><style>
:root{{--bg:#fcfcfb;--tx:#1c1c1b;--sec:#52514e;--line:#e4e3df}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--tx:#f0efec;--sec:#c3c2b7;--line:#33322f}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,Segoe UI,sans-serif}}
main{{max-width:1100px;margin:0 auto}} h1{{margin:0 0 4px}} .sub{{color:var(--sec);margin:0 0 20px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px}}
.kpi{{border:1px solid var(--line);border-radius:10px;padding:12px 14px}} .kpi b{{display:block;font-size:22px}} .kpi span{{color:var(--sec);font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}
.card{{border:1px solid var(--line);border-radius:10px;padding:10px;background:#fff}} .card img{{width:100%;height:auto;display:block}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}} th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}}
h2{{margin:28px 0 8px}} p.n{{color:var(--sec);font-size:13px}}
</style></head><body><main>
<h1>IDM dos municípios goianos</h1><p class="sub">Fonte: IMB / BDE-Goiás · {w.Localidade.nunique()} municípios · edições {", ".join(map(str, ANOS))}</p>
<div class="kpis">
<div class="kpi"><b>{top.Localidade}</b><span>maior IDM Geral em 2024 ({fmt(top['IDM Geral'])})</span></div>
<div class="kpi"><b>{pior.Localidade}</b><span>menor IDM Geral em 2024 ({fmt(pior['IDM Geral'])})</span></div>
<div class="kpi"><b>R² = {fmt(rf_b['R2 (teste)'])}</b><span>Random Forest prevendo IDM Economia (teste)</span></div>
<div class="kpi"><b>R² = {fmt(lr_b['R2 (teste)'])}</b><span>Regressão Linear prevendo IDM Economia (teste)</span></div></div>
<div class="grid">
<div class="card"><img alt="10 melhores e 10 piores municípios" src="data:image/png;base64,{b64('01_ranking.png')}"></div>
<div class="card"><img alt="Mapa de calor de correlação" src="data:image/png;base64,{b64('02_correlacao.png')}"></div>
<div class="card"><img alt="Evolução por dimensão" src="data:image/png;base64,{b64('03_evolucao.png')}"></div>
<div class="card"><img alt="Boxplot por dimensão" src="data:image/png;base64,{b64('04_boxplot.png')}"></div></div>
<h2>Modelos (alvo: IDM Economia)</h2>
<div class="grid"><div class="card"><img alt="Previsto x real" src="data:image/png;base64,{b64('05_previsto_vs_real.png')}"></div>
<div class="card"><img alt="Importância das variáveis" src="data:image/png;base64,{b64('06_importancia.png')}"></div></div>
<table><thead><tr>{cab}</tr></thead><tbody>{tab_html}</tbody></table>
<p class="n">Teste = 20% dos municípios separados antes do treino. CV = validação cruzada 5-fold agrupada por município.</p>
</main></body></html>"""
(OUT_FIG / "dashboard.html").write_text(html, encoding="utf-8")
print("Dashboard HTML:", OUT_FIG / "dashboard.html")
