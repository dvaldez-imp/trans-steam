# app.py
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict


# =========================
# Config
# =========================
st.set_page_config(page_title="Rutas - Optimización de Capacidad", layout="wide")

SPANISH_WEEKDAY_ORDER = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
WEEKDAY_MAP_ES = {
    "Monday": "Lunes",
    "Tuesday": "Martes",
    "Wednesday": "Miércoles",
    "Thursday": "Jueves",
    "Friday": "Viernes",
    "Saturday": "Sábado",
    "Sunday": "Domingo",
}

DEFAULT_COMMON_ROUTES_TEXT = """SM02\tSM03
SM04\tSM06
SG01\tSG05
SG04\tSG06
MFA1\tMFA2
MFA3\tMFA1
"""


@dataclass
class ParsedData:
    resumen: pd.DataFrame
    dinamo: Optional[pd.DataFrame]
    warnings: List[str]


# =========================
# Text utils (normalización para joins robustos)
# =========================
def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return s.lower().strip()


def norm_jornada(x: object) -> str:
    s = norm_text(x)
    if s in {"ordinario", "ordinaria"}:
        return "ordinario"
    if s == "turno":
        return "turno"
    return s


def parse_money_like(x: object) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = s.replace("Q", "").replace("q", "")
    s = s.replace(",", "")
    s = s.strip()
    try:
        return float(s)
    except:
        return np.nan


# =========================
# Parsing helpers (resumen)
# =========================
def _parse_percent_series(s: pd.Series) -> pd.Series:
    def to_float(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.strip()
            if x.endswith("%"):
                x = x[:-1].strip()
                try:
                    return float(x) / 100.0
                except:
                    return np.nan
            try:
                return float(x)
            except:
                return np.nan
        try:
            return float(x)
        except:
            return np.nan

    out = s.apply(to_float)
    # si viniera 48 en lugar de 0.48, lo interpretamos como %
    out = out.where(out <= 1.0, out / 100.0)
    # ✅ CAP 0–100%
    out = out.clip(lower=0.0, upper=1.0)
    return out


def _coerce_datetime(s: pd.Series, dayfirst: bool) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)

def parse_coef_text(text: str) -> dict:
    """
    Acepta líneas tipo:
      feature<TAB>coef
      feature  coef
    Ej:
      intercept    8213.80
      bus_Mercedes Benz 4768.05
    """
    coef = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue

        # intenta split por tab, si no por espacios múltiples
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
        else:
            parts = re.split(r"\s{2,}|\s+(?=[-+]?\d+(\.\d+)?$)", line)

        if len(parts) < 2:
            continue

        name = parts[0].strip()
        try:
            val = float(parts[-1])
        except:
            continue

        coef[name] = val
    return coef


def build_basic_contrib_row(row: pd.Series, coef: dict) -> tuple[float, pd.DataFrame, list[str]]:
    """
    Construye contribuciones usando:
      - intercept como baseline
      - numéricas: coef * valor_columna
      - categóricas:
          bus_<Tipo de bus> => 1 si coincide
          jornada_<Jornada> => 1 si coincide
    """
    warnings = []
    intercept = float(coef.get("intercept", 0.0))

    contribs = []

    # 1) variables numéricas: si el feature existe como columna exacta
    for feat, b in coef.items():
        if feat == "intercept":
            continue
        if feat.startswith("bus_") or feat.startswith("jornada_"):
            continue

        if feat in row.index:
            x = row[feat]
            x = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
            if pd.isna(x):
                warnings.append(f"'{feat}' está NaN → contrib=0")
                x = 0.0
            contribs.append((feat, float(b) * float(x), float(x), float(b)))
        else:
            # si no existe la columna, solo la ignoramos (sin reventar)
            warnings.append(f"No encontré columna '{feat}' en la fila → se ignora")

    # 2) categóricas: Tipo de bus
    bus_type = str(row.get("Tipo de bus", "")).strip()
    if bus_type:
        key = f"bus_{bus_type}"
        if key in coef:
            b = float(coef[key])
            contribs.append((key, b * 1.0, 1.0, b))
        # si no existe coef para ese bus (p.ej. categoría base), contrib=0

    # 3) categóricas: Jornada
    jornada = str(row.get("Jornada", "")).strip()
    if jornada:
        key = f"jornada_{jornada}"
        if key in coef:
            b = float(coef[key])
            contribs.append((key, b * 1.0, 1.0, b))

    # tabla
    dfc = pd.DataFrame(contribs, columns=["feature", "contrib_Q", "x", "coef"])
    dfc = dfc.sort_values("contrib_Q", key=lambda s: s.abs(), ascending=False)

    pred = intercept + dfc["contrib_Q"].sum() if not dfc.empty else intercept
    return float(pred), dfc, warnings


def plot_basic_waterfall(intercept: float, pred: float, contrib_df: pd.DataFrame, title: str):
    # Waterfall: todas las contribuciones como relative, y un total al final.
    labels = contrib_df["feature"].tolist() + ["Predicción"]
    values = contrib_df["contrib_Q"].tolist() + [0]
    measures = ["relative"] * len(contrib_df) + ["total"]

    fig = go.Figure(go.Waterfall(
        x=labels,
        y=values,
        measure=measures
    ))
    fig.update_layout(
        title=f"{title} | baseline(intercept)={intercept:.2f} → pred={pred:.2f}",
        xaxis_title="Variable",
        yaxis_title="Aporte al Precio (Q)"
    )
    return fig


@st.cache_data(show_spinner=False)
def load_excel_sheets(file_bytes: bytes, sheet_resumen: str, sheet_dinamo: str, dayfirst: bool) -> ParsedData:
    warnings: List[str] = []

    # --- Resumen ---
    df = pd.read_excel(file_bytes, sheet_name=sheet_resumen, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    col_inicio = "hora de inicio"
    col_fin = "hora de finalizacion"
    col_date = "date"
    col_pct = "% de ocupación"
    col_ie = "Ingreso / Egreso"

    # datetimes
    if col_inicio in df.columns:
        df["inicio_dt"] = _coerce_datetime(df[col_inicio], dayfirst=dayfirst)
        if df["inicio_dt"].isna().mean() > 0.2:
            warnings.append(f"Muchos valores no se pudieron parsear en '{col_inicio}'. Revisá formato/idioma.")
    elif col_date in df.columns:
        df["inicio_dt"] = _coerce_datetime(df[col_date], dayfirst=dayfirst)
        warnings.append(f"No encontré '{col_inicio}'. Estoy usando '{col_date}' como inicio.")
    else:
        df["inicio_dt"] = pd.NaT
        warnings.append("No encontré 'hora de inicio' ni 'date' en resumen.")

    if col_fin in df.columns:
        df["fin_dt"] = _coerce_datetime(df[col_fin], dayfirst=dayfirst)
    else:
        df["fin_dt"] = pd.NaT

    df["duracion_s"] = (df["fin_dt"] - df["inicio_dt"]).dt.total_seconds()

    # % ocupación
    if col_pct in df.columns:
        df["pct_ocup"] = _parse_percent_series(df[col_pct])
    else:
        df["pct_ocup"] = np.nan

    # numéricos clave
    for c in ["Cantidad de correlativos", "Ocupación máxima"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # categoricos
    for c in ["jornada", "ruta", "sede", "piloto", "horario de turno", col_ie]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    # normalizar Ingreso/Egreso
    if col_ie in df.columns:
        ie = df[col_ie].astype(str).str.strip().str.lower()
        ie = np.where(ie.str.contains("egre"), "egreso",
                      np.where(ie.str.contains("ingre"), "ingreso", ie))
        df["ie_norm"] = ie
    else:
        df["ie_norm"] = ""

    # derivados fecha
    df["fecha"] = df["inicio_dt"].dt.date
    df["mes_ym"] = df["inicio_dt"].dt.to_period("M").astype(str)
    df["dow_en"] = df["inicio_dt"].dt.day_name()
    df["dia_semana"] = pd.Categorical(df["dow_en"].map(WEEKDAY_MAP_ES), categories=SPANISH_WEEKDAY_ORDER, ordered=True)

    # normalizadores para joins
    df["ruta_norm"] = df["ruta"].map(norm_text) if "ruta" in df.columns else ""
    df["sede_norm"] = df["sede"].map(norm_text) if "sede" in df.columns else ""
    df["jornada_norm"] = df["jornada"].map(norm_jornada) if "jornada" in df.columns else ""

        # --- Dinamo ---
    dinamo = None
    try:
        d = pd.read_excel(file_bytes, sheet_name=sheet_dinamo, engine="openpyxl")
        d.columns = [str(c).strip() for c in d.columns]

        # normalizar nombres esperados
        rename_map = {}
        for col in d.columns:
            n = norm_text(col)
            if n == "precio q (diario)":
                rename_map[col] = "Precio Q (Diario)"
            if n in {"# dias", "# d ias", "dias"}:
                rename_map[col] = "# Días"
            if n == "tiempo (minutos)":
                rename_map[col] = "Tiempo (minutos)"
            if n in {"tipo de bus", "tipo bus", "tipodebus"}:
                rename_map[col] = "Tipo de bus"
        if rename_map:
            d = d.rename(columns=rename_map)

        # numéricos
        if "Precio Q (Diario)" in d.columns:
            d["Precio Q (Diario)"] = d["Precio Q (Diario)"].apply(parse_money_like)
        for c in ["Capacidad", "# Días", "Tiempo (minutos)", "Km"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")

        # texto
        for c in ["Sede", "Ruta", "Jornada", "Destino", "Tipo de bus"]:
            if c in d.columns:
                d[c] = d[c].astype(str).fillna("").str.strip()

        # normalizadores para joins
        d["ruta_norm"] = d["Ruta"].map(norm_text) if "Ruta" in d.columns else ""
        d["sede_norm"] = d["Sede"].map(norm_text) if "Sede" in d.columns else ""
        d["jornada_norm"] = d["Jornada"].map(norm_jornada) if "Jornada" in d.columns else ""

        # normalizador de tipo de bus (para categorías limpias)
        if "Tipo de bus" in d.columns:
            d["bus_tipo_norm"] = d["Tipo de bus"].map(norm_text)

        dinamo = d
    except Exception as e:
        warnings.append(f"No pude leer hoja '{sheet_dinamo}'. Error: {e}")

    return ParsedData(resumen=df, dinamo=dinamo, warnings=warnings)


# =========================
# Corredores (rutas en común)
# =========================
def parse_edges_from_text(text: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    warnings: List[str] = []
    edges: List[Tuple[str, str]] = []
    if not text or not text.strip():
        return edges, warnings

    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        if "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
        elif "," in line:
            parts = [p.strip() for p in line.split(",") if p.strip()]
        else:
            parts = [p.strip() for p in line.split() if p.strip()]

        if len(parts) < 2:
            warnings.append(f"Línea {i}: no pude leer par en '{raw}'.")
            continue

        a, b = parts[0], parts[1]
        if a == b:
            warnings.append(f"Línea {i}: '{a}' con '{b}' es el mismo; ignorado.")
            continue

        edges.append((a, b))

    return edges, warnings


def build_corridors_from_edges(edges: List[Tuple[str, str]]) -> List[Tuple[str, List[str]]]:
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    groups: Dict[str, set] = {}
    for node in list(parent.keys()):
        root = find(node)
        groups.setdefault(root, set()).add(node)

    members_sorted = sorted([sorted(list(s)) for s in groups.values()], key=lambda x: (-len(x), x))
    corridors: List[Tuple[str, List[str]]] = []
    for i, members in enumerate(members_sorted):
        corridors.append((f"CORR-{i+1:03d}", members))
    return corridors


def assign_corridor(df: pd.DataFrame, edges: List[Tuple[str, str]], route_col: str = "ruta") -> Tuple[pd.DataFrame, List[Tuple[str, List[str]]]]:
    corridors = build_corridors_from_edges(edges)
    route_to_corr: Dict[str, str] = {}
    for corr_id, routes in corridors:
        for r in routes:
            route_to_corr[str(r).strip()] = corr_id

    out = df.copy()
    if route_col not in out.columns:
        out["corredor_id"] = "SIN_CORREDOR"
        out["corredor_desc"] = "SIN_CORREDOR"
        return out, corridors

    out[route_col] = out[route_col].astype(str).str.strip()
    out["corredor_id"] = out[route_col].map(route_to_corr).fillna("SIN_CORREDOR")

    corr_desc = {cid: f"{cid} ({', '.join(routes)})" for cid, routes in corridors}
    out["corredor_desc"] = out["corredor_id"].map(corr_desc).fillna("SIN_CORREDOR")
    return out, corridors


# =========================
# Diagnóstico Ingreso/Egreso (solo reporta)
# =========================
def make_bus_key(df: pd.DataFrame, key_mode: str) -> pd.Series:
    def col_or_empty(c):
        return df[c].astype(str).fillna("").str.strip() if c in df.columns else ""

    sede = col_or_empty("sede")
    ruta = col_or_empty("ruta")
    jornada = col_or_empty("jornada")
    piloto = col_or_empty("piloto")

    if key_mode == "sede+ruta":
        return sede + " | " + ruta
    if key_mode == "sede+ruta+jornada":
        return sede + " | " + ruta + " | " + jornada
    if key_mode == "sede+ruta+piloto":
        return sede + " | " + ruta + " | " + piloto
    return sede + " | " + ruta + " | " + jornada + " | " + piloto


def pairing_quality(events_df: pd.DataFrame, key_mode: str) -> Dict[str, int]:
    if events_df.empty or "ie_norm" not in events_df.columns or "inicio_dt" not in events_df.columns:
        return {"orphan_egreso": 0, "consecutive_ingreso": 0, "open_ingreso_end": 0}

    w = events_df.dropna(subset=["inicio_dt"]).copy()
    w["bus_key"] = make_bus_key(w, key_mode=key_mode)
    w = w.sort_values(["bus_key", "inicio_dt"])

    open_ing: Dict[str, pd.Timestamp] = {}
    orphan_eg = 0
    consec_in = 0

    for _, r in w.iterrows():
        k = r["bus_key"]
        ie = r["ie_norm"]

        if ie == "ingreso":
            if k in open_ing:
                consec_in += 1
            open_ing[k] = r["inicio_dt"]

        elif ie == "egreso":
            if k not in open_ing:
                orphan_eg += 1
            else:
                open_ing.pop(k, None)

    open_end = len(open_ing)
    return {"orphan_egreso": orphan_eg, "consecutive_ingreso": consec_in, "open_ingreso_end": open_end}


# =========================
# Filtros globales
# =========================
@dataclass
class FilterState:
    date_range: Optional[Tuple]
    corredor_sel: Optional[List[str]]
    sede_sel: Optional[List[str]]
    ruta_sel: Optional[List[str]]  # ahora será None o [una]
    jornada_sel: Optional[List[str]]
    piloto_sel: Optional[List[str]]
    ie_sel: Optional[List[str]]


def _multiselect_if_exists(df: pd.DataFrame, col: str, label: str):
    if col not in df.columns:
        return None
    vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v).strip() != ""])
    if not vals:
        return None
    return st.multiselect(label, vals, default=[])


def _select_one_or_all(df: pd.DataFrame, col: str, label: str) -> Optional[List[str]]:
    if col not in df.columns:
        return None
    vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v).strip() != ""])
    if not vals:
        return None
    choice = st.selectbox(label, ["Todas"] + vals, index=0)
    if choice == "Todas":
        return None
    return [choice]


def get_filters(events_df: pd.DataFrame) -> FilterState:
    with st.sidebar:
        st.header("Filtros (globales)")

        min_dt = events_df["inicio_dt"].min() if "inicio_dt" in events_df.columns else pd.NaT
        max_dt = events_df["inicio_dt"].max() if "inicio_dt" in events_df.columns else pd.NaT

        date_range = None
        if pd.notna(min_dt) and pd.notna(max_dt):
            date_range = st.date_input(
                "Rango de fechas",
                value=(min_dt.date(), max_dt.date()),
                min_value=min_dt.date(),
                max_value=max_dt.date(),
            )

        corredor_sel = _multiselect_if_exists(events_df, "corredor_desc", "Corredor (rutas en común)")
        sede_sel = _multiselect_if_exists(events_df, "sede", "Sede")

        # ✅ RUTA: “una o todas” (no multiruta)
        ruta_sel = _select_one_or_all(events_df, "ruta", "Ruta (una o todas)")

        jornada_sel = _multiselect_if_exists(events_df, "jornada", "Jornada")
        piloto_sel = _multiselect_if_exists(events_df, "piloto", "Piloto")
        ie_sel = _multiselect_if_exists(events_df, "Ingreso / Egreso", "Ingreso / Egreso")

    return FilterState(
        date_range=date_range,
        corredor_sel=corredor_sel,
        sede_sel=sede_sel,
        ruta_sel=ruta_sel,
        jornada_sel=jornada_sel,
        piloto_sel=piloto_sel,
        ie_sel=ie_sel,
    )


def apply_filters_events(df: pd.DataFrame, fs: FilterState) -> pd.DataFrame:
    out = df.copy()

    if fs.date_range and isinstance(fs.date_range, tuple) and len(fs.date_range) == 2 and "inicio_dt" in out.columns:
        d0, d1 = fs.date_range
        out = out[(out["inicio_dt"].dt.date >= d0) & (out["inicio_dt"].dt.date <= d1)]

    def apply_multi(col: str, selected: Optional[List[str]]):
        nonlocal out
        if selected is not None and len(selected) > 0 and col in out.columns:
            out = out[out[col].isin(selected)]

    apply_multi("corredor_desc", fs.corredor_sel)
    apply_multi("sede", fs.sede_sel)
    apply_multi("ruta", fs.ruta_sel)
    apply_multi("jornada", fs.jornada_sel)
    apply_multi("piloto", fs.piloto_sel)
    apply_multi("Ingreso / Egreso", fs.ie_sel)

    return out


# =========================
# Demanda = Cantidad de correlativos (hard)
# =========================
def compute_demand(events_df: pd.DataFrame) -> pd.Series:
    if "Cantidad de correlativos" not in events_df.columns:
        return pd.Series([np.nan] * len(events_df), index=events_df.index)
    return pd.to_numeric(events_df["Cantidad de correlativos"], errors="coerce")


# =========================
# KPI
# =========================
def kpi_row_events(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Movimientos (registros)", f"{len(df):,}".replace(",", " "))

    if "ie_norm" in df.columns:
        ing = int((df["ie_norm"] == "ingreso").sum())
        egr = int((df["ie_norm"] == "egreso").sum())
        c2.metric("Ingreso / Egreso", f"{ing} / {egr}")
    else:
        c2.metric("Ingreso / Egreso", "—")

    if "pct_ocup" in df.columns:
        mean_pct = df["pct_ocup"].mean()
        c3.metric("% ocupación (prom.)", f"{(mean_pct * 100):.1f}%" if pd.notna(mean_pct) else "—")
    else:
        c3.metric("% ocupación (prom.)", "—")

    if "Ocupación máxima" in df.columns:
        med_cap = pd.to_numeric(df["Ocupación máxima"], errors="coerce").median()
        c4.metric("Capacidad típica (mediana)", f"{med_cap:.0f}" if pd.notna(med_cap) else "—")
    else:
        c4.metric("Capacidad típica (mediana)", "—")


# =========================
# Boxplot + cascada + series generales
# =========================
def boxplot_section_events(df: pd.DataFrame):
    st.subheader("Caja y bigotes — por mes / día semana (con filtros locales)")

    if df.empty:
        st.info("No hay datos.")
        return

    left, _ = st.columns([1, 3])
    with left:
        group = st.radio("Agrupar por", ["Mes", "Día de la semana"], horizontal=False, key="box_group")
        metric = st.selectbox("Métrica", ["Demanda (correlativos)", "% ocupación"], key="box_metric")

        # ✅ rutas: “una o todas”
        rutas = sorted([r for r in df["ruta"].dropna().unique().tolist()]) if "ruta" in df.columns else []
        ruta_choice = st.selectbox("Ruta (local)", ["Todas"] + rutas, index=0, key="box_ruta_one")
        # jornada: sigue siendo multi (todas por default)
        jornadas = sorted([j for j in df["jornada"].dropna().unique().tolist()]) if "jornada" in df.columns else []
        jornadas_sel = st.multiselect("Jornadas (local)", jornadas, default=jornadas, key="box_jornadas")

        dir_sel = st.selectbox("Movimiento (local)", ["Ambos", "Ingreso", "Egreso"], index=0, key="box_dir")

    group_col = "mes_ym" if group == "Mes" else "dia_semana"

    base = df.copy()
    if ruta_choice != "Todas":
        base = base[base["ruta"] == ruta_choice]
    if jornadas_sel:
        base = base[base["jornada"].isin(jornadas_sel)]
    if dir_sel != "Ambos" and "ie_norm" in base.columns:
        base = base[base["ie_norm"] == ("ingreso" if dir_sel == "Ingreso" else "egreso")]

    if base.empty:
        st.info("No hay datos después del filtro local.")
        return

    if metric == "% ocupación":
        if "pct_ocup" not in base.columns:
            st.info("No existe % ocupación.")
            return
        plot_df = base[[group_col, "pct_ocup"]].dropna().copy()
        plot_df["valor"] = plot_df["pct_ocup"] * 100.0
        fig = px.box(plot_df, x=group_col, y="valor", points="outliers",
                     title=f"% ocupación por {group} ({dir_sel})")
        fig.update_layout(xaxis_title=group, yaxis_title="% ocupación")
        st.plotly_chart(fig, use_container_width=True)
        return

    base["demanda"] = compute_demand(base)
    plot_df = base[[group_col, "demanda"]].dropna()
    if plot_df.empty:
        st.info("No hay demanda (Cantidad de correlativos) para el boxplot.")
        return

    fig = px.box(plot_df, x=group_col, y="demanda", points="outliers",
                 title=f"Demanda (correlativos) por {group} ({dir_sel})")
    fig.update_layout(xaxis_title=group, yaxis_title="Demanda (Cantidad de correlativos)")
    st.plotly_chart(fig, use_container_width=True)


def waterfall_by_group_events(df: pd.DataFrame):
    st.subheader("Cascada — Total → descomposición por jornada")

    if df.empty or "jornada" not in df.columns:
        st.info("No hay datos o falta 'jornada'.")
        return

    left, _ = st.columns([1, 3])
    with left:
        group = st.radio("Grupo", ["Mes", "Día de la semana"], horizontal=False, key="wf_group")
        measure = st.selectbox("Métrica", ["Movimientos (conteo)", "Demanda (suma)"], key="wf_measure")
        max_groups = st.slider("Máx. gráficos", 1, 30, 8, key="wf_max")
        dir_sel = st.selectbox("Movimiento (local)", ["Ambos", "Ingreso", "Egreso"], index=0, key="wf_dir")

        jornadas = sorted([j for j in df["jornada"].dropna().unique().tolist()])
        jornadas_sel = st.multiselect("Jornadas (local)", jornadas, default=jornadas, key="wf_jornadas")

    group_col = "mes_ym" if group == "Mes" else "dia_semana"

    base = df.copy()
    base = base[base["jornada"].isin(jornadas_sel)]
    if dir_sel != "Ambos" and "ie_norm" in base.columns:
        base = base[base["ie_norm"] == ("ingreso" if dir_sel == "Ingreso" else "egreso")]

    if base.empty:
        st.info("No hay datos para cascada.")
        return

    if measure == "Movimientos (conteo)":
        agg = base.groupby([group_col, "jornada"]).size().reset_index(name="valor")
        total = base.groupby(group_col).size().reset_index(name="total")
        y_title = "Movimientos"
    else:
        base["demanda"] = compute_demand(base)
        agg = base.groupby([group_col, "jornada"])["demanda"].sum().reset_index(name="valor")
        total = base.groupby(group_col)["demanda"].sum().reset_index(name="total")
        y_title = "Demanda (suma)"

    show_groups = total.sort_values("total", ascending=False).head(max_groups)[group_col].tolist()

    cols = st.columns(2)
    col_i = 0
    for g in show_groups:
        g_rows = agg[agg[group_col] == g].sort_values("valor", ascending=False)
        if g_rows.empty:
            continue

        labels = g_rows["jornada"].tolist() + ["Total"]
        values = g_rows["valor"].tolist() + [0]
        measures = ["relative"] * len(g_rows) + ["total"]

        fig = go.Figure(go.Waterfall(x=labels, y=values, measure=measures))
        fig.update_layout(title=f"{y_title} – {group}: {g} ({dir_sel})",
                          xaxis_title="Jornada", yaxis_title=y_title)
        cols[col_i].plotly_chart(fig, use_container_width=True)
        col_i = 1 - col_i


def time_bin(df: pd.DataFrame, dt_col: str, gran: str) -> pd.Series:
    t = pd.to_datetime(df[dt_col], errors="coerce")
    if gran == "Día":
        return t.dt.to_period("D").dt.to_timestamp()
    if gran == "Semana":
        return t.dt.to_period("W").dt.start_time
    return t.dt.to_period("M").dt.to_timestamp()


def timeseries_overview(df: pd.DataFrame):
    st.subheader("Series de tiempo (sin hora): movimientos + demanda")

    if df.empty:
        st.info("No hay datos.")
        return

    left, _ = st.columns([1, 3])
    with left:
        gran = st.selectbox("Granularidad", ["Día", "Semana", "Mes"], index=1, key="ts_gran")
        metric = st.selectbox("Métrica", ["Movimientos", "Demanda (suma)", "Demanda (P95)"], key="ts_metric")
        split = st.selectbox("Separar por", ["Nada", "Ingreso/Egreso", "Jornada"], key="ts_split")

    base = df.dropna(subset=["inicio_dt"]).copy()
    base["t"] = time_bin(base, "inicio_dt", gran)
    base["demanda"] = compute_demand(base)

    color_col = None
    if split == "Ingreso/Egreso" and "ie_norm" in base.columns:
        base["ie_label"] = base["ie_norm"].map({"ingreso": "Ingreso", "egreso": "Egreso"}).fillna("Otro")
        color_col = "ie_label"
    elif split == "Jornada" and "jornada" in base.columns:
        color_col = "jornada"

    if metric == "Movimientos":
        agg = base.groupby(["t", color_col]).size().reset_index(name="valor") if color_col else base.groupby("t").size().reset_index(name="valor")
        fig = px.line(agg, x="t", y="valor", color=color_col, markers=True)
        fig.update_layout(xaxis_title="Tiempo", yaxis_title="Movimientos")
        st.plotly_chart(fig, use_container_width=True)
        return

    if base["demanda"].isna().all():
        st.info("No hay demanda (Cantidad de correlativos) para series.")
        return

    if metric == "Demanda (suma)":
        agg = base.groupby(["t", color_col])["demanda"].sum().reset_index(name="valor") if color_col else base.groupby("t")["demanda"].sum().reset_index(name="valor")
        fig = px.line(agg, x="t", y="valor", color=color_col, markers=True)
        fig.update_layout(xaxis_title="Tiempo", yaxis_title="Demanda (suma)")
        st.plotly_chart(fig, use_container_width=True)
        return

    agg = base.groupby(["t", color_col])["demanda"].quantile(0.95).reset_index(name="valor") if color_col else base.groupby("t")["demanda"].quantile(0.95).reset_index(name="valor")
    fig = px.line(agg, x="t", y="valor", color=color_col, markers=True)
    fig.update_layout(xaxis_title="Tiempo", yaxis_title="Demanda (P95)")
    st.plotly_chart(fig, use_container_width=True)


# =========================
# ✅ Nivel de servicio (acumulado / ECDF)
# =========================
def service_level_section(events_df: pd.DataFrame, dinamo_df: Optional[pd.DataFrame]):
    st.subheader("Acumulado: nivel de servicio vs capacidad (demanda = correlativos)")

    if events_df.empty:
        st.info("No hay datos.")
        return

    left, right = st.columns([1, 2])

    with left:
        rutas = sorted([r for r in events_df["ruta"].dropna().unique().tolist()]) if "ruta" in events_df.columns else []
        ruta_choice = st.selectbox("Ruta", ["Todas"] + rutas, index=0, key="svc_ruta")

        dir_sel = st.selectbox("Movimiento", ["Ambos", "Ingreso", "Egreso"], index=0, key="svc_dir")

        unidad = st.selectbox(
            "Unidad de análisis (para evitar dependencia por hora)",
            ["Registro", "Día (máximo)", "Día+Turno (máximo)"],
            index=1,
            key="svc_unit",
        )

        levels = st.multiselect(
            "Porcentajes objetivo (nivel de servicio)",
            [0.80, 0.85, 0.90, 0.95, 0.97, 0.99],
            default=[0.90, 0.95],
            key="svc_levels",
        )

    base = events_df.copy()
    if ruta_choice != "Todas":
        base = base[base["ruta"] == ruta_choice]
    if dir_sel != "Ambos" and "ie_norm" in base.columns:
        base = base[base["ie_norm"] == ("ingreso" if dir_sel == "Ingreso" else "egreso")]

    base["demanda"] = compute_demand(base)
    base = base.dropna(subset=["demanda"])

    if base.empty:
        st.info("No hay demanda válida (Cantidad de correlativos) después del filtro.")
        return

    # unidad -> serie de demanda
    if unidad == "Registro":
        s = base["demanda"].astype(float)
    elif unidad == "Día (máximo)":
        if "fecha" not in base.columns:
            st.info("No existe columna fecha para agrupar por día.")
            return
        s = base.groupby("fecha")["demanda"].max().astype(float)
    else:  # Día+Turno
        if "fecha" not in base.columns or "horario de turno" not in base.columns:
            st.info("No existe fecha/horario de turno para agrupar.")
            return
        s = base.groupby(["fecha", "horario de turno"])["demanda"].max().astype(float)

    s = s.dropna()
    if s.empty:
        st.info("Serie de demanda vacía.")
        return

    # referencia de capacidad actual (Dinamo)
    current_cap = None
    if dinamo_df is not None and not dinamo_df.empty and ruta_choice != "Todas":
        m = dinamo_df[dinamo_df["Ruta"].astype(str).str.strip() == ruta_choice]
        if not m.empty and "Capacidad" in m.columns:
            try:
                current_cap = float(pd.to_numeric(m["Capacidad"], errors="coerce").dropna().iloc[0])
            except:
                current_cap = None

    # tabla de capacidades recomendadas por nivel de servicio (quantiles)
    rec_rows = []
    for p in sorted(levels):
        cap_need = float(np.ceil(s.quantile(p)))
        rec_rows.append({"nivel_servicio_objetivo": f"{int(p*100)}%", "capacidad_minima_recomendada": cap_need})
    rec_df = pd.DataFrame(rec_rows)

    with right:
        st.markdown(
            """
**Interpretación rápida:**
- El *nivel de servicio* para una capacidad **C** es: **% de veces que demanda ≤ C**.  
- Entonces, la capacidad mínima para **95%** es el **P95** (percentil 95) de la demanda.
"""
        )
        if not rec_df.empty:
            st.dataframe(rec_df, use_container_width=True)

        # slider “¿qué pasa si pongo capacidad C?”
        cmin = int(max(0, np.floor(s.min())))
        cmax = int(np.ceil(s.max()))
        cap_test = st.slider("Probar capacidad C", min_value=cmin, max_value=max(cmin + 1, cmax), value=min(max(cmin + 1, cmax), int(np.ceil(s.quantile(0.95)))), step=1)

        achieved = float((s <= cap_test).mean() * 100.0)
        overflow = int((s > cap_test).sum())
        st.metric("Nivel de servicio logrado", f"{achieved:.1f}%")
        st.metric("Casos excedidos", f"{overflow:,}".replace(",", " "))

    # ECDF plot (acumulado)
    df_ecdf = pd.DataFrame({"demanda": s.values})

    try:
        fig = px.ecdf(df_ecdf, x="demanda", title=f"ECDF (acumulado) — {ruta_choice} | {dir_sel} | unidad: {unidad}")
    except Exception:
        # fallback manual si px.ecdf no está
        xs = np.sort(df_ecdf["demanda"].to_numpy(dtype=float))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name="ECDF"))
        fig.update_layout(title=f"ECDF (manual) — {ruta_choice}", xaxis_title="Demanda", yaxis_title="Acumulado")

    # líneas verticales: capacidades recomendadas + capacidad actual + cap_test
    for p in sorted(levels):
        cap_need = float(np.ceil(s.quantile(p)))
        fig.add_vline(x=cap_need, line_dash="dot")
        fig.add_annotation(x=cap_need, y=0.02, text=f"P{int(p*100)}={int(cap_need)}", showarrow=False, yanchor="bottom")

    fig.add_vline(x=cap_test, line_dash="solid")
    fig.add_annotation(x=cap_test, y=0.98, text=f"C probada={cap_test}", showarrow=False, yanchor="top")

    if current_cap is not None:
        fig.add_vline(x=current_cap, line_dash="dash")
        fig.add_annotation(x=current_cap, y=0.90, text=f"Cap. Dinamo={int(current_cap)}", showarrow=False, yanchor="top")

    fig.update_layout(xaxis_title="Capacidad / Demanda (correlativos)", yaxis_title="Acumulado (nivel de servicio)")
    st.plotly_chart(fig, use_container_width=True)


# =========================
# TAB Dinamo: precio vs km/tiempo/jornada + pilotos (igual que antes)
# =========================
def pilot_metrics_from_resumen(events_df: pd.DataFrame, min_share: float, min_total_events: int, bucket_min_events: int) -> pd.DataFrame:
    if events_df.empty or "ruta" not in events_df.columns or "piloto" not in events_df.columns:
        return pd.DataFrame(columns=["ruta_norm"])

    df = events_df.copy()
    df = df.dropna(subset=["ruta", "piloto"])
    df["ruta_norm"] = df["ruta"].map(norm_text)
    df["piloto_norm"] = df["piloto"].map(norm_text)
    df["horario_norm"] = df["horario de turno"].map(norm_text) if "horario de turno" in df.columns else ""

    rp = df.groupby(["ruta_norm", "piloto_norm"]).size().reset_index(name="events")
    rt = df.groupby("ruta_norm").size().reset_index(name="events_total")
    rp = rp.merge(rt, on="ruta_norm", how="left")
    rp["share"] = rp["events"] / rp["events_total"]

    rp["is_significant"] = (rp["share"] >= min_share) & (rp["events"] >= min_total_events)

    agg = rp.groupby("ruta_norm").agg(
        pilots_total=("piloto_norm", "nunique"),
        pilots_significant=("is_significant", "sum"),
        pilot_top_share=("share", "max"),
    ).reset_index()

    if "fecha" in df.columns:
        bucket = df.groupby(["ruta_norm", "fecha", "horario_norm", "piloto_norm"]).size().reset_index(name="bucket_events")
        bucket = bucket[bucket["bucket_events"] >= bucket_min_events].copy()

        bucket = bucket.merge(rp[["ruta_norm", "piloto_norm", "is_significant"]], on=["ruta_norm", "piloto_norm"], how="left")
        bucket["is_significant"] = bucket["is_significant"].fillna(False)

        b2 = bucket.groupby(["ruta_norm", "fecha", "horario_norm"]).agg(
            pilots_in_bucket=("piloto_norm", "nunique"),
            sig_pilots_in_bucket=("is_significant", "sum"),
        ).reset_index()

        b_route = b2.groupby("ruta_norm").agg(
            buckets=("pilots_in_bucket", "count"),
            avg_pilots_per_bucket=("pilots_in_bucket", "mean"),
            pct_buckets_2plus_pilots=("pilots_in_bucket", lambda s: (s >= 2).mean() * 100.0),
            pct_buckets_2plus_sig_pilots=("sig_pilots_in_bucket", lambda s: (s >= 2).mean() * 100.0),
        ).reset_index()

        out = agg.merge(b_route, on="ruta_norm", how="left")
    else:
        out = agg.copy()
        out["buckets"] = np.nan
        out["avg_pilots_per_bucket"] = np.nan
        out["pct_buckets_2plus_pilots"] = np.nan
        out["pct_buckets_2plus_sig_pilots"] = np.nan

    return out


def ols_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return beta, r2

def dinamo_price_tab(events_df: pd.DataFrame, dinamo_df: Optional[pd.DataFrame]):
    st.subheader("Dinamo: Precio Q (Diario) — correlación + SHAP básico (cascada)")

    if dinamo_df is None or dinamo_df.empty:
        st.info("No pude cargar la hoja Dinamo (o viene vacía).")
        return

    # ---------- controles ----------
    left, right = st.columns([1, 2])
    with left:
        join_mode = st.selectbox(
            "Cómo unir Dinamo ↔ Resumen",
            ["Ruta", "Ruta + Jornada", "Ruta + Sede", "Ruta + Sede + Jornada"],
            index=0,
        )
        corr_method = st.selectbox("Correlación (solo numéricas)", ["pearson", "spearman"], index=1)

        min_share = st.slider("Piloto significativo: share mínimo", 0.05, 0.50, 0.15, 0.01)
        min_total_events = st.number_input("Piloto significativo: eventos mínimos en la ruta", min_value=1, value=30, step=1)
        bucket_min_events = st.number_input("Simultáneo proxy: eventos mínimos por piloto en bucket (día+turno)", min_value=1, value=2, step=1)

        st.caption("SHAP básico: baseline = predicción promedio; aportes = (x - promedio) * coef.")

    # ---------- métricas desde resumen (por ruta) ----------
    pm = pilot_metrics_from_resumen(
        events_df,
        min_share=float(min_share),
        min_total_events=int(min_total_events),
        bucket_min_events=int(bucket_min_events),
    )

    if not events_df.empty and "ruta_norm" in events_df.columns:
        d = events_df.copy()
        d["demanda"] = compute_demand(d)  # demanda = Cantidad de correlativos
        dem_stats = d.groupby("ruta_norm").agg(
            demanda_p50=("demanda", lambda s: s.quantile(0.50)),
            demanda_p90=("demanda", lambda s: s.quantile(0.90)),
            demanda_p95=("demanda", lambda s: s.quantile(0.95)),
            demanda_mean=("demanda", "mean"),
            mov=("demanda", "count"),
        ).reset_index()
        pm = pm.merge(dem_stats, on="ruta_norm", how="left")

    # ---------- merge dinamo + resumen ----------
    din = dinamo_df.copy()

    keys = ["ruta_norm"]
    if join_mode in {"Ruta + Jornada", "Ruta + Sede + Jornada"}:
        keys.append("jornada_norm")
    if join_mode in {"Ruta + Sede", "Ruta + Sede + Jornada"}:
        keys.append("sede_norm")

    merged = din.merge(pm, on=keys, how="left")

    if "Precio Q (Diario)" not in merged.columns:
        st.error("En Dinamo no encontré 'Precio Q (Diario)'.")
        return

    st.info(f"Filas Dinamo: {len(din):,} | Filas unidas: {len(merged):,}")

    with st.expander("Tabla unida Dinamo↔Resumen", expanded=False):
        st.dataframe(merged, use_container_width=True)

    # ======================================================
    # 1) Correlación SOLO vs precio (numéricas)
    # ======================================================
    st.markdown("## 1) Correlación numérica vs Precio")

    numeric_vars = [c for c in [
        "Km", "Tiempo (minutos)", "Capacidad", "# Días",
        "pilots_total", "pilots_significant", "pilot_top_share",
        "avg_pilots_per_bucket", "pct_buckets_2plus_sig_pilots",
        "demanda_p95", "demanda_mean",
    ] if c in merged.columns]

    corr_rows = []
    base_corr = merged.dropna(subset=["Precio Q (Diario)"]).copy()
    for v in numeric_vars:
        tmp = base_corr[[v, "Precio Q (Diario)"]].dropna()
        if len(tmp) >= 3:
            corr_val = tmp[v].corr(tmp["Precio Q (Diario)"], method=corr_method)
            corr_rows.append({"variable": v, f"corr_{corr_method}": float(corr_val), "n": int(len(tmp))})

    if corr_rows:
        corr_tbl = pd.DataFrame(corr_rows)
        corr_tbl["abs"] = corr_tbl[f"corr_{corr_method}"].abs()
        corr_tbl = corr_tbl.sort_values("abs", ascending=False).drop(columns=["abs"])
        st.dataframe(corr_tbl, use_container_width=True)
    else:
        st.info("No hubo suficientes datos para correlaciones numéricas.")

    # ======================================================
    # 2) Categóricas vs Precio (Tipo de bus + Jornada)
    # ======================================================
    st.markdown("## 2) Categóricas vs Precio")

    cat_cols = []
    if "Jornada" in merged.columns:
        cat_cols.append("Jornada")
    if "Tipo de bus" in merged.columns:
        cat_cols.append("Tipo de bus")

    if cat_cols:
        cat = st.selectbox("Categoría", cat_cols, index=0, key="price_cat_pick")
        plot_df = merged.dropna(subset=["Precio Q (Diario)"]).copy()
        fig = px.box(plot_df, x=cat, y="Precio Q (Diario)", points="outliers", title=f"Precio por {cat}")
        fig.update_layout(xaxis_title=cat, yaxis_title="Precio Q (Diario)")
        st.plotly_chart(fig, use_container_width=True)

        stats = plot_df.groupby(cat)["Precio Q (Diario)"].agg(
            n="count", mean="mean", median="median", std="std", min="min", max="max"
        ).reset_index().sort_values("mean", ascending=False)
        st.dataframe(stats, use_container_width=True)
    else:
        st.info("No hay Jornada/Tipo de bus en Dinamo para comparar vs precio.")

    st.markdown("## 3) Cascada de influencia en Precio (ΔR² por variable/bloque)")

    model_df = merged.dropna(subset=["Precio Q (Diario)"]).copy()
    if model_df.empty:
        st.info("No hay filas con precio para analizar.")
        return

    # --- selección de variables numéricas
    numeric_vars = [c for c in [
        "Km", "Tiempo (minutos)", "Capacidad", "# Días",
        "pilots_total", "pilots_significant", "pilot_top_share",
        "avg_pilots_per_bucket", "pct_buckets_2plus_sig_pilots",
        "demanda_p95", "demanda_mean",
    ] if c in model_df.columns]

    default_feats = [c for c in ["# Días", "Km", "Tiempo (minutos)", "demanda_p95", "pilots_total"] if c in numeric_vars]
    feats_sel = st.multiselect("Variables numéricas (para ΔR²)", numeric_vars, default=default_feats, key="r2_feats")

    include_jornada = ("Jornada" in model_df.columns)
    include_bus = ("Tipo de bus" in model_df.columns)

    order_mode = st.selectbox(
        "Orden de entrada (afecta ΔR²)",
        ["Por |correlación| (numéricas primero)", "Manual (como están seleccionadas)"],
        index=0,
        key="r2_order"
    )

    # --- preparar datos: numéricas -> mediana para no botar filas (exploratorio)
    for c in feats_sel:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce")
        model_df[c] = model_df[c].fillna(model_df[c].median())

    y = model_df["Precio Q (Diario)"].to_numpy(dtype=float)

    # --- construir dummies (bloques)
    jd = None
    if include_jornada:
        j = model_df["Jornada"].astype(str).fillna("")
        jd = pd.get_dummies(j, prefix="jornada", drop_first=True)

    bd = None
    if include_bus:
        b = model_df["Tipo de bus"].astype(str).fillna("")
        bd = pd.get_dummies(b, prefix="bus", drop_first=True)

    # --- orden: por |correlación| para numéricas (más estable visualmente)
    feats_ordered = feats_sel[:]
    if order_mode.startswith("Por |correlación|"):
        rows = []
        for c in feats_sel:
            tmp = model_df[[c, "Precio Q (Diario)"]].dropna()
            if len(tmp) >= 3:
                # Spearman suele ser más robusto
                corr_val = tmp[c].corr(tmp["Precio Q (Diario)"], method="spearman")
                rows.append((c, abs(float(corr_val)) if pd.notna(corr_val) else 0.0))
            else:
                rows.append((c, 0.0))
        feats_ordered = [c for c, _ in sorted(rows, key=lambda t: t[1], reverse=True)]

    # --- definir “bloques” a entrar
    blocks = []
    for c in feats_ordered:
        blocks.append((c, model_df[c].to_numpy(dtype=float).reshape(-1, 1)))

    if include_jornada and jd is not None and jd.shape[1] > 0:
        blocks.append(("Jornada (dummies)", jd.to_numpy(dtype=float)))

    if include_bus and bd is not None and bd.shape[1] > 0:
        blocks.append(("Tipo de bus (dummies)", bd.to_numpy(dtype=float)))

    # --- función local: calcula R² con intercept + bloques
    def r2_for(block_mats: list[np.ndarray]) -> float:
        X_parts = [np.ones((len(model_df), 1))] + block_mats
        X = np.concatenate(X_parts, axis=1)
        _, r2 = ols_fit(X, y)
        return float(r2)

    # --- calcular ΔR² por bloque (nested models / hierarchical)
    r2_vals = []
    deltas = []
    mats_so_far = []

    prev = 0.0  # intercept-only => R² ~ 0
    for name, mat in blocks:
        mats_so_far.append(mat)
        r2_now = r2_for(mats_so_far)
        delta = r2_now - prev
        r2_vals.append(r2_now)
        deltas.append(delta)
        prev = r2_now

    total_r2 = prev

    # --- tabla resumen
    out = pd.DataFrame({
        "bloque": [b[0] for b in blocks],
        "delta_R2": deltas,
        "R2_acumulado": r2_vals
    }).sort_values("delta_R2", key=lambda s: s.abs(), ascending=False)

    st.write(f"R² total del modelo (con todo): **{total_r2:.3f}**")
    st.dataframe(out, use_container_width=True)

    # --- waterfall: empieza en 0, suma ΔR², termina en R² total
    labels = ["R² inicial"] + [b[0] for b in blocks] + ["R² total"]
    values = [0.0] + deltas + [0.0]
    measures = ["absolute"] + ["relative"] * len(deltas) + ["total"]

    fig = go.Figure(go.Waterfall(x=labels, y=values, measure=measures))
    fig.update_layout(
        title="Cascada (ΔR²): cuánto aporta cada variable/bloque a explicar el Precio",
        xaxis_title="Variable / Bloque",
        yaxis_title="ΔR² (incremento en varianza explicada)"
    )
    st.plotly_chart(fig, use_container_width=True)
# =========================
# Main
# =========================
def main():
    st.title("Rutas — Demanda por movimiento y optimización de capacidad")

    with st.sidebar:
        st.header("Entrada")
        uploaded = st.file_uploader("Subí el Excel", type=["xlsx", "xls"])
        sheet_resumen = st.text_input("Hoja resumen", value="resumen")
        sheet_dinamo = st.text_input("Hoja Dinamo", value="Dinamo")
        dayfirst = st.checkbox("Interpretar fechas como día/mes (dayfirst)", value=False)

        st.markdown("---")
        st.subheader("Rutas en común (para corredores)")
        common_text = st.text_area(
            "Pegá pares de rutas (una línea por par). Separá con TAB, coma o espacio.",
            value=DEFAULT_COMMON_ROUTES_TEXT,
            height=140,
        )

        st.markdown("---")
        st.subheader("Diagnóstico Ingreso↔Egreso (NO filtra)")
        key_mode = st.selectbox(
            "Heurística bus_key",
            ["sede+ruta", "sede+ruta+jornada", "sede+ruta+piloto", "sede+ruta+jornada+piloto"],
            index=1,
        )

    if not uploaded:
        st.info("Subí tu archivo Excel para empezar.")
        return

    parsed = load_excel_sheets(uploaded.getvalue(), sheet_resumen, sheet_dinamo, dayfirst=dayfirst)
    events = parsed.resumen
    dinamo = parsed.dinamo

    edges, edge_warn = parse_edges_from_text(common_text)
    if not edges:
        edge_warn.append("No se detectaron pares de rutas en común; todo quedará como SIN_CORREDOR.")
    events, _ = assign_corridor(events, edges)

    all_warn = list(parsed.warnings) + edge_warn
    if all_warn:
        with st.expander("⚠️ Avisos", expanded=False):
            for w in all_warn:
                st.warning(w)

    pq = pairing_quality(events, key_mode=key_mode)
    st.info(
        f"Diagnóstico (solo reporte):\n"
        f"- Egresos sin ingreso previo (bus_key): {pq['orphan_egreso']}\n"
        f"- Ingresos consecutivos sin egreso (bus_key): {pq['consecutive_ingreso']}\n"
        f"- Ingresos abiertos al final (sin egreso): {pq['open_ingreso_end']}"
    )

    fs = get_filters(events)
    events_f = apply_filters_events(events, fs)

    tab1, tab2, tab3 = st.tabs(["Dashboard general", "Series / cascada / servicio", "Dinamo: Precio"])

    with tab1:
        kpi_row_events(events_f)
        st.markdown("---")
        boxplot_section_events(events_f)

    with tab2:
        kpi_row_events(events_f)
        st.markdown("---")
        waterfall_by_group_events(events_f)
        st.markdown("---")
        timeseries_overview(events_f)
        st.markdown("---")
        # ✅ el acumulado que te faltaba
        service_level_section(events_f, dinamo)

        st.markdown("---")
        st.markdown("### Descargar datos filtrados")
        csv = events_f.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV", csv, file_name="eventos_filtrados.csv", mime="text/csv")

    with tab3:
        kpi_row_events(events_f)
        st.markdown("---")
        dinamo_price_tab(events_f, dinamo)


if __name__ == "__main__":
    main()