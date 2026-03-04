# 02_app_streamlit.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# CONFIG / Defaults
# =========================
TURN_COL = "Horario de turno"
DT_COL   = "Momento de inicio"
HOUR_COL = "Hora de inicio (hora)"
PAX_COL  = "Pasajeros"
DAY_COL  = "Dia inicio (Dia de la semana)"

TURN_BINS   = [0, 8, 16, 24]
TURN_LABELS = ["Amanecer", "Dia", "Tarde"]
TURN_MAP = {
    "amanecer": "Amanecer",
    "amanece": "Amanecer",
    "dia": "Dia", "día": "Dia", "diurno": "Dia",
    "tarde": "Tarde", "vespertino": "Tarde",
}

DAY_ORDER = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
DAY_MAP = {
    "lun": "Lunes", "lunes": "Lunes",
    "mar": "Martes", "martes": "Martes",
    "mie": "Miercoles", "mié": "Miercoles", "miércoles": "Miercoles", "miercoles": "Miercoles",
    "jue": "Jueves", "jueves": "Jueves",
    "vie": "Viernes", "viernes": "Viernes",
    "sab": "Sabado", "sábado": "Sabado", "sabado": "Sabado",
    "dom": "Domingo", "domingo": "Domingo",
}

SCENARIOS_DEFAULT = {
    "timid": {"target_cov": 0.90, "slack": 0.02, "max_step_drop": 1},
    "aggressive": {"target_cov": 0.80, "slack": 0.06, "max_step_drop": 3},
}

GROUP_COL_OPTIONS = ["Ruta", TURN_COL, DAY_COL]
GROUP_COLS_DEFAULT = ["Ruta", TURN_COL]

# ✅ reglas nuevas (corregidas)
MIN_COV_TIMID = 0.95          # timid: DEBE cubrir >=95% en 12m del grupo (o fallback)
MIN_COV_AGGR_3M = 0.95        # aggressive: DEBE cubrir >=95% en 3m del grupo (o fallback)
MONTHS_3M = 3                 # ventana 3 meses


# =========================
# Helpers
# =========================
def _safe_num(s):
    if isinstance(s, pd.Series):
        x = s.astype("string")
    else:
        x = pd.Series(s, dtype="string")
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(x, errors="coerce")


def _winsor_upper_arr(x: np.ndarray, p: float):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return x
    hi = np.quantile(x, p)
    return np.clip(x, None, hi)


def _winsor_upper_series(s: pd.Series, p: float):
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().shape[0] < 5:
        return s
    hi = s.quantile(p)
    return s.clip(upper=hi)


def _coverage_unused(pax: np.ndarray, cap: int):
    if cap is None or pd.isna(cap) or cap <= 0 or pax.size == 0:
        return (np.nan, np.nan, np.nan)
    cov = float((pax <= cap).mean())
    unused = float(np.maximum(cap - pax, 0).mean())
    unused_pct = float(unused / cap)
    return cov, unused, unused_pct


def _nearest_available(x, available_caps):
    if pd.isna(x) or not available_caps:
        return pd.NA
    x = float(x)
    return int(min(available_caps, key=lambda c: abs(c - x)))


def _limit_step_drop(cap_actual, cap_candidate, available_caps, max_step_drop: int):
    if max_step_drop is None or max_step_drop <= 0:
        return cap_candidate
    if pd.isna(cap_actual) or pd.isna(cap_candidate) or not available_caps:
        return cap_candidate

    caps = sorted(set(int(c) for c in available_caps))
    ca = _nearest_available(cap_actual, caps)
    cc = _nearest_available(cap_candidate, caps)
    if pd.isna(ca) or pd.isna(cc) or ca not in caps or cc not in caps:
        return cap_candidate

    i = caps.index(ca)
    min_allowed = caps[max(0, i - int(max_step_drop))]
    return int(max(cc, min_allowed))


def _pick_cap_min_cov(arr, caps_try, min_cov: float, slack: float = 0.0):
    """
    Escoge la MENOR capacidad que cumpla cobertura >= min_cov (estricto).
    Si no existe, intenta cobertura >= (min_cov - slack) (soft).
    Si tampoco, toma la que maximiza cobertura (best-effort).
    Retorna: (cap, cov, unused, tag)
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not caps_try:
        return None, np.nan, np.nan, "no_data"

    rows = []
    for c in caps_try:
        cov, unused, _ = _coverage_unused(arr, int(c))
        if np.isnan(cov):
            continue
        rows.append((int(c), float(cov), float(unused)))

    if not rows:
        return None, np.nan, np.nan, "no_rows"

    # 1) estricto
    strict = [(c, cov, unused) for (c, cov, unused) in rows if cov >= float(min_cov)]
    if strict:
        strict.sort(key=lambda t: (t[0], t[2]))  # menor cap; desempate menor unused
        c, cov, unused = strict[0]
        return c, cov, unused, "strict"

    # 2) soft
    soft_min = max(0.0, float(min_cov) - float(slack))
    soft = [(c, cov, unused) for (c, cov, unused) in rows if cov >= soft_min]
    if soft:
        soft.sort(key=lambda t: (t[0], t[2]))
        c, cov, unused = soft[0]
        return c, cov, unused, "soft"

    # 3) best-effort (max cov; desempate menor cap)
    rows.sort(key=lambda t: (-t[1], t[0], t[2]))
    c, cov, unused = rows[0]
    return c, cov, unused, "best_effort"


def _normalize_turn_text(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    s_norm = s.str.lower().map(TURN_MAP)
    s_keep = s.where(s.isin(TURN_LABELS), pd.NA)
    return s_norm.fillna(s_keep)


def _normalize_day_text(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    s2 = s.str.lower()
    s2 = s2.str.replace("á", "a").str.replace("é", "e").str.replace("í", "i").str.replace("ó", "o").str.replace("ú", "u")
    s_norm = s2.map(DAY_MAP)
    s_keep = s.where(s.isin(DAY_ORDER), pd.NA)
    return s_norm.fillna(s_keep)


def _detect_ordinario(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["Jornada_x", "Jornada_y", "Jornada"] if c in df.columns]
    if not cols:
        return pd.Series(False, index=df.index)
    s = pd.Series("", index=df.index, dtype="string")
    for c in cols:
        s = s.fillna("").astype("string") + " " + df[c].astype("string").fillna("")
    s = s.str.strip().str.lower()
    return s.str.contains("ordinario", na=False)


def _fmt_group_label(row: pd.Series, group_cols) -> str:
    parts = []
    for c in group_cols:
        v = row.get(c, "")
        if pd.isna(v) or v is None:
            v = "NA"
        parts.append(f"{c}={v}")
    return " | ".join(parts)


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d[DT_COL] = pd.to_datetime(d.get(DT_COL, pd.NaT), errors="coerce")

    # Hora
    if HOUR_COL not in d.columns:
        d[HOUR_COL] = d[DT_COL].dt.hour
    d[HOUR_COL] = pd.to_numeric(d[HOUR_COL], errors="coerce").round()
    d.loc[d[HOUR_COL] == 24, HOUR_COL] = 0
    d.loc[(d[HOUR_COL] < 0) | (d[HOUR_COL] > 23), HOUR_COL] = np.nan

    # num cols
    for c in [PAX_COL, "Capacidad", "# Días", "Km", "Tiempo (minutos)", "Precio Q (Diario)"]:
        if c in d.columns:
            d[c] = _safe_num(d[c])

    # Ordinario
    d["_is_ordinario"] = _detect_ordinario(d)

    # Turno
    if TURN_COL not in d.columns:
        d[TURN_COL] = pd.NA
    d["_turn_raw_norm"] = _normalize_turn_text(d[TURN_COL])

    turn_from_hour = pd.cut(
        d[HOUR_COL],
        bins=TURN_BINS,
        labels=TURN_LABELS,
        right=False,
        include_lowest=True
    ).astype("string")

    mode_by_route = (d.loc[~d["_is_ordinario"]]
                       .dropna(subset=["Ruta", "_turn_raw_norm"])
                       .groupby("Ruta")["_turn_raw_norm"]
                       .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else pd.NA))
    d["_route_mode_turn"] = d["Ruta"].map(mode_by_route)

    missing = d["_turn_raw_norm"].isna()
    turn_nonord = d["_turn_raw_norm"].fillna(turn_from_hour).fillna(d["_route_mode_turn"]).fillna("TODOS")
    d[TURN_COL] = np.where(d["_is_ordinario"], "Ordinario", turn_nonord)
    d["_turn_imputed"] = (missing & ~d["_is_ordinario"] & d[TURN_COL].isin(TURN_LABELS))

    # Día semana
    if DAY_COL not in d.columns:
        dow = d[DT_COL].dt.dayofweek
        map_dow = {0: "Lunes", 1: "Martes", 2: "Miercoles", 3: "Jueves", 4: "Viernes", 5: "Sabado", 6: "Domingo"}
        d[DAY_COL] = dow.map(map_dow)
    d[DAY_COL] = _normalize_day_text(d[DAY_COL]).fillna(d[DAY_COL])

    # Limpieza base
    d[PAX_COL] = _safe_num(d[PAX_COL])
    needed = ["Ruta", DT_COL, PAX_COL]
    d = d.dropna(subset=needed).copy()
    d = d[d[PAX_COL] >= 0].copy()

    return d.sort_values(DT_COL)


def build_windows(d: pd.DataFrame, months_12m: int, recent_weeks: int, backtest_weeks: int):
    tmax = d[DT_COL].max()
    d12 = d[d[DT_COL] >= (tmax - pd.DateOffset(months=months_12m))].copy()
    dR  = d12[d12[DT_COL] >= (tmax - pd.Timedelta(days=7 * recent_weeks))].copy()
    dBT = d12[d12[DT_COL] >= (tmax - pd.Timedelta(days=7 * backtest_weeks))].copy()
    return tmax, d12, dR, dBT


def get_groups(d12: pd.DataFrame, group_cols):
    grp_df = (
        d12[group_cols]
          .dropna()
          .drop_duplicates()
          .sort_values(group_cols)
          .reset_index(drop=True)
    )
    return pd.MultiIndex.from_frame(grp_df, names=group_cols)


def get_avail_caps_and_actual(d12: pd.DataFrame, group_cols, avail_caps_fixed):
    avail_caps = list(avail_caps_fixed)
    groups = get_groups(d12, group_cols)

    if "Capacidad" in d12.columns:
        cap_actual = (
            d12.groupby(group_cols)["Capacidad"]
              .mean().round().astype("Int64")
              .reindex(groups)
        )
    else:
        cap_actual = pd.Series(index=groups, dtype="Int64")

    if "# Días" in d12.columns:
        dias_contract = d12.groupby(group_cols)["# Días"].mean().reindex(groups)
    else:
        dias_contract = pd.Series(index=groups, dtype="float64")

    dias_contract = pd.to_numeric(dias_contract, errors="coerce").fillna(30).clip(lower=0)
    return avail_caps, cap_actual, dias_contract


def _mask_by_keys(df: pd.DataFrame, cols, vals: dict):
    m = pd.Series(True, index=df.index)
    for c in cols:
        m = m & (df[c] == vals[c])
    return m


def pooled_eval_samples(d12, dR, dBT, group_cols, key_vals: dict):
    """
    Pools:
    - exact group: BT / recent / 12m (todas las cols del grupo)
    - fallbacks: subsets por prioridad (Ruta > Turno > Día) usando BT y 12m
    - global: 12m_all
    """
    priority = {"Ruta": 0, TURN_COL: 1, DAY_COL: 2}
    ordered = sorted(list(group_cols), key=lambda c: priority.get(c, 99))

    pools = []

    # exact
    m_bt = _mask_by_keys(dBT, group_cols, key_vals)
    m_r  = _mask_by_keys(dR,  group_cols, key_vals)
    m_12 = _mask_by_keys(d12, group_cols, key_vals)

    pools.append(("BT_group",  dBT.loc[m_bt, PAX_COL].dropna().to_numpy()))
    pools.append(("R_group",   dR.loc[m_r,  PAX_COL].dropna().to_numpy()))
    pools.append(("12m_group", d12.loc[m_12, PAX_COL].dropna().to_numpy()))

    # partial subsets
    if len(ordered) >= 2:
        for k in range(len(ordered) - 1, 0, -1):
            cols_k = ordered[:k]
            name = "+".join(cols_k)
            m_bt_k = _mask_by_keys(dBT, cols_k, key_vals)
            m_12_k = _mask_by_keys(d12, cols_k, key_vals)
            pools.append((f"BT_{name}",  dBT.loc[m_bt_k, PAX_COL].dropna().to_numpy()))
            pools.append((f"12m_{name}", d12.loc[m_12_k, PAX_COL].dropna().to_numpy()))

    pools.append(("12m_all", d12[PAX_COL].dropna().to_numpy()))
    return pools


def pick_group_exact_array(pools: list, min_eval_n: int):
    """
    - Siempre usar el grupo exacto si tiene algo (aunque sea poco).
    - Solo si exacto está vacío, caemos a fallback.
    - Damos más peso a recent para que grupos cambien con comportamiento reciente.
    """
    d = {src: arr for src, arr in pools}
    arr_bt = d.get("BT_group", np.array([], dtype=float))
    arr_r  = d.get("R_group",  np.array([], dtype=float))
    arr_12 = d.get("12m_group", np.array([], dtype=float))

    if (arr_12.size > 0) or (arr_r.size > 0) or (arr_bt.size > 0):
        parts = []
        srcs = []
        if arr_12.size > 0:
            parts.append(arr_12)
            srcs.append(f"12m_group({arr_12.size})")
        if arr_r.size > 0:
            parts.append(arr_r); parts.append(arr_r)  # x2
            srcs.append(f"R_groupx2({arr_r.size})")
        elif arr_bt.size > 0:
            parts.append(arr_bt)
            srcs.append(f"BT_group({arr_bt.size})")

        arr = np.concatenate(parts) if parts else np.array([], dtype=float)
        return " + ".join(srcs), arr, int(arr_12.size)

    for src, arr in pools[3:]:
        if arr.size >= min_eval_n:
            return src, arr, 0
    src, arr = max(pools[3:], key=lambda x: x[1].size)
    return src, arr, 0


def _is_exact_src(src: str) -> bool:
    return ("12m_group" in src) or ("R_group" in src) or ("BT_group" in src)


def recommend_caps_with_rules(
    arr_t_rule: np.ndarray,   # timid rule array (ideal: 12m_group exact)
    arr_a_rule: np.ndarray,   # aggressive rule array (ideal: 3m_group exact)
    ca: int,
    avail_caps,
    timid_cfg,
    aggr_cfg,
    src_eval: str,
    clip_p_eval: float,
):
    """
    Reglas pedidas (corregidas):
      - Timid: cobertura >= 95% (MIN_COV_TIMID) en 12m del grupo (o fallback).
      - Aggressive: cobertura >= 95% (MIN_COV_AGGR_3M) en 3m del grupo (o fallback)
        y además ser <= timid (más bajo o igual).

    Step-drop SOLO se aplica en fallbacks (cuando no es exact group),
    para no aplanar recomendaciones cuando el grupo sí tiene datos.
    """
    caps = sorted(set(int(c) for c in avail_caps))
    ca_int = int(ca)

    # Permití que la capacidad actual sea opción aunque no esté en avail_caps
    caps_try = sorted(set([c for c in caps if c <= ca_int] + [ca_int]))
    if not caps_try:
        return ca_int, ca_int, np.nan, np.nan, "No hay caps para evaluar"

    # winsor (robustez contra outliers)
    t_arr = _winsor_upper_arr(np.asarray(arr_t_rule, dtype=float), clip_p_eval)
    a_arr = _winsor_upper_arr(np.asarray(arr_a_rule, dtype=float), clip_p_eval)

    # --- Timid: >=95% (12m)
    t_best, t_cov, _, t_tag = _pick_cap_min_cov(
        t_arr,
        caps_try,
        min_cov=float(MIN_COV_TIMID),
        slack=float(timid_cfg.get("slack", 0.0)),
    )
    if t_best is None:
        t_best = ca_int

    # --- Aggressive: >=95% (3m) y <= timid
    caps_try_a = [c for c in caps_try if c <= int(t_best)]
    if not caps_try_a:
        caps_try_a = [int(t_best)]

    a_best, a_cov, _, a_tag = _pick_cap_min_cov(
        a_arr,
        caps_try_a,
        min_cov=float(MIN_COV_AGGR_3M),
        slack=float(aggr_cfg.get("slack", 0.0)),
    )
    if a_best is None:
        a_best = int(t_best)

    # --- Step drop SOLO si NO es exact
    if not _is_exact_src(src_eval):
        t_best = _limit_step_drop(ca_int, int(t_best), caps_try, int(timid_cfg.get("max_step_drop", 0)))
        a_best = _limit_step_drop(ca_int, int(a_best), caps_try, int(aggr_cfg.get("max_step_drop", 0)))

        # re-enforzar aggressive <= timid
        if int(a_best) > int(t_best):
            a_best = int(t_best)

    cap_t = int(min(ca_int, int(t_best)))
    cap_a = int(min(ca_int, int(a_best)))

    # sanity final: aggressive <= timid
    if cap_a > cap_t:
        cap_a = cap_t

    msg = (
        f"OK | timid(min>={MIN_COV_TIMID:.2f}) cov_t={t_cov:.3f} tag_t={t_tag} "
        f"| aggr3m(min>={MIN_COV_AGGR_3M:.2f}) cov_a={a_cov:.3f} tag_a={a_tag}"
    )
    return cap_t, cap_a, float(t_cov), float(a_cov), msg


def metrics_for_cap_series(dwin, groups, group_cols, cap_series):
    rows = []
    for key in groups:
        key_vals = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        m = _mask_by_keys(dwin, group_cols, key_vals)
        pax = dwin.loc[m, PAX_COL].dropna().to_numpy()

        cap = cap_series.get(key, pd.NA)
        cap_int = int(cap) if pd.notna(cap) else None
        cov, unused, unused_pct = _coverage_unused(pax, cap_int) if cap_int else (np.nan, np.nan, np.nan)

        rows.append((*list(key_vals.values()), cov, unused, unused_pct))

    cols = [*group_cols, "coverage", "unused_seats", "unused_pct"]
    return pd.DataFrame(rows, columns=cols).set_index(group_cols)


def build_pmf_figure(d12: pd.DataFrame, group_cols, key_vals: dict, group_label: str, cap0, capt, capa) -> go.Figure:
    dd = d12.copy()
    dd["pax_int"] = pd.to_numeric(dd[PAX_COL], errors="coerce").round().astype("Int64")
    m = _mask_by_keys(dd, group_cols, key_vals)
    dr = dd.loc[m].dropna(subset=["pax_int"])

        # --- Cobertura (12m vs 3m) PARA ESTE GRUPO (y para cada cap: actual/timid/aggressive)
    pax12 = pd.to_numeric(dr[PAX_COL], errors="coerce").dropna().to_numpy()

    # 3m: usando el máximo timestamp del grupo (mismo filtro actual)
    tmax_g = dr[DT_COL].max() if DT_COL in dr.columns and dr[DT_COL].notna().any() else dd[DT_COL].max()
    cut3 = tmax_g - pd.DateOffset(months=MONTHS_3M)
    dr3 = dr[dr[DT_COL] >= cut3] if DT_COL in dr.columns else dr.iloc[0:0]
    pax3 = pd.to_numeric(dr3[PAX_COL], errors="coerce").dropna().to_numpy()

    def _fmt_pct(v):
        return "N/A" if (v is None or pd.isna(v)) else f"{100*v:.2f}%"

    def cov_for(cap, pax):
        if pd.isna(cap) or cap is None:
            return np.nan
        cov, _, _ = _coverage_unused(pax, int(cap))
        return cov

    cov12_act = cov_for(cap0, pax12)
    cov12_t   = cov_for(capt, pax12)
    cov12_a   = cov_for(capa, pax12)

    cov3_act = cov_for(cap0, pax3)
    cov3_t   = cov_for(capt, pax3)
    cov3_a   = cov_for(capa, pax3)

    # Texto arriba del gráfico
    cov_text = (
        f"<b>Cobertura 12m</b> — Actual: {_fmt_pct(cov12_act)} | Timid: {_fmt_pct(cov12_t)} | Agg: {_fmt_pct(cov12_a)}"
        f"<br><b>Cobertura 3m</b> — Actual: {_fmt_pct(cov3_act)} | Timid: {_fmt_pct(cov3_t)} | Agg: {_fmt_pct(cov3_a)}"
    )
            # Para waterfall conviene mostrar acumulado hasta 100%
    fig = go.Figure()

    fig.add_annotation(
        x=0, y=1.13, xref="paper", yref="paper",
        text=cov_text,
        showarrow=False,
        align="left"
    )

    pmf = dr["pax_int"].value_counts(normalize=True).sort_index()
    x = pmf.index.astype(int).tolist()
    y = pmf.values.tolist()


    cum = np.cumsum(y).tolist()
    x_w = x + ["Total"]
    y_w = y + [0]  # el "total" lo calcula plotly por measure="total"
    measure = (["relative"] * len(x)) + ["total"]
    text = cum + [cum[-1] if cum else 0.0]

    fig.add_trace(go.Waterfall(
        x=x_w,
        y=y_w,
        measure=measure,
        name="Probabilidad",
        text=text,
        hovertemplate="P(X=%{x})=%{y:.2%}<br>Acum=%{text:.2%}<extra></extra>"
    ))

    ymax = 1.05

    def vline(cap, name, dash):
        if pd.isna(cap):
            return
        c = int(cap)
        fig.add_trace(go.Scatter(
            x=[c, c], y=[0, ymax],
            mode="lines", name=name,
            line=dict(dash=dash),
            hovertemplate=f"{name}=%{{x}}<extra></extra>"
        ))

    vline(cap0, "Actual", "dash")
    vline(capt, "Timid", "dot")
    vline(capa, "Aggressive", "dashdot")

    fig.update_layout(
        title=f"PMF — {group_label}",
        xaxis_title="Pasajeros (X)",
        yaxis_title="Probabilidad P(X)",
        yaxis=dict(range=[0, ymax]),
        legend=dict(orientation="h", x=0, y=-0.25, xanchor="left", yanchor="top"),
        margin=dict(t=80, b=90, r=30, l=30)
    )
    return fig


def build_box_month_figure(
    d12: pd.DataFrame,
    group_cols,
    key_vals: dict,
    group_label: str,
    cap0, capt, capa,
    min_n_mes: int,
    view_mode: str = "box",      # "box" | "points"
    max_points: int = 8000       # para no matar el navegador
) -> go.Figure:
    dd = d12.copy()
    dd[DT_COL] = pd.to_datetime(dd[DT_COL], errors="coerce")
    dd["Mes"] = dd[DT_COL].dt.to_period("M").astype(str)

    m = _mask_by_keys(dd, group_cols, key_vals)
    dr = dd.loc[m].copy()

    cnt = dr.groupby("Mes")[PAX_COL].size()
    months_ok = sorted(cnt[cnt >= min_n_mes].index.tolist())
    if months_ok:
        dr = dr[dr["Mes"].isin(months_ok)]
        X = months_ok
    else:
        X = sorted(dd["Mes"].dropna().unique().tolist())

    fig = go.Figure()

    # =========================
    # Vista: BOX o PUNTOS
    # =========================
    if len(dr) > 0:
        if view_mode == "points":
            # datos
            dplot = dr[["Mes", PAX_COL]].copy()
            dplot[PAX_COL] = pd.to_numeric(dplot[PAX_COL], errors="coerce")
            dplot = dplot.dropna(subset=["Mes", PAX_COL])

            if len(dplot) > max_points:
                dplot = dplot.sample(max_points, random_state=7)

            # mapeo Mes -> índice
            X_idx = list(range(len(X)))
            month_to_i = {m: i for i, m in enumerate(X)}

            dplot["x_i"] = dplot["Mes"].map(month_to_i)
            dplot = dplot.dropna(subset=["x_i"])
            dplot["x_i"] = dplot["x_i"].astype(float)

            # jitter
            rng = np.random.default_rng(7)
            dplot["x_j"] = dplot["x_i"] + rng.uniform(-0.35, 0.35, size=len(dplot))

            Trace = go.Scattergl if len(dplot) > 5000 else go.Scatter

            fig.add_trace(Trace(
                x=dplot["x_j"],
                y=dplot[PAX_COL],
                mode="markers",
                marker=dict(size=6, opacity=0.45),
                text=dplot["Mes"],
                hovertemplate="Mes=%{text}<br>Pasajeros=%{y}<extra></extra>",
                name="Pasajeros"
            ))
        else:
            # boxplot clásico
            fig.add_trace(px.box(dr, x="Mes", y=PAX_COL, points=False).data[0])
            fig.data[0].name = "Pasajeros"

    def hline(cap, name, dash):
        if pd.isna(cap) or len(X) == 0:
            return

        x_vals = list(range(len(X))) if view_mode == "points" else X

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=[float(cap)] * len(X),
            mode="lines",
            name=name,
            line=dict(dash=dash),
            hovertemplate=f"{name}≈%{{y:.0f}}<extra></extra>"
        ))

    hline(cap0, "Actual", "dash")
    hline(capt, "Timid", "dot")
    hline(capa, "Aggressive", "dashdot")

    title = (
        f"Distribución mensual (puntos) — {group_label} (meses con n≥{min_n_mes})"
        if view_mode == "points"
        else f"Boxplot mensual — {group_label} (meses con n≥{min_n_mes})"
    )

    xaxis_cfg = (
        dict(
            type="linear",
            tickmode="array",
            tickvals=list(range(len(X))),
            ticktext=X,
            range=[-0.5, len(X) - 0.5]
        )
        if view_mode == "points"
        else dict(
            type="category",
            categoryorder="array",
            categoryarray=X
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Mes",
        yaxis_title="Pasajeros",
        xaxis=xaxis_cfg,
        legend=dict(orientation="h", x=0, y=-0.25, xanchor="left", yanchor="top"),
        margin=dict(t=80, b=90, r=30, l=30)
    )
    return fig

def run_pipeline_prepared(
    d_clean: pd.DataFrame,
    group_cols,
    months_12m: int,
    recent_weeks: int,
    backtest_weeks: int,
    active_months: int,
    min_eval_n: int,
    min_group_n: int,
    clip_p_eval: float,
    clip_p_p90: float,
    avail_caps_fixed,
    scenarios,
):
    d = d_clean.copy()
    tmax, d12, dR, dBT = build_windows(d, months_12m, recent_weeks, backtest_weeks)

    # ventana 3m (para regla del aggressive + tabla)
    d3 = d12[d12[DT_COL] >= (tmax - pd.DateOffset(months=MONTHS_3M))].copy()

    groups_all = get_groups(d12, group_cols)

    active_cut = tmax - pd.DateOffset(months=active_months)
    d_active = d12[d12[DT_COL] >= active_cut]
    n_active = d_active.groupby(group_cols)[PAX_COL].size().reindex(groups_all).fillna(0).astype(int)

    groups = pd.MultiIndex.from_tuples([k for k in groups_all if n_active.loc[k] > 0], names=group_cols)

    # filtrar a grupos activos
    active_df = pd.DataFrame(list(groups), columns=group_cols)
    d12 = d12.merge(active_df, on=group_cols, how="inner")
    dR  = dR.merge(active_df, on=group_cols, how="inner")
    dBT = dBT.merge(active_df, on=group_cols, how="inner")
    d3  = d3.merge(active_df, on=group_cols, how="inner") if len(d3) else d3

    # caps y días contrato por grupo
    avail_caps, cap_actual_all, dias_contract_all = get_avail_caps_and_actual(d12, group_cols, avail_caps_fixed)
    cap_actual = cap_actual_all.reindex(groups)
    dias_contract = dias_contract_all.reindex(groups)

    # outputs
    cap_t = pd.Series(index=groups, dtype="Int64")
    cap_a = pd.Series(index=groups, dtype="Int64")
    src_eval = pd.Series(index=groups, dtype="object")
    n_eval   = pd.Series(index=groups, dtype="int64")
    cov_eval_t = pd.Series(index=groups, dtype="float64")
    cov_eval_a = pd.Series(index=groups, dtype="float64")
    motivo = pd.Series(index=groups, dtype="object")

    # loop por grupo
    for key in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_vals = dict(zip(group_cols, key_tuple))

        ca = cap_actual.get(key, pd.NA)
        if pd.isna(ca):
            cap_t.loc[key] = pd.NA
            cap_a.loc[key] = pd.NA
            motivo.loc[key] = "Sin cap_actual"
            continue

        pools = pooled_eval_samples(d12, dR, dBT, group_cols, key_vals)

        # eval del grupo (para display/fallback general)
        src, arr_eval, n_exact_12m = pick_group_exact_array(pools, min_eval_n=min_eval_n)

        # diccionario para agarrar exactos
        dp = {k: v for k, v in pools}
        arr_12g = dp.get("12m_group", np.array([], dtype=float))
        arr_rg  = dp.get("R_group",  np.array([], dtype=float))
        arr_btg = dp.get("BT_group", np.array([], dtype=float))

        # 3m exacto (si existe)
        if len(d3) > 0:
            m3 = _mask_by_keys(d3, group_cols, key_vals)
            arr_3m = d3.loc[m3, PAX_COL].dropna().to_numpy()
        else:
            arr_3m = np.array([], dtype=float)

        # ✅ regla timid: ideal 12m exacto; si no hay, usa arr_eval
        arr_t_rule = arr_12g if arr_12g.size > 0 else arr_eval

        # ✅ regla aggressive: ideal 3m exacto; si no hay, usa recent; si no hay, usa timid/eval
        if arr_3m.size > 0:
            arr_a_rule = arr_3m
            src_a_rule = "3m_group"
        elif arr_rg.size > 0:
            arr_a_rule = arr_rg
            src_a_rule = "R_group_fallback"
        elif arr_t_rule.size > 0:
            arr_a_rule = arr_t_rule
            src_a_rule = "12m_or_eval_fallback"
        else:
            arr_a_rule = arr_eval
            src_a_rule = "eval_fallback"

        src_eval.loc[key] = src
        n_eval.loc[key] = int(arr_eval.size)

        ct, ca2, tcov, acov, msg = recommend_caps_with_rules(
            arr_t_rule=arr_t_rule,
            arr_a_rule=arr_a_rule,
            ca=int(ca),
            avail_caps=avail_caps,
            timid_cfg=scenarios["timid"],
            aggr_cfg=scenarios["aggressive"],
            src_eval=src,
            clip_p_eval=clip_p_eval,
        )

        cap_t.loc[key] = ct
        cap_a.loc[key] = ca2
        cov_eval_t.loc[key] = tcov
        cov_eval_a.loc[key] = acov

        small_flag = f" | n12m_group={n_exact_12m} (bajo)" if (n_exact_12m < max(3, min_group_n)) else ""
        n3_flag = f" | n3m={int(arr_3m.size)}"
        motivo.loc[key] = f"{msg} | src={src} | src_a_rule={src_a_rule}{n3_flag}{small_flag}"

    # métricas 12m por cap
    m_act = metrics_for_cap_series(d12, groups, group_cols, cap_actual)
    m_t   = metrics_for_cap_series(d12, groups, group_cols, cap_t)
    m_a   = metrics_for_cap_series(d12, groups, group_cols, cap_a)

    # métricas 3m (para tabla): cobertura en 3m para actual / timid / aggressive
    if len(d3) > 0:
        m3_act = metrics_for_cap_series(d3, groups, group_cols, cap_actual)
        m3_t   = metrics_for_cap_series(d3, groups, group_cols, cap_t)
        m3_a   = metrics_for_cap_series(d3, groups, group_cols, cap_a)
    else:
        m3_act = pd.DataFrame(index=groups, columns=["coverage", "unused_seats", "unused_pct"])
        m3_t   = pd.DataFrame(index=groups, columns=["coverage", "unused_seats", "unused_pct"])
        m3_a   = pd.DataFrame(index=groups, columns=["coverage", "unused_seats", "unused_pct"])

    # reco df
    reco = pd.DataFrame(index=groups).reset_index()
    reco["grupo"] = reco.apply(lambda r: _fmt_group_label(r, group_cols), axis=1)

    reco["n_active_recent"] = n_active.reindex(groups).values
    reco["cap_actual"] = cap_actual.reindex(groups).values
    reco["cap_reco_timid"] = cap_t.reindex(groups).values
    reco["cap_reco_aggressive"] = cap_a.reindex(groups).values

    reco["src_eval"] = src_eval.reindex(groups).values
    reco["n_eval"] = n_eval.reindex(groups).values
    reco["cov_eval_timid"] = cov_eval_t.reindex(groups).values
    reco["cov_eval_aggressive"] = cov_eval_a.reindex(groups).values
    reco["motivo"] = motivo.reindex(groups).values

    # P90 diagnóstico (clipeado) por grupo
    d12[PAX_COL] = pd.to_numeric(d12[PAX_COL], errors="coerce").astype("float64")
    dR[PAX_COL]  = pd.to_numeric(dR[PAX_COL], errors="coerce").astype("float64")

    if len(d12) > 0:
        d12["pax_clip_p90"] = d12.groupby(group_cols)[PAX_COL].transform(lambda s: _winsor_upper_series(s, clip_p_p90))
    else:
        d12["pax_clip_p90"] = np.nan

    if len(dR) > 0:
        dR["pax_clip_p90"] = dR.groupby(group_cols)[PAX_COL].transform(lambda s: _winsor_upper_series(s, clip_p_p90))
    else:
        dR["pax_clip_p90"] = np.nan

    p90_12m = d12.groupby(group_cols)["pax_clip_p90"].quantile(0.90).reindex(groups) if len(d12) else pd.Series(index=groups, dtype="float64")
    p90_recent = dR.groupby(group_cols)["pax_clip_p90"].quantile(0.90).reindex(groups) if len(dR) else pd.Series(index=groups, dtype="float64")
    reco["p90_12m_clip"] = p90_12m.values
    reco["p90_recent_clip"] = p90_recent.values

    # merges métricas 12m
    reco = reco.merge(m_act.reset_index().rename(columns={
        "coverage": "coverage_actual",
        "unused_seats": "unused_seats_actual",
        "unused_pct": "unused_pct_actual",
    }), on=group_cols, how="left")

    reco = reco.merge(m_t.reset_index().rename(columns={
        "coverage": "coverage_timid",
        "unused_seats": "unused_seats_timid",
        "unused_pct": "unused_pct_timid",
    }), on=group_cols, how="left")

    reco = reco.merge(m_a.reset_index().rename(columns={
        "coverage": "coverage_aggressive",
        "unused_seats": "unused_seats_aggressive",
        "unused_pct": "unused_pct_aggressive",
    }), on=group_cols, how="left")

    # merges cobertura 3m (actual + timid + aggressive)
    reco = reco.merge(
        m3_act.reset_index()[[*group_cols, "coverage"]].rename(columns={"coverage": "coverage_3m_actual"}),
        on=group_cols, how="left"
    )
    reco = reco.merge(
        m3_t.reset_index()[[*group_cols, "coverage"]].rename(columns={"coverage": "coverage_3m_timid"}),
        on=group_cols, how="left"
    )
    reco = reco.merge(
        m3_a.reset_index()[[*group_cols, "coverage"]].rename(columns={"coverage": "coverage_3m_aggressive"}),
        on=group_cols, how="left"
    )

    reco["dias_contract"] = dias_contract.reindex(groups).astype("float64").values

    # summary base
    n12 = d12.groupby(group_cols)[PAX_COL].size().rename("n_12m") if len(d12) else pd.Series(dtype="int64", name="n_12m")
    summary = reco.merge(n12.reset_index(), on=group_cols, how="left")

    for c in ["coverage_actual", "coverage_timid", "coverage_aggressive",
              "coverage_3m_actual", "coverage_3m_timid", "coverage_3m_aggressive"]:
        if c in summary.columns:
            summary[c + "_pct"] = (pd.to_numeric(summary[c], errors="coerce") * 100)

    return d12, reco, summary, avail_caps


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Rutas — Capacidad vs Demanda", layout="wide")
st.title("Rutas — Capacidad vs Demanda (Timid vs Aggressive) — Grupos dinámicos")

with st.sidebar:
    st.header("Carga de datos")
    up = st.file_uploader("Subí el CSV limpio (ocupa_clean.csv)", type=["csv"])
    default_path = st.text_input("O path local", value="ocupa_clean.csv")

    st.divider()
    st.header("Cómo agrupar")
    group_cols = st.multiselect("Crear grupos por", options=GROUP_COL_OPTIONS, default=GROUP_COLS_DEFAULT)
    if not group_cols:
        group_cols = GROUP_COLS_DEFAULT

    st.divider()
    st.header("Ventanas")
    months_12m = st.slider("Meses para 12m (window)", 3, 24, 12, 1)
    recent_weeks = st.slider("Recent weeks", 1, 26, 8, 1)
    backtest_weeks = st.slider("Backtest weeks", 1, 26, 4, 1)
    active_months = st.slider("Filtrar grupos activos últimos (meses)", 1, 12, 5, 1)

    st.divider()
    st.header("Robustez")
    clip_p_eval = st.slider("Winsor eval (p)", 0.90, 0.999, 0.995, 0.001)
    clip_p_p90 = st.slider("Winsor P90 (p)", 0.90, 0.999, 0.99, 0.001)
    min_eval_n = st.slider("MIN_EVAL_N", 3, 50, 8, 1)
    min_group_n = st.slider("MIN_GROUP_N", 1, 20, 3, 1)

    st.divider()
    st.header("Escenarios")
    timid_cov = st.slider("Timid target_cov", 0.50, 0.99, 0.90, 0.01)
    timid_slack = st.slider("Timid slack", 0.00, 0.20, 0.02, 0.01)
    timid_step = st.slider("Timid max_step_drop", 0, 8, 1, 1)

    aggr_cov = st.slider("Aggressive target_cov", 0.50, 0.99, 0.80, 0.01)
    aggr_slack = st.slider("Aggressive slack", 0.00, 0.30, 0.06, 0.01)
    aggr_step = st.slider("Aggressive max_step_drop", 0, 12, 3, 1)

    st.divider()
    st.header("Caps disponibles")
    caps_text = st.text_input(
        "Lista (comma-separated)",
        value="8,10,12,14,18,20,23,32,34,36,39,42,44,45,46,48"
    )
    avail_caps_fixed = [int(x.strip()) for x in caps_text.split(",") if x.strip().isdigit()]


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_csv_from_upload(bytes_data) -> pd.DataFrame:
    return pd.read_csv(bytes_data)


@st.cache_data(show_spinner=False)
def prepare_cached(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_df(df)


# Load data
if up is not None:
    df_raw = load_csv_from_upload(up)
else:
    try:
        df_raw = load_csv_from_path(default_path)
    except Exception as e:
        st.error(f"No pude leer el archivo. Subilo o revisá el path. Error: {e}")
        st.stop()

d_clean = prepare_cached(df_raw.copy())

scenarios = {
    "timid": {"target_cov": timid_cov, "slack": timid_slack, "max_step_drop": timid_step},
    "aggressive": {"target_cov": aggr_cov, "slack": aggr_slack, "max_step_drop": aggr_step},
}

# =========================
# Global filters
# =========================
st.subheader("Filtros globales (aplican a todo)")
c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.3, 1.3, 1.3])

if d_clean[DT_COL].notna().any():
    min_dt = d_clean[DT_COL].min()
    max_dt = d_clean[DT_COL].max()
else:
    min_dt = pd.Timestamp("2024-01-01")
    max_dt = pd.Timestamp("2026-12-31")

with c1:
    date_from = st.date_input("Desde", value=min_dt.date())
with c2:
    date_to = st.date_input("Hasta", value=max_dt.date())

routes = sorted(d_clean["Ruta"].dropna().unique().tolist()) if "Ruta" in d_clean.columns else []
turns = sorted(d_clean[TURN_COL].dropna().unique().tolist()) if TURN_COL in d_clean.columns else []
days  = sorted(d_clean[DAY_COL].dropna().unique().tolist()) if DAY_COL in d_clean.columns else []
days = [d for d in DAY_ORDER if d in set(days)] + [d for d in days if d not in set(DAY_ORDER)]

with c3:
    rutas_sel = st.multiselect("Rutas", options=routes, default=routes[:])
with c4:
    turnos_sel = st.multiselect("Turnos", options=turns, default=turns[:])
with c5:
    dias_sel = st.multiselect("Días", options=days, default=days[:])

df_f = d_clean.copy()
df_f = df_f[(df_f[DT_COL].dt.date >= date_from) & (df_f[DT_COL].dt.date <= date_to)].copy()
if rutas_sel:
    df_f = df_f[df_f["Ruta"].isin(rutas_sel)].copy()
if TURN_COL in df_f.columns and turnos_sel:
    df_f = df_f[df_f[TURN_COL].isin(turnos_sel)].copy()
if DAY_COL in df_f.columns and dias_sel:
    df_f = df_f[df_f[DAY_COL].isin(dias_sel)].copy()

# =========================
# Run pipeline
# =========================
with st.spinner("Corriendo análisis…"):
    d12, reco, summary, avail_caps = run_pipeline_prepared(
        df_f,
        group_cols=group_cols,
        months_12m=months_12m,
        recent_weeks=recent_weeks,
        backtest_weeks=backtest_weeks,
        active_months=active_months,
        min_eval_n=min_eval_n,
        min_group_n=min_group_n,
        clip_p_eval=clip_p_eval,
        clip_p_p90=clip_p_p90,
        avail_caps_fixed=avail_caps_fixed,
        scenarios=scenarios,
    )

# =========================
# KPIs (incluye desuso actual + timid + aggressive)
# =========================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Grupos activos", f"{len(reco):,}")
k2.metric("Cobertura promedio actual", f"{reco['coverage_actual'].mean():.2%}" if "coverage_actual" in reco.columns else "N/A")
k3.metric("Desuso promedio actual", f"{reco['unused_pct_actual'].mean():.2%}" if "unused_pct_actual" in reco.columns else "N/A")
k4.metric("Desuso promedio (Timid)", f"{reco['unused_pct_timid'].mean():.2%}" if "unused_pct_timid" in reco.columns else "N/A")
k5.metric("Desuso promedio (Aggressive)", f"{reco['unused_pct_aggressive'].mean():.2%}" if "unused_pct_aggressive" in reco.columns else "N/A")

st.caption(f"Caps usados: {avail_caps}")

# =========================
# Tabs
# =========================
tabA, tabB, tabD = st.tabs(["Tablas", "Desuso", "Boxplot & PMF"])

with tabA:
    st.subheader("Tabla resumen (cuantiles + coberturas)")

    if len(d12) == 0:
        st.info("No hay data en la ventana/filters seleccionados.")
    else:
        q_mode = st.selectbox(
            "Cuantiles de Pasajeros",
            ["Cuartiles (25/50/75)", "Quintiles (20/40/60/80)", "Deciles (10..90)"],
            index=2
        )

        if q_mode.startswith("Cuartiles"):
            q_list = [0.25, 0.50, 0.75]
        elif q_mode.startswith("Quintiles"):
            q_list = [0.20, 0.40, 0.60, 0.80]
        else:
            q_list = [i / 10 for i in range(1, 10)]  # 0.1 ... 0.9

        q_cols = [f"p{int(round(q * 100))}" for q in q_list]

        # cuantiles
        q_tbl = d12.groupby(group_cols)[PAX_COL].quantile(q_list).unstack()

        # anti-crash y anti-merge-fail
        if (q_tbl is None) or (q_tbl.shape[0] == 0) or (q_tbl.shape[1] == 0):
            q_tbl = pd.DataFrame(columns=[*group_cols, *q_cols])
        else:
            q_tbl.columns = q_cols
            q_tbl = q_tbl.reset_index()

        n12 = d12.groupby(group_cols)[PAX_COL].size().rename("n_12m")

        summary_dyn = (
            reco.merge(q_tbl, on=group_cols, how="left")
                .merge(n12.reset_index(), on=group_cols, how="left")
        )

        # porcentajes
        for c in ["coverage_actual", "coverage_timid", "coverage_aggressive",
                  "coverage_3m_actual", "coverage_3m_timid", "coverage_3m_aggressive"]:
            if c in summary_dyn.columns:
                summary_dyn[c + "_pct"] = pd.to_numeric(summary_dyn[c], errors="coerce") * 100

        to_r = {
            "n_active_recent": f"Obs activas ({active_months}m)",
            "n_12m": "Obs 12m",
            "cap_actual": "Capacidad actual",
            "cap_reco_timid": "Capacidad reco. timid",
            "cap_reco_aggressive": "Capacidad reco. aggressive",
            "coverage_actual_pct": "Cobertura actual (%)",
            "coverage_timid_pct": "Cobertura timid (%)",
            "coverage_aggressive_pct": "Cobertura aggressive (%)",
            "coverage_3m_actual_pct": "Cobertura 3m actual (%)",
            "coverage_3m_timid_pct": "Cobertura 3m timid (%)",
            "coverage_3m_aggressive_pct": "Cobertura 3m aggressive (%)",
            "p90_12m_clip": "P90 12m (clip)",
            "p90_recent_clip": "P90 recent (clip)",
        }
        q_rename = {c: c.upper() for c in q_cols}

        summary_show = summary_dyn.rename(columns={**to_r, **q_rename})

        for c in ["Cobertura actual (%)", "Cobertura timid (%)", "Cobertura aggressive (%)",
                  "Cobertura 3m actual (%)", "Cobertura 3m timid (%)", "Cobertura 3m aggressive (%)"]:
            if c in summary_show.columns:
                summary_show[c] = summary_show[c].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

        q_cols_show = [c.upper() for c in q_cols]
        cols = [
            *group_cols, "grupo",
            "Obs 12m", f"Obs activas ({active_months}m)",
            "Cobertura 3m actual (%)", "Cobertura 3m timid (%)", "Cobertura 3m aggressive (%)",
            *q_cols_show,
            "P90 12m (clip)", "P90 recent (clip)",
            "Capacidad actual", "Capacidad reco. timid", "Capacidad reco. aggressive",
            "Cobertura actual (%)", "Cobertura timid (%)", "Cobertura aggressive (%)",
            "src_eval", "n_eval", "motivo",
        ]
        cols = [c for c in cols if c in summary_show.columns]

        st.dataframe(summary_show[cols].sort_values(group_cols), use_container_width=True, height=420)

        st.subheader("Tabla detallada (reco)")
        st.dataframe(
            reco.sort_values("unused_pct_actual", ascending=False, na_position="last"),
            use_container_width=True,
            height=420
        )

        st.download_button(
            "Descargar summary CSV",
            data=summary_dyn.to_csv(index=False).encode("utf-8-sig"),
            file_name="summary.csv",
            mime="text/csv"
        )
        st.download_button(
            "Descargar reco CSV",
            data=reco.to_csv(index=False).encode("utf-8-sig"),
            file_name="reco.csv",
            mime="text/csv"
        )

with tabB:
    st.subheader("Desuso 12m (% asientos vacíos) — por grupo")

    if len(reco) == 0:
        st.info("No hay grupos con data en el rango seleccionado.")
    else:
        un = reco[["grupo", "unused_pct_actual", "unused_pct_timid", "unused_pct_aggressive"]].copy()
        un = un.melt(id_vars=["grupo"], var_name="escenario", value_name="unused_pct")
        un["escenario"] = un["escenario"].map({
            "unused_pct_actual": "Actual",
            "unused_pct_timid": "Timid",
            "unused_pct_aggressive": "Aggressive"
        })

        fig = px.bar(
            un.sort_values("unused_pct", ascending=False),
            x="grupo", y="unused_pct", color="escenario", barmode="group",
            title="Desuso 12m — Actual vs Timid vs Aggressive"
        )
        fig.update_layout(xaxis_title="Grupo", yaxis_tickformat=".0%", yaxis_title="% asientos vacíos")
        st.plotly_chart(fig, use_container_width=True)

with tabD:
    st.subheader("Boxplot mensual y PMF por grupo (dropdowns dinámicos)")

    if len(reco) == 0:
        st.info("No hay grupos con data en el rango seleccionado.")
    else:
        keys_df = reco[group_cols].drop_duplicates().copy()

        # orden bonito (si aplica)
        if DAY_COL in keys_df.columns:
            keys_df[DAY_COL] = pd.Categorical(keys_df[DAY_COL], categories=DAY_ORDER, ordered=True)
        if TURN_COL in keys_df.columns:
            turn_order = ["Ordinario", *TURN_LABELS, "TODOS"]
            keys_df[TURN_COL] = pd.Categorical(keys_df[TURN_COL], categories=turn_order, ordered=True)

        keys_df = keys_df.sort_values(group_cols)

        sel = {}
        df_opts = keys_df.copy()
        cols_ui = st.columns(len(group_cols))

        for i, c in enumerate(group_cols):
            with cols_ui[i]:
                opts = df_opts[c].dropna().unique().tolist()
                if not opts:
                    opts = keys_df[c].dropna().unique().tolist()
                chosen = st.selectbox(f"{c}", options=opts, index=0, key=f"box_sel_{c}")
                sel[c] = chosen
            df_opts = df_opts[df_opts[c] == chosen].copy()

        key_vals = sel
        group_label = " | ".join([f"{c}={key_vals[c]}" for c in group_cols])

        m = pd.Series(True, index=reco.index)
        for c in group_cols:
            m = m & (reco[c] == key_vals[c])

        if not m.any():
            st.info("No encontré data para ese grupo con los filtros actuales.")
        else:
            rec_one = reco.loc[m].iloc[0]
            cap0 = rec_one.get("cap_actual", np.nan)
            capt = rec_one.get("cap_reco_timid", np.nan)
            capa = rec_one.get("cap_reco_aggressive", np.nan)

            min_n_mes = st.slider("MIN_N_MES (boxplot)", 1, 60, 10, 1)
            view_points = st.checkbox("Ver distribución por puntos (en vez de boxplot)", value=False)

            max_points = 8000
            if view_points:
                max_points = st.slider("Máx puntos a dibujar (performance)", 1000, 30000, 8000, 500)

            fig_box = build_box_month_figure(
                d12, group_cols, key_vals, group_label,
                cap0, capt, capa,
                min_n_mes=min_n_mes,
                view_mode="points" if view_points else "box",
                max_points=max_points
            )
            st.plotly_chart(fig_box, use_container_width=True)

            fig_pmf = build_pmf_figure(d12, group_cols, key_vals, group_label, cap0, capt, capa)
            st.plotly_chart(fig_pmf, use_container_width=True)