# 02_app_streamlit.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import statsmodels.api as sm


# =========================
# CONFIG / Defaults
# =========================
TURN_COL = "Horario de turno"
DT_COL   = "Momento de inicio"
HOUR_COL = "Hora de inicio (hora)"
PAX_COL  = "Pasajeros"

TURN_BINS   = [0, 8, 16, 24]
TURN_LABELS = ["Amanecer", "Dia", "Tarde"]
TURN_MAP = {
    "amanecer": "Amanecer",
    "amanece": "Amanecer",
    "dia": "Dia", "día": "Dia", "diurno": "Dia",
    "tarde": "Tarde", "vespertino": "Tarde",
}

UBER_COL = "Precio Uber"
UBER_SEATS_DEFAULT = 4

SCENARIOS_DEFAULT = {
    "timid": {
        "target_cov": 0.90,
        "slack": 0.02,
        "max_step_drop": 1,
    },
    "aggressive": {
        "target_cov": 0.80,
        "slack": 0.06,
        "max_step_drop": 3,
    }
}


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


def _fmt_key(route, turno):
    return f"{route} | {turno}"


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


def _pick_cap(arr, caps_try, target, slack):
    if arr.size == 0:
        return None, np.nan, np.nan

    best = None
    best_cov = None
    best_unused = None

    for c in caps_try:
        cov, unused, _ = _coverage_unused(arr, c)
        if np.isnan(cov):
            continue
        if cov >= target:
            if (best is None) or (c < best) or (c == best and unused < best_unused):
                best, best_cov, best_unused = c, cov, unused

    if best is None:
        soft_target = max(0.0, target - slack)
        for c in caps_try:
            cov, unused, _ = _coverage_unused(arr, c)
            if np.isnan(cov):
                continue
            if cov >= soft_target:
                if (best is None) or (c < best) or (c == best and unused < best_unused):
                    best, best_cov, best_unused = c, cov, unused

    if best is None:
        best = None
        best_cov = -1
        best_unused = np.inf
        for c in caps_try:
            cov, unused, _ = _coverage_unused(arr, c)
            if np.isnan(cov):
                continue
            if (cov > best_cov) or (cov == best_cov and (best is None or c < best)) or (cov == best_cov and c == best and unused < best_unused):
                best, best_cov, best_unused = c, cov, unused

    return best, float(best_cov), float(best_unused)


def _normalize_turn_text(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    s_norm = s.str.lower().map(TURN_MAP)
    s_keep = s.where(s.isin(TURN_LABELS), pd.NA)
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


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d[DT_COL] = pd.to_datetime(d.get(DT_COL, pd.NaT), errors="coerce")

    if HOUR_COL not in d.columns:
        d[HOUR_COL] = d[DT_COL].dt.hour
    d[HOUR_COL] = pd.to_numeric(d[HOUR_COL], errors="coerce").round()
    d.loc[d[HOUR_COL] == 24, HOUR_COL] = 0
    d.loc[(d[HOUR_COL] < 0) | (d[HOUR_COL] > 23), HOUR_COL] = np.nan

    for c in [PAX_COL, "Capacidad", "# Días", "Precio Q (Diario)", UBER_COL, "Km", "Tiempo (minutos)"]:
        if c in d.columns:
            d[c] = _safe_num(d[c])

    d["_is_ordinario"] = _detect_ordinario(d)

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

    obs = (d.loc[~d["_is_ordinario"]]
             .dropna(subset=["Ruta", "_turn_raw_norm"])
             .groupby("Ruta")["_turn_raw_norm"]
             .agg(lambda s: set(s.dropna().unique().tolist())))
    mode_by_route = (d.loc[~d["_is_ordinario"]]
                       .dropna(subset=["Ruta", "_turn_raw_norm"])
                       .groupby("Ruta")["_turn_raw_norm"]
                       .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else pd.NA))

    d["_route_mode_turn"] = d["Ruta"].map(mode_by_route)

    missing = d["_turn_raw_norm"].isna()
    fill_hour = turn_from_hour
    turn_nonord = d["_turn_raw_norm"].fillna(fill_hour).fillna(d["_route_mode_turn"]).fillna("TODOS")

    d[TURN_COL] = np.where(d["_is_ordinario"], "Ordinario", turn_nonord)
    d["_turn_imputed"] = (missing & ~d["_is_ordinario"] & d[TURN_COL].isin(TURN_LABELS))

    d[PAX_COL] = _safe_num(d[PAX_COL])
    d = d.dropna(subset=["Ruta", TURN_COL, DT_COL, PAX_COL]).copy()
    d = d[d[PAX_COL] >= 0].copy()

    return d.sort_values(DT_COL)


def build_windows(d: pd.DataFrame, months_12m: int, recent_weeks: int, backtest_weeks: int):
    tmax = d[DT_COL].max()
    d12 = d[d[DT_COL] >= (tmax - pd.DateOffset(months=months_12m))].copy()
    dR  = d12[d12[DT_COL] >= (tmax - pd.Timedelta(days=7*recent_weeks))].copy()
    dBT = d12[d12[DT_COL] >= (tmax - pd.Timedelta(days=7*backtest_weeks))].copy()
    return tmax, d12, dR, dBT


def get_groups(d12: pd.DataFrame):
    grp_df = (d12[["Ruta", TURN_COL]]
              .dropna()
              .drop_duplicates()
              .sort_values(["Ruta", TURN_COL])
              .reset_index(drop=True))
    return pd.MultiIndex.from_frame(grp_df, names=["Ruta", TURN_COL])


def get_avail_caps_and_actual(d12: pd.DataFrame, avail_caps_fixed):
    avail_caps = list(avail_caps_fixed)

    groups = get_groups(d12)

    if "Capacidad" in d12.columns:
        cap_actual = (d12.groupby(["Ruta", TURN_COL])["Capacidad"].mean().round().astype("Int64").reindex(groups))
    else:
        cap_actual = pd.Series(index=groups, dtype="Int64")

    if "# Días" in d12.columns:
        dias_contract = (d12.groupby(["Ruta", TURN_COL])["# Días"].mean().reindex(groups))
    else:
        dias_contract = pd.Series(index=groups, dtype="float64")

    dias_contract = pd.to_numeric(dias_contract, errors="coerce").fillna(30).clip(lower=0)

    return avail_caps, cap_actual, dias_contract


def pooled_eval_samples(d12, dR, dBT, route, turno):
    out = {}
    out["BT_group"] = dBT[(dBT["Ruta"] == route) & (dBT[TURN_COL] == turno)][PAX_COL].dropna().to_numpy()
    out["R_group"]  = dR[(dR["Ruta"] == route) & (dR[TURN_COL] == turno)][PAX_COL].dropna().to_numpy()
    out["12m_group"]= d12[(d12["Ruta"] == route) & (d12[TURN_COL] == turno)][PAX_COL].dropna().to_numpy()
    out["BT_route"] = dBT[dBT["Ruta"] == route][PAX_COL].dropna().to_numpy()
    out["12m_route"]= d12[d12["Ruta"] == route][PAX_COL].dropna().to_numpy()
    out["12m_all"]  = d12[PAX_COL].dropna().to_numpy()
    return out


def pick_eval_pool(pools: dict, min_eval_n: int, min_group_n: int):
    order = ["BT_group","R_group","12m_group","BT_route","12m_route","12m_all"]

    for src in ["BT_group","R_group","12m_group"]:
        if pools[src].size >= min_group_n:
            return src, pools[src]

    for src in order:
        if pools[src].size >= min_eval_n:
            return src, pools[src]

    src = max(order, key=lambda s: pools[s].size)
    return src, pools[src]


def recommend_two_caps_for_group(arr, ca, avail_caps, timid_cfg, aggr_cfg, src, clip_p_eval: float):
    caps = sorted(set(int(c) for c in avail_caps))
    ca_int = int(ca)

    caps_try = [c for c in caps if c <= ca_int]
    if not caps_try:
        return ca_int, ca_int, np.nan, np.nan, "No hay cap menor disponible"

    arr = _winsor_upper_arr(arr, clip_p_eval)

    t_best, t_cov, _ = _pick_cap(arr, caps_try, timid_cfg["target_cov"], timid_cfg["slack"])
    a_best, a_cov, _ = _pick_cap(arr, caps_try, aggr_cfg["target_cov"], aggr_cfg["slack"])

    t_step = int(timid_cfg["max_step_drop"])
    a_step = int(aggr_cfg["max_step_drop"])

    if src in ("12m_all", "12m_route"):
        a_step = min(a_step, 2)
    if src == "12m_all":
        a_step = 1

    cap_t = _limit_step_drop(ca_int, int(t_best), caps, t_step)
    cap_a = _limit_step_drop(ca_int, int(a_best), caps, a_step)

    cap_t = int(min(ca_int, cap_t))
    cap_a = int(min(ca_int, cap_a))

    if cap_t in caps and cap_a in caps:
        it = caps.index(cap_t)
        ia = caps.index(cap_a)
        if ia < it - 2:
            cap_a = caps[max(0, it - 2)]

    cov_ca, _, unusedpct_ca = _coverage_unused(arr, ca_int)
    if (cap_t == ca_int) and (unusedpct_ca >= 0.40):
        idx = caps.index(_nearest_available(ca_int, caps))
        if idx > 0:
            c1 = caps[idx - 1]
            cov1, _, _ = _coverage_unused(arr, c1)
            if not np.isnan(cov1) and cov1 >= (timid_cfg["target_cov"] - 0.03):
                cap_t = c1

    return cap_t, cap_a, t_cov, a_cov, "OK"


def metrics_for_cap_series(d12, groups, cap_series):
    rows = []
    for (r,t) in groups:
        pax = d12[(d12["Ruta"] == r) & (d12[TURN_COL] == t)][PAX_COL].dropna().to_numpy()
        cap = cap_series.get((r,t), pd.NA)
        cap_int = int(cap) if pd.notna(cap) else None
        cov, unused, unused_pct = _coverage_unused(pax, cap_int) if cap_int else (np.nan, np.nan, np.nan)
        exceed = np.nan if np.isnan(cov) else 1 - cov
        rows.append((r,t,cov,exceed,unused,unused_pct))
    return pd.DataFrame(rows, columns=["Ruta",TURN_COL,"coverage","exceed","unused_seats","unused_pct"]).set_index(["Ruta",TURN_COL])


def compute_viajes_dia_mean(d12: pd.DataFrame, groups: pd.MultiIndex) -> pd.Series:
    daily_trips = (
        d12.assign(fecha=d12[DT_COL].dt.date)
           .groupby(["Ruta",TURN_COL,"fecha"]).size()
           .reset_index(name="viajes_dia")
    )
    return daily_trips.groupby(["Ruta",TURN_COL])["viajes_dia"].mean().reindex(groups)


def add_excess_costs(reco: pd.DataFrame, d12: pd.DataFrame, uber_seats: int) -> pd.DataFrame:
    if UBER_COL not in d12.columns:
        for p in ["actual","timid","aggressive"]:
            for c in [f"excess_prob_{p}", f"excess_pax_mean_{p}",
                      f"uber_rides_mean_{p}", f"excess_cost_trip_mean_{p}",
                      f"excess_cost_day_{p}", f"excess_cost_month_{p}"]:
                reco[c] = np.nan
        return reco

    dd = d12.copy()
    dd[UBER_COL] = _safe_num(dd[UBER_COL])

    groups_here = pd.MultiIndex.from_frame(reco[["Ruta",TURN_COL]])
    viajes_dia_mean = compute_viajes_dia_mean(dd, groups_here)

    if "dias_contract" not in reco.columns:
        reco["dias_contract"] = 30.0
    reco["dias_contract"] = pd.to_numeric(reco["dias_contract"], errors="coerce").fillna(30).clip(lower=0)

    base = reco[["Ruta",TURN_COL,"dias_contract",
                 "cap_actual","cap_reco_timid","cap_reco_aggressive"]].copy()

    def _calc(prefix: str, cap_col: str):
        tmp = dd.merge(
            base[["Ruta",TURN_COL,cap_col]].rename(columns={cap_col:"cap"}),
            on=["Ruta",TURN_COL], how="left"
        )
        tmp["cap"] = pd.to_numeric(tmp["cap"], errors="coerce")
        tmp["pax"] = pd.to_numeric(tmp[PAX_COL], errors="coerce")

        tmp["excess_pax"] = (tmp["pax"] - tmp["cap"]).clip(lower=0)
        tmp["uber_rides"] = np.ceil(tmp["excess_pax"] / max(1, int(uber_seats))).fillna(0)
        tmp["excess_cost"] = (tmp["uber_rides"] * tmp[UBER_COL]).fillna(0)
        tmp["has_excess"] = tmp["excess_pax"] > 0

        g = tmp.groupby(["Ruta",TURN_COL])
        out = pd.DataFrame({
            f"excess_prob_{prefix}": g["has_excess"].mean(),
            f"excess_pax_mean_{prefix}": g["excess_pax"].mean(),
            f"uber_rides_mean_{prefix}": g["uber_rides"].mean(),
            f"excess_cost_trip_mean_{prefix}": g["excess_cost"].mean(),
        }).reset_index()

        out = out.merge(viajes_dia_mean.rename("viajes_dia").reset_index(), on=["Ruta",TURN_COL], how="left") \
                 .merge(base[["Ruta",TURN_COL,"dias_contract"]], on=["Ruta",TURN_COL], how="left")

        out[f"excess_cost_day_{prefix}"] = out[f"excess_cost_trip_mean_{prefix}"] * out["viajes_dia"].fillna(0)
        out[f"excess_cost_month_{prefix}"] = out[f"excess_cost_day_{prefix}"] * out["dias_contract"].fillna(30)

        return out.drop(columns=["viajes_dia","dias_contract"])

    out_a = _calc("actual", "cap_actual")
    out_t = _calc("timid", "cap_reco_timid")
    out_g = _calc("aggressive", "cap_reco_aggressive")

    reco2 = reco.merge(out_a, on=["Ruta",TURN_COL], how="left") \
                .merge(out_t, on=["Ruta",TURN_COL], how="left") \
                .merge(out_g, on=["Ruta",TURN_COL], how="left")
    return reco2


def build_pmf_figure(d12: pd.DataFrame, ruta: str, turno: str, cap0, capt, capa) -> go.Figure:
    dd = d12.copy()
    dd["pax_int"] = pd.to_numeric(dd[PAX_COL], errors="coerce").round().astype("Int64")
    dr = dd[(dd["Ruta"]==ruta) & (dd[TURN_COL]==turno)].dropna(subset=["pax_int"])
    pmf = dr["pax_int"].value_counts(normalize=True).sort_index()
    x = pmf.index.astype(int).tolist()
    y = pmf.values.tolist()
    ymax = float(pmf.max()) if len(pmf) else 0.1
    ymax = max(ymax * 1.20, 0.05)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=y, name="Probabilidad",
        hovertemplate="P(X=%{x})=%{y:.2%}<extra></extra>"
    ))

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
        title=f"PMF — {ruta} | {turno}",
        xaxis_title="Pasajeros (X)",
        yaxis_title="Probabilidad P(X)",
        yaxis=dict(range=[0, ymax]),
        legend=dict(orientation="h", x=0, y=-0.25, xanchor="left", yanchor="top"),
        margin=dict(t=80, b=90, r=30, l=30)
    )
    return fig


def build_box_month_figure(d12: pd.DataFrame, ruta: str, turno: str, cap0, capt, capa, min_n_mes: int) -> go.Figure:
    dd = d12.copy()
    dd[DT_COL] = pd.to_datetime(dd[DT_COL], errors="coerce")
    dd["Mes"] = dd[DT_COL].dt.to_period("M").astype(str)

    dr = dd[(dd["Ruta"]==ruta) & (dd[TURN_COL]==turno)].copy()
    cnt = dr.groupby("Mes")[PAX_COL].size()
    months_ok = sorted(cnt[cnt >= min_n_mes].index.tolist())
    if months_ok:
        dr = dr[dr["Mes"].isin(months_ok)]
        X = months_ok
    else:
        X = sorted(dd["Mes"].dropna().unique().tolist())

    fig = go.Figure()
    fig.add_trace(px.box(dr, x="Mes", y=PAX_COL, points=False).data[0])
    fig.data[0].name = "Pasajeros"

    def hline(cap, name, dash):
        if pd.isna(cap):
            return
        fig.add_trace(go.Scatter(
            x=X, y=[float(cap)]*len(X),
            mode="lines", name=name,
            line=dict(dash=dash),
            hovertemplate=f"{name}≈%{{y:.0f}}<extra></extra>"
        ))

    hline(cap0, "Actual", "dash")
    hline(capt, "Timid", "dot")
    hline(capa, "Aggressive", "dashdot")

    fig.update_layout(
        title=f"Boxplot mensual — {ruta} | {turno} (meses con n≥{min_n_mes})",
        xaxis_title="Mes",
        yaxis_title="Pasajeros",
        xaxis=dict(type="category", categoryorder="array", categoryarray=X),
        legend=dict(orientation="h", x=0, y=-0.25, xanchor="left", yanchor="top"),
        margin=dict(t=80, b=90, r=30, l=30)
    )
    return fig


def run_pipeline(df_raw: pd.DataFrame,
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
                 uber_seats: int):
    d = prepare_df(df_raw)
    tmax, d12, dR, dBT = build_windows(d, months_12m, recent_weeks, backtest_weeks)

    groups_all = get_groups(d12)

    active_cut = tmax - pd.DateOffset(months=active_months)
    d_active = d12[d12[DT_COL] >= active_cut]
    n_active = d_active.groupby(["Ruta", TURN_COL])[PAX_COL].size().reindex(groups_all).fillna(0).astype(int)

    groups = pd.MultiIndex.from_tuples([k for k in groups_all if n_active.loc[k] > 0], names=["Ruta",TURN_COL])

    active_df = pd.DataFrame(list(groups), columns=["Ruta",TURN_COL])
    d12 = d12.merge(active_df, on=["Ruta",TURN_COL], how="inner")
    dR  = dR.merge(active_df, on=["Ruta",TURN_COL], how="inner")
    dBT = dBT.merge(active_df, on=["Ruta",TURN_COL], how="inner")

    avail_caps, cap_actual_all, dias_contract_all = get_avail_caps_and_actual(d12, avail_caps_fixed)
    cap_actual = cap_actual_all.reindex(groups)
    dias_contract = dias_contract_all.reindex(groups)

    cap_t = pd.Series(index=groups, dtype="Int64")
    cap_a = pd.Series(index=groups, dtype="Int64")
    src_eval = pd.Series(index=groups, dtype="object")
    n_eval   = pd.Series(index=groups, dtype="int64")
    cov_eval_t = pd.Series(index=groups, dtype="float64")
    cov_eval_a = pd.Series(index=groups, dtype="float64")
    motivo = pd.Series(index=groups, dtype="object")

    for (r,t) in groups:
        ca = cap_actual.get((r,t), pd.NA)
        if pd.isna(ca):
            cap_t.loc[(r,t)] = pd.NA
            cap_a.loc[(r,t)] = pd.NA
            motivo.loc[(r,t)] = "Sin cap_actual"
            continue

        pools = pooled_eval_samples(d12, dR, dBT, r, t)
        src, arr = pick_eval_pool(pools, min_eval_n, min_group_n)
        src_eval.loc[(r,t)] = src
        n_eval.loc[(r,t)] = int(arr.size)

        ct, ca2, tcov, acov, msg = recommend_two_caps_for_group(
            arr, int(ca), avail_caps, scenarios["timid"], scenarios["aggressive"], src, clip_p_eval
        )
        cap_t.loc[(r,t)] = ct
        cap_a.loc[(r,t)] = ca2
        cov_eval_t.loc[(r,t)] = tcov
        cov_eval_a.loc[(r,t)] = acov
        motivo.loc[(r,t)] = f"{msg} | src={src} n={arr.size}"

    m_act = metrics_for_cap_series(d12, groups, cap_actual)
    m_t   = metrics_for_cap_series(d12, groups, cap_t)
    m_a   = metrics_for_cap_series(d12, groups, cap_a)

    reco = pd.DataFrame(index=groups).reset_index()
    reco["grupo"] = reco.apply(lambda x: _fmt_key(x["Ruta"], x[TURN_COL]), axis=1)
    reco["n_active_5m"] = n_active.reindex(groups).values

    reco["cap_actual"] = cap_actual.reindex(groups).values
    reco["cap_reco_timid"] = cap_t.reindex(groups).values
    reco["cap_reco_aggressive"] = cap_a.reindex(groups).values

    reco["src_eval"] = src_eval.reindex(groups).values
    reco["n_eval"] = n_eval.reindex(groups).values
    reco["cov_eval_timid"] = cov_eval_t.reindex(groups).values
    reco["cov_eval_aggressive"] = cov_eval_a.reindex(groups).values
    reco["motivo"] = motivo.reindex(groups).values

    # p90 diagnóstico (clipeado)
    d12[PAX_COL] = pd.to_numeric(d12[PAX_COL], errors="coerce").astype("float64")
    dR[PAX_COL]  = pd.to_numeric(dR[PAX_COL], errors="coerce").astype("float64")

    d12["pax_clip_p90"] = d12.groupby(["Ruta",TURN_COL])[PAX_COL].transform(lambda s: _winsor_upper_series(s, clip_p_p90))
    dR["pax_clip_p90"]  = dR.groupby(["Ruta",TURN_COL])[PAX_COL].transform(lambda s: _winsor_upper_series(s, clip_p_p90))

    p90_12m = d12.groupby(["Ruta",TURN_COL])["pax_clip_p90"].quantile(0.90).reindex(groups)
    p90_recent = dR.groupby(["Ruta",TURN_COL])["pax_clip_p90"].quantile(0.90).reindex(groups)
    reco["p90_12m_clip"] = p90_12m.values
    reco["p90_recent_clip"] = p90_recent.values

    reco = reco.merge(m_act.reset_index().rename(columns={
        "coverage":"coverage_actual","exceed":"exceed_actual","unused_pct":"unused_pct_actual","unused_seats":"unused_seats_actual"
    }), on=["Ruta",TURN_COL], how="left")
    reco = reco.merge(m_t.reset_index().rename(columns={
        "coverage":"coverage_timid","exceed":"exceed_timid","unused_pct":"unused_pct_timid","unused_seats":"unused_seats_timid"
    }), on=["Ruta",TURN_COL], how="left")
    reco = reco.merge(m_a.reset_index().rename(columns={
        "coverage":"coverage_aggressive","exceed":"exceed_aggressive","unused_pct":"unused_pct_aggressive","unused_seats":"unused_seats_aggressive"
    }), on=["Ruta",TURN_COL], how="left")

    # =========================
    # AHORRO (RLM Huber) si existe Precio Q (Diario)
    # =========================
    if "Precio Q (Diario)" in d12.columns:
        daily_trips = (
            d12.assign(fecha=d12[DT_COL].dt.date)
               .groupby(["Ruta",TURN_COL,"fecha"]).size()
               .reset_index(name="viajes_dia")
        )
        viajes_dia_mean = daily_trips.groupby(["Ruta",TURN_COL])["viajes_dia"].mean().reindex(groups)

        if "Km" in d12.columns:
            daily_km = (
                d12.assign(fecha=d12[DT_COL].dt.date)
                   .groupby(["Ruta",TURN_COL,"fecha"])["Km"].sum()
                   .reset_index(name="km_dia")
            )
            km_dia_mean = daily_km.groupby(["Ruta",TURN_COL])["km_dia"].mean().reindex(groups)
        else:
            km_dia_mean = pd.Series(index=groups, dtype="float64")

        precio_por_dia = d12.groupby(["Ruta",TURN_COL])["Precio Q (Diario)"].mean().reindex(groups)

        model_df = pd.DataFrame({
            "Ruta": [k[0] for k in groups],
            TURN_COL: [k[1] for k in groups],
            "precio_por_dia": precio_por_dia.values,
            "cap_actual": cap_actual.reindex(groups).astype("float64").values,
            "cap_timid": cap_t.reindex(groups).astype("float64").values,
            "cap_aggressive": cap_a.reindex(groups).astype("float64").values,
            "viajes_dia": viajes_dia_mean.values,
            "km_dia": km_dia_mean.values,
            "dias_contract": dias_contract.reindex(groups).astype("float64").values,
        }).dropna(subset=["precio_por_dia","cap_actual"]).copy()

        model_df["log_precio"] = np.log1p(model_df["precio_por_dia"])
        model_df["log_km_dia"] = np.log1p(pd.to_numeric(model_df["km_dia"], errors="coerce").fillna(0).clip(lower=0))
        model_df["log_viajes"] = np.log1p(pd.to_numeric(model_df["viajes_dia"], errors="coerce").fillna(0).clip(lower=0))
        model_df["log_dias"]   = np.log1p(pd.to_numeric(model_df["dias_contract"], errors="coerce").fillna(30).clip(lower=0))
        model_df["cap"]        = pd.to_numeric(model_df["cap_actual"], errors="coerce")

        for c in ["log_km_dia","log_viajes","log_dias"]:
            model_df[c] = model_df[c].fillna(model_df[c].median())

        for c in ["log_precio","log_km_dia","log_viajes","log_dias","cap","cap_timid","cap_aggressive"]:
            model_df[c] = pd.to_numeric(model_df[c], errors="coerce").astype("float64")

        model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["log_precio","log_km_dia","log_viajes","log_dias","cap"]
        ).copy()

        X = sm.add_constant(
            model_df[["log_km_dia","log_viajes","log_dias","cap"]].astype("float64"),
            has_constant="add"
        )
        y = model_df["log_precio"].astype("float64")

        rlm = sm.RLM(
            y.to_numpy(dtype="float64"),
            X.to_numpy(dtype="float64"),
            M=sm.robust.norms.HuberT()
        ).fit()

        X_cur = sm.add_constant(
            model_df[["log_km_dia","log_viajes","log_dias","cap"]].astype("float64"),
            has_constant="add"
        ).to_numpy(dtype="float64")
        model_df["precio_pred_actual"] = np.expm1(rlm.predict(X_cur))

        X_t = sm.add_constant(pd.DataFrame({
            "log_km_dia": model_df["log_km_dia"],
            "log_viajes": model_df["log_viajes"],
            "log_dias": model_df["log_dias"],
            "cap": model_df["cap_timid"].fillna(model_df["cap"]),
        }).astype("float64"), has_constant="add").to_numpy(dtype="float64")
        model_df["precio_pred_timid"] = np.expm1(rlm.predict(X_t))

        X_a2 = sm.add_constant(pd.DataFrame({
            "log_km_dia": model_df["log_km_dia"],
            "log_viajes": model_df["log_viajes"],
            "log_dias": model_df["log_dias"],
            "cap": model_df["cap_aggressive"].fillna(model_df["cap"]),
        }).astype("float64"), has_constant="add").to_numpy(dtype="float64")
        model_df["precio_pred_aggressive"] = np.expm1(rlm.predict(X_a2))

        cut_t = model_df["cap_timid"] < model_df["cap"]
        cut_a2 = model_df["cap_aggressive"] < model_df["cap"]

        model_df["ahorro_dia_timid"] = 0.0
        model_df.loc[cut_t, "ahorro_dia_timid"] = (model_df.loc[cut_t, "precio_pred_actual"] - model_df.loc[cut_t, "precio_pred_timid"]).clip(lower=0)

        model_df["ahorro_dia_aggressive"] = 0.0
        model_df.loc[cut_a2, "ahorro_dia_aggressive"] = (model_df.loc[cut_a2, "precio_pred_actual"] - model_df.loc[cut_a2, "precio_pred_aggressive"]).clip(lower=0)

        model_df["ahorro_mes_timid"] = model_df["ahorro_dia_timid"] * model_df["dias_contract"]
        model_df["ahorro_mes_aggressive"] = model_df["ahorro_dia_aggressive"] * model_df["dias_contract"]

        reco = reco.merge(
            model_df[["Ruta",TURN_COL,"precio_por_dia","dias_contract",
                      "precio_pred_actual","precio_pred_timid","precio_pred_aggressive",
                      "ahorro_dia_timid","ahorro_mes_timid",
                      "ahorro_dia_aggressive","ahorro_mes_aggressive"]],
            on=["Ruta",TURN_COL], how="left"
        )
    else:
        for c in ["precio_por_dia","precio_pred_actual","precio_pred_timid","precio_pred_aggressive",
                  "ahorro_dia_timid","ahorro_mes_timid","ahorro_dia_aggressive","ahorro_mes_aggressive","dias_contract"]:
            if c not in reco.columns:
                reco[c] = np.nan

    reco["ahorro_mes_timid_pos"] = pd.to_numeric(reco.get("ahorro_mes_timid", 0), errors="coerce").fillna(0).clip(lower=0)
    reco["ahorro_mes_aggressive_pos"] = pd.to_numeric(reco.get("ahorro_mes_aggressive", 0), errors="coerce").fillna(0).clip(lower=0)

    reco = add_excess_costs(reco, d12, uber_seats)

    # tabla resumen cuartiles + coverage %
    q = (d12.groupby(["Ruta",TURN_COL])[PAX_COL]
            .quantile([0.25, 0.50, 0.75])
            .unstack())
    q.columns = ["q25", "q50", "q75"]
    n12 = d12.groupby(["Ruta",TURN_COL])[PAX_COL].size().rename("n_12m")

    summary = (reco.merge(q.reset_index(), on=["Ruta",TURN_COL], how="left")
                  .merge(n12.reset_index(), on=["Ruta",TURN_COL], how="left"))

    for c in ["coverage_actual","coverage_timid","coverage_aggressive"]:
        summary[c + "_pct"] = (pd.to_numeric(summary[c], errors="coerce") * 100)

    return d12, reco, summary, avail_caps


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Rutas — Capacidad vs Demanda", layout="wide")

st.title("Rutas — Capacidad vs Demanda (Timid vs Aggressive)")

with st.sidebar:
    st.header("Carga de datos")
    up = st.file_uploader("Subí el CSV limpio (ocupa_clean.csv)", type=["csv"])
    default_path = st.text_input("O path local", value="ocupa_clean.csv")

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
    st.header("Uber excedentes")
    uber_seats = st.number_input("Asientos por Uber (para dividir excedente)", min_value=1, max_value=10, value=UBER_SEATS_DEFAULT, step=1)

    st.divider()
    st.header("Caps disponibles")
    caps_text = st.text_input("Lista (comma-separated)", value="8,10,12,14,18,20,23,32,34,36,39,42,44,45,46,48")
    avail_caps_fixed = [int(x.strip()) for x in caps_text.split(",") if x.strip().isdigit()]


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_csv_from_upload(bytes_data) -> pd.DataFrame:
    return pd.read_csv(bytes_data)


# Load data
if up is not None:
    df_raw = load_csv_from_upload(up)
else:
    try:
        df_raw = load_csv_from_path(default_path)
    except Exception as e:
        st.error(f"No pude leer el archivo. Subilo o revisá el path. Error: {e}")
        st.stop()

# Normalize minimal
df_raw = df_raw.copy()
if DT_COL in df_raw.columns:
    df_raw[DT_COL] = pd.to_datetime(df_raw[DT_COL], errors="coerce")

scenarios = {
    "timid": {"target_cov": timid_cov, "slack": timid_slack, "max_step_drop": timid_step},
    "aggressive": {"target_cov": aggr_cov, "slack": aggr_slack, "max_step_drop": aggr_step},
}

# Global filters (en el CSV limpio ya no vienen sede/destino/piloto)
st.subheader("Filtros")
c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.3, 1.3])

if DT_COL in df_raw.columns and df_raw[DT_COL].notna().any():
    min_dt = df_raw[DT_COL].min()
    max_dt = df_raw[DT_COL].max()
else:
    min_dt = pd.Timestamp("2024-01-01")
    max_dt = pd.Timestamp("2026-12-31")

with c1:
    date_from = st.date_input("Desde", value=min_dt.date())
with c2:
    date_to = st.date_input("Hasta", value=max_dt.date())

routes = sorted([x for x in df_raw.get("Ruta", pd.Series([], dtype=str)).dropna().unique().tolist()])
turns = sorted([x for x in df_raw.get(TURN_COL, pd.Series([], dtype=str)).dropna().unique().tolist()])

with c3:
    rutas_sel = st.multiselect("Rutas", options=routes, default=routes[:])
with c4:
    turnos_sel = st.multiselect("Turnos", options=turns, default=turns[:])

df_f = df_raw.copy()
if DT_COL in df_f.columns:
    df_f = df_f[(df_f[DT_COL].dt.date >= date_from) & (df_f[DT_COL].dt.date <= date_to)].copy()
if "Ruta" in df_f.columns and rutas_sel:
    df_f = df_f[df_f["Ruta"].isin(rutas_sel)].copy()
if TURN_COL in df_f.columns and turnos_sel:
    df_f = df_f[df_f[TURN_COL].isin(turnos_sel)].copy()

# Run pipeline
with st.spinner("Corriendo análisis…"):
    d12, reco, summary, avail_caps = run_pipeline(
        df_f,
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
        uber_seats=int(uber_seats),
    )

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Grupos activos", f"{len(reco):,}")
k2.metric("Ahorro mensual timid (+)", f"Q{reco['ahorro_mes_timid_pos'].sum():,.2f}")
k3.metric("Ahorro mensual aggressive (+)", f"Q{reco['ahorro_mes_aggressive_pos'].sum():,.2f}")
k4.metric("Desuso promedio actual", f"{reco['unused_pct_actual'].mean():.2%}" if "unused_pct_actual" in reco.columns else "N/A")

st.caption(f"Caps usados: {avail_caps}")

# Tabs
tab4, tab1, tab2, tab3 = st.tabs(["Tablas", "Ahorro", "Desuso", "Boxplot & PMF"])

with tab1:
    st.subheader("Ahorro mensual estimado (solo positivos)")
    sav = reco[["grupo","ahorro_mes_timid_pos","ahorro_mes_aggressive_pos"]].copy()
    sav = sav.melt(id_vars=["grupo"], var_name="escenario", value_name="ahorro_mes")
    sav["escenario"] = sav["escenario"].map({
        "ahorro_mes_timid_pos":"Timid",
        "ahorro_mes_aggressive_pos":"Aggressive"
    })
    fig = px.bar(
        sav.sort_values("ahorro_mes", ascending=False),
        x="grupo", y="ahorro_mes", color="escenario", barmode="group",
        title="Ahorro mensual estimado — Timid vs Aggressive"
    )
    fig.update_layout(xaxis_title="Ruta | Turno", yaxis_title="Ahorro mensual (Q)")
    st.plotly_chart(fig, use_container_width=True)

    if "excess_cost_month_timid" in reco.columns:
        st.subheader("Costo excedentes (Uber) — mensual")
        ex = reco[["grupo","excess_cost_month_timid","excess_cost_month_aggressive"]].copy()
        ex = ex.melt(id_vars=["grupo"], var_name="escenario", value_name="costo_mes")
        ex["escenario"] = ex["escenario"].map({
            "excess_cost_month_timid":"Timid",
            "excess_cost_month_aggressive":"Aggressive"
        })
        fig2 = px.bar(
            ex.sort_values("costo_mes", ascending=False),
            x="grupo", y="costo_mes", color="escenario", barmode="group",
            title="Costo mensual excedentes (Uber) — Timid vs Aggressive"
        )
        fig2.update_layout(xaxis_title="Ruta | Turno", yaxis_title="Costo mensual (Q)")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Desuso 12m (% asientos vacíos)")
    un = reco[["grupo","unused_pct_actual","unused_pct_timid","unused_pct_aggressive"]].copy()
    un = un.melt(id_vars=["grupo"], var_name="escenario", value_name="unused_pct")
    un["escenario"] = un["escenario"].map({
        "unused_pct_actual":"Actual",
        "unused_pct_timid":"Timid",
        "unused_pct_aggressive":"Aggressive"
    })
    fig = px.bar(un, x="grupo", y="unused_pct", color="escenario", barmode="group",
                 title="Desuso 12m — Actual vs Timid vs Aggressive")
    fig.update_layout(xaxis_title="Ruta | Turno", yaxis_tickformat=".0%", yaxis_title="% asientos vacíos")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Boxplot mensual y PMF por grupo (seleccionable)")

    groups_list = reco[["Ruta",TURN_COL,"grupo"]].drop_duplicates().sort_values(["Ruta",TURN_COL])
    if len(groups_list) == 0:
        st.info("No hay grupos con data en el rango seleccionado.")
    else:
        gsel = st.selectbox("Elegí grupo", options=groups_list["grupo"].tolist(), index=0)
        row = groups_list[groups_list["grupo"] == gsel].iloc[0]
        ruta = row["Ruta"]
        turno = row[TURN_COL]

        rec_row = reco[(reco["Ruta"]==ruta) & (reco[TURN_COL]==turno)].iloc[0]
        cap0 = rec_row.get("cap_actual", np.nan)
        capt = rec_row.get("cap_reco_timid", np.nan)
        capa = rec_row.get("cap_reco_aggressive", np.nan)

        min_n_mes = st.slider("MIN_N_MES (boxplot)", 1, 60, 10, 1)

        fig_box = build_box_month_figure(d12, ruta, turno, cap0, capt, capa, min_n_mes=min_n_mes)
        st.plotly_chart(fig_box, use_container_width=True)

        fig_pmf = build_pmf_figure(d12, ruta, turno, cap0, capt, capa)
        st.plotly_chart(fig_pmf, use_container_width=True)

with tab4:
    st.subheader("Tabla resumen (cuantiles + coberturas)")

    # ✅ Nuevo: selector de cuantiles
    q_mode = st.selectbox(
        "Cuantiles de Pasajeros",
        [
            "Cuartiles (25/50/75)",
            "Quintiles (20/40/60/80)",
            "Deciles (10..90)"
        ],
        index=0
    )

    if q_mode.startswith("Cuartiles"):
        q_list = [0.25, 0.50, 0.75]
    elif q_mode.startswith("Quintiles"):
        q_list = [0.20, 0.40, 0.60, 0.80]
    else:
        q_list = [i / 10 for i in range(1, 10)]  # 0.1 ... 0.9

    # ✅ Recalcular cuantiles dinámicos por (Ruta, Turno)
    q_tbl = (
        d12.groupby(["Ruta", TURN_COL])[PAX_COL]
           .quantile(q_list)
           .unstack()
    )

    # nombres tipo p25, p50, p75 / p20.. / p10..p90
    q_cols = [f"p{int(round(q*100))}" for q in q_list]
    q_tbl.columns = q_cols

    # n_12m
    n12 = d12.groupby(["Ruta", TURN_COL])[PAX_COL].size().rename("n_12m")

    # ✅ reconstruir summary con cuantiles elegidos
    summary_dyn = (
        reco.merge(q_tbl.reset_index(), on=["Ruta", TURN_COL], how="left")
            .merge(n12.reset_index(), on=["Ruta", TURN_COL], how="left")
    )

    # coberturas en %
    for c in ["coverage_actual", "coverage_timid", "coverage_aggressive"]:
        if c in summary_dyn.columns:
            summary_dyn[c + "_pct"] = pd.to_numeric(summary_dyn[c], errors="coerce") * 100

    # Renombres “amenos”
    to_r = {
        "n_active_5m": "Pasajeros activos (5m)",
        "n_12m": "Pasajeros 12m",
        "cap_actual": "Capacidad actual",
        "cap_reco_timid": "Capacidad reco. timid",
        "cap_reco_aggressive": "Capacidad reco. aggressive",
        "coverage_actual_pct": "Cobertura actual (%)",
        "coverage_timid_pct": "Cobertura timid (%)",
        "coverage_aggressive_pct": "Cobertura aggressive (%)",
        "excess_cost_month_timid": "Costo excedentes mensual (Timid)",
        "excess_cost_month_aggressive": "Costo excedentes mensual (Aggressive)",
    }

    # también renombrar cuantiles a algo legible
    # p25 -> P25, p50 -> P50, etc.
    q_rename = {c: c.upper() for c in q_cols}

    summary_show = summary_dyn.rename(columns={**to_r, **q_rename})

    # formateo de coberturas
    for c in ["Cobertura actual (%)", "Cobertura timid (%)", "Cobertura aggressive (%)"]:
        if c in summary_show.columns:
            summary_show[c] = summary_show[c].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

    # columnas a mostrar (dinámicas según cuantiles)
    q_cols_show = [c.upper() for c in q_cols]
    cols = [
        "Ruta", TURN_COL, "grupo",
        "Pasajeros 12m", "Pasajeros activos (5m)",
        *q_cols_show,
        "Capacidad actual", "Capacidad reco. timid", "Capacidad reco. aggressive",
        "Cobertura actual (%)", "Cobertura timid (%)", "Cobertura aggressive (%)",
        "Costo excedentes mensual (Timid)", "Costo excedentes mensual (Aggressive)",
    ]
    cols = [c for c in cols if c in summary_show.columns]

    st.dataframe(
        summary_show[cols].sort_values(["Ruta", TURN_COL]),
        use_container_width=True,
        height=420
    )

    st.subheader("Tabla detallada (reco)")
    st.dataframe(
        reco.sort_values("ahorro_mes_aggressive_pos", ascending=False, na_position="last"),
        use_container_width=True,
        height=420
    )

    # downloads (usá summary_dyn para exportar con cuantiles elegidos)
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