# app.py
import re
import unicodedata
import json
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict


# =========================
# ✅ JSON helpers (FIX: date not JSON serializable)
# =========================
def _json_default(o):
    """
    Hace serializable para json.dumps:
    - datetime/date/pandas Timestamp/Period
    - numpy scalars/arrays
    - pandas NA
    """
    # pandas / datetime
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, pd.Period):
        return str(o)

    # numpy
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()

    # pandas NA
    try:
        if pd.isna(o):
            return None
    except Exception:
        pass

    # fallback
    return str(o)


def build_dashboard_html(events_df: pd.DataFrame, dinamo_df: Optional[pd.DataFrame], initial_filters: dict) -> bytes:
    # -------------------------
    # 1) Preparar data “limpia” para el navegador (keys sin espacios)
    # -------------------------
    ev = events_df.copy()

    # columnas base que usás en el dashboard
    # (si alguna no existe, la creamos vacía)
    need_cols = [
        "inicio_dt", "fin_dt", "duracion_s",
        "pct_ocup", "fecha", "mes_ym", "dia_semana",
        "ruta", "sede", "jornada", "piloto",
        "ie_norm", "corredor_desc",
        "Cantidad de correlativos", "Ocupación máxima", "horario de turno",
        "ruta_norm", "sede_norm", "jornada_norm",
    ]
    for c in need_cols:
        if c not in ev.columns:
            ev[c] = np.nan

    # renombrar a keys JS-friendly
    ev_out = pd.DataFrame({
        "inicio_dt": pd.to_datetime(ev["inicio_dt"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "fin_dt": pd.to_datetime(ev["fin_dt"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "duracion_s": pd.to_numeric(ev["duracion_s"], errors="coerce"),
        "pct_ocup": pd.to_numeric(ev["pct_ocup"], errors="coerce"),  # 0..1
        "fecha": ev["fecha"].astype(str),
        "mes_ym": ev["mes_ym"].astype(str),
        "dia_semana": ev["dia_semana"].astype(str),
        "ruta": ev["ruta"].astype(str),
        "sede": ev["sede"].astype(str),
        "jornada": ev["jornada"].astype(str),
        "piloto": ev["piloto"].astype(str),
        "ie_norm": ev["ie_norm"].astype(str),
        "corredor_desc": ev["corredor_desc"].astype(str),
        "correlativos": pd.to_numeric(ev["Cantidad de correlativos"], errors="coerce"),
        "ocup_max": pd.to_numeric(ev["Ocupación máxima"], errors="coerce"),
        "horario_turno": ev["horario de turno"].astype(str),
        "ruta_norm": ev["ruta_norm"].astype(str),
        "sede_norm": ev["sede_norm"].astype(str),
        "jornada_norm": ev["jornada_norm"].astype(str),
    })

    # NaN -> None (para JSON)
    ev_out = ev_out.where(pd.notna(ev_out), None)

    din_records = []
    cap_map = {}
    if dinamo_df is not None and not dinamo_df.empty:
        d = dinamo_df.copy()

        # asegurar columnas esperadas
        d_need = [
            "Ruta", "Sede", "Jornada", "Tipo de bus", "Km", "Tiempo (minutos)", "Capacidad", "# Días", "Precio Q (Diario)",
            "ruta_norm", "sede_norm", "jornada_norm"
        ]
        for c in d_need:
            if c not in d.columns:
                d[c] = np.nan

        d_out = pd.DataFrame({
            "ruta": d["Ruta"].astype(str),
            "sede": d["Sede"].astype(str),
            "jornada": d["Jornada"].astype(str),
            "bus_tipo": d["Tipo de bus"].astype(str),
            "km": pd.to_numeric(d["Km"], errors="coerce"),
            "tiempo_min": pd.to_numeric(d["Tiempo (minutos)"], errors="coerce"),
            "capacidad": pd.to_numeric(d["Capacidad"], errors="coerce"),
            "dias": pd.to_numeric(d["# Días"], errors="coerce"),
            "precio_q": pd.to_numeric(d["Precio Q (Diario)"], errors="coerce"),
            "ruta_norm": d["ruta_norm"].astype(str),
            "sede_norm": d["sede_norm"].astype(str),
            "jornada_norm": d["jornada_norm"].astype(str),
        })
        d_out = d_out.where(pd.notna(d_out), None)
        din_records = d_out.to_dict(orient="records")

        # mapa de capacidad por ruta (para la línea “Cap. Dinamo”)
        for r in din_records:
            rr = (r.get("ruta") or "").strip()
            cap = r.get("capacidad")
            if rr and cap is not None and rr not in cap_map:
                cap_map[rr] = cap

    events_records = ev_out.to_dict(orient="records")

    # -------------------------
    # 2) Template HTML (Plotly + MathJS por CDN)
    # -------------------------
    init = initial_filters or {}
    payload = {
        "events": events_records,
        "dinamo": din_records,
        "cap_by_route": cap_map,
        "init": init,
        "weekday_order": ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
    }

    # ✅ FIX: default=_json_default para date/datetime/numpy
    html = HTML_TEMPLATE.replace("__DASH_DATA__", json.dumps(payload, ensure_ascii=False, default=_json_default))

    return html.encode("utf-8")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Dashboard Rutas (HTML)</title>

  <!-- CDN (si querés offline, bajás estos .js y apuntás local) -->
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/mathjs@12.4.2/lib/browser/math.min.js"></script>

  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",Arial; margin:0; background:#0b1020; color:#e9edf7;}
    .wrap{max-width:1400px; margin:0 auto; padding:18px;}
    .card{background:#111a33; border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:14px; box-shadow:0 10px 24px rgba(0,0,0,.25);}
    .grid{display:grid; gap:12px;}
    .grid.kpis{grid-template-columns:repeat(4,minmax(0,1fr));}
    .kpi .label{opacity:.8; font-size:12px;}
    .kpi .value{font-size:22px; font-weight:700; margin-top:6px;}
    .row{display:flex; gap:12px; flex-wrap:wrap;}
    label{font-size:12px; opacity:.85; display:block; margin-bottom:6px;}
    select,input{background:#0b1020; color:#e9edf7; border:1px solid rgba(255,255,255,.14); border-radius:10px; padding:8px; min-width:180px;}
    select[multiple]{min-height:110px;}
    .tabs{display:flex; gap:8px; margin:14px 0;}
    .tabbtn{cursor:pointer; padding:10px 12px; border-radius:12px; border:1px solid rgba(255,255,255,.14); background:#0b1020; color:#e9edf7;}
    .tabbtn.active{background:#1b2a55;}
    .tab{display:none;}
    .tab.active{display:block;}
    .h2{font-size:16px; font-weight:800; margin:0 0 10px;}
    .muted{opacity:.75;}
    .plots-grid{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px;}
    table{width:100%; border-collapse:collapse; font-size:12px;}
    th,td{border-bottom:1px solid rgba(255,255,255,.10); padding:8px; text-align:left;}
  </style>
</head>
<body>
<div class="wrap">

  <div class="card">
    <div class="h2">Rutas — Dashboard (HTML dinámico)</div>
    <div class="muted">Filtros globales (aplican a todas las pestañas)</div>
    <div class="row" style="margin-top:10px;">
      <div>
        <label>Fecha desde</label>
        <input id="f_date0" type="date"/>
      </div>
      <div>
        <label>Fecha hasta</label>
        <input id="f_date1" type="date"/>
      </div>
      <div>
        <label>Corredor (multi)</label>
        <select id="f_corr" multiple></select>
      </div>
      <div>
        <label>Sede (multi)</label>
        <select id="f_sede" multiple></select>
      </div>
      <div>
        <label>Ruta (una o todas)</label>
        <select id="f_ruta"></select>
      </div>
      <div>
        <label>Jornada (multi)</label>
        <select id="f_jornada" multiple></select>
      </div>
      <div>
        <label>Piloto (multi)</label>
        <select id="f_piloto" multiple></select>
      </div>
      <div>
        <label>Ingreso/Egreso (multi)</label>
        <select id="f_ie" multiple></select>
      </div>
    </div>
  </div>

  <div class="tabs">
    <button class="tabbtn active" data-tab="t1">Dashboard general</button>
    <button class="tabbtn" data-tab="t2">Series / cascada / servicio</button>
    <button class="tabbtn" data-tab="t3">Dinamo: Precio</button>
  </div>

  <!-- TAB 1 -->
  <div id="t1" class="tab active">
    <div class="grid kpis" style="margin-bottom:12px;">
      <div class="card kpi"><div class="label">Movimientos</div><div id="k_mov" class="value">—</div></div>
      <div class="card kpi"><div class="label">Ingreso / Egreso</div><div id="k_ie" class="value">—</div></div>
      <div class="card kpi"><div class="label">% ocupación (prom.)</div><div id="k_pct" class="value">—</div></div>
      <div class="card kpi"><div class="label">Capacidad típica (mediana)</div><div id="k_cap" class="value">—</div></div>
    </div>

    <div class="card">
      <div class="h2">Caja y bigotes — por mes / día semana</div>
      <div class="row" style="margin:10px 0;">
        <div>
          <label>Agrupar por</label>
          <select id="box_group">
            <option>Mes</option>
            <option>Día de la semana</option>
          </select>
        </div>
        <div>
          <label>Métrica</label>
          <select id="box_metric">
            <option>Demanda (correlativos)</option>
            <option>% ocupación</option>
          </select>
        </div>
        <div>
          <label>Ruta (local)</label>
          <select id="box_ruta"></select>
        </div>
        <div>
          <label>Jornadas (local, multi)</label>
          <select id="box_jornadas" multiple></select>
        </div>
        <div>
          <label>Movimiento (local)</label>
          <select id="box_dir">
            <option>Ambos</option>
            <option>Ingreso</option>
            <option>Egreso</option>
          </select>
        </div>
      </div>
      <div id="boxplot" style="height:520px;"></div>
    </div>
  </div>

  <!-- TAB 2 -->
  <div id="t2" class="tab">
    <div class="card" style="margin-bottom:12px;">
      <div class="h2">Cascada — Total → descomposición por jornada</div>
      <div class="row" style="margin:10px 0;">
        <div>
          <label>Grupo</label>
          <select id="wf_group">
            <option>Mes</option>
            <option>Día de la semana</option>
          </select>
        </div>
        <div>
          <label>Métrica</label>
          <select id="wf_measure">
            <option>Movimientos (conteo)</option>
            <option>Demanda (suma)</option>
          </select>
        </div>
        <div>
          <label>Máx. gráficos</label>
          <input id="wf_max" type="number" min="1" max="30" value="8"/>
        </div>
        <div>
          <label>Movimiento</label>
          <select id="wf_dir">
            <option>Ambos</option>
            <option>Ingreso</option>
            <option>Egreso</option>
          </select>
        </div>
        <div>
          <label>Jornadas (multi)</label>
          <select id="wf_jornadas" multiple></select>
        </div>
      </div>
      <div id="wf_grid" class="plots-grid"></div>
    </div>

    <div class="card" style="margin-bottom:12px;">
      <div class="h2">Series de tiempo</div>
      <div class="row" style="margin:10px 0;">
        <div>
          <label>Granularidad</label>
          <select id="ts_gran">
            <option>Día</option>
            <option selected>Semana</option>
            <option>Mes</option>
          </select>
        </div>
        <div>
          <label>Métrica</label>
          <select id="ts_metric">
            <option>Movimientos</option>
            <option>Demanda (suma)</option>
            <option>Demanda (P95)</option>
          </select>
        </div>
        <div>
          <label>Separar por</label>
          <select id="ts_split">
            <option>Nada</option>
            <option>Ingreso/Egreso</option>
            <option>Jornada</option>
          </select>
        </div>
      </div>
      <div id="ts_plot" style="height:520px;"></div>
    </div>

    <div class="card">
      <div class="h2">Acumulado: nivel de servicio vs capacidad</div>
      <div class="row" style="margin:10px 0;">
        <div>
          <label>Ruta</label>
          <select id="svc_ruta"></select>
        </div>
        <div>
          <label>Movimiento</label>
          <select id="svc_dir">
            <option>Ambos</option>
            <option>Ingreso</option>
            <option>Egreso</option>
          </select>
        </div>
        <div>
          <label>Unidad</label>
          <select id="svc_unit">
            <option>Registro</option>
            <option selected>Día (máximo)</option>
            <option>Día+Turno (máximo)</option>
          </select>
        </div>
        <div>
          <label>Objetivos (multi)</label>
          <select id="svc_levels" multiple>
            <option value="0.80">80%</option>
            <option value="0.85">85%</option>
            <option value="0.90" selected>90%</option>
            <option value="0.95" selected>95%</option>
            <option value="0.97">97%</option>
            <option value="0.99">99%</option>
          </select>
        </div>
      </div>

      <div class="row" style="margin:10px 0;">
        <div class="card" style="flex:1;">
          <div class="muted" style="margin-bottom:8px;">Capacidades recomendadas</div>
          <div id="svc_table"></div>
        </div>
        <div class="card" style="flex:1;">
          <div class="muted" style="margin-bottom:8px;">Probar capacidad C</div>
          <input id="svc_cap_test" type="range" min="0" max="1" value="1" step="1" style="width:100%;">
          <div class="row" style="margin-top:10px;">
            <div style="flex:1;"><div class="label">Nivel logrado</div><div id="svc_ach" class="value">—</div></div>
            <div style="flex:1;"><div class="label">Casos excedidos</div><div id="svc_over" class="value">—</div></div>
          </div>
        </div>
      </div>

      <div id="svc_ecdf" style="height:520px; margin-top:12px;"></div>
    </div>
  </div>

  <!-- TAB 3 -->
  <div id="t3" class="tab">
    <div class="card">
      <div class="h2">Dinamo: Precio Q (Diario) — correlación + cascada ΔR²</div>

      <div class="row" style="margin:10px 0;">
        <div>
          <label>Join mode</label>
          <select id="p_join">
            <option>Ruta</option>
            <option>Ruta + Jornada</option>
            <option>Ruta + Sede</option>
            <option>Ruta + Sede + Jornada</option>
          </select>
        </div>
        <div>
          <label>Correlación</label>
          <select id="p_corr">
            <option>pearson</option>
            <option selected>spearman</option>
          </select>
        </div>
        <div>
          <label>Piloto significativo: share mínimo</label>
          <input id="p_min_share" type="number" min="0.05" max="0.50" step="0.01" value="0.15"/>
        </div>
        <div>
          <label>Eventos mínimos por piloto</label>
          <input id="p_min_total" type="number" min="1" step="1" value="30"/>
        </div>
        <div>
          <label>Bucket min eventos (día+turno)</label>
          <input id="p_bucket_min" type="number" min="1" step="1" value="2"/>
        </div>
      </div>

      <div class="card" style="margin-top:10px;">
        <div class="h2">1) Correlación numérica vs Precio</div>
        <div id="p_corr_tbl"></div>
      </div>

      <div class="card" style="margin-top:10px;">
        <div class="h2">2) Categóricas vs Precio</div>
        <div class="row" style="margin:10px 0;">
          <div>
            <label>Categoría</label>
            <select id="p_cat">
              <option>Jornada</option>
              <option>Tipo de bus</option>
            </select>
          </div>
        </div>
        <div id="p_cat_plot" style="height:520px;"></div>
        <div id="p_cat_tbl" style="margin-top:10px;"></div>
      </div>

      <div class="card" style="margin-top:10px;">
        <div class="h2">3) Cascada ΔR² (nested OLS)</div>
        <div class="muted">Ojo: si hay demasiadas filas/categorías puede tardar; se muestrea si es enorme.</div>
        <div id="p_r2_plot" style="height:520px; margin-top:10px;"></div>
        <div id="p_r2_tbl" style="margin-top:10px;"></div>
      </div>
    </div>
  </div>

</div>

<script>
  // ============================
  // DATA
  // ============================
  const DASH = __DASH_DATA__;
  const WEEKDAY_ORDER = DASH.weekday_order || ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"];

  // ============================
  // Helpers (arrow functions only)
  // ============================
  const uniqSorted = (arr) => [...new Set(arr.filter((x) => x !== null && x !== undefined && String(x).trim() !== ""))].sort((a,b)=>String(a).localeCompare(String(b)));
  const getMulti = (sel) => [...sel.selectedOptions].map((o) => o.value);
  const setOptions = (sel, values, {includeAll=false, allLabel="Todas"} = {}) => {
    const prev = new Set([...sel.options].filter((o)=>o.selected).map((o)=>o.value));
    sel.innerHTML = "";
    if (includeAll) {
      const opt = document.createElement("option");
      opt.value = "__ALL__";
      opt.textContent = allLabel;
      sel.appendChild(opt);
    }
    values.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      if (prev.has(v)) opt.selected = true;
      sel.appendChild(opt);
    });
  };

  const parseDateOnly = (iso) => {
    if (!iso) return null;
    const s = String(iso).slice(0,10);
    if (!s || s.length !== 10) return null;
    return s;
  };

  const corrPearson = (xs, ys) => {
    const n = xs.length;
    if (n < 3) return null;
    const mx = xs.reduce((a,b)=>a+b,0)/n;
    const my = ys.reduce((a,b)=>a+b,0)/n;
    let num=0, dx=0, dy=0;
    for (let i=0;i<n;i++){
      const vx = xs[i]-mx;
      const vy = ys[i]-my;
      num += vx*vy;
      dx += vx*vx;
      dy += vy*vy;
    }
    const den = Math.sqrt(dx*dy);
    if (!isFinite(den) || den===0) return null;
    return num/den;
  };

  const rankAvg = (arr) => {
    const idx = arr.map((v,i)=>[v,i]).sort((a,b)=>a[0]-b[0]);
    const ranks = Array(arr.length).fill(0);
    let i=0;
    while(i<idx.length){
      let j=i;
      while(j<idx.length && idx[j][0]===idx[i][0]) j++;
      const r = (i+1 + j)/2; // promedio
      for(let k=i;k<j;k++) ranks[idx[k][1]] = r;
      i=j;
    }
    return ranks;
  };

  const corrSpearman = (xs, ys) => corrPearson(rankAvg(xs), rankAvg(ys));

  const quantile = (arr, q) => {
    const xs = arr.filter((v)=>v!==null && v!==undefined && isFinite(v)).slice().sort((a,b)=>a-b);
    if (!xs.length) return null;
    const pos = (xs.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if ((xs[base+1] !== undefined)) return xs[base] + rest * (xs[base+1] - xs[base]);
    return xs[base];
  };

  const timeBin = (iso, gran) => {
    if (!iso) return null;
    const d = new Date(iso);
    if (!isFinite(d)) return null;
    const yyyy = d.getFullYear();
    const mm = d.getMonth();
    const dd = d.getDate();

    if (gran === "Día") return new Date(yyyy, mm, dd).toISOString().slice(0,10);

    if (gran === "Mes") return new Date(yyyy, mm, 1).toISOString().slice(0,10);

    // Semana (inicio lunes)
    const day = d.getDay(); // 0 dom ... 1 lun ...
    const delta = (day === 0 ? -6 : 1 - day);
    const w = new Date(yyyy, mm, dd + delta);
    return new Date(w.getFullYear(), w.getMonth(), w.getDate()).toISOString().slice(0,10);
  };

  const groupAgg = (rows, keyFn, valFn, aggFn) => {
    const m = new Map();
    rows.forEach((r) => {
      const k = keyFn(r);
      if (k === null || k === undefined) return;
      const v = valFn(r);
      if (v === null || v === undefined || !isFinite(v)) return;
      if (!m.has(k)) m.set(k, []);
      m.get(k).push(v);
    });
    const out = [];
    m.forEach((vals, k) => out.push([k, aggFn(vals)]));
    return out;
  };

  const renderTable = (rows, cols) => {
    if (!rows || !rows.length) return "<div class='muted'>Sin datos</div>";
    const thead = "<tr>" + cols.map((c)=>`<th>${c}</th>`).join("") + "</tr>";
    const tbody = rows.map((r)=>"<tr>" + cols.map((c)=>`<td>${r[c] ?? "—"}</td>`).join("") + "</tr>").join("");
    return `<table><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;
  };

  // ============================
  // Global filters
  // ============================
  const els = {
    f_date0: document.getElementById("f_date0"),
    f_date1: document.getElementById("f_date1"),
    f_corr: document.getElementById("f_corr"),
    f_sede: document.getElementById("f_sede"),
    f_ruta: document.getElementById("f_ruta"),
    f_jornada: document.getElementById("f_jornada"),
    f_piloto: document.getElementById("f_piloto"),
    f_ie: document.getElementById("f_ie"),
  };

  const state = {
    global: {
      date0: null,
      date1: null,
      corr: [],
      sede: [],
      ruta: "__ALL__",
      jornada: [],
      piloto: [],
      ie: [],
    }
  };

  const applyGlobal = (rows) => {
    const g = state.global;
    return rows.filter((r) => {
      const d = parseDateOnly(r.inicio_dt);
      if (g.date0 && d && d < g.date0) return false;
      if (g.date1 && d && d > g.date1) return false;

      if (g.corr.length && !g.corr.includes(r.corredor_desc)) return false;
      if (g.sede.length && !g.sede.includes(r.sede)) return false;
      if (g.ruta !== "__ALL__" && String(r.ruta).trim() !== String(g.ruta).trim()) return false;
      if (g.jornada.length && !g.jornada.includes(r.jornada)) return false;
      if (g.piloto.length && !g.piloto.includes(r.piloto)) return false;
      if (g.ie.length && !g.ie.includes(r.ie_norm)) return false;
      return true;
    });
  };

  // ============================
  // Render: KPI
  // ============================
  const renderKPI = (rows) => {
    document.getElementById("k_mov").textContent = (rows.length || 0).toLocaleString("es-GT");

    const ing = rows.filter((r)=>r.ie_norm==="ingreso").length;
    const egr = rows.filter((r)=>r.ie_norm==="egreso").length;
    document.getElementById("k_ie").textContent = `${ing} / ${egr}`;

    const pcts = rows.map((r)=> (r.pct_ocup!==null && r.pct_ocup!==undefined) ? Number(r.pct_ocup) : null).filter((v)=>v!==null && isFinite(v));
    const mp = pcts.length ? (pcts.reduce((a,b)=>a+b,0)/pcts.length) : null;
    document.getElementById("k_pct").textContent = (mp===null) ? "—" : `${(mp*100).toFixed(1)}%`;

    const caps = rows.map((r)=> (r.ocup_max!==null && r.ocup_max!==undefined) ? Number(r.ocup_max) : null).filter((v)=>v!==null && isFinite(v)).sort((a,b)=>a-b);
    const med = caps.length ? caps[Math.floor(caps.length/2)] : null;
    document.getElementById("k_cap").textContent = (med===null) ? "—" : `${Math.round(med)}`;
  };

  // ============================
  // TAB 1: Boxplot
  // ============================
  const renderBoxplot = (rows) => {
    const group = document.getElementById("box_group").value;
    const metric = document.getElementById("box_metric").value;
    const ruta = document.getElementById("box_ruta").value;
    const jornadasSel = getMulti(document.getElementById("box_jornadas"));
    const dir = document.getElementById("box_dir").value;

    let base = rows.slice();
    if (ruta !== "__ALL__") base = base.filter((r)=>String(r.ruta)===String(ruta));
    if (jornadasSel.length) base = base.filter((r)=>jornadasSel.includes(r.jornada));
    if (dir !== "Ambos") base = base.filter((r)=> r.ie_norm === (dir==="Ingreso" ? "ingreso" : "egreso"));

    const gcol = (group==="Mes") ? "mes_ym" : "dia_semana";

    const groups = new Map();
    base.forEach((r) => {
      const k = r[gcol];
      if (!k) return;
      let v = null;
      if (metric.startsWith("%")) v = (r.pct_ocup!==null && isFinite(r.pct_ocup)) ? (Number(r.pct_ocup)*100) : null;
      else v = (r.correlativos!==null && isFinite(r.correlativos)) ? Number(r.correlativos) : null;
      if (v===null) return;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(v);
    });

    let keys = [...groups.keys()];
    if (gcol === "dia_semana") keys = WEEKDAY_ORDER.filter((d)=>keys.includes(d));
    else keys = keys.sort();

    const traces = keys.map((k)=>({
      type: "box",
      name: k,
      y: groups.get(k),
      boxpoints: "outliers"
    }));

    const layout = {
      title: metric + " por " + group,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {color:"#e9edf7"},
      xaxis: {title: group},
      yaxis: {title: metric},
      margin: {l:60,r:20,t:50,b:60}
    };
    Plotly.react("boxplot", traces, layout, {responsive:true});
  };

  // ============================
  // TAB 2: Waterfall por grupo
  // ============================
  const renderWaterfalls = (rows) => {
    const group = document.getElementById("wf_group").value;
    const measure = document.getElementById("wf_measure").value;
    const maxN = Math.max(1, Math.min(30, Number(document.getElementById("wf_max").value || 8)));
    const dir = document.getElementById("wf_dir").value;
    const jornadasSel = getMulti(document.getElementById("wf_jornadas"));

    const gcol = (group==="Mes") ? "mes_ym" : "dia_semana";

    let base = rows.slice();
    if (jornadasSel.length) base = base.filter((r)=>jornadasSel.includes(r.jornada));
    if (dir !== "Ambos") base = base.filter((r)=> r.ie_norm === (dir==="Ingreso" ? "ingreso" : "egreso"));

    const totalMap = new Map();
    base.forEach((r)=>{
      const k = r[gcol];
      if (!k) return;
      const add = (measure.startsWith("Mov")) ? 1 : (isFinite(r.correlativos) ? Number(r.correlativos) : null);
      if (add===null) return;
      totalMap.set(k, (totalMap.get(k) || 0) + add);
    });

    let groups = [...totalMap.entries()].sort((a,b)=>b[1]-a[1]).slice(0, maxN).map((x)=>x[0]);
    if (gcol === "dia_semana") groups = WEEKDAY_ORDER.filter((d)=>groups.includes(d));

    const grid = document.getElementById("wf_grid");
    grid.innerHTML = "";

    groups.forEach((g, idx) => {
      const div = document.createElement("div");
      div.className = "card";
      div.style.height = "420px";
      div.id = `wf_${idx}`;
      grid.appendChild(div);

      const byJ = new Map();
      base.forEach((r)=>{
        if (r[gcol] !== g) return;
        const j = r.jornada || "—";
        const add = (measure.startsWith("Mov")) ? 1 : (isFinite(r.correlativos) ? Number(r.correlativos) : null);
        if (add===null) return;
        byJ.set(j, (byJ.get(j) || 0) + add);
      });

      const rowsJ = [...byJ.entries()].sort((a,b)=>b[1]-a[1]);
      const labels = rowsJ.map((x)=>x[0]).concat(["Total"]);
      const values = rowsJ.map((x)=>x[1]).concat([0]);
      const measures = rowsJ.map(()=> "relative").concat(["total"]);

      const fig = [{
        type:"waterfall",
        x: labels,
        y: values,
        measure: measures
      }];

      const layout = {
        title: `${measure} – ${group}: ${g} (${dir})`,
        paper_bgcolor:"rgba(0,0,0,0)",
        plot_bgcolor:"rgba(0,0,0,0)",
        font:{color:"#e9edf7"},
        margin:{l:60,r:20,t:60,b:60},
        xaxis:{title:"Jornada"},
        yaxis:{title: measure.startsWith("Mov") ? "Movimientos" : "Demanda (suma)"}
      };

      Plotly.newPlot(div.id, fig, layout, {responsive:true});
    });
  };

  // ============================
  // TAB 2: Timeseries
  // ============================
  const renderTimeseries = (rows) => {
    const gran = document.getElementById("ts_gran").value;
    const metric = document.getElementById("ts_metric").value;
    const split = document.getElementById("ts_split").value;

    const base = rows.filter((r)=>!!r.inicio_dt);
    const colorFn = (r) => {
      if (split === "Ingreso/Egreso") return (r.ie_norm==="ingreso" ? "Ingreso" : (r.ie_norm==="egreso" ? "Egreso" : "Otro"));
      if (split === "Jornada") return r.jornada || "—";
      return "Total";
    };

    const agg = new Map();
    base.forEach((r)=>{
      const t = timeBin(r.inicio_dt, gran);
      if (!t) return;
      const c = colorFn(r);
      const k = `${t}||${c}`;
      if (!agg.has(k)) agg.set(k, []);
      if (metric === "Movimientos") agg.get(k).push(1);
      else {
        const d = (isFinite(r.correlativos) ? Number(r.correlativos) : null);
        if (d===null) return;
        agg.get(k).push(d);
      }
    });

    const points = [];
    agg.forEach((vals, k)=>{
      const [t,c] = k.split("||");
      let v = null;
      if (metric === "Movimientos") v = vals.length;
      else if (metric === "Demanda (suma)") v = vals.reduce((a,b)=>a+b,0);
      else v = quantile(vals, 0.95);
      if (v===null) return;
      points.push({t, c, v});
    });

    points.sort((a,b)=>a.t.localeCompare(b.t));

    const colors = uniqSorted(points.map((p)=>p.c));
    const traces = colors.map((c)=>({
      type:"scatter",
      mode:"lines+markers",
      name: c,
      x: points.filter((p)=>p.c===c).map((p)=>p.t),
      y: points.filter((p)=>p.c===c).map((p)=>p.v)
    }));

    const layout = {
      title: `${metric} (${gran})`,
      paper_bgcolor:"rgba(0,0,0,0)",
      plot_bgcolor:"rgba(0,0,0,0)",
      font:{color:"#e9edf7"},
      margin:{l:60,r:20,t:60,b:60},
      xaxis:{title:"Tiempo"},
      yaxis:{title: metric}
    };

    Plotly.react("ts_plot", traces, layout, {responsive:true});
  };

  // ============================
  // TAB 2: Service level (ECDF)
  // ============================
  const buildDemandSeries = (rows, unit) => {
    const xs = [];
    if (unit === "Registro") {
      rows.forEach((r)=>{
        if (isFinite(r.correlativos)) xs.push(Number(r.correlativos));
      });
      return xs;
    }
    if (unit === "Día (máximo)") {
      const m = new Map();
      rows.forEach((r)=>{
        const f = r.fecha;
        if (!f || !isFinite(r.correlativos)) return;
        const v = Number(r.correlativos);
        m.set(f, Math.max(m.get(f) || -Infinity, v));
      });
      return [...m.values()].filter((v)=>isFinite(v));
    }
    // Día+Turno
    const m = new Map();
    rows.forEach((r)=>{
      const f = r.fecha;
      const h = r.horario_turno || "";
      if (!f || !isFinite(r.correlativos)) return;
      const k = `${f}||${h}`;
      const v = Number(r.correlativos);
      m.set(k, Math.max(m.get(k) || -Infinity, v));
    });
    return [...m.values()].filter((v)=>isFinite(v));
  };

  const renderService = (rows) => {
    const ruta = document.getElementById("svc_ruta").value;
    const dir = document.getElementById("svc_dir").value;
    const unit = document.getElementById("svc_unit").value;
    const levels = getMulti(document.getElementById("svc_levels")).map((x)=>Number(x)).filter((x)=>isFinite(x)).sort((a,b)=>a-b);

    let base = rows.slice();
    if (ruta !== "__ALL__") base = base.filter((r)=>String(r.ruta)===String(ruta));
    if (dir !== "Ambos") base = base.filter((r)=> r.ie_norm === (dir==="Ingreso" ? "ingreso" : "egreso"));

    const s = buildDemandSeries(base, unit);
    if (!s.length) {
      document.getElementById("svc_table").innerHTML = "<div class='muted'>Sin demanda válida</div>";
      Plotly.react("svc_ecdf", [], {paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{color:"#e9edf7"}});
      return;
    }

    const rec = levels.map((p)=>({
      nivel_servicio_objetivo: `${Math.round(p*100)}%`,
      capacidad_minima_recomendada: Math.ceil(quantile(s, p))
    }));

    document.getElementById("svc_table").innerHTML = renderTable(rec, ["nivel_servicio_objetivo", "capacidad_minima_recomendada"]);

    const sMin = Math.floor(Math.min(...s));
    const sMax = Math.ceil(Math.max(...s));
    const capTest = document.getElementById("svc_cap_test");
    capTest.min = String(Math.max(0, sMin));
    capTest.max = String(Math.max(Math.max(1, sMin+1), sMax));
    const p95 = Math.ceil(quantile(s, 0.95));
    capTest.value = String(Math.min(Number(capTest.max), Math.max(Number(capTest.min), p95)));

    const calcAndRender = () => {
      const C = Number(capTest.value);
      const ach = (s.filter((x)=>x<=C).length / s.length) * 100;
      const over = s.filter((x)=>x>C).length;
      document.getElementById("svc_ach").textContent = `${ach.toFixed(1)}%`;
      document.getElementById("svc_over").textContent = over.toLocaleString("es-GT");

      const xs = s.slice().sort((a,b)=>a-b);
      const ys = xs.map((_,i)=>(i+1)/xs.length);

      const shapes = [];
      const ann = [];

      levels.forEach((p)=>{
        const cap = Math.ceil(quantile(s, p));
        shapes.push({type:"line", x0:cap,x1:cap,y0:0,y1:1, xref:"x",yref:"y", line:{dash:"dot", width:1}});
        ann.push({x:cap, y:0.02, xref:"x", yref:"y", text:`P${Math.round(p*100)}=${cap}`, showarrow:false});
      });

      shapes.push({type:"line", x0:C,x1:C,y0:0,y1:1, xref:"x",yref:"y", line:{dash:"solid", width:2}});
      ann.push({x:C, y:0.98, xref:"x", yref:"y", text:`C probada=${C}`, showarrow:false, yanchor:"top"});

      const capDin = (ruta !== "__ALL__") ? DASH.cap_by_route[String(ruta)] : null;
      if (capDin !== null && capDin !== undefined && isFinite(capDin)) {
        shapes.push({type:"line", x0:capDin,x1:capDin,y0:0,y1:1, xref:"x",yref:"y", line:{dash:"dash", width:2}});
        ann.push({x:capDin, y:0.90, xref:"x", yref:"y", text:`Cap. Dinamo=${Math.round(capDin)}`, showarrow:false, yanchor:"top"});
      }

      const traces = [{
        type:"scatter",
        mode:"lines",
        name:"ECDF",
        x: xs,
        y: ys
      }];

      const layout = {
        title: `ECDF — ${ruta} | ${dir} | ${unit}`,
        paper_bgcolor:"rgba(0,0,0,0)",
        plot_bgcolor:"rgba(0,0,0,0)",
        font:{color:"#e9edf7"},
        margin:{l:60,r:20,t:60,b:60},
        xaxis:{title:"Capacidad / Demanda (correlativos)"},
        yaxis:{title:"Acumulado (nivel de servicio)", range:[0,1]},
        shapes,
        annotations: ann
      };

      Plotly.react("svc_ecdf", traces, layout, {responsive:true});
    };

    capTest.oninput = () => calcAndRender();
    calcAndRender();
  };

  // ============================
  // TAB 3: Dinamo Precio (PM + merge + corr + cat + ΔR²)
  // ============================
  const buildPilotMetrics = (rows, minShare, minTotal, bucketMin) => {
    // rp: route_norm + piloto => events
    const rp = new Map();
    const rt = new Map();

    rows.forEach((r)=>{
      const rn = r.ruta_norm || "";
      const pn = (r.piloto || "").trim().toLowerCase();
      if (!rn || !pn) return;
      const k = `${rn}||${pn}`;
      rp.set(k, (rp.get(k) || 0) + 1);
      rt.set(rn, (rt.get(rn) || 0) + 1);
    });

    // share + significant
    const sig = new Map(); // route||pilot => bool
    const pilotSetByRoute = new Map();
    const topShare = new Map();
    rp.forEach((cnt, k)=>{
      const [rn,pn] = k.split("||");
      const tot = rt.get(rn) || 0;
      const share = tot ? cnt/tot : 0;
      const isSig = (share >= minShare) && (cnt >= minTotal);
      sig.set(k, isSig);

      if (!pilotSetByRoute.has(rn)) pilotSetByRoute.set(rn, new Set());
      pilotSetByRoute.get(rn).add(pn);

      topShare.set(rn, Math.max(topShare.get(rn) || 0, share));
    });

    const pilotsTotal = new Map();
    pilotSetByRoute.forEach((s, rn)=>pilotsTotal.set(rn, s.size));

    const pilotsSig = new Map();
    sig.forEach((isSig, k)=>{
      if (!isSig) return;
      const [rn,_] = k.split("||");
      pilotsSig.set(rn, (pilotsSig.get(rn) || 0) + 1);
    });

    // bucket: route_norm + fecha + horario + piloto => count
    const bucket = new Map();
    rows.forEach((r)=>{
      const rn = r.ruta_norm || "";
      const f = r.fecha || "";
      const h = (r.horario_turno || "").trim().toLowerCase();
      const pn = (r.piloto || "").trim().toLowerCase();
      if (!rn || !f || !pn) return;
      const k = `${rn}||${f}||${h}||${pn}`;
      bucket.set(k, (bucket.get(k) || 0) + 1);
    });

    // keep bucket_events >= bucketMin
    const bucket2 = [];
    bucket.forEach((cnt, k)=>{
      if (cnt < bucketMin) return;
      const [rn,f,h,pn] = k.split("||");
      const isSig = sig.get(`${rn}||${pn}`) || false;
      bucket2.push({rn,f,h,pn,isSig});
    });

    // per (rn,f,h): pilots_in_bucket + sig_pilots_in_bucket
    const bf = new Map();
    bucket2.forEach((b)=>{
      const k = `${b.rn}||${b.f}||${b.h}`;
      if (!bf.has(k)) bf.set(k, {pilots:new Set(), sig:0});
      const o = bf.get(k);
      o.pilots.add(b.pn);
    });

    // sig count: need unique significant pilots per bucket
    const bfs = new Map();
    bucket2.forEach((b)=>{
      if (!b.isSig) return;
      const k = `${b.rn}||${b.f}||${b.h}`;
      if (!bfs.has(k)) bfs.set(k, new Set());
      bfs.get(k).add(b.pn);
    });

    // per route aggregate
    const bucketsCount = new Map();
    const sumPilots = new Map();
    const pct2plus = new Map();      // counts
    const pct2plusSig = new Map();   // counts

    bf.forEach((o, k)=>{
      const [rn] = k.split("||");
      const pilotsN = o.pilots.size;
      const sigN = (bfs.get(k) ? bfs.get(k).size : 0);

      bucketsCount.set(rn, (bucketsCount.get(rn) || 0) + 1);
      sumPilots.set(rn, (sumPilots.get(rn) || 0) + pilotsN);
      pct2plus.set(rn, (pct2plus.get(rn) || 0) + (pilotsN >= 2 ? 1 : 0));
      pct2plusSig.set(rn, (pct2plusSig.get(rn) || 0) + (sigN >= 2 ? 1 : 0));
    });

    const pm = new Map();
    [...rt.keys()].forEach((rn)=>{
      const bN = bucketsCount.get(rn) || 0;
      pm.set(rn, {
        pilots_total: pilotsTotal.get(rn) || 0,
        pilots_significant: pilotsSig.get(rn) || 0,
        pilot_top_share: topShare.get(rn) || 0,
        buckets: bN || null,
        avg_pilots_per_bucket: bN ? (sumPilots.get(rn) || 0) / bN : null,
        pct_buckets_2plus_pilots: bN ? ((pct2plus.get(rn) || 0)/bN)*100 : null,
        pct_buckets_2plus_sig_pilots: bN ? ((pct2plusSig.get(rn) || 0)/bN)*100 : null
      });
    });

    // demanda stats por ruta_norm
    const demByRoute = new Map();
    rows.forEach((r)=>{
      const rn = r.ruta_norm || "";
      if (!rn || !isFinite(r.correlativos)) return;
      if (!demByRoute.has(rn)) demByRoute.set(rn, []);
      demByRoute.get(rn).push(Number(r.correlativos));
    });

    demByRoute.forEach((arr, rn)=>{
      const p50 = quantile(arr, 0.50);
      const p90 = quantile(arr, 0.90);
      const p95 = quantile(arr, 0.95);
      const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
      const mov = arr.length;
      const cur = pm.get(rn) || {};
      pm.set(rn, {...cur, demanda_p50:p50, demanda_p90:p90, demanda_p95:p95, demanda_mean:mean, mov});
    });

    return pm; // Map route_norm -> metrics
  };

  const mergeDinamo = (pm, joinMode) => {
    const keyPm = new Map();
    // pm está por ruta_norm; para modos con sede/jornada, usamos “lo disponible” => se pega por ruta y listo
    // (si querés hacerlo exacto por sede/jornada, necesitás pm a ese nivel; acá lo dejamos por ruta_norm para que sea estable)
    pm.forEach((v, rn)=> keyPm.set(rn, v));

    const merged = DASH.dinamo.map((d)=>{
      const rn = d.ruta_norm || "";
      const m = keyPm.get(rn) || {};
      return {...d, ...m};
    }).filter((r)=> r.precio_q !== null && isFinite(r.precio_q));

    return merged;
  };

  const renderPriceTab = (eventsFiltered) => {
    const joinMode = document.getElementById("p_join").value;
    const corrMethod = document.getElementById("p_corr").value;
    const minShare = Number(document.getElementById("p_min_share").value || 0.15);
    const minTotal = Number(document.getElementById("p_min_total").value || 30);
    const bucketMin = Number(document.getElementById("p_bucket_min").value || 2);

    const pm = buildPilotMetrics(eventsFiltered, minShare, minTotal, bucketMin);
    const merged = mergeDinamo(pm, joinMode);

    // --- 1) corr numérica vs precio
    const numericVars = ["km","tiempo_min","capacidad","dias","pilots_total","pilots_significant","pilot_top_share",
                         "avg_pilots_per_bucket","pct_buckets_2plus_sig_pilots","demanda_p95","demanda_mean"].filter((c)=> merged.some((r)=> isFinite(r[c])));

    const corrRows = numericVars.map((v)=>{
      const xs = [];
      const ys = [];
      merged.forEach((r)=>{
        const x = r[v];
        const y = r.precio_q;
        if (isFinite(x) && isFinite(y)) { xs.push(Number(x)); ys.push(Number(y)); }
      });
      const c = (corrMethod==="pearson") ? corrPearson(xs, ys) : corrSpearman(xs, ys);
      return {variable:v, [`corr_${corrMethod}`]:(c===null ? null : Number(c.toFixed(4))), n: xs.length};
    }).sort((a,b)=> Math.abs(b[`corr_${corrMethod}`]||0) - Math.abs(a[`corr_${corrMethod}`]||0));

    document.getElementById("p_corr_tbl").innerHTML = renderTable(corrRows, ["variable", `corr_${corrMethod}`, "n"]);

    // --- 2) categórica box
    const cat = document.getElementById("p_cat").value;
    const catKey = (cat==="Jornada") ? "jornada" : "bus_tipo";
    const byCat = new Map();
    merged.forEach((r)=>{
      const k = (r[catKey] || "—");
      const y = r.precio_q;
      if (!isFinite(y)) return;
      if (!byCat.has(k)) byCat.set(k, []);
      byCat.get(k).push(Number(y));
    });
    const cats = [...byCat.keys()].sort((a,b)=> String(a).localeCompare(String(b)));
    const traces = cats.map((k)=>({type:"box", name:k, y:byCat.get(k), boxpoints:"outliers"}));
    Plotly.react("p_cat_plot", traces, {
      title:`Precio por ${cat}`,
      paper_bgcolor:"rgba(0,0,0,0)",
      plot_bgcolor:"rgba(0,0,0,0)",
      font:{color:"#e9edf7"},
      margin:{l:60,r:20,t:60,b:80},
      xaxis:{title:cat},
      yaxis:{title:"Precio Q (Diario)"}
    }, {responsive:true});

    const stats = cats.map((k)=>{
      const arr = byCat.get(k).slice().sort((a,b)=>a-b);
      const n = arr.length;
      const mean = arr.reduce((a,b)=>a+b,0)/n;
      const med = arr[Math.floor(n/2)];
      const min = arr[0], max = arr[n-1];
      const std = Math.sqrt(arr.reduce((a,b)=>a+(b-mean)*(b-mean),0)/Math.max(1,n-1));
      return {[cat]:k, n, mean:mean.toFixed(2), median:med.toFixed(2), std:std.toFixed(2), min:min.toFixed(2), max:max.toFixed(2)};
    }).sort((a,b)=> Number(b.mean) - Number(a.mean));

    document.getElementById("p_cat_tbl").innerHTML = renderTable(stats, [cat,"n","mean","median","std","min","max"]);

    // --- 3) ΔR² waterfall (OLS con mathjs)
    const sampleMax = 3500;
    const data = (merged.length > sampleMax) ? merged.slice(0, sampleMax) : merged.slice();

    // features default
    const feats = ["dias","km","tiempo_min","demanda_p95","pilots_total"].filter((c)=> numericVars.includes(c));
    const blocks = feats.map((c)=>({name:c, mat: data.map((r)=>[ isFinite(r[c]) ? Number(r[c]) : 0 ])}));

    // dummies (jornada + bus) si existen
    const addDummies = (key, prefix) => {
      const vals = uniqSorted(data.map((r)=>(r[key] || "")));
      if (vals.length <= 1) return null;
      const base = vals[0];
      const cols = vals.slice(1);
      const mat = data.map((r)=>{
        const v = (r[key] || "");
        return cols.map((c)=> (v===c ? 1 : 0));
      });
      return {name:`${prefix} (dummies)`, mat};
    };

    const jd = addDummies("jornada","Jornada");
    const bd = addDummies("bus_tipo","Tipo de bus");
    if (jd) blocks.push(jd);
    if (bd) blocks.push(bd);

    const yAll = data.map((r)=>Number(r.precio_q));

    const fitR2 = (mats) => {
      const n = data.length;
      const X = [];
      for (let i=0;i<n;i++){
        const row = [1];
        mats.forEach((mat)=>{
          row.push(...mat[i]);
        });
        X.push(row);
      }
      const yv = yAll.map((v)=> (isFinite(v) ? v : 0));

      const Xm = math.matrix(X);
      const ym = math.matrix(yv);

      const Xt = math.transpose(Xm);
      const XtX = math.multiply(Xt, Xm);
      const Xty = math.multiply(Xt, ym);
      const beta = math.lusolve(XtX, Xty);

      const yhat = math.multiply(Xm, beta).toArray().map((v)=>Array.isArray(v)?v[0]:v);
      const ymean = yv.reduce((a,b)=>a+b,0)/yv.length;

      let ssRes = 0, ssTot = 0;
      for (let i=0;i<yv.length;i++){
        ssRes += (yv[i]-yhat[i])*(yv[i]-yhat[i]);
        ssTot += (yv[i]-ymean)*(yv[i]-ymean);
      }
      if (!isFinite(ssTot) || ssTot===0) return null;
      return 1 - (ssRes/ssTot);
    };

    const r2Vals = [];
    const deltas = [];
    const matsSoFar = [];
    let prev = 0;

    blocks.forEach((b)=>{
      matsSoFar.push(b.mat);
      const r2 = fitR2(matsSoFar);
      const r2v = (r2===null ? 0 : r2);
      const d = r2v - prev;
      r2Vals.push(r2v);
      deltas.push(d);
      prev = r2v;
    });

    const totalR2 = prev;

    const wfTrace = [{
      type:"waterfall",
      x: ["R² inicial"].concat(blocks.map((b)=>b.name)).concat(["R² total"]),
      y: [0].concat(deltas).concat([0]),
      measure: ["absolute"].concat(deltas.map(()=> "relative")).concat(["total"])
    }];

    Plotly.react("p_r2_plot", wfTrace, {
      title:`Cascada (ΔR²) — R² total ≈ ${totalR2.toFixed(3)}`,
      paper_bgcolor:"rgba(0,0,0,0)",
      plot_bgcolor:"rgba(0,0,0,0)",
      font:{color:"#e9edf7"},
      margin:{l:60,r:20,t:60,b:80},
      xaxis:{title:"Variable / Bloque"},
      yaxis:{title:"ΔR²"}
    }, {responsive:true});

    const r2Tbl = blocks.map((b,i)=>({bloque:b.name, delta_R2:Number(deltas[i].toFixed(4)), R2_acumulado:Number(r2Vals[i].toFixed(4))}))
      .sort((a,b)=> Math.abs(b.delta_R2) - Math.abs(a.delta_R2));

    document.getElementById("p_r2_tbl").innerHTML = renderTable(r2Tbl, ["bloque","delta_R2","R2_acumulado"]);
  };

  // ============================
  // Wiring / bootstrap
  // ============================
  const initUI = () => {
    // tabs
    document.querySelectorAll(".tabbtn").forEach((b)=>{
      b.addEventListener("click", ()=>{
        document.querySelectorAll(".tabbtn").forEach((x)=>x.classList.remove("active"));
        document.querySelectorAll(".tab").forEach((x)=>x.classList.remove("active"));
        b.classList.add("active");
        document.getElementById(b.dataset.tab).classList.add("active");
      });
    });

    // populate filter options from raw events
    const all = DASH.events;

    const minD = uniqSorted(all.map((r)=>parseDateOnly(r.inicio_dt))).slice(0,1)[0] || null;
    const maxD = uniqSorted(all.map((r)=>parseDateOnly(r.inicio_dt))).slice(-1)[0] || null;

    els.f_date0.value = minD || "";
    els.f_date1.value = maxD || "";

    setOptions(els.f_corr, uniqSorted(all.map((r)=>r.corredor_desc)));
    setOptions(els.f_sede, uniqSorted(all.map((r)=>r.sede)));
    setOptions(els.f_ruta, uniqSorted(all.map((r)=>r.ruta)), {includeAll:true, allLabel:"Todas"});
    setOptions(els.f_jornada, uniqSorted(all.map((r)=>r.jornada)));
    setOptions(els.f_piloto, uniqSorted(all.map((r)=>r.piloto)));
    setOptions(els.f_ie, uniqSorted(all.map((r)=>r.ie_norm)));

    // local selects
    setOptions(document.getElementById("box_ruta"), uniqSorted(all.map((r)=>r.ruta)), {includeAll:true, allLabel:"Todas"});
    setOptions(document.getElementById("box_jornadas"), uniqSorted(all.map((r)=>r.jornada)));
    setOptions(document.getElementById("wf_jornadas"), uniqSorted(all.map((r)=>r.jornada)));
    setOptions(document.getElementById("svc_ruta"), uniqSorted(all.map((r)=>r.ruta)), {includeAll:true, allLabel:"Todas"});

    // load initial from streamlit if viene
    const ini = DASH.init || {};
    if (ini.date_range && Array.isArray(ini.date_range) && ini.date_range.length===2) {
      els.f_date0.value = ini.date_range[0] || els.f_date0.value;
      els.f_date1.value = ini.date_range[1] || els.f_date1.value;
    }
    if (ini.corredor_sel && ini.corredor_sel.length) {
      [...els.f_corr.options].forEach((o)=>o.selected = ini.corredor_sel.includes(o.value));
    }
    if (ini.sede_sel && ini.sede_sel.length) {
      [...els.f_sede.options].forEach((o)=>o.selected = ini.sede_sel.includes(o.value));
    }
    if (ini.ruta_sel && ini.ruta_sel.length===1) {
      els.f_ruta.value = ini.ruta_sel[0];
    } else {
      els.f_ruta.value = "__ALL__";
    }

    // ✅ jornada multi default viene desde Streamlit en ini.jornada_sel
    if (ini.jornada_sel && ini.jornada_sel.length) {
      [...els.f_jornada.options].forEach((o)=>o.selected = ini.jornada_sel.includes(o.value));
    }

    if (ini.piloto_sel && ini.piloto_sel.length) {
      [...els.f_piloto.options].forEach((o)=>o.selected = ini.piloto_sel.includes(o.value));
    }
    if (ini.ie_sel && ini.ie_sel.length) {
      [...els.f_ie.options].forEach((o)=>o.selected = ini.ie_sel.includes(o.value));
    }

    const onGlobalChange = () => {
      state.global.date0 = els.f_date0.value || null;
      state.global.date1 = els.f_date1.value || null;
      state.global.corr = getMulti(els.f_corr);
      state.global.sede = getMulti(els.f_sede);
      state.global.ruta = els.f_ruta.value || "__ALL__";
      state.global.jornada = getMulti(els.f_jornada);
      state.global.piloto = getMulti(els.f_piloto);
      state.global.ie = getMulti(els.f_ie);

      const filtered = applyGlobal(DASH.events);

      // render all
      renderKPI(filtered);
      renderBoxplot(filtered);
      renderWaterfalls(filtered);
      renderTimeseries(filtered);
      renderService(filtered);
      renderPriceTab(filtered);
    };

    // bind all relevant controls
    [els.f_date0, els.f_date1, els.f_corr, els.f_sede, els.f_ruta, els.f_jornada, els.f_piloto, els.f_ie].forEach((x)=>x.addEventListener("change", onGlobalChange));
    ["box_group","box_metric","box_ruta","box_jornadas","box_dir"].forEach((id)=>document.getElementById(id).addEventListener("change", onGlobalChange));
    ["wf_group","wf_measure","wf_max","wf_dir","wf_jornadas"].forEach((id)=>document.getElementById(id).addEventListener("change", onGlobalChange));
    ["ts_gran","ts_metric","ts_split"].forEach((id)=>document.getElementById(id).addEventListener("change", onGlobalChange));
    ["svc_ruta","svc_dir","svc_unit","svc_levels"].forEach((id)=>document.getElementById(id).addEventListener("change", onGlobalChange));
    ["p_join","p_corr","p_min_share","p_min_total","p_bucket_min","p_cat"].forEach((id)=>document.getElementById(id).addEventListener("change", onGlobalChange));

    onGlobalChange();
  };

  initUI();
</script>
</body>
</html>
"""


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

        # ✅ Jornada multi, DEFAULT: Turno + Ordinaria/Ordinario (según exista)
        jornada_sel = None
        if "jornada" in events_df.columns:
            jornadas_vals = sorted([v for v in events_df["jornada"].dropna().unique().tolist() if str(v).strip() != ""])
            if jornadas_vals:
                default_j = [v for v in jornadas_vals if norm_jornada(v) in {"turno", "ordinario"}]
                jornada_sel = st.multiselect("Jornada", jornadas_vals, default=default_j)

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
        cap_test = st.slider(
            "Probar capacidad C",
            min_value=cmin,
            max_value=max(cmin + 1, cmax),
            value=min(max(cmin + 1, cmax), int(np.ceil(s.quantile(0.95)))),
            step=1
        )

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

    # Spearman/Kendall require scipy in pandas. If it is missing, use Pearson.
    corr_method_effective = corr_method
    if corr_method_effective in {"spearman", "kendall"}:
        try:
            import scipy  # noqa: F401
        except Exception:
            corr_method_effective = "pearson"
            st.warning("No se encontró scipy en el entorno; se usará correlación Pearson.")

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
            corr_val = tmp[v].corr(tmp["Precio Q (Diario)"], method=corr_method_effective)
            corr_rows.append({"variable": v, f"corr_{corr_method_effective}": float(corr_val), "n": int(len(tmp))})

    if corr_rows:
        corr_tbl = pd.DataFrame(corr_rows)
        corr_tbl["abs"] = corr_tbl[f"corr_{corr_method_effective}"].abs()
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
                corr_val = tmp[c].corr(tmp["Precio Q (Diario)"], method=corr_method_effective)
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

    with st.sidebar:
        st.markdown("---")
        st.subheader("Exportar (HTML dinámico)")

        export_scope = st.selectbox(
            "Qué datos meter en el HTML",
            ["Todo (permite filtrar después)", "Solo lo filtrado (archivo más liviano)"],
            index=0
        )

        if export_scope.startswith("Todo"):
            ev_for_html = events
        else:
            ev_for_html = events_f

        # ✅ FIX: date_range debe ir como strings YYYY-MM-DD
        date_range_for_json = None
        if fs.date_range and isinstance(fs.date_range, tuple) and len(fs.date_range) == 2:
            date_range_for_json = [
                (d.isoformat() if hasattr(d, "isoformat") else str(d)) for d in fs.date_range
            ]

        html_bytes = build_dashboard_html(
            ev_for_html,
            dinamo,
            initial_filters={
                "date_range": date_range_for_json,
                "corredor_sel": fs.corredor_sel,
                "sede_sel": fs.sede_sel,
                "ruta_sel": fs.ruta_sel,
                "jornada_sel": fs.jornada_sel,
                "piloto_sel": fs.piloto_sel,
                "ie_sel": fs.ie_sel,
            }
        )

        st.download_button(
            "Descargar dashboard.html",
            data=html_bytes,
            file_name="dashboard.html",
            mime="text/html"
        )

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