# 01_prepare_data_export.py
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# Helpers
# =========================
def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_num(s):
    """
    Convierte a número manejando:
    - comas
    - símbolos como 'Q'
    - espacios
    """
    if isinstance(s, pd.Series):
        x = s.astype("string")
    else:
        x = pd.Series(s, dtype="string")
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(x, errors="coerce")


def drop_by_patterns(df: pd.DataFrame, patterns):
    """
    Elimina columnas si el nombre matchea cualquiera de los patterns (regex, case-insensitive).
    """
    cols = list(df.columns)
    drop_cols = []
    for c in cols:
        for pat in patterns:
            if re.search(pat, str(c), flags=re.IGNORECASE):
                drop_cols.append(c)
                break
    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df, drop_cols


def parse_datetime_col(df: pd.DataFrame, col: str, out_col: str):
    """
    Parsea columnas tipo: 6/21/24 10:19:59 (mm/dd/yy HH:MM:SS)
    """
    if col not in df.columns:
        df[out_col] = pd.NaT
        return df

    df[out_col] = pd.to_datetime(df[col], format="%m/%d/%y %H:%M:%S", errors="coerce")
    return df


def ensure_turn_from_hour(df: pd.DataFrame, dt_col: str, turn_col: str):
    """
    Si no existe Horario de turno, lo crea desde la hora:
      [0,8)=Amanecer, [8,16)=Dia, [16,24)=Tarde
    """
    if turn_col in df.columns and df[turn_col].notna().any():
        return df

    if dt_col not in df.columns:
        df[turn_col] = pd.NA
        return df

    hour = pd.to_datetime(df[dt_col], errors="coerce").dt.hour
    bins = [0, 8, 16, 24]
    labels = ["Amanecer", "Dia", "Tarde"]
    df[turn_col] = pd.cut(hour, bins=bins, labels=labels, right=False, include_lowest=True).astype("string")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocupa", required=True, help="Path a ocupa.csv")
    ap.add_argument("--resumen", required=True, help="Path a reporte_viajes_resumen2.xlsx")
    ap.add_argument("--sheet", default="Dinamo", help="Nombre de hoja en el excel (default: Dinamo)")
    ap.add_argument("--out", default="ocupa_clean.csv", help="CSV de salida (default: ocupa_clean.csv)")
    ap.add_argument("--encoding", default="latin-1", help="Encoding del csv (default: latin-1)")
    args = ap.parse_args()

    ocupa_path = Path(args.ocupa)
    resumen_path = Path(args.resumen)
    out_path = Path(args.out)

    # =========================
    # Load
    # =========================
    df = pd.read_csv(ocupa_path, encoding=args.encoding)
    df = strip_columns(df)

    df_resumen2 = pd.read_excel(resumen_path, sheet_name=args.sheet)
    df_resumen2 = strip_columns(df_resumen2)
    # caso típico: "Ruta " -> "Ruta"
    if "Ruta " in df_resumen2.columns and "Ruta" not in df_resumen2.columns:
        df_resumen2 = df_resumen2.rename(columns={"Ruta ": "Ruta"})

    # =========================
    # Merge por Ruta
    # =========================
    if "Ruta" not in df.columns:
        raise ValueError("No existe columna 'Ruta' en ocupa.csv")

    if "Ruta" not in df_resumen2.columns:
        raise ValueError("No existe columna 'Ruta' en la hoja Dinamo del excel")

    df = pd.merge(df, df_resumen2, on="Ruta", how="left")

    # =========================
    # Datetimes
    # =========================
    # Hora de inicio viene así: 6/21/24 10:19:59
    # Hay una columna "mes", "dia" y otra "año". Probar esas y si no vienen usar Hora de inicio. Lo mismo para finalización. La idea es tener columnas tipo datetime para poder sacar derivados como fecha, hora, turno, etc.
    df["Momento de inicio"] = df.apply(lambda row: row["Hora de inicio"] if pd.notna(row.get("Hora de inicio", np.nan)) else f"{row.get('mes', '')}/{row.get('dia', '')}/{row.get('año', '')} 00:00:00", axis=1)
    df["Dia inicio (Dia de la semana)"] = pd.to_datetime(df["Momento de inicio"], errors="coerce").dt.day_name(locale="es_ES")
    df = parse_datetime_col(df, "Hora de finalización", "Momento de finalización")

    # derivados (opcionales)
    df["Fecha al iniciar"] = pd.to_datetime(df["Momento de inicio"], errors="coerce").dt.date
    df["Hora de inicio (hora)"] = pd.to_datetime(df["Momento de inicio"], errors="coerce").dt.hour
    df["Hora de inicio (minuto)"] = pd.to_datetime(df["Momento de inicio"], errors="coerce").dt.minute

    df["Fecha al finalizar"] = pd.to_datetime(df["Momento de finalización"], errors="coerce").dt.date
    df["Hora de finalización (hora)"] = pd.to_datetime(df["Momento de finalización"], errors="coerce").dt.hour
    df["Hora de finalización (minuto)"] = pd.to_datetime(df["Momento de finalización"], errors="coerce").dt.minute

    # Asegurar turno si no viene
    TURN_COL = "Horario de turno"
    df = ensure_turn_from_hour(df, "Momento de inicio", TURN_COL)

    # =========================
    # Pasajeros desde Correlativos
    # =========================
    correlativo_cols = [c for c in df.columns if str(c).startswith("Correlativo")]
    desea_cols = [c for c in df.columns if str(c).startswith("Desea agregar otro usuario")]

    if correlativo_cols:
        df["Pasajeros"] = df[correlativo_cols].notna().sum(axis=1)
    else:
        # fallback si ya viene una columna Pasajeros
        if "Pasajeros" not in df.columns:
            df["Pasajeros"] = np.nan

    # drop correlativos + desea
    df = df.drop(columns=correlativo_cols + desea_cols, errors="ignore")

    # =========================
    # Limpieza numérica (por si vienen como texto con Q, comas, etc.)
    # =========================
    for c in ["Pasajeros", "Capacidad", "# Días", "Precio Q (Diario)", "Precio Uber", "Km", "Tiempo (minutos)"]:
        if c in df.columns:
            df[c] = safe_num(df[c])

    # =========================
    # Drop columnas sensibles / irrelevantes
    # =========================
    # Pediste específicamente: piloto, destino, sede. También saco nombres y cualquier cosa obvia de persona.
    patterns = [
        r"\bpiloto\b",
        r"\bdestino\b",
        r"\bsede\b",
        r"\bnombre\b",
        r"correo|email|tel[eé]fono|celular",
        r"cui|dpi|nit",          # por si por error viene algo identificable
        r"ingreso\s*/\s*egreso|ingreso/egreso",
    ]

    df, dropped = drop_by_patterns(df, patterns)

    # También: columnas que no sirven casi nunca en el análisis final
    # (las dejamos opcionalmente, pero por defecto las quitamos)
    drop_exact = [
        "Hora de inicio",
        "Hora de finalización",
        "Ingreso/Egreso",
        "Ingreso / Egreso",
        "Nombre de Piloto de cobertura",
        "Nombre",
    ]
    df = df.drop(columns=[c for c in drop_exact if c in df.columns], errors="ignore")

    # =========================
    # Selección final (keeplist) – conservador
    # =========================
    keep = [
        "Ruta",
        "Momento de inicio",
        "Momento de finalización",
        "Fecha al iniciar",
        "Hora de inicio (hora)",
        "Hora de inicio (minuto)",
        "Dia inicio (Dia de la semana)",
        TURN_COL,
        "Pasajeros",
        # de resumen (si existen)
        "Capacidad",
        "# Días",
        "Precio Q (Diario)",
        "Precio Uber",
        "Km",
        "Tiempo (minutos)",
        # a veces útil para ordinario
        "Jornada",
        "Jornada_x",
        "Jornada_y",
        "Tipo de bus",
    ]
    keep = [c for c in keep if c in df.columns]
    df_out = df[keep].copy()

    # Quitar filas sin lo mínimo
    df_out["Momento de inicio"] = pd.to_datetime(df_out["Momento de inicio"], errors="coerce")
    df_out = df_out.dropna(subset=["Ruta", "Momento de inicio", "Pasajeros"], how="any")

    # Por ultimo comparar el numero de Pasajeros con la capacidad y si es mayor, bajarlo a la capacidad (asumiendo que es un error de digitación)
    if "Capacidad" in df_out.columns:
        df_out["Pasajeros"] = df_out.apply(lambda row: min(row["Pasajeros"], row["Capacidad"]) if pd.notna(row["Pasajeros"]) and pd.notna(row["Capacidad"]) else row["Pasajeros"], axis=1)

    # Export
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("✅ Export listo")
    print(f"  - Output: {out_path.resolve()}")
    print(f"  - Filas: {len(df_out):,}")
    print(f"  - Columnas: {len(df_out.columns)}")
    if dropped:
        print("🧹 Columnas eliminadas por patrones (sensibles/irrelevantes):")
        for c in dropped:
            print("  -", c)


if __name__ == "__main__":
    main()