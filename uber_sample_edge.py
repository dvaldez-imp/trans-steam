import os
import re
import csv
import time
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains



UBER_URL = "https://www.uber.com/global/en/price-estimate/"

# Soporta "68,00 GTQ" y también "GTQ 68,00"
PRICE_RE = re.compile(
    r"(?i)^(?=.*\b(?:Q|GTQ|USD)\b)\s*"
    r"(?:\b(?:Q|GTQ|USD)\b\s*)?"
    r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?"
    r"(?:\s*(?:-|–)\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)?"
    r"\s*(?:\b(?:Q|GTQ|USD)\b)?\s*$"
)

# Solo para "ya hay precios en pantalla"
CURRENCY_HINT_RE = re.compile(r"\b(Q|GTQ|USD)\b", re.IGNORECASE)

# ==========
# 👇 Pegás tu tabla aquí una sola vez (ya te la dejo metida).
# Formato TSV: columnas separadas por TAB
# ==========
ROUTES_TSV = """Sede\tRuta\tJornada\tCapacidad\tDestino
San Miguel\tSM01\tOrdinario\t48\tParque Sanarate
San Miguel\tSM02\tOrdinario\t45\tParque Guastatoya
San Miguel\tSM03\tTurno\t42\tPta. La Pedrera
San Miguel\tSM04\tOrdinario\t44\tPlaza La Florida
San Miguel\tSM05\tOrdinario\t46\tParque Guastatoya
San Miguel\tSM06\tTurno\t23\tPta. La Pedrera
San Gabriel\tSG01\tOrdinario\t42\tPta. La Pedrera
San Gabriel\tSG02\tOrdinario\t18\tMiraflores
San Gabriel\tSG03\tTurno\t12\tEl Parador
San Gabriel\tSG04\tOrdinario\t18\tEstadio San Juan Sacatepéquez
San Gabriel\tSG05\tOrdinario\t18\tPta. La Pedrera
San Gabriel\tSG06\tTurno\t18\tEstadio San Juan Sacatepéquez
Sacos Atlántico\tSA01\tTurno\t12\tTeculután
Sacos Atlántico\tSA02\tTurno\t12\tZacapa
La Pedrera\tLP01\tOrdinario\t14\tSan Pedro Ayampuc
"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_csv_rows(file_path: str, rows: list[dict]):
    exists = Path(file_path).exists()
    header = [
        "ts",
        "sede",
        "ruta",
        "jornada",
        "capacidad",
        "destino",
        "pickup_query",
        "dropoff_query",
        "product",
        "price_text",
        "source",
        "page_url",

        # Agregar fecha y hora
        "date",
        "time",
    ]
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        for r in rows:
            r["date"] = now_iso()[:10]
            r["time"] = now_iso()[11:19]
            w.writerow([r.get(k, "") for k in header])


def parse_routes_from_tsv(tsv_text: str) -> list[dict]:
    lines = [ln.strip() for ln in tsv_text.splitlines() if ln.strip()]
    if not lines:
        return []

    header = [h.strip() for h in lines[0].split("\t")]
    out = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split("\t")]
        if len(parts) != len(header):
            # si viene medio chueco, lo saltamos
            continue
        row = dict(zip(header, parts))
        # normalizaciones
        row["Capacidad"] = int(row["Capacidad"])
        row["Sede"] = " ".join(row["Sede"].split())  # colapsa doble espacio
        row["Destino"] = " ".join(row["Destino"].split())
        out.append(row)
    return out


def launch_edge_driver() -> webdriver.Edge:
    """
    A) ATTACH a Edge real:
       ATTACH_DEBUGGER=1
       DEBUGGER_ADDR=127.0.0.1:9222
       DETACH=1
       (En remote debugging hay comandos no soportados; normal.) :contentReference[oaicite:1]{index=1}

    B) Normal:
       EDGE_USER_DATA_DIR / EDGE_PROFILE_DIR (opcional)
    """
    opts = EdgeOptions()

    # Para SPAs pesadas, eager ayuda a que no se quede pegado esperando "complete". :contentReference[oaicite:2]{index=2}
    opts.page_load_strategy = os.getenv("PAGE_LOAD_STRATEGY", "eager")  # normal|eager|none

    if os.getenv("ATTACH_DEBUGGER", "").strip() == "1":
        addr = os.getenv("DEBUGGER_ADDR", "127.0.0.1:9222")
        opts.add_experimental_option("debuggerAddress", addr)
        driver = webdriver.Edge(options=opts)
        return driver

    user_data = os.getenv(
        "EDGE_USER_DATA_DIR",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data"),
    )
    profile_dir = os.getenv("EDGE_PROFILE_DIR", "Default")

    opts.add_argument(f'--user-data-dir={user_data}')
    opts.add_argument(f'--profile-directory={profile_dir}')
    opts.add_argument("--lang=es-GT")
    opts.add_argument("--disable-notifications")
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1280,900")

    driver = webdriver.Edge(options=opts)
    driver.set_page_load_timeout(int(os.getenv("PAGE_LOAD_TIMEOUT", "25")))
    return driver


def safe_get(driver, url: str):
    try:
        driver.get(url)
    except TimeoutException:
        pass


def dismiss_cookies(driver):
    for xp in [
        "//button[contains(., 'Reject')]",
        "//button[contains(., 'Accept')]",
        "//button[contains(., 'Got it')]",
        "//button[contains(., 'Opt out')]",
        "//button[contains(., 'Rechazar')]",
        "//button[contains(., 'Aceptar')]",
    ]:
        try:
            WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.XPATH, xp))).click()
            time.sleep(0.2)
            break
        except Exception:
            pass


def find_inputs(wait: WebDriverWait):
    pickup_css = (
        'input[aria-label*="pickup" i], '
        'input[placeholder*="pickup" i], '
        'input[placeholder*="Where from" i], '
        'input[aria-label*="Enter a pickup" i]'
    )
    dropoff_css = (
        'input[aria-label*="where to" i], '
        'input[aria-label*="destination" i], '
        'input[placeholder*="destination" i], '
        'input[placeholder*="Where to" i], '
        'input[aria-label*="dropoff" i]'
    )
    pickup_el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, pickup_css)))
    dropoff_el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, dropoff_css)))
    return pickup_el, dropoff_el


def type_and_select_first_option(driver, el, value: str, tries: int =1):
    """
    Selecciona la primera sugerencia REAL del autocomplete evitando Saved Places.
    No depende de aria-activedescendant (Uber a veces no lo usa en el input).
    """

    bad_keywords = (
        "saved places", "lugares guardados", "ubicaciones guardadas",
        "home", "casa", "work", "trabajo",
        "add", "agregar", "manage", "administrar",
    )

    def norm(s: str) -> str:
        return (s or "").replace("\u00a0", " ").strip()

    def is_bad(t: str) -> bool:
        tl = (t or "").lower()
        return any(k in tl for k in bad_keywords)

    def js_click(elem):
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", elem)
        driver.execute_script("arguments[0].click();", elem)

    def clear_and_type():
        # Cerrar cualquier dropdown raro previo
        try:
            el.send_keys(Keys.ESCAPE)
        except Exception:
            pass

        # Foco + limpiar (React/SPAs a veces necesitan evento input)
        driver.execute_script("arguments[0].focus();", el)
        el.click()
        el.send_keys(Keys.CONTROL, "a")
        el.send_keys(Keys.BACKSPACE)

        # Type con mini delay para que autocomplete reaccione
        for ch in value:
            el.send_keys(ch)
            time.sleep(0.06)

        time.sleep(2)

    def get_listbox():
        # 1) Preferir el listbox asociado al input (aria-controls / aria-owns)
        lb_id = el.get_attribute("aria-controls") or el.get_attribute("aria-owns")
        if lb_id:
            try:
                lb = driver.find_element(By.ID, lb_id)
                if lb.is_displayed():
                    return lb
            except Exception:
                pass

        # 2) Fallback: primer listbox visible
        lbs = driver.find_elements(By.CSS_SELECTOR, '[role="listbox"]')
        for lb in lbs:
            try:
                if lb.is_displayed():
                    return lb
            except Exception:
                pass
        return None

    def get_candidates(lb):
        # OJO: en Uber no siempre son role=option. Probamos varios.
        sels = [
            '[role="option"]',
            'button',
            'li',
            'div[tabindex]',
            'a',
        ]
        cands = []
        for sel in sels:
            try:
                cands.extend(lb.find_elements(By.CSS_SELECTOR, sel))
            except Exception:
                pass

        # dedupe por id (o por objeto)
        seen = set()
        uniq = []
        for c in cands:
            try:
                key = c.id
            except Exception:
                key = id(c)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        return uniq

    def pick_first_real_option(lb):
        cands = get_candidates(lb)

        cleaned = []
        for c in cands:
            try:
                if not c.is_displayed():
                    continue
            except Exception:
                continue

            t = norm(c.get_attribute("aria-label")) or norm(c.text)
            if not t:
                continue
            if is_bad(t):
                continue
            # a veces hay headers tipo "Saved places" sin ser exacto → ya los filtra keywords
            cleaned.append((c, t))

        if not cleaned:
            return None

        # Preferir opciones “reales”: suelen tener coma o Guatemala
        for c, t in cleaned:
            tl = t.lower()
            if "guatemala" in tl or "," in t:
                return c

        # Si ninguna tiene Guatemala/coma, agarrá la primera válida
        return cleaned[0][0]

    for _ in range(tries):
        clear_and_type()

        # Esperar a que aparezca el dropdown
        lb = None
        t0 = time.time()
        while time.time() - t0 < 8:
            lb = get_listbox()
            if lb:
                # si ya tiene opciones visibles, seguimos
                break
            time.sleep(0.1)

        if not lb:
            # fallback: Enter por si la UI acepta el texto
            el.send_keys(Keys.ENTER)
            time.sleep(0.2)
            if norm(el.get_attribute("value")):
                return
            continue

        # Elegir primera sugerencia REAL (no Saved Places)
        opt = pick_first_real_option(lb)
        time.sleep(0.2)

    # Último recurso
    el.send_keys(Keys.ENTER)
    return


def click_element_by_partial_text(driver, partial_text):
    """
    Funcion para con JavaScript hacer click en un elemento con parte de su texto
    
    :param driver: El driver de Selenium
    :param partial_text: Parte del texto del elemento al que se le quiere hacer click
    """
    try:
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, f"//*[contains(text(), '{partial_text}')]"))
        )
        driver.execute_script("arguments[0].click();", element)
    except Exception as e:
        print(f"Error al hacer click en el elemento con texto parcial '{partial_text}': {e}")


def click_see_prices(driver, wait: WebDriverWait, pickup_el, dropoff_el):
    """
    Intenta disparar el cálculo de precios:
    - Scroll + JS click al botón/link
    - Fallback: Enter en el campo destino
    - Fallback2: submit del form (si existe)
    """

    def js_click(elem):
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", elem)
        driver.execute_script("arguments[0].click();", elem)

    # 1) Asegurarse que ambos inputs tienen valor
    def has_values(_):
        pv = (pickup_el.get_attribute("value") or "").strip()
        dv = (dropoff_el.get_attribute("value") or "").strip()
        return len(pv) > 0 and len(dv) > 0

    try:
        wait.until(has_values)
    except Exception:
        pass

    # 2) Intentar click por texto (link o botón)
    xpaths = [
        "//a[contains(., 'See prices') or contains(., 'Ver precios')]",
        "//button[contains(., 'See prices') or contains(., 'Ver precios')]",
        # a veces solo hay un submit “hidden” o un botón sin texto visible
        "//*[@role='button' and (contains(., 'See prices') or contains(., 'Ver precios'))]",
    ]

    for xp in xpaths:
        try:
            el = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.XPATH, xp)))
            # Esperar que sea visible (no solo presente)
            WebDriverWait(driver, 3).until(lambda d: el.is_displayed())
            try:
                js_click(el)          # ✅ lo más efectivo contra overlays
            except Exception:
                el.click()
            return
        except Exception:
            pass

    # 3) Fallback: Enter en el destino (Uber muchas veces dispara así)
    try:
        dropoff_el.send_keys(Keys.ENTER)
        return
    except Exception:
        pass

    # 4) Fallback2: submit form si existe
    try:
        form = driver.find_element(By.CSS_SELECTOR, "form")
        driver.execute_script("arguments[0].submit();", form)
    except Exception:
        pass

def extract_products_from_dom(driver):
    """
    Extrae (producto, precio) desde los tiles de productos:
    [data-testid="product_selector.list_item"]
    """
    items = driver.find_elements(By.CSS_SELECTOR, '[data-testid="product_selector.list_item"]')
    results = []
    seen = set()

    for li in items:
        name = None
        try:
            img = li.find_element(By.CSS_SELECTOR, "img[alt]")
            alt = (img.get_attribute("alt") or "").strip()
            if alt:
                name = alt
        except Exception:
            pass

        price = None
        ps = li.find_elements(By.CSS_SELECTOR, "p")
        for p in ps:
            t = (p.text or "").replace("\u00a0", " ").strip()
            if t and PRICE_RE.match(t):
                price = t
                break

        if not name:
            for p in ps:
                t = (p.text or "").replace("\u00a0", " ").strip()
                if not t:
                    continue
                if PRICE_RE.match(t):
                    continue
                tl = t.lower()
                if tl.startswith("a "):  # "A 2 minutos • ..."
                    continue
                if len(t) > 60:
                    continue
                name = t.splitlines()[0]
                break

        if price:
            key = (name or "Unknown", price)
            if key not in seen:
                seen.add(key)
                results.append(key)

    return results


def write_debug(driver, tag: str):
    tag = re.sub(r"[^a-zA-Z0-9_\-]+", "_", tag)[:80]
    try:
        driver.save_screenshot(f"debug_{tag}.png")
    except Exception:
        pass
    try:
        with open(f"debug_{tag}.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
    except Exception:
        pass


def build_pickup_query(sede: str) -> str:
    # 👇 lo que pediste: "Cementos Progreso" + sede
    suffix = os.getenv("LOCATION_SUFFIX", ", Guatemala")
    return f"Cementos Progreso {sede}{suffix}"


def build_dropoff_query(destino: str) -> str:
    suffix = os.getenv("LOCATION_SUFFIX", ", Guatemala")
    return f"{destino}{suffix}"


def run_one_route(driver, wait: WebDriverWait, route: dict) -> list[dict]:
    sede = route["Sede"]
    ruta = route["Ruta"]
    jornada = route["Jornada"]
    capacidad = route["Capacidad"]
    destino = route["Destino"]

    pickup = build_pickup_query(sede)
    dropoff = build_dropoff_query(destino)

    # asegurar que estamos en el estimator
    safe_get(driver, UBER_URL)
    dismiss_cookies(driver)

    # a veces ayuda forzar top
    try:
        driver.execute_script("window.scrollTo(0, 0);")
    except Exception:
        pass

    pickup_el, dropoff_el = find_inputs(wait)
    type_and_select_first_option(driver, pickup_el, pickup)
    type_and_select_first_option(driver, dropoff_el, dropoff)
    time.sleep(2)
    click_element_by_partial_text(driver, "See prices")

    # esperar lista de productos
    try:
        WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="product_selector.list_item"]'))
        )
    except Exception:
        # fallback: esperar que haya moneda en pantalla
        try:
            wait.until(lambda d: CURRENCY_HINT_RE.search(d.find_element(By.TAG_NAME, "body").text or "") is not None)
        except Exception:
            pass

    time.sleep(1.5)

    pairs = extract_products_from_dom(driver)

    # filtro opcional por producto (ej: uberx)
    product_filter = (os.getenv("PRODUCT_FILTER", "") or "").strip().lower()
    if product_filter:
        pairs = [(n, p) for (n, p) in pairs if product_filter in (n or "").lower()]

    ts = now_iso()
    url = driver.current_url

    rows = []
    if pairs:
        for name, price in pairs:
            rows.append({
                "ts": ts,
                "sede": sede,
                "ruta": ruta,
                "jornada": jornada,
                "capacidad": capacidad,
                "destino": destino,
                "pickup_query": pickup,
                "dropoff_query": dropoff,
                "product": name,
                "price_text": price,
                "source": "dom_tiles",
                "page_url": url,
            })
    else:
        rows.append({
            "ts": ts,
            "sede": sede,
            "ruta": ruta,
            "jornada": jornada,
            "capacidad": capacidad,
            "destino": destino,
            "pickup_query": pickup,
            "dropoff_query": dropoff,
            "product": "NOT_FOUND",
            "price_text": "NOT_FOUND",
            "source": "none",
            "page_url": url,
        })

    return rows


def main():
    out_file = os.getenv("OUT", "samples_all_routes.csv")
    sleep_between = float(os.getenv("SLEEP_BETWEEN", "1.0"))

    routes = parse_routes_from_tsv(ROUTES_TSV)
    if not routes:
        raise SystemExit("No hay rutas para procesar.")

    driver = launch_edge_driver()
    wait = WebDriverWait(driver, 30)

    try:
        all_rows = []
        for idx, r in enumerate(routes, start=1):
            tag = f"{r['Ruta']}_{r['Sede']}".replace(" ", "_")
            print(f"[{idx}/{len(routes)}] {r['Ruta']} | {r['Sede']} -> {r['Destino']}")

            try:
                rows = run_one_route(driver, wait, r)
                append_csv_rows(out_file, rows)
                print(f"   -> guardé {len(rows)} precios")
            except Exception as e:
                print(f"   !! falló {r['Ruta']}: {e}")
                write_debug(driver, tag)
            finally:
                time.sleep(sleep_between)

        print(f"\nListo ✅ CSV: {out_file}")

    finally:
        if os.getenv("DETACH", "").strip() == "1":
            return
        driver.quit()


if __name__ == "__main__":
    main()