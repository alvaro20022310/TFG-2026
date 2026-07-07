"""
================================================================================
OPTIMIZACIÓN DE SUPERFICIES — Disposición de paneles (Etapa 1)
TFG: Estudio de viabilidad de instalación de autoconsumo fotovoltaico
     (cubierta de oficina, Madrid)
================================================================================

QUÉ HACE
--------
Dado un MODELO de panel, una INCLINACIÓN y una ORIENTACIÓN (azimut), calcula
cuántos paneles caben sobre la cubierta. Recorre la rejilla discreta de
inclinaciones y azimuts (la misma que se usa en PVGIS, para poder cruzar después
conteo y generación) sobre las DOS zonas útiles de la cubierta (franja dentada y
saliente superior, separadas por la torre de comunicaciones), y produce:
  - las tablas de nº de paneles por combinación, para cada zona y el total,
  - los mapas de calor (heatmaps) de esas tablas,
  - esquemas gráficos de la disposición de paneles para una configuración dada.

MÉTODO
------
Las filas (mesas) se colocan paralelas y separadas por el "pitch". El problema 2D
se reduce a: (1) nº de filas = profundidad disponible / pitch, y (2) nº de paneles
por fila = se rebana la zona en franjas y se mide, en cada una, el tramo que cae
dentro del polígono. La orientación se resuelve girando el polígono al marco de
las mesas, contando, y devolviendo los paneles a su posición real. Las pérdidas de
superficie se descuentan geométricamente (retranqueo perimetral y obstáculos como
zonas excluidas); se cuentan solo paneles enteros.

El barrido da el MÁXIMO que cabe en cada combinación; NO elige la mejor: la
elección es económica (se cruza con la generación de PVGIS y el consumo en etapas
posteriores). El edificio está orientado norte-sur (desviación de 0,7°, despre-
ciable), por lo que azimut 0 equivale a paneles al sur, igual que en PVGIS.

REQUISITOS
----------
Instalar las librerías:  pip install shapely matplotlib numpy

USO
---
Ejecutar el archivo. Cambiar, si se desea, el panel, la orientación y la
configuración de los esquemas en la sección CONFIGURACIÓN del bloque principal.

Autor: Álvaro (TFG). Código de elaboración propia.
================================================================================
"""

import math
from dataclasses import dataclass
from shapely.geometry import Polygon, LineString, box
from shapely.affinity import rotate


# ============================================================ 1. PARÁMETROS
LATITUD_DEG = 40.49          # latitud del emplazamiento (Madrid)
RETRANQUEO_M = 1.0           # franja perimetral de seguridad libre de paneles
PASO_OPERARIO_M = 0.70       # paso mínimo de mantenimiento entre filas
SEP_PANELES_M = 0.02         # holgura de montaje entre paneles de una misma fila


# ============================================================ 2. MODELO PANEL
@dataclass
class Panel:
    modelo: str
    potencia_wp: int
    largo_mm: int
    ancho_mm: int

    @property
    def largo_m(self):
        return self.largo_mm / 1000.0

    @property
    def ancho_m(self):
        return self.ancho_mm / 1000.0


# Catálogo (muestra del estudio de mercado, fabricantes Tier 1)
CATALOGO = {
    "Jinko 580W": Panel("Jinko Tiger Neo JKM580N-72HL4", 580, 2278, 1134),
    "Jinko 450W": Panel("Jinko Tiger Neo JKM450N-54HL4R", 450, 1722, 1134),
    "Trina 450W": Panel("Trina Vertex S+ TSM-450NEG9R", 450, 1762, 1134),
    "Longi 650W": Panel("Longi Hi-MO X10 LR7-72HVH-650M", 650, 2382, 1134),
}


# ============================================================ 3. GEOMETRÍA
# Coordenadas reales (m) exportadas de SolidWorks (medición in situ + catastro).
# Dos zonas separadas por la torre de comunicaciones.

# Zona 1: franja dentada (sur). Contiene un obstáculo (registro) de 0,70x0,70 m.
FRANJA_VERTS = [
    (9.61, 0.0), (9.61, 32.82), (-1.79963643, 32.82), (-2.18894517, 32.70983673),
    (-0.85749476, 28.00459074), (-2.16611123, 27.63428961), (-0.83466082, 22.92904362),
    (-2.14327729, 22.55874248), (-0.81182688, 17.8534965), (-2.12044336, 17.48319536),
    (-0.78899295, 12.77794937), (-2.09760942, 12.40764824), (-0.76615901, 7.70240225),
    (-2.07477548, 7.33210111), (0.0, 0.0),
]
FRANJA_OBSTACULO = box(3.96, 9.1, 4.66, 9.8)   # registro a descontar

# Zona 2: saliente superior (norte).
SALIENTE_VERTS = [
    (-1.99099413, 42.0), (11.17, 42.0), (11.17, 38.2), (19.47, 38.2),
    (19.47, 49.32), (4.08, 49.32), (4.08, 43.93624543), (-1.25, 43.93624543),
    (-2.23461304, 42.86093098),
]


# ============================================================ 4. REJILLA
# Conjunto discreto de inclinaciones y azimuts (coincide con el usado en PVGIS).
INCLINACIONES = [0, 10, 15, 20, 25, 30, 35, 40]
AZIMUTS = [-90, -60, -30, 0, 30, 60, 90]


# ============================================== 5. CÁLCULOS GEOMÉTRICOS (motor)
def dimensiones_en_mesa(panel, orientacion):
    """(canto_en_pendiente, ancho_a_lo_largo_de_la_fila) en metros."""
    if orientacion == "apaisado":      # lado corto en la pendiente
        return panel.ancho_m, panel.largo_m
    else:                               # "vertical": lado largo en la pendiente
        return panel.largo_m, panel.ancho_m


def calcular_pitch(panel, inclinacion_deg, orientacion):
    """Separación entre filas (m) y profundidad de la mesa (m).

    pitch = base de la mesa + hueco trasero, siendo el hueco el mayor entre el
    exigido por sombras (criterio IDAE, solsticio de invierno) y el paso de
    operario. base = canto*cos(incl); altura = canto*sin(incl).
    """
    canto, _ = dimensiones_en_mesa(panel, orientacion)
    beta = math.radians(inclinacion_deg)
    altura = canto * math.sin(beta)
    base = canto * math.cos(beta)
    k_idae = 1.0 / math.tan(math.radians(61.0 - LATITUD_DEG))
    hueco_sombra = altura * k_idae
    hueco = max(hueco_sombra, PASO_OPERARIO_M)
    return base + hueco, base


def colocar_paneles(zona, panel, inclinacion_deg, azimut_deg,
                    orientacion="apaisado", obstaculos=None):
    """Coloca los paneles y devuelve la lista de rectángulos (en coords reales).

    azimut_deg: orientación de las filas. 0 = filas este-oeste (paneles al sur).
    """
    # (1) Descontar pérdidas geométricas: retranqueo y obstáculos
    util = zona.buffer(-RETRANQUEO_M)
    if obstaculos is not None and not util.is_empty:
        util = util.difference(obstaculos)
    if util.is_empty:
        return []

    # (2) Pitch (de la inclinación) y ancho de panel a lo largo de la fila
    pitch, base = calcular_pitch(panel, inclinacion_deg, orientacion)
    _, ancho_fila = dimensiones_en_mesa(panel, orientacion)
    paso = ancho_fila + SEP_PANELES_M

    # (3) Girar la zona al marco de las mesas (para el azimut)
    cx, cy = zona.centroid.x, zona.centroid.y
    util_rot = rotate(util, -azimut_deg, origin=(cx, cy))
    minx, miny, maxx, maxy = util_rot.bounds

    # (4) Rebanar en filas separadas por el pitch y contar paneles por fila
    paneles_rot = []
    y = miny + base / 2.0
    while y <= maxy - base / 2.0 + 1e-9:
        linea = LineString([(minx - 2, y), (maxx + 2, y)])
        inter = linea.intersection(util_rot)
        if inter.is_empty:
            tramos = []
        elif inter.geom_type == "LineString":
            tramos = [inter]
        elif inter.geom_type == "MultiLineString":
            tramos = list(inter.geoms)
        else:
            tramos = []
        for tr in tramos:
            xs = list(tr.xy[0])
            x0, x1 = min(xs), max(xs)
            longitud = x1 - x0
            n = int((longitud + SEP_PANELES_M) // paso)
            for k in range(n):
                px = x0 + k * paso
                paneles_rot.append(box(px, y - base / 2, px + ancho_fila, y + base / 2))
        y += pitch

    # (5) Devolver los paneles a su posición real (deshacer el giro)
    return [rotate(p, azimut_deg, origin=(cx, cy)) for p in paneles_rot]


def maximo_paneles(zona, panel, inclinacion_deg, azimut_deg,
                   orientacion="apaisado", obstaculos=None):
    """Número máximo de paneles que caben para la combinación dada."""
    return len(colocar_paneles(zona, panel, inclinacion_deg, azimut_deg,
                               orientacion, obstaculos))


def generar_configuraciones(panel, n_max):
    """Rango de configuraciones de 1 a n_max paneles, con su potencia (kWp)."""
    return [{"n_paneles": n, "potencia_kwp": round(n * panel.potencia_wp / 1000.0, 2)}
            for n in range(1, n_max + 1)]


# ============================================== 6. BARRIDO (recorre la rejilla)
def barrido_zona(zona, panel, orientacion="apaisado", obstaculos=None):
    """Matriz {(inclinacion, azimut): n_paneles} para una zona."""
    res = {}
    for incl in INCLINACIONES:
        for azim in AZIMUTS:
            res[(incl, azim)] = maximo_paneles(zona, panel, incl, azim,
                                               orientacion, obstaculos)
    return res


def imprimir_tabla(res, nombre):
    """Tabla de texto: filas = inclinación, columnas = azimut."""
    print(f"\n  {nombre}  (nº de paneles)")
    print("  incl\\azim " + "".join(f"{a:>6}" for a in AZIMUTS))
    for incl in INCLINACIONES:
        fila = "".join(f"{res[(incl, a)]:>6}" for a in AZIMUTS)
        print(f"  {incl:>6}°  {fila}")


# ============================================================ 7. GRÁFICOS
def dibujar_disposicion(zona, paneles, titulo, ruta, obstaculos=None,
                        retranqueo=RETRANQUEO_M):
    """Esquema gráfico de la disposición de paneles (para el TFG)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 10))
    xs, ys = zona.exterior.xy
    ax.plot(xs, ys, "-", color="#222", lw=2, label="Contorno de la zona")

    util = zona.buffer(-retranqueo)
    if util.geom_type == "Polygon" and not util.is_empty:
        ux, uy = util.exterior.xy
        ax.plot(ux, uy, "--", color="#999", lw=1.1,
                label=f"Retranqueo ({retranqueo:.1f} m)")

    if obstaculos is not None:
        ox, oy = obstaculos.exterior.xy
        ax.fill(ox, oy, color="#c0392b", alpha=0.85, label="Obstáculo")

    for i, p in enumerate(paneles):
        px, py = p.exterior.xy
        ax.fill(px, py, facecolor="#1f6fb4", edgecolor="white", lw=0.5, alpha=0.9,
                label="Paneles" if i == 0 else None)

    ax.set_aspect("equal")
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("Este-Oeste (m)")
    ax.set_ylabel("Norte-Sur (m)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ruta, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Esquema guardado: {ruta}")


def dibujar_heatmap(res, titulo, ruta):
    """Mapa de calor de nº de paneles por inclinación (filas) y azimut (cols)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    M = np.array([[res[(incl, a)] for a in AZIMUTS] for incl in INCLINACIONES])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(AZIMUTS)), [f"{a}°" for a in AZIMUTS])
    ax.set_yticks(range(len(INCLINACIONES)), [f"{i}°" for i in INCLINACIONES])
    ax.set_xlabel("Azimut (0° = sur)")
    ax.set_ylabel("Inclinación")
    for i in range(len(INCLINACIONES)):
        for j in range(len(AZIMUTS)):
            ax.text(j, i, M[i, j], ha="center", va="center", color="black", fontsize=9)
    ax.set_title(titulo, fontsize=11)
    fig.colorbar(im, ax=ax, label="nº de paneles")
    plt.tight_layout()
    plt.savefig(ruta, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Mapa guardado: {ruta}")


# ============================================================ 8. PRINCIPAL
if __name__ == "__main__":
    # ---------------------- CONFIGURACIÓN (cambiar aquí lo que se quiera probar)
    PANEL = CATALOGO["Jinko 580W"]   # modelo de panel
    ORIENTACION = "apaisado"          # "apaisado" o "vertical"
    INCL_ESQUEMA = 30                 # inclinación del esquema gráfico de ejemplo
    AZIM_ESQUEMA = 0                  # azimut del esquema gráfico de ejemplo
    # -------------------------------------------------------------------------

    franja = Polygon(FRANJA_VERTS)
    saliente = Polygon(SALIENTE_VERTS)

    print("=" * 64)
    print(f"BARRIDO — panel {PANEL.modelo} ({PANEL.potencia_wp} Wp), {ORIENTACION}")
    print(f"Inclinaciones: {INCLINACIONES}")
    print(f"Azimuts:       {AZIMUTS}")
    print("=" * 64)

    # Barrido de las dos zonas
    res_franja = barrido_zona(franja, PANEL, ORIENTACION, FRANJA_OBSTACULO)
    res_saliente = barrido_zona(saliente, PANEL, ORIENTACION, None)

    imprimir_tabla(res_franja, "ZONA 1 - FRANJA dentada")
    imprimir_tabla(res_saliente, "ZONA 2 - SALIENTE superior")

    total = {k: res_franja[k] + res_saliente[k] for k in res_franja}
    imprimir_tabla(total, "TOTAL (franja + saliente)")

    # Máximo geométrico (sólo informativo: NO es la mejor opción económica)
    mejor = max(total, key=total.get)
    n_mejor = total[mejor]
    print("\n" + "-" * 64)
    print(f"Máximo geométrico: {n_mejor} paneles "
          f"({n_mejor * PANEL.potencia_wp / 1000:.1f} kWp) en "
          f"inclinación {mejor[0]}°, azimut {mejor[1]}° "
          f"(franja {res_franja[mejor]} + saliente {res_saliente[mejor]})")
    print("  NOTA: es el máximo que CABE, no la mejor opción económica.")

    # ---------------------- Gráficos (se generan al ejecutar)
    print("\nGenerando gráficos...")
    dibujar_heatmap(res_franja,
                    "Paneles en la FRANJA según inclinación y azimut",
                    "heatmap_franja.png")
    dibujar_heatmap(res_saliente,
                    "Paneles en el SALIENTE según inclinación y azimut",
                    "heatmap_saliente.png")

    # Esquemas de disposición para la configuración de ejemplo
    pitch, _ = calcular_pitch(PANEL, INCL_ESQUEMA, ORIENTACION)
    pan_fr = colocar_paneles(franja, PANEL, INCL_ESQUEMA, AZIM_ESQUEMA,
                             ORIENTACION, FRANJA_OBSTACULO)
    pan_sa = colocar_paneles(saliente, PANEL, INCL_ESQUEMA, AZIM_ESQUEMA,
                             ORIENTACION, None)
    dibujar_disposicion(
        franja, pan_fr,
        f"Franja — {PANEL.potencia_wp}Wp, {INCL_ESQUEMA}°, azimut {AZIM_ESQUEMA}° "
        f"(pitch {pitch:.2f} m)\n{len(pan_fr)} paneles · "
        f"{len(pan_fr) * PANEL.potencia_wp / 1000:.1f} kWp",
        "disposicion_franja.png", FRANJA_OBSTACULO)
    dibujar_disposicion(
        saliente, pan_sa,
        f"Saliente — {PANEL.potencia_wp}Wp, {INCL_ESQUEMA}°, azimut {AZIM_ESQUEMA}°\n"
        f"{len(pan_sa)} paneles · {len(pan_sa) * PANEL.potencia_wp / 1000:.1f} kWp",
        "disposicion_saliente.png", None)

    print("\nHecho. Revisa los archivos .png generados en esta carpeta.")
