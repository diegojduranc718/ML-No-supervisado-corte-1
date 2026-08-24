# Detección de fraude en tarjetas de crédito — análisis probabilístico

> Por qué las métricas de "precisión" de un sistema antifraude pueden ser engañosas, y qué variables son realmente informativas cuando solo el **0.1727%** de las transacciones es fraudulento.

Proyecto de Aprendizaje de Máquina No Supervisado — Universidad de La Sabana, 2026-II.

---

## El problema de negocio

Una red de tarjetas de crédito reporta que su sistema de detección de fraude opera con más del 99% de "precisión". El encargo, en rol de consultor externo, es evaluar si esa cifra significa algo.

No significa nada. Con 492 fraudes en 284,807 transacciones, un clasificador que declare **todas** las transacciones legítimas alcanza **99.827% de accuracy** y deja pasar la totalidad del fraude: 60,128 en montos no interceptados en apenas dos días de operación. El análisis responde tres preguntas concretas:

1. ¿Por qué el desbalance extremo invalida las métricas que la red usa hoy?
2. ¿Sirve el monto de la transacción como disparador de alertas?
3. ¿Qué variables sí separan fraude de no-fraude, y cuánto?

## Dataset

**Credit Card Fraud Detection** — Machine Learning Group, Université Libre de Bruxelles (vía Kaggle).

| | |
|---|---|
| Transacciones | 284,807 (tarjetahabientes europeos, septiembre 2013) |
| Fraudes | 492 (0.1727%) |
| Variables | `Time`, `Amount`, `Class` + 28 componentes PCA (`V1`–`V28`) |
| Período cubierto | 2 días |
| Fuente | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |

Por confidencialidad, ULB no revela qué variable original representa cada componente PCA — una limitación real del dataset que condiciona la interpretación de los hallazgos.

**El CSV no está versionado en este repositorio** (~150 MB y licencia de Kaggle). Hay que descargarlo y colocarlo en `data/`, o subirlo directamente desde el computador con la celda de carga incluida en el notebook.

## Hallazgos principales

**1. El prior domina cualquier prueba.** Una regla de alerta sobre `V14` con 83.5% de sensibilidad y 99.14% de especificidad —números que en cualquier otro contexto serían excelentes— produce una probabilidad posterior de fraude de solo **14.43%**: 411 aciertos contra 2,438 falsas alarmas. Es la paradoja de los falsos positivos, aplicada a un caso real.

**2. El monto es una señal débil.** De ocho umbrales evaluados, el mejor (`Amount > 500`) apenas lleva la probabilidad de fraude a **0.383%** —un lift de 2.2x— y captura solo el 7.1% de los fraudes. La divergencia KL lo confirma: `Amount` obtiene 0.628 bits, puesto 22 de 29 variables.

**3. El fraude típico es de monto bajo.** La media de `Amount` en fraude es mayor (122.21 vs 88.29), pero la **mediana es menos de la mitad** (9.25 vs 22.00). El fraude característico es una micro-transacción de prueba de tarjeta; la media alta la produce una cola minoritaria. Este hallazgo contradice la intuición operativa de alertar por montos altos.

**4. Dos señales encadenadas sí resuelven el problema.** Actualización bayesiana secuencial: prior 0.1727% → 14.43% (`V14`, LR=97.4) → **93.97%** (`V10`, LR=92.5). El orden de las pruebas es irrelevante (el producto de razones de verosimilitud es conmutativo), lo que se verifica explícitamente en el notebook.

**5. Optimizar la entropía cruzada desalinea el modelo del negocio.** El modelo sin balanceo de clases logra la mejor log-loss (0.0048 nats) con recall de 61.5%; el modelo balanceado empeora la pérdida a 0.0961 nats pero eleva el recall a 87.8%. La métrica "mejor" corresponde al modelo que detecta menos fraude.

**6. Ranking de informatividad por divergencia KL** (fraude ‖ legítimas, en bits): `V14` 3.487 · `V10` 3.209 · `V12` 3.109 · `V17` 2.983 · `V11` 2.937. Contra `Amount` con 0.628.

### Cifras de referencia

| Concepto | Resultado |
|---|---|
| Prior P(fraude) | 0.1727% (492 / 284,807) |
| P(fraude \| Amount > 500) | 0.383% (lift 2.22x) |
| Posterior tras prueba V14 | 14.43% (84x el prior) |
| Posterior secuencial V14 + V10 | 93.97% |
| MLE V14 \| fraude | μ = −6.97, σ = 4.27 (d de Cohen = −2.26) |
| E[Amount] fraude vs legítima | 122.21 vs 88.29 (medianas: 9.25 vs 22.00) |
| \|corr\| media entre componentes PCA | 6.05e−16 |
| χ² (Amount>200 vs Class) | 26.9 (p = 2.1e−07), Cramér's V = 0.0097 |
| H(Class) | 0.0183 bits (1.83% del máximo) |
| Ganancia de información con V14 | 47.1% |
| Entropía cruzada test (balanced) | 0.0961 nats · PR-AUC 0.705 |
| KL máxima | V14, 3.487 bits |

## Conceptos aplicados

Probabilidad condicional · Teorema de Bayes · Verosimilitud · Máxima verosimilitud (MLE) · Distribuciones paramétricas · Esperanza y varianza · Independencia y correlación · Prior y posterior · Entropía · Entropía cruzada · Divergencia KL.

Cada uno está implementado sobre el dataset con cálculo verificable, no como definición teórica. Las secciones del notebook siguen ese mismo orden.

## Estructura del repositorio

```
.
├── data/                                  # creditcard.csv (no versionado)
├── notebooks/
│   └── proyecto_fraude_propuesta_C.ipynb  # análisis completo, 11 secciones
├── informe/
│   └── informe_gerencial.pdf              # informe ejecutivo (2 páginas)
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

### Google Colab

Abrir el notebook y ejecutar las celdas en orden. La sección 0.1 abre el selector de archivos del computador y acepta tanto el `creditcard.csv` como el `.zip` de Kaggle (lo descomprime automáticamente). Como la subida se pierde al reiniciar el entorno, la celda siguiente incluye la alternativa vía Google Drive.

### Local

```bash
git clone https://github.com/USUARIO/REPO.git
cd REPO
pip install -r requirements.txt

# descargar creditcard.csv de Kaggle y dejarlo en data/
jupyter notebook notebooks/proyecto_fraude_propuesta_C.ipynb
```

Dependencias: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`. El notebook fija `RANDOM_STATE = 42`, de modo que todas las cifras citadas arriba son reproducibles.

La última celda imprime un resumen consolidado con todos los números del informe.

## Decisiones metodológicas

- **Divergencia KL:** discretización en 30 bins por cuantiles con suavizado de Laplace (+1) para evitar divisiones por cero. Se reporta KL(fraude ‖ legítimas) y se verifica su asimetría calculando también la dirección inversa.
- **Prueba bayesiana:** el umbral de `V14` es el percentil 1 de la distribución — una elección deliberada para emular una regla de alerta operativa, no un óptimo ajustado a los datos.
- **Normalidad:** con n = 284,807 el test de Kolmogórov-Smirnov rechaza normalidad casi siempre, así que el criterio de decisión son los QQ-plots y la curtosis en exceso (23.9, 32.0 y 94.8 para `V14`, `V10` y `V17`), no el p-valor.
- **Duplicados:** se detectaron 1,081 registros duplicados exactos y se conservaron, para no alterar el prior sobre el que descansa todo el análisis.

## Limitaciones

Los datos cubren dos días de septiembre de 2013 y provienen de un solo emisor europeo; los patrones de fraude cambian rápido y estas reglas requerirían revalidación sobre datos vigentes. Las componentes PCA son anónimas, de modo que `V14` y `V10` no admiten interpretación de negocio directa. Todos los hallazgos son asociaciones observacionales, no relaciones causales.

## Referencia

Machine Learning Group – ULB. (2018). *Credit card fraud detection* [Data set]. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

**Autor:** Diego Duran · Universidad de La Sabana, Facultad de Ingeniería
