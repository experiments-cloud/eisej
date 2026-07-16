# Operational Strain Signatures — Data & Code

Pipeline completo de extracción, limpieza, generación de Ground Truth,
análisis de estabilidad temporal y modelado predictivo utilizado en el
manuscrito *"Detecting Operational Strain Signatures in Software
Development: A Behavioral Telemetry Approach with Trait-State
Validation"*, enviado a e-Informatica Software Engineering Journal.

## Estructura

```
eisej/
├── scripts/
│   ├── 01_extract_5repos.py                  # Sec. 3.1 — react, tensorflow, vscode, linux, bitcoin
│   ├── 01b_extract_2repos_community.py        # Sec. 3.1 — neovim, svelte
│   ├── 02_clean_and_feature_engineer.py        # Sec. 3.2, 3.3
│   ├── 03_groundtruth_windows.py               # Sec. 3.4 (ventanas + agregación)
│   ├── 03b_groundtruth_compare_methods.py      # Sec. 3.4 (comparación percentil/IsolationForest/GMM)
│   ├── 04_trait_state_stability.py             # Sec. 3.6, 4.2
│   ├── 05_oracle_modeling_cv.py                # Sec. 3.7, 4.3
│   ├── figures/                                # Scripts que generan las Figuras 1-6 del paper
│   └── exploratory_superseded/                 # Intentos metodológicos descartados,
│                                                #   documentados en el paper (Sec. 3.4, 3.7)
│                                                #   como parte del proceso de corrección.
│                                                #   NO forman parte del pipeline final.
├── data/
│   ├── raw/                                    # Salida de 01 + 01b
│   └── processed/                              # Salida de 02, 03, 03b, 04, 05
├── paper/
│   ├── eisej_paper.tex
│   └── eisej_paper.bib
└── requirements.txt
```

## Orden de ejecución (reproducción desde cero)

```bash
pip install -r requirements.txt

export GITHUB_TOKEN=tu_token_aqui   # scope: ninguno (clásico) o "Public Repositories read-only" (fine-grained)

python scripts/01_extract_5repos.py
python scripts/01b_extract_2repos_community.py
# consolidar manualmente los dos parquet de salida en data/raw/github_raw_final.parquet
# (concat + drop_duplicates por 'sha'; ver Sección 3.1 del paper)

python scripts/02_clean_and_feature_engineer.py
python scripts/03_groundtruth_windows.py
python scripts/03b_groundtruth_compare_methods.py
python scripts/04_trait_state_stability.py
python scripts/05_oracle_modeling_cv.py

python scripts/figures/generar_figura1.py   # ... hasta generar_figura6.py
```

**Nota sobre reproducibilidad exacta:** `05_oracle_modeling_cv.py` fija semilla
en NumPy, scikit-learn y PyTorch (`torch.manual_seed(42)`). Sin esa fijación,
el entrenamiento del LSTM no es determinista (ver limitación documentada en
la Sección 4.3 del paper). Los resultados publicados corresponden a la
versión con semilla fija, verificados por ejecución duplicada con resultados
idénticos.

## Carpeta `exploratory_superseded/`

Contiene dos scripts que representan decisiones metodológicas **descartadas**
durante el desarrollo del estudio, incluidos aquí por transparencia y no
porque formen parte del pipeline final:

- `commit_level_kmeans_FAILED.py`: primer intento de Ground Truth, agrupando
  *k*-means sobre commits individuales en vez de ventanas. Falló por
  dominancia de variable binaria y ausencia de minoría diferenciada
  (documentado en la Sección 3.4 del paper).
- `single_split_modeling_SUPERSEDED.py`: primera versión del modelado oracle
  con un único `GroupShuffleSplit`, reemplazada por validación cruzada de 5
  particiones (`05_oracle_modeling_cv.py`) por mayor robustez estadística
  (Sección 4.3 del paper).

## Datasets incluidos en `data/`

| Archivo | Filas | Descripción |
|---|---|---|
| `raw/github_raw_final.parquet` | 21,638 | Commits crudos post-filtro de bots/identidades espurias en extracción |
| `processed/github_features_final.parquet` | 17,428 | Corpus limpio con features de la Sección 3.3 |
| `processed/github_windows_groundtruth.parquet` | 15,564 | Ventanas de 5 commits con etiqueta de Isolation Forest |
| `processed/fase4_author_profiles.parquet` | 238 | Perfiles de estabilidad temporal por autor |
| `processed/fase5_oof_predictions.npz` | 15,098 | Predicciones out-of-fold (RF y LSTM) usadas en Tabla 9 y Figuras 4-5 |

## Datos NO incluidos

El contenido textual completo de los mensajes de commit se conserva en los
parquet (`commit_message`) tal como se extrajo de la API pública de GitHub,
sujeto a los términos de servicio de GitHub. No se realizó anonimización de
`author_id` dado que corresponde a nombres de usuario ya públicos en
repositorios públicos.

## Licencia y cita

[Pendiente: agregar licencia del repositorio (ej. MIT para código, CC-BY 4.0
para datos, consistente con la política de datos abiertos de EISEJ) y
formato de cita del paper una vez asignado el DOI].
