# Sistema de Procesamiento MeSH

## Estructura del Proyecto

```
proyecto/
├── mesh/
│   ├── data/
│   │   ├── desc2025.xml              # Archivo XML oficial de MeSH
│   │   └── mesh_data.json            # Datos MeSH procesados
│   ├── src/
│   │   ├── mesh_downloader.py        # Clase para descargar datos MeSH
│   │   ├── mesh_processor_local.py   # Clase para procesar artículos
│   │   ├── run_mesh_downloader.py    # Script ejecutor del downloader
│   │   └── run_mesh_processor.py     # Script ejecutor del procesador
│   └── README.md                     # Este archivo
├── pubmed/
│   ├── data/
│   │   ├── spanish_medical_articles.json
│   │   └── translated_dataset.json
│   ├── sample/
│   │   ├── spanish_medical_articles_small.json
│   │   └── translated_dataset_small.json
│   └── src/
│       └── [otros scripts de pubmed]
└── dataset/
    └── [archivos_mesh_generados].csv
```

## Uso del Sistema

### Paso 1: Configurar Datos MeSH (Solo una vez)

```bash
# Ejecutar desde la raíz del proyecto
python mesh/src/run_mesh_downloader.py
```

**¿Qué hace?**
- Descarga `desc2025.xml` (~200MB) desde NCBI
- Procesa el XML y extrae términos MeSH
- Genera `mesh_data.json` optimizado
- Guarda todo en la carpeta `mesh/data/`

**Tiempo estimado:** 10-15 minutos (solo la primera vez)

### Paso 2: Procesar Artículos

```bash
# Sintaxis
python mesh/src/run_mesh_processor.py <archivo_entrada> <archivo_salida>

# Ejemplos con archivos pequeños (para testing)
python mesh/src/run_mesh_processor.py pubmed/sample/translated_dataset_small.json dataset/translated_dataset_small.csv
python mesh/src/run_mesh_processor.py pubmed/sample/spanish_medical_articles_small.json dataset/spanish_medical_articles_small.csv

# Ejemplos con datasets completos
python mesh/src/run_mesh_processor.py pubmed/data/translated_dataset.json dataset/translated_dataset.csv
python mesh/src/run_mesh_processor.py pubmed/data/spanish_medical_articles.json dataset/spanish_medical_articles.csv

# Con debugging (muestra detalles de los primeros 3 artículos)
python mesh/src/run_mesh_processor.py --debug pubmed/sample/translated_dataset_small.json dataset/debug_test.csv
```

**¿Qué hace?**
- Lee artículos del archivo JSON especificado
- Extrae categorías MeSH (solo major subjects)
- Incluye abstracts en español
- Guarda resultado en la ruta especificada

## Archivos de Entrada Disponibles

### En `pubmed/sample/` (para testing)
- `translated_dataset_small.json` - 4 artículos con traducción
- `spanish_medical_articles_small.json` - Artículos pequeños en español

### En `pubmed/data/` (datasets completos)
- `translated_dataset.json` - Dataset completo con traducciones
- `spanish_medical_articles.json` - Dataset completo en español

## Resultado Final

El archivo CSV generado incluye estas columnas:

| Columna | Descripción | Ejemplo |
|---------|-------------|---------|
| `pmid` | ID del artículo | `"31516003"` |
| `title` | Título del artículo | `"[Influence of overweight...]"` |
| `year` | Año de publicación | `2019` |
| `spanish_abstract` | **Abstract en español** | `"Introducción: el sobrepeso..."` |
| `level1_codes` | Códigos nivel 1 | `["E", "G"]` |
| `level1_names` | Nombres nivel 1 | `["Phenomena and Processes", ...]` |
| `level2_codes` | Códigos nivel 2 | `["E01", "G11"]` |
| `level2_names` | Nombres nivel 2 | `["Diagnosis", ...]` |
| `level1_categories_str` | Categorías L1 (string) | `"Phenomena and Processes; Health Care"` |
| `level2_categories_str` | Categorías L2 (string) | `"Diagnosis; Physiological Phenomena"` |
| `major_subjects_count` | Número de major subjects | `2` |

## Ejemplos de Categorías MeSH

### Nivel 1 (16 categorías principales)
- `A` - Anatomy
- `B` - Organisms  
- `C` - Diseases
- `D` - Chemicals and Drugs
- `E` - Analytical, Diagnostic and Therapeutic Techniques and Equipment
- `G` - Phenomena and Processes
- `N` - Health Care
- etc.

### Nivel 2 (subcategorías)
- `B01` - Eukaryota
- `C14` - Cardiovascular Diseases
- `E01` - Diagnosis
- `G11` - Physiological Phenomena
- etc.


## Ventajas del Sistema

- **🚀 Súper rápido:** Sin APIs, todo local
- **📚 Completo:** Todos los términos MeSH 2025
- **🎯 Preciso:** Solo major subjects relevantes
- **🌍 Multilingüe:** Incluye abstracts en español
- **📊 Listo para ML:** DataFrame estructurado para entrenamiento
- **🔧 Flexible:** Especifica archivos de entrada y salida

## Solución de Problemas

### Error: "No se pudieron cargar los datos de MeSH"
```bash
python mesh/src/run_mesh_downloader.py
```

### Error: "Archivo de entrada no encontrado"
- Verificar la ruta del archivo JSON
- Usar rutas relativas desde la raíz del proyecto
- Revisar archivos disponibles en `pubmed/sample/` o `pubmed/data/`

### Error: "El archivo de salida debe tener extensión .csv"
```bash
# ❌ Incorrecto
python mesh/src/run_mesh_processor.py input.json output

# ✅ Correcto  
python mesh/src/run_mesh_processor.py input.json dataset/output.csv
```

### Error: Sin abstracts en español
- Verificar que los artículos tengan `spanish_translation` en el JSON
- Algunos archivos pueden no tener traducciones

### Procesamiento lento
- Normal para datasets grandes
- Usar `--debug` solo para debugging (hace más lento)
- El primer procesamiento puede ser más lento (carga de datos)
