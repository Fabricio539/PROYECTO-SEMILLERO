# 🟡 Mesa de Ayuda IA — Patito S.A.

Sistema **multi-agente de Inteligencia Artificial** para el equipo de Marketing de Patito S.A. Implementado como un Jupyter Notebook autocontenido que permite resolver consultas especializadas sobre identidad de marca, planificación de campañas, cumplimiento publicitario y análisis visual de piezas gráficas, sin necesidad de buscar manualmente en documentos internos.

> **Semillero de Inteligencia Artificial**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-green)](https://www.langchain.com/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash-orange?logo=google)](https://aistudio.google.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.9-red)](https://www.trychroma.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#licencia)

---

## Índice

1. [Descripción del sistema](#descripción-del-sistema)
2. [Arquitectura](#arquitectura)
3. [Tecnologías utilizadas](#tecnologías-utilizadas)
4. [Agentes implementados](#agentes-implementados)
5. [Estructura del notebook](#estructura-del-notebook)
6. [Archivos del repositorio](#archivos-del-repositorio)
7. [Requisitos previos](#requisitos-previos)
8. [Instalación](#instalación)
9. [Configuración de la API Key](#configuración-de-la-api-key)
10. [Ejecución paso a paso](#ejecución-paso-a-paso)
11. [Guía de uso y ejemplos](#guía-de-uso-y-ejemplos)
12. [Flujo de registro de campaña](#flujo-de-registro-de-campaña)
13. [Análisis multimodal de imágenes](#análisis-multimodal-de-imágenes)
14. [Base de conocimiento](#base-de-conocimiento)
15. [Archivos generados en runtime](#archivos-generados-en-runtime)
16. [Personalización](#personalización)
17. [Licencia](#licencia)

---

## Descripción del sistema

La Mesa de Ayuda IA de Patito S.A. es un sistema **RAG** (*Retrieval-Augmented Generation*) **multi-agente** construido con LangChain y Google Gemini. El sistema procesa consultas en lenguaje natural, identifica automáticamente la intención del usuario y enruta la solicitud al agente especializado correspondiente.

Todo el sistema —documentos de base de conocimiento, lógica de agentes, orquestador y demos— se encuentra en un único archivo `.ipynb`, sin dependencias de archivos externos en el repositorio.

**Capacidades principales:**

- Responde consultas sobre marca, campañas y cumplimiento leyendo exclusivamente los documentos internos de Patito S.A. (no usa conocimiento general del LLM).
- Analiza imágenes de piezas gráficas con Gemini Vision y las contrasta con el Manual de Marca.
- Registra solicitudes de campaña con validación de datos obligatorios, flujo de confirmación y persistencia en archivo `.txt`.
- Clasifica la intención de cada consulta con un prompt de clasificación y enruta automáticamente al agente correcto.

---

## Arquitectura

```
┌──────────────────────────────────────────────────────────────┐
│                   USUARIO (celda del notebook)               │
└─────────────────────────┬────────────────────────────────────┘
                          │ orquestador.process(query, image_path?)
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                      ORQUESTADOR                             │
│  1. Detecta imagen adjunta → enruta directo a Multimodal     │
│  2. Detecta confirmación pendiente → flujo AccionAgent        │
│  3. Clasifica intención con Gemini (JSON estructurado)        │
│     → marca | campanas | cumplimiento | accion | general     │
└──────┬──────────┬──────────┬────────────┬──────────┬─────────┘
       │          │          │            │          │
       ▼          ▼          ▼            ▼          ▼
  [Marca]   [Campañas]  [Cumpli-    [Multimodal] [Acción]
  RAG+Chroma RAG+Chroma  miento]    Gemini Vision Registro
                        RAG+Chroma               en .txt
       │          │          │
       ▼          ▼          ▼
  chroma_db/ chroma_db/ chroma_db/
  marca_     campanas_  cumplimiento_
  lineamientos  kpis    publicitario
```

### Flujo RAG interno de cada agente

```
Consulta del usuario
      │
      ▼
Retriever Chroma (búsqueda semántica por similitud coseno)
      │
      ▼
Top-K chunks más relevantes (K=4 por defecto)
      │
      ▼
Prompt template: system + contexto + pregunta
      │
      ▼
ChatGoogleGenerativeAI (Gemini 3.5 Flash, temperature=0.1)
      │
      ▼
StrOutputParser → texto de respuesta
      │
      ▼
AgentResponse(agent_name, answer, sources, chunks_used, found_info)
```

---

## Tecnologías utilizadas

| Componente | Librería / Versión instalada | Rol |
|---|---|---|
| LLM principal | `langchain-google-genai` (4.3.1) | Interfaz con Google Gemini para chat, embeddings y visión |
| Modelo LLM | Google Gemini 3.5 Flash | Generación de respuestas, clasificación de intención, extracción de datos |
| Embeddings | `GoogleGenerativeAIEmbeddings` (text-embedding-004) | Vectorización de chunks de texto para búsqueda semántica |
| Vector store | `langchain-chroma` + `chromadb` (1.5.9) | Almacenamiento y consulta de índices vectoriales con persistencia en disco |
| Framework IA | `langchain` (1.3.14) + `langchain-community` (0.4.2) | Chains, retrievers, loaders, splitters |
| Core LangChain | `langchain-core` (1.5.0) | Prompts, parsers, runnables |
| SDK Google | `google-genai` (2.13.0) | SDK unificado de Google para Gemini (usado internamente por langchain-google-genai 4.x) |
| SDK legacy | `google-generativeai` (0.8.6) | Requerido por algunos componentes de embeddings |
| Visión artificial | Gemini Vision (vía `ChatGoogleGenerativeAI`) | Análisis multimodal de imágenes en base64 |
| Chunking | `RecursiveCharacterTextSplitter` | División de documentos en chunks con solapamiento |
| Carga de docs | `TextLoader` (langchain-community) | Lectura de archivos `.txt` de la base de conocimiento |
| Variables de entorno | `python-dotenv` | Carga de `.env` para la API Key |
| Procesamiento de imágenes | `Pillow` (12.3.0) | Lectura y codificación base64 de imágenes |
| Entorno de ejecución | Jupyter Notebook / JupyterLab | Interfaz interactiva del sistema |
| Python | 3.10+ (probado en 3.14.6 en Windows) | Lenguaje base del proyecto |

### Dependencias transitivas relevantes

| Paquete | Versión | Motivo |
|---|---|---|
| `langgraph` | 1.2.9 | Requerido por `langchain` 1.x |
| `pydantic` | 2.13.4 | Validación de datos y schemas |
| `google-genai` | 2.13.0 | SDK unificado Google (requerido por langchain-google-genai 4.x) |
| `chromadb` | 1.5.9 | Motor de vector store con persistencia SQLite |
| `onnxruntime` | 1.27.0 | Modelos de embeddings locales de Chroma |
| `numpy` | 2.5.1 | Operaciones vectoriales |

---

## Agentes implementados

### 1. 🎨 Agente de Marca y Lineamientos (`MarcaAgent`)

- **Clase:** `MarcaAgent(BaseRAGAgent)`
- **Fuente de conocimiento:** `01_Manual_de_Marca.txt`
- **Colección Chroma:** `marca_lineamientos`
- **Responde sobre:** logotipo (versiones, zonas de exclusión, tamaño mínimo 24px digital / 15mm impreso), paleta de colores primarios (Amarillo Patito `#FFC400`, Azul Patito `#1F4E79`) y secundarios, tipografías oficiales (Montserrat para titulares, Open Sans para cuerpo), tono de voz, usos correctos e incorrectos del logo, plantillas oficiales.

### 2. 📊 Agente de Campañas y Performance (`CampanasAgent`)

- **Clase:** `CampanasAgent(BaseRAGAgent)`
- **Fuente de conocimiento:** `02_Guia_Campanas_KPIs.txt`
- **Colección Chroma:** `campanas_kpis`
- **Responde sobre:** tipos de campaña (awareness, leads, conversión, fidelización), canales disponibles (Meta Ads, Instagram, TikTok, LinkedIn, Google Ads, email, landing pages), proceso de campaña en 6 pasos, KPIs obligatorios por objetivo (CPM, CPL, CPA, CTR, ROAS, tasa de conversión, leads, MQL), requisitos del reporte de cierre.

### 3. ⚖️ Agente de Cumplimiento Publicitario (`CumplimientoAgent`)

- **Clase:** `CumplimientoAgent(BaseRAGAgent)`
- **Fuente de conocimiento:** `03_Cumplimiento_Publicitario.txt`
- **Colección Chroma:** `cumplimiento_publicitario`
- **Responde sobre:** publicidad responsable y veracidad, claims permitidos y prohibidos, consentimiento explícito de marketing (opt-in) para email/SMS/WhatsApp, mecanismo de baja (opt-out), uso de datos de clientes, identificación de contenido pagado con influencers, checklist de validación pre-lanzamiento.

### 4. 🖼️ Agente Multimodal de Imagen (`MultimodalAgent`)

- **Clase:** `MultimodalAgent`
- **Tecnología:** Gemini Vision + RAG del Manual de Marca
- **Funcionamiento:** codifica la imagen en base64, recupera los chunks más relevantes del Manual de Marca como contexto, y construye un mensaje multimodal `HumanMessage` con imagen + prompt. Gemini analiza la imagen y la contrasta con los lineamientos de marca.
- **Output:** descripción del contenido visual, elementos de marca detectados (logo, colores, tipografías), incumplimientos específicos, veredicto final: **CUMPLE / NO CUMPLE / CUMPLE PARCIALMENTE**.
- **Formatos soportados:** JPG, JPEG, PNG, WEBP, GIF.

### 5. 📝 Agente de Acción — Registro de Campañas (`AccionAgent`)

- **Clase:** `AccionAgent`
- **Tecnología:** Gemini para extracción de datos estructurados (JSON) + validación de reglas de negocio
- **Funcionamiento en 3 pasos:**
  1. Extrae campos de la solicitud en lenguaje natural usando un prompt que retorna JSON.
  2. Valida que estén presentes todos los campos obligatorios; si faltan, los solicita.
  3. Muestra resumen de confirmación y espera respuesta `sí` / `no` antes de guardar.
- **Persistencia:** escribe el registro en `registro_campanas.txt` con UUID de 8 caracteres y timestamp.
- **Campos obligatorios:** nombre, objetivo, canales, público objetivo, presupuesto, fecha de inicio, fecha de fin. Además: consentimiento de marketing si el canal es email, SMS o WhatsApp.

### 6. 🧠 Orquestador (`Orchestrator`)

- **Clase:** `Orchestrator`
- **Funcionamiento:** gestiona el estado de sesión (`_awaiting_confirmation`), detecta imágenes adjuntas, y clasifica la intención de cada consulta usando Gemini con un prompt que retorna JSON con campos `agent`, `confidence` y `reasoning`. Instancia todos los agentes una sola vez al inicializarse.

---

## Estructura del notebook

El notebook contiene **41 celdas** organizadas en **14 secciones secuenciales**:

| # | Sección | Tipo | Contenido |
|---|---|---|---|
| — | Portada | Markdown | Descripción del sistema y tabla de agentes |
| 1 | Instalación de dependencias | Código | `%pip install` de todos los paquetes |
| 2 | Configuración | Código | Captura de API Key, parámetros del sistema (modelo, chunking, colecciones) |
| 3 | Base de conocimiento | Código | Strings con los 3 documentos; se escriben como `.txt` en disco |
| 4 | Servicio de embeddings | Código | `get_embeddings()`, `build_vector_store()`, `get_retriever()` |
| 5 | Clase base RAG | Código | `AgentResponse` (dataclass) y `BaseRAGAgent` con chain RAG |
| 6 | Agentes RAG especializados | Código | `MarcaAgent`, `CampanasAgent`, `CumplimientoAgent` |
| 7 | Agente Multimodal | Código | `MultimodalAgent` con Gemini Vision |
| 8 | Agente de Acción | Código | `AccionAgent` con extracción JSON, validación y escritura en `.txt` |
| 9 | Orquestador | Código | `Orchestrator` con clasificador de intención y enrutamiento |
| 10 | Inicialización | Código | `orquestador = Orchestrator(force_rebuild=False)` |
| 11 | Demos por agente | Código | 5 demos ejecutables (una por agente) |
| 12 | Consultas libres | Código | Celdas editables `MI_CONSULTA = "..."` |
| 13 | Ver campañas registradas | Código | Lectura de `registro_campanas.txt` |
| 14 | Reconstruir índices | Código | `Orchestrator(force_rebuild=True)` (comentado por defecto) |

---

## Archivos del repositorio

```
mesa-de-ayuda-ia-patito/
├── PROYECTO_SEMILLERO__MARKETING_.ipynb   ← notebook principal (único archivo de código)
└── README.md                              ← este archivo
```

**Archivos generados al ejecutar el notebook** (no se suben al repositorio):

```
├── chroma_db/                        ← índices vectoriales (ChromaDB en disco)
│   ├── marca_lineamientos/
│   ├── campanas_kpis/
│   └── cumplimiento_publicitario/
├── 01_Manual_de_Marca.txt            ← generado por Sección 3
├── 02_Guia_Campanas_KPIs.txt         ← generado por Sección 3
├── 03_Cumplimiento_Publicitario.txt  ← generado por Sección 3
└── registro_campanas.txt             ← generado al confirmar un registro
```

---

## Requisitos previos

- **Python 3.10 o superior** (probado en Python 3.14.6 en Windows)
- **Jupyter Notebook**, **JupyterLab** o **Google Colab**
- **Cuenta en Google AI Studio** con acceso a la API de Gemini
- **API Key de Google Gemini** (capa gratuita disponible)
- Conexión a internet activa (requerida para llamadas a Gemini y descarga de paquetes)

---

## Instalación

### Opción A — Jupyter local (Windows / Linux / macOS)

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/<tu-repositorio>.git
cd <tu-repositorio>

# 2. Crear y activar entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Instalar Jupyter
pip install notebook

# 4. Abrir el notebook
jupyter notebook
```

Abre `PROYECTO_SEMILLERO__MARKETING_.ipynb` desde el explorador de Jupyter.

### Opción B — Google Colab (sin instalación local)

1. Ve a [colab.research.google.com](https://colab.research.google.com/).
2. Haz clic en **Archivo → Subir notebook**.
3. Selecciona `PROYECTO_SEMILLERO__MARKETING_.ipynb`.
4. Ejecuta las celdas en orden.

> La **Sección 1** del notebook instala automáticamente todas las dependencias con `%pip install`. No necesitas `requirements.txt` externo.

### Paquetes que instala la Sección 1

```
langchain-google-genai   → integración LangChain con Gemini 4.x (instala google-genai 2.x)
langchain                → framework principal
langchain-community      → TextLoader y componentes de comunidad
langchain-chroma         → integración LangChain + ChromaDB
chromadb                 → motor de vector store con persistencia SQLite
google-generativeai      → SDK legacy de Google (requerido por algunos componentes)
python-dotenv            → carga de variables de entorno
Pillow                   → procesamiento de imágenes (codificación base64)
```

---

## Configuración de la API Key

### Obtener la API Key

1. Ve a [Google AI Studio](https://aistudio.google.com/).
2. Inicia sesión con tu cuenta de Google.
3. Haz clic en **"Get API key"** → **"Create API key"**.
4. Copia la key generada

### Configurar en el notebook

En la **Sección 2**, el notebook solicita la API Key de dos formas:

**Opción 1 — Input interactivo:**
```python
GOOGLE_API_KEY = input("🔑 Ingresa tu Google API Key: ").strip()
```

**Opción 2 — Variable de entorno con `.env`:**
```env
GOOGLE_API_KEY=AIzaSy...tu_key_aqui
```

> ⚠️ Nunca subas tu API Key al repositorio. Antes de hacer commit, ejecuta `Kernel → Restart & Clear Output` para limpiar los outputs de todas las celdas.

---

## Ejecución paso a paso

Ejecuta el notebook **de arriba hacia abajo**. Usa `Shift + Enter` para ejecutar celda por celda, o `Kernel → Restart & Run All` para ejecutar todo de una vez.

### Orden de ejecución

```
Sección 1  →  Instalar dependencias
               ↓  (reiniciar kernel si Jupyter lo pide)
Sección 2  →  Configurar API Key y parámetros
Sección 3  →  Crear archivos .txt de la base de conocimiento en disco
Sección 4  →  Definir funciones de embeddings y vector store
Sección 5  →  Definir BaseRAGAgent y AgentResponse
Sección 6  →  Definir los 3 agentes RAG especializados
Sección 7  →  Definir el agente multimodal
Sección 8  →  Definir el agente de acción
Sección 9  →  Definir el orquestador
Sección 10 →  Inicializar el sistema
               ↓  PRIMERA VEZ: ~15-30s (genera embeddings y persiste en chroma_db/)
               ↓  EJECUCIONES SIGUIENTES: instantáneo (carga desde disco)
Secciones 11-14 → Demos y consultas
```

### Salida esperada de la Sección 10

**Primera ejecución:**
```
⏳ Iniciando agentes...
  ⏳ Construyendo índice: marca_lineamientos
     12 chunks generados
  ⏳ Construyendo índice: campanas_kpis
     10 chunks generados
  ⏳ Construyendo índice: cumplimiento_publicitario
     11 chunks generados
✅ Orquestador listo. Todos los agentes activos.
```

**Ejecuciones posteriores:**
```
⏳ Iniciando agentes...
  ↩️  Cargando índice existente: marca_lineamientos
  ↩️  Cargando índice existente: campanas_kpis
  ↩️  Cargando índice existente: cumplimiento_publicitario
✅ Orquestador listo. Todos los agentes activos.
```

---

## Guía de uso y ejemplos

### Sintaxis básica

```python
resultado = orquestador.process("tu consulta aquí")
resultado.display()
```

### Formato de salida

```
════════════════════════════════════════════════════════════
  📨 Consulta : ¿Cuáles son los colores primarios de Patito S.A.?
  🎯 Agente   : Agente de Marca y Lineamientos
  📊 Confianza: 95%  |  Pregunta sobre paleta de colores → agente de marca
════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────
🤖  Agente de Marca y Lineamientos
────────────────────────────────────────────────────────────
Los colores primarios de Patito S.A. son el Amarillo Patito (#FFC400)
y el Azul Patito (#1F4E79)...

📄 Fuente: 01_Manual_de_Marca.txt
────────────────────────────────────────────────────────────
```

### Ejemplos por agente

**Agente de Marca:**
```python
orquestador.process("¿Cuándo debo usar la versión blanca del logotipo?").display()
orquestador.process("¿Puedo cambiar los colores primarios para una campaña de verano?").display()
orquestador.process("¿Cuál es la tipografía oficial para titulares?").display()
```

**Agente de Campañas:**
```python
orquestador.process("¿Qué KPIs son obligatorios en el reporte de cierre de una campaña de leads?").display()
orquestador.process("¿Qué canales están disponibles para campañas de conversión?").display()
orquestador.process("¿Qué debe incluir el reporte de cierre de una campaña de awareness?").display()
```

**Agente de Cumplimiento:**
```python
orquestador.process("¿Puedo usar el claim 'el mejor precio garantizado'?").display()
orquestador.process("¿Puedo enviar emails a clientes que no dieron consentimiento?").display()
orquestador.process("¿Cómo debo identificar el contenido pagado con influencers?").display()
```

---

## Flujo de registro de campaña

El `AccionAgent` implementa un flujo de validación en 3 pasos con confirmación antes de guardar:

```
Usuario envía solicitud
         │
         ▼
AccionAgent extrae campos con Gemini (JSON)
         │
         ├─ Faltan campos → respuesta con lista de campos faltantes
         │
         └─ Todos los campos presentes
               → Gemini genera resumen legible
               → "¿Confirmas el registro? (sí/no)"
               → Orchestrator._awaiting_confirmation = True
                      │
                      ├─ "sí" → genera UUID + timestamp → append a .txt
                      └─ "no" → cancela sin guardar nada
```

**Campos obligatorios:**

| Campo | Descripción |
|---|---|
| `nombre` | Nombre de la campaña |
| `objetivo` | `awareness` / `leads` / `conversión` / `fidelización` |
| `canales` | Lista de canales (Instagram, Facebook, Google Ads, etc.) |
| `publico_objetivo` | Descripción del segmento target |
| `presupuesto` | Monto y moneda (ej: `USD 800`) |
| `fecha_inicio` | Fecha de inicio |
| `fecha_fin` | Fecha de fin |
| `consentimiento_marketing` | Solo requerido si el canal es email, SMS o WhatsApp |

**Ejemplo — Demo 4 (Sección 11):**

```python
# Paso 1: solicitud incompleta
orquestador.process("Quiero registrar una campaña de Instagram").display()
# → Solicita: nombre, objetivo, público, presupuesto, fechas

# Paso 2: solicitud completa
orquestador.process(
    "Campaña 'Verano Patito' en Instagram y Facebook, objetivo conversión, "
    "público jóvenes 18-30 años, presupuesto USD 800, del 1 al 31 de julio"
).display()
# → Muestra resumen y pide confirmación

# Paso 3: confirmar
orquestador.process("sí").display()
# → ✅ Campaña registrada. ID: A04378B7
```

**Registro guardado en `registro_campanas.txt`:**

```
=======================================================
ID: A04378B7 | Registrado: 2026-07-01 13:19:09
Nombre: Verano Patito
Objetivo: conversión
Canales: Instagram, Facebook
Público: jóvenes 18-30 años
Presupuesto: USD 800
Período: 1 de julio → 31 de julio
=======================================================
```

---

## Análisis multimodal de imágenes

```python
resultado = orquestador.process(
    query="¿Los colores y tipografías cumplen con el Manual de Marca?",
    image_path="./mi_banner.png"
)
resultado.display()
```

### Cómo subir una imagen

**Jupyter local:** copia la imagen a la misma carpeta del notebook.

**Google Colab:** panel izquierdo → ícono de carpeta 📁 → arrastra la imagen desde tu PC.

> En Colab las imágenes se eliminan al reiniciar el entorno. Debes volver a subirlas en cada sesión.

### Qué imagen analizar

Cualquier pieza gráfica de marketing: banner, post de redes, flyer, portada de presentación. Puede ser de cualquier marca — el agente la contrastará contra el Manual de Marca de Patito S.A. y señalará lo que no coincide, lo que es válido para demostrar el funcionamiento del sistema.

**Formatos soportados:** JPG, JPEG, PNG, WEBP, GIF.

---

## Base de conocimiento

Los tres documentos están embebidos como strings en la **Sección 3** y se generan en disco al ejecutarla:

| Archivo | Contenido principal |
|---|---|
| `01_Manual_de_Marca.txt` | Logotipo (versiones, exclusiones, tamaño mínimo 24px / 15mm), paleta (`#FFC400`, `#1F4E79`, `#555555`, `#6FCF97`), tipografías (Montserrat, Open Sans), tono de voz, plantillas |
| `02_Guia_Campanas_KPIs.txt` | 4 tipos de campaña, 5 canales, proceso de 6 pasos, KPIs por objetivo (CPM, CPL, CTR, ROAS, CPA, MQL, tasa de conversión), reporte de cierre |
| `03_Cumplimiento_Publicitario.txt` | Claims permitidos/prohibidos, consentimiento opt-in, opt-out, protección de datos, contenido pagado con influencers, checklist pre-lanzamiento |

> Documentos ficticios creados exclusivamente para fines académicos del semillero.

### Parámetros de chunking (Sección 2)

```python
CHUNK_SIZE      = 400   # caracteres por chunk
CHUNK_OVERLAP   = 80    # solapamiento entre chunks consecutivos
TOP_K_RETRIEVAL = 4     # chunks recuperados por consulta (similitud coseno)
```

### Actualizar la base de conocimiento

1. Edita el string del documento en la Sección 3.
2. Vuelve a ejecutar la Sección 3.
3. Reconstruye los índices desde la Sección 14:

```python
orquestador = Orchestrator(force_rebuild=True)
```

---

## Archivos generados en runtime

| Archivo / Carpeta | Cuándo se crea | Descripción |
|---|---|---|
| `chroma_db/marca_lineamientos/` | Sección 10, primera ejecución | Índice vectorial del Manual de Marca |
| `chroma_db/campanas_kpis/` | Sección 10, primera ejecución | Índice vectorial de Campañas y KPIs |
| `chroma_db/cumplimiento_publicitario/` | Sección 10, primera ejecución | Índice vectorial de Cumplimiento |
| `01_Manual_de_Marca.txt` | Sección 3 | Documento fuente de marca |
| `02_Guia_Campanas_KPIs.txt` | Sección 3 | Documento fuente de campañas |
| `03_Cumplimiento_Publicitario.txt` | Sección 3 | Documento fuente de cumplimiento |
| `registro_campanas.txt` | Al confirmar un registro | Historial acumulativo de campañas (append) |

Ninguno de estos archivos debe subirse al repositorio de GitHub.

---

## Personalización

### Cambiar el modelo Gemini

```python
GEMINI_MODEL = "gemini-2.0-flash"   # rápido, por defecto
GEMINI_MODEL = "gemini-1.5-pro"     # mayor razonamiento, más lento
GEMINI_MODEL = "gemini-3.5-flash"   # versión más reciente
```

### Ajustar parámetros RAG

```python
CHUNK_SIZE      = 400   # aumentar para más contexto por chunk
CHUNK_OVERLAP   = 80    # aumentar para menos cortes abruptos entre chunks
TOP_K_RETRIEVAL = 4     # aumentar para más contexto, disminuir para más precisión
```

### Agregar un nuevo agente

1. Define una nueva clase en la Sección 6 heredando de `BaseRAGAgent`.
2. Agrega su documento fuente en la Sección 3 y su colección en la Sección 2.
3. Instáncialo en la clase `Orchestrator` (Sección 9).
4. Agrega la categoría al `CLASSIFIER_PROMPT` del orquestador.
5. Reconstruye los índices con `Orchestrator(force_rebuild=True)`.

---

## Licencia

Este proyecto se distribuye bajo la licencia **MIT**. Los documentos de la base de conocimiento son contenido ficticio creado exclusivamente con fines académicos para el Semillero de Inteligencia Artificial.

```
MIT License — Copyright (c) 2025 <tu-nombre>

Se permite el uso, copia, modificación y distribución de este software
sin restricción, siempre que se incluya este aviso de copyright.
```
