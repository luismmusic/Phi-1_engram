# Microsoft Phi-1 con DeepSeek Engram: Guía Completa para Principiantes

Esta guía te enseñará, paso a paso y sin omitir detalles, cómo utilizar esta versión avanzada del modelo de lenguaje **Phi-1** que incluye el módulo **Engram** de DeepSeek. Engram permite al modelo tener una "memoria" eficiente para patrones de texto repetitivos, mejorando su capacidad sin hacerlo más lento.

---

## 🚀 Flujo de Trabajo en la Nube con Git (Método Profesional)

Dado que estás trabajando en la nube (Google Colab, Kaggle), la forma más rápida y asertiva de obtener este código es utilizando **Git**. Git te permite "clonar" (copiar exactamente) todo este proyecto en tu entorno virtual en segundos.

### ¿Por qué usar Git en lugar de subir archivos a mano?
1.  **Velocidad**: No tienes que descargar y luego subir archivos. Todo sucede directamente en la nube.
2.  **Integridad**: Te aseguras de tener todos los archivos necesarios y sus versiones correctas.
3.  **Persistencia**: Si tu entorno de nube se reinicia, solo tienes que volver a ejecutar un comando para recuperar todo.

---

## 1. Entendiendo los Archivos

En este repositorio encontrarás dos archivos principales de código:

1.  **`phi1_engram.py`**: Este es el "cerebro" del modelo. Contiene todas las fórmulas matemáticas y la estructura necesaria para que Phi-1 y Engram trabajen juntos. **No necesitas modificarlo**.
2.  **`verify_phi_engram.py`**: Es un script de prueba. Su función es verificar que el modelo esté bien instalado y que funcione perfectamente.

---

## 2. Cómo probarlo en Google Colab

Sigue estos pasos exactos y exhaustivos:

1.  **Abre Google Colab**: Ve a [colab.research.google.com](https://colab.research.google.com).
2.  **Crea un Notebook nuevo**: Haz clic en el botón "Nuevo cuaderno" (New notebook).
3.  **🔴 PASO CRÍTICO: Activar GPU**:
    *   Sin esto, el modelo será extremadamente lento y consumirá demasiados recursos de CPU.
    *   Ve al menú superior: **Entorno de ejecución** -> **Cambiar tipo de entorno de ejecución**.
    *   En "Acelerador de hardware", selecciona **T4 GPU** y haz clic en "Guardar".
5.  **Paso A: Instalar librerías**: Copia y ejecuta este comando en la primera celda:
    ```bash
    !pip install torch transformers tokenizers numpy sympy
    ```
6.  **Paso B: Clonar el proyecto con Git**: Crea una celda nueva y ejecuta este comando (reemplaza la URL si es necesario):
    ```bash
    !git clone https://github.com/tu-usuario/Phi-1_engram.git
    %cd Phi-1_engram
    ```
    *Nota: El símbolo `!` indica a Colab que ejecute un comando del sistema, y `%cd` cambia la carpeta de trabajo.*
7.  **Paso C: Ejecutar la verificación**: Ejecuta este comando en otra celda:
    ```bash
    !python verify_phi_engram.py
    ```
    Si ves un mensaje de éxito con un check verde (✅), ¡el modelo está listo!

---

## 3. Cómo usarlo en Kaggle

Kaggle es ideal para entrenamiento pesado. Sigue estas instrucciones detalladas:

1.  **Inicia sesión**: Ve a [kaggle.com](https://www.kaggle.com).
2.  **Crea un Notebook**: Haz clic en `+ Create` -> `New Notebook`.
3.  **Configura el entorno**:
    *   En el panel derecho ("Settings"), activa **Internet on**.
    *   En **Accelerator**, selecciona **GPU T4 x2**.
4.  **Descarga el código con Git**: En la primera celda, escribe y ejecuta:
    ```bash
    !pip install torch transformers tokenizers numpy sympy
    !git clone https://github.com/tu-usuario/Phi-1_engram.git
    ```
5.  **Entra en la carpeta**:
    ```python
    import os
    os.chdir("/kaggle/working/Phi-1_engram")
    ```
6.  **Prueba el modelo**:
    ```bash
    !python verify_phi_engram.py
    ```

---

## 4. Cómo integrarlo con Hugging Face

Hugging Face es el repositorio central donde guardarás tus modelos entrenados.

1.  **Crea una cuenta**: Regístrate en [huggingface.co](https://huggingface.co).
2.  **Uso desde cualquier lugar**: Una vez subido tu modelo, puedes cargarlo directamente así:
    ```python
    from transformers import AutoModelForCausalLM
    # Reemplaza 'tu-usuario' por tu nombre real
    model = AutoModelForCausalLM.from_pretrained("tu-usuario/phi1-engram", trust_remote_code=True)
    ```

---

## 5. Ejemplo de "Uso Básico" Explicado

Aquí tienes un código que puedes copiar y pegar. Está diseñado para ser asertivo y directo:

```python
import torch
from phi1_engram import PhiEngramConfig, PhiEngramForCausalLM
from transformers import AutoTokenizer

# 1. Configuración: Definimos las características del modelo
config = PhiEngramConfig(
    hidden_size=256,
    num_hidden_layers=2,
    vocab_size=51200
)

# 2. Creación: Construimos el modelo basado en la configuración
model = PhiEngramForCausalLM(config)

# 3. Traducción: Preparamos el tokenizer oficial de Microsoft
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")

# 4. Procesamiento: Convertimos texto a números y pedimos un resultado
texto = "Hola, mundo!"
inputs = tokenizer(texto, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print("¡Éxito! El modelo generó un tensor de tamaño:", outputs.logits.shape)
```

---

## 6. Ciclo de Trabajo Interrelacionado (El Flujo Ideal)

Para un desarrollo profesional en la nube, sigue este orden:

1.  **GIT**: Usa Git para clonar este código en cualquier plataforma.
2.  **KAGGLE**: Entrena el modelo usando sus GPUs potentes.
3.  **HUGGING FACE**: Sube el modelo entrenado a tu Hub para guardarlo.
4.  **COLAB**: Descarga tu modelo desde Hugging Face para mostrar resultados o hacer pruebas rápidas.

---

## 📚 Documentación Técnica Detallada

Si eres un desarrollador avanzado o investigador, hemos preparado un documento exhaustivo que explica las ecuaciones matemáticas, la arquitectura de hashing y los detalles de integración:

👉 **[Consulta la Documentación Técnica Completa aquí](./DOCUMENTACION_TECNICA.md)**

---

## 🛠️ Próximos Pasos Técnicos y Hoja de Ruta

Para avanzar de este prototipo a un modelo entrenado y optimizado, consulta nuestra **Hoja de Ruta Detallada** que incluye:
1.  **Bootstrapping**: Cómo cargar los pesos de Phi-1 manteniendo la memoria Engram.
2.  **Fine-tuning Estratégico**: Fases de entrenamiento (Warm-up vs Conjunto).
3.  **Optimizaciones de Memoria**: Implementación de Prefetching y CPU Offloading.
4.  **Validación de Profundidad**: Uso de LogitLens y CKA.

👉 **[Ver Hoja de Ruta en la Documentación Técnica](./DOCUMENTACION_TECNICA.md#7-próximos-pasos-técnicos-y-hoja-de-ruta)**

---

## 🎓 Tutorial: Carga de Pesos y Entrenamiento

Para usar el modelo real con el conocimiento de Microsoft, sigue este flujo de trabajo completo:

### 1. Cargar Pesos Oficiales
Ejecuta el script `load_phi_engram.py`. Este script descargará los pesos de Phi-1 (aprox. 2.6GB) y los inyectará en la nueva arquitectura. Las partes de Engram se mantendrán nuevas (inicializadas aleatoriamente) listas para aprender.
```bash
python load_phi_engram.py
```

### 2. Entrenamiento en Dos Fases
El archivo `train_phi_engram.py` contiene la lógica para entrenar el modelo correctamente:

- **Fase 1 (Warm-up)**: Se congela el "cerebro" (Phi-1) y solo se entrena la "memoria" (Engram). Esto evita que el modelo olvide lo que ya sabe mientras se adapta a la nueva estructura.
- **Fase 2 (Joint Fine-tuning)**: Se entrena todo el modelo. Engram usa un Learning Rate 5 veces más alto para capturar patrones rápidamente, mientras que Phi-1 se ajusta suavemente.

**Para iniciar el entrenamiento**:
```bash
python train_phi_engram.py
```

### 3. Guardar y Compartir
Al finalizar, el script creará una carpeta `phi1-engram-trained` con todo lo necesario para subirlo a Hugging Face o usarlo en tus proyectos.

---

## 💬 Cómo Hablar con el Modelo (Modo Chat)

Si quieres probar la capacidad conversacional del modelo, utiliza el script interactivo:

1.  **Ejecuta el Chat**:
    ```bash
    python chat_phi_engram.py
    ```
2.  **Cómo interactuar**:
    - El script te pedirá que escribas un mensaje: `👤 Tú:`.
    - Escribe tu pregunta y presiona `Enter`.
    - El modelo responderá como `🤖 Phi-Engram:`.
3.  **Optimización de Recursos**:
    El script de chat ahora utiliza **precisión reducida (FP16)** automáticamente si detecta una GPU. Esto reduce el consumo de memoria RAM tanto del sistema como de la GPU a la mitad, evitando el cierre del entorno por falta de memoria.
4.  **En caso de error "Out of Memory" (OOM)**:
    Si recibes un error de memoria agotada:
    - Reinicia el entorno de ejecución (**Entorno de ejecución** -> **Reiniciar sesión**).
    - No ejecutes otros modelos pesados en la misma sesión.
5.  **Consejo de experto**:
    Como Phi-1 es un modelo base (no entrenado específicamente para chat), funciona mejor si le haces preguntas directas o le pides completar frases.
