# Microsoft Phi-1 con DeepSeek Engram: Guía Completa para Principiantes

Esta guía te enseñará, paso a paso y sin omitir detalles, cómo utilizar esta versión avanzada del modelo de lenguaje **Phi-1** que incluye el módulo **Engram** de DeepSeek. Engram permite al modelo tener una "memoria" eficiente para patrones de texto repetitivos, mejorando su capacidad sin hacerlo más lento.

---

## 1. Entendiendo los Archivos

En este repositorio encontrarás dos archivos principales de código:

1.  **`phi1_engram.py`**: Este es el "cerebro" del modelo. Contiene todas las fórmulas matemáticas y la estructura necesaria para que Phi-1 y Engram trabajen juntos. **No necesitas modificarlo**, solo asegúrate de que esté en la misma carpeta que tus scripts.
2.  **`verify_phi_engram.py`**: Es un script de prueba. Su función es verificar que el modelo esté bien instalado y que pueda generar texto tanto de forma rápida (por bloques) como paso a paso (token por token).

---

## 2. Cómo probarlo en Google Colab (Recomendado para empezar)

Google Colab te permite ejecutar código en la nube de forma gratuita. Sigue estos pasos exactos:

1.  **Abre Google Colab**: Ve a [colab.research.google.com](https://colab.research.google.com).
2.  **Crea un Notebook nuevo**: Haz clic en el botón "Nuevo cuaderno" (New notebook).
3.  **Configura el entorno**:
    *   Copia y pega este comando en la primera celda y presiona el botón de "Play" (o pulsa `Ctrl+Enter`):
        ```bash
        !pip install torch transformers tokenizers numpy sympy
        ```
4.  **Sube los archivos**:
    *   En el menú de la izquierda, haz clic en el icono de la **carpeta**.
    *   Arrastra los archivos `phi1_engram.py` y `verify_phi_engram.py` desde tu computadora al panel de la izquierda en Colab.
5.  **Ejecuta la prueba**:
    *   Crea una celda nueva abajo y escribe:
        ```bash
        !python verify_phi_engram.py
        ```
    *   Presiona "Play". Verás mensajes indicando que el modelo se está instanciando y verificando. Si ves un mensaje de éxito con un check verde (✅), ¡todo está perfecto!

---

## 3. Cómo usarlo en Kaggle (Para entrenamiento pesado)

Kaggle es ideal porque ofrece GPUs (procesadores gráficos) muy potentes de forma gratuita por tiempo limitado.

1.  **Inicia sesión**: Ve a [kaggle.com](https://www.kaggle.com) y crea una cuenta si no tienes una.
2.  **Crea un Notebook**: Haz clic en `+ Create` -> `New Notebook`.
3.  **Activa Internet y GPU**:
    *   En el panel derecho ("Settings"), busca **Internet on** y actívalo (necesario para descargar el modelo base).
    *   En **Accelerator**, selecciona **GPU T4 x2**.
4.  **Instala las herramientas**:
    *   En la primera celda escribe y ejecuta:
        ```bash
        !pip install torch transformers tokenizers numpy sympy
        ```
5.  **Carga el código**: Puedes copiar el contenido de `phi1_engram.py` directamente en una celda de Kaggle o subir el archivo usando el botón `+ Add Data` -> `Upload` (en la pestaña superior).

---

## 4. Cómo integrarlo con Hugging Face

Hugging Face es como el "GitHub" de la Inteligencia Artificial. Sirve para guardar y compartir tus modelos.

1.  **Crea una cuenta**: Regístrate en [huggingface.co](https://huggingface.co).
2.  **Crea un Repositorio (Model)**:
    *   Haz clic en tu perfil -> `New Model`.
    *   Ponle un nombre (ej: `phi1-engram-test`).
3.  **Sube tus archivos**:
    *   Ve a la pestaña `Files and versions`.
    *   Haz clic en `Add file` -> `Upload files`.
    *   Sube `phi1_engram.py`.
4.  **Uso desde cualquier lugar**:
    *   Una vez subido, puedes cargar el modelo en cualquier computadora del mundo usando este código:
        ```python
        from transformers import AutoModelForCausalLM
        # Reemplaza 'tu-usuario' por tu nombre real en Hugging Face
        model = AutoModelForCausalLM.from_pretrained("tu-usuario/phi1-engram-test", trust_remote_code=True)
        ```

---

## 5. Ejemplo de "Uso Básico" Explicado

Si quieres usar el modelo para generar texto por tu cuenta, aquí tienes un ejemplo que puedes copiar en Colab. Está explicado línea por línea:

```python
import torch
# Importamos las clases que creamos en phi1_engram.py
from phi1_engram import PhiEngramConfig, PhiEngramForCausalLM
from transformers import AutoTokenizer

# 1. Definimos la 'receta' (configuración) del modelo
config = PhiEngramConfig(
    hidden_size=256,        # Tamaño de las capas internas
    num_hidden_layers=2,    # Número de capas del modelo
    vocab_size=51200        # Tamaño del diccionario de palabras
)

# 2. Creamos el modelo físico basado en esa receta
model = PhiEngramForCausalLM(config)

# 3. Preparamos el 'traductor' (tokenizer) para que el modelo entienda el texto
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")

# 4. Convertimos una frase en números que el modelo puede procesar
texto = "Hola, mundo!"
inputs = tokenizer(texto, return_tensors="pt")

# 5. Le pedimos al modelo que procese los números y nos dé un resultado (logits)
with torch.no_grad(): # Esto apaga el 'modo entrenamiento' para ahorrar memoria
    outputs = model(**inputs)

# 6. Imprimimos el tamaño del resultado
print("¡Éxito! El modelo generó un tensor de tamaño:", outputs.logits.shape)
```

---

## 6. Ciclo de Trabajo Recomendado (Flujo Pro)

Si quieres ser un experto, usa los tres recursos juntos:

1.  **Entrena en Kaggle**: Usa sus GPUs gratuitas para que el modelo aprenda.
2.  **Guarda en Hugging Face**: Sube el resultado a tu perfil de Hugging Face para no perderlo.
3.  **Prueba en Colab**: Descarga el modelo desde Hugging Face a Colab para hacer demos rápidas o compartirlo con amigos.

---

## Próximos Pasos Técnicos
Si ya dominas lo anterior, consulta la sección técnica en el archivo `phi1_engram.py` para aprender a cargar los pesos reales de Microsoft y empezar un entrenamiento formal (Fine-tuning).
