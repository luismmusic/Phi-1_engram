import torch
from transformers import AutoModelForCausalLM, AutoConfig
from phi1_engram import PhiEngramConfig, PhiEngramForCausalLM

def load_phi_1_with_engram(engram_layers=[1, 15]):
    """
    Carga los pesos oficiales de Microsoft Phi-1 y los transfiere a nuestra
    arquitectura aumentada con DeepSeek Engram.
    """
    print(f"--- Iniciando carga de Microsoft Phi-1 ---")
    model_id = "microsoft/phi-1"

    # 1. Cargamos la configuración oficial
    print(f"[1/4] Descargando configuración de {model_id}...")
    base_config = AutoConfig.from_pretrained(model_id)

    # 2. Creamos nuestra configuración de Engram basada en la oficial
    print(f"[2/4] Preparando configuración con Engram en capas: {engram_layers}...")
    config = PhiEngramConfig(
        **base_config.to_dict(),
        engram_layer_ids=engram_layers,
        tokenizer_name_or_path=model_id # Usamos el mismo tokenizador
    )

    # 3. Instanciamos nuestro modelo (inicializado aleatoriamente al principio)
    print(f"[3/4] Instanciando arquitectura Phi-1 + Engram...")

    # Optimizamos para no agotar la RAM del sistema
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_default_dtype(torch.float16)

    model = PhiEngramForCausalLM(config).to(device)

    torch.set_default_dtype(torch.float32)

    # 4. Cargamos los pesos oficiales del Hub
    print(f"[4/4] Cargando pesos oficiales de Microsoft (esto puede tardar)...")
    official_model = AutoModelForCausalLM.from_pretrained(model_id)

    # Transferimos los pesos. strict=False es CRÍTICO aquí porque nuestro modelo
    # tiene capas nuevas (las de Engram) que NO están en el modelo oficial.
    missing_keys, unexpected_keys = model.load_state_dict(official_model.state_dict(), strict=False)

    print("\n--- Carga Completada con Éxito ---")
    print(f"Capas cargadas (Backbone): {len(official_model.state_dict().keys())}")
    print(f"Capas nuevas (Engram) inicializadas aleatoriamente: {len(missing_keys)}")

    # Liberamos memoria del modelo oficial
    del official_model

    return model

if __name__ == "__main__":
    # Prueba rápida de carga (sin descargar pesos pesados si es posible para evitar timeouts)
    # En un entorno real, esto cargaría todo el modelo.
    print("Módulo de carga listo.")
