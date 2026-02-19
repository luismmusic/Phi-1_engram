import torch
from transformers import AutoTokenizer
from phi1_engram import PhiEngramConfig, PhiEngramForCausalLM
import os

# =============================================================================
# CHAT INTERACTIVO CON PHI-1 ENGRAM
# =============================================================================

def run_chat():
    # Detección de dispositivo y optimización de memoria
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "microsoft/phi-1"

    print("--------------------------------------------------")
    print("🤖 CARGANDO CHAT DE PHI-1 ENGRAM")
    print(f"Dispositivo detectado: {device.upper()}")
    print(f"Precisión utilizada: {dtype}")

    if device == "cpu":
        print("⚠️  ADVERTENCIA: Estás usando la CPU. El modelo será MUY lento.")
        print("En Colab, ve a: Entorno de ejecución -> Cambiar tipo de entorno -> T4 GPU")
    print("--------------------------------------------------")

    # 1. Configuración del modelo
    # Nota: Usamos una configuración estándar compatible con los pesos de Phi-1
    config = PhiEngramConfig(
        vocab_size=51200,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_key_value_heads=32,
        engram_layer_ids=[1, 15] # Ubicación de la memoria Engram
    )

    # 2. Cargar modelo y pesos
    print("[1/2] Instanciando arquitectura (Modo Eficiente)...")
    # Usamos half precision (fp16) para reducir el consumo de VRAM a la mitad
    model = PhiEngramForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()

    # 3. Cargar Tokenizador
    print("[2/2] Cargando traductor (tokenizer)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n✅ ¡SISTEMA LISTO!")
    print("Nota: Este es un modelo base. Para mejores resultados,")
    print("usa el formato 'User: pregunta \nAssistant:'")
    print("(Escribe 'salir' para terminar)\n")

    while True:
        user_input = input("👤 Tú: ")

        if user_input.lower() in ["salir", "exit", "quit"]:
            print("👋 ¡Adiós!")
            break

        # Formateamos el prompt para guiar al modelo
        prompt = f"User: {user_input}\nAssistant:"

        # Convertimos texto a números
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generamos la respuesta
        print("🤖 Phi-Engram escribiendo...", end="", flush=True)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=100,     # Aumentamos un poco el límite
                do_sample=True,         # Permite creatividad
                temperature=0.7,        # Nivel de aleatoriedad
                top_p=0.9,              # Filtro de palabras probables
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True          # Crucial para rendimiento
            )

        # Traducimos de números a palabras
        full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Extraemos solo la parte que escribió el modelo (después de 'Assistant:')
        response = full_text.split("Assistant:")[-1].strip()

        print(f"\r🤖 Phi-Engram: {response}\n")

        # Limpieza de memoria para evitar picos de consumo (solo en GPU)
        if device == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        run_chat()
    except KeyboardInterrupt:
        print("\n\n👋 Chat interrumpido por el usuario.")
