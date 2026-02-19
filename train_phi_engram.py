import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from load_phi_engram import load_phi_1_with_engram

# =============================================================================
# 1. PREPARACIÓN DE DATOS (Dataset de ejemplo)
# =============================================================================
class ToyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings.input_ids[idx],
            'attention_mask': self.encodings.attention_mask[idx],
            'labels': self.encodings.input_ids[idx].clone() # En CLM, labels son los mismos inputs
        }

# =============================================================================
# 2. LÓGICA DE ENTRENAMIENTO (Dos Fases)
# =============================================================================
def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Entrenando en: {device}")

    # Cargar modelo y pesos de Phi-1
    # Nota: En una ejecución real, load_phi_1_with_engram descargará >2GB de pesos.
    model = load_phi_1_with_engram().to(device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datos de prueba (Novela de ejemplo)
    texts = [
        "Alexander the Great was a king of the ancient Greek kingdom of Macedon.",
        "The Milky Way is the galaxy that contains our Solar System.",
        "Princess of Wales is a title usually held by the wife of the Prince of Wales."
    ]
    dataset = ToyDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # ---------------------------------------------------------
    # FASE 1: Warm-up de Engram (Backbone Congelado)
    # ---------------------------------------------------------
    print("\n>>> INICIANDO FASE 1: Warm-up (Solo entrena Engram)")

    # Congelamos todo el modelo
    for param in model.parameters():
        param.requires_grad = False

    # Descongelamos SOLO los módulos Engram
    engram_params = []
    for name, module in model.named_modules():
        if "engram" in name.lower():
            for param in module.parameters():
                param.requires_grad = True
                engram_params.append(param)

    optimizer_fase1 = torch.optim.Adam(engram_params, lr=1e-4)

    model.train()
    for batch in dataloader:
        optimizer_fase1.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer_fase1.step()
        print(f"Fase 1 - Loss: {loss.item():.4f}")

    # ---------------------------------------------------------
    # FASE 2: Fine-tuning Conjunto (Todo Descongelado)
    # ---------------------------------------------------------
    print("\n>>> INICIANDO FASE 2: Fine-tuning Conjunto")

    # Descongelamos todo
    for param in model.parameters():
        param.requires_grad = True

    # Usamos diferentes tasas de aprendizaje (LR) según el paper:
    # Engram LR debe ser mayor que el Backbone LR.
    backbone_params = []
    engram_params = []
    for name, param in model.named_parameters():
        if "engram" in name.lower():
            engram_params.append(param)
        else:
            backbone_params.append(param)

    optimizer_fase2 = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-5}, # LR bajo para Phi-1
        {'params': engram_params, 'lr': 5e-5}   # LR alto para Engram (5x)
    ])

    for batch in dataloader:
        optimizer_fase2.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer_fase2.step()
        print(f"Fase 2 - Loss: {loss.item():.4f}")

    print("\n✅ Proceso de entrenamiento finalizado correctamente.")

    # Guardar el modelo entrenado
    print("Guardando modelo...")
    model.save_pretrained("phi1-engram-trained")
    tokenizer.save_pretrained("phi1-engram-trained")

if __name__ == "__main__":
    # print("Script de entrenamiento listo.")
    # Evitamos la ejecución completa para prevenir timeouts en el sandbox
    pass
