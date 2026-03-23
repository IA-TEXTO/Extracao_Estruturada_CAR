from transformers import pipeline
import torch
import os
import re

MODEL = "pierreguillou/ner-bert-large-cased-pt-lenerbr"

device = 0 if torch.cuda.is_available() else -1

print("GPU ativada" if device == 0 else "Rodando na CPU")

ner = pipeline(
    "ner",
    model=MODEL,
    aggregation_strategy="simple",
    device=device
)

def limpar_token(texto):
    """
    Remove ## e corrige espaços quebrados
    """
    texto = texto.replace("##", "")
    texto = re.sub(r"\s+", " ", texto)
    texto = texto.strip()
    return texto


def pos_processar(entidades):
    """
    Junta entidades consecutivas iguais
    """
    resultado = []

    buffer_texto = ""
    buffer_tipo = None

    for ent in entidades:

        palavra = limpar_token(ent["word"])
        tipo = ent["entity_group"]

        if buffer_tipo == tipo:
            buffer_texto += "" + palavra
        else:
            if buffer_texto:
                resultado.append((buffer_texto, buffer_tipo))
            buffer_texto = palavra
            buffer_tipo = tipo

    if buffer_texto:
        resultado.append((buffer_texto, buffer_tipo))

    return resultado


entidades = []

with open("OCR/output/Documento_OCR.txt", "r", encoding="utf-8") as f:
    linhas = f.readlines()

for linha in linhas:
    resultado = ner(linha)
    entidades.extend(resultado)

# ⭐ aplica pós-processamento
entidades_final = pos_processar(entidades)

os.makedirs("NER/output", exist_ok=True)

with open("NER/output/entidades.txt", "w", encoding="utf-8") as f:

    for palavra, tipo in entidades_final:

        linha = f"{palavra} -> {tipo}"

        print(linha)
        f.write(linha + "\n")