import spacy

# usar GPU se disponível
try:
    spacy.require_gpu()
    print("GPU ativada")
except:
    print("Rodando na CPU")

# carregar modelo de português
nlp = spacy.load("pt_core_news_lg")

# ler arquivo txt
with open("/home/hiper-pmc/Documentos/Extracao_Estruturada_CAR-dev/OCR/Output/Documento_OCR.txt", "r", encoding="utf-8") as f:
    texto = f.read()

# processar texto
doc = nlp(texto)

print("\nENTIDADES ENCONTRADAS:\n")

# caminho do arquivo de saída
saida_path = "/home/hiper-pmc/Documentos/Extracao_Estruturada_CAR-dev/NER/Output/pessoas_extraidas.txt"

# garantir que a pasta existe (opcional)
import os
os.makedirs(os.path.dirname(saida_path), exist_ok=True)

# abrir arquivo de saída
with open(saida_path, "w", encoding="utf-8") as f_out:

    for ent in doc.ents:
        print(ent.text)
        f_out.write(ent.text + "\n")

print(f"\nArquivo salvo em: {saida_path}")