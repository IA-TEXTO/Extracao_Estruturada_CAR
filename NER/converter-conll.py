arquivo_entrada = "OCR/project-14-at-2026-03-23-12-22-be13856f.conll"
arquivo_saida = "OCR/project-14-corrigido.conll"

with open(arquivo_entrada, "r", encoding="utf-8") as f:
    linhas = f.readlines()

linhas_corrigidas = []
for linha in linhas:
    linha = linha.rstrip("\n")
    
    if not linha.strip():
        linhas_corrigidas.append("\n")
        continue

    partes = linha.split()

    # troca "_" da última coluna por "O"
    if partes[-1] == "_":
        partes[-1] = "O"

    linhas_corrigidas.append(" ".join(partes) + "\n")

with open(arquivo_saida, "w", encoding="utf-8") as f:
    f.writelines(linhas_corrigidas)

print("Arquivo corrigido salvo em:", arquivo_saida)