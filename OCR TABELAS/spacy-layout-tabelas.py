import spacy
from spacy_layout import spaCyLayout
import requests
from pathlib import Path

# =========================
# CONFIGURAÇÕES
# =========================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral-nemo:12b"

pdf_path = "/home/hiper-pmc/Documentos/Extracao_Estruturada_CAR-dev/OCR TABELAS/pdf-com-ocr/paddle-teste.pdf"
output_dir = Path("/home/hiper-pmc/Documentos/Extracao_Estruturada_CAR-dev/OCR TABELAS/Output")

output_dir.mkdir(parents=True, exist_ok=True)

output_path_tabelas = output_dir / "tabelas_extraidas-paddle.txt"
output_path_atp = output_dir / "ATP_limpa.txt"
output_path_arl = output_dir / "ARL_limpa.txt"
output_path_desconhecidas = output_dir / "tabelas_desconhecidas.txt"

# =========================
# MODELO SPACY + LAYOUT
# =========================
nlp = spacy.load("pt_core_news_lg")
layout = spaCyLayout(nlp)

# =========================
# FUNÇÕES AUXILIARES
# =========================
import re

def remover_pontos(texto: str) -> str:
    return texto.replace(".", "")

def normalizar_numero(num: str) -> str:
    num = num.strip().replace(".", "").replace(" ", "")
    if "," not in num:
        if len(num) >= 3:
            num = num[:-2] + "," + num[-2:]
    return num

def linha_valida_coordenada(linha: str) -> bool:
    padrao = r"^\d+\s+\d+,\d{2}\s+\d+,\d{2}$"
    return re.match(padrao, linha.strip()) is not None

def limpar_resposta_llm(coordenadas: str) -> str:
    linhas = coordenadas.split("\n")
    linhas_limpas = []

    for linha in linhas:
        low = linha.lower().strip()

        if not low:
            continue

        if "vertice" in low and "coord" in low:
            continue

        linha = remover_pontos(linha.strip())
        partes = linha.split()

        if len(partes) >= 3 and partes[0].isdigit():
            vertice = partes[0]
            coord_n = normalizar_numero(partes[1])
            coord_e = normalizar_numero(partes[2])

            nova_linha = f"{vertice} {coord_n} {coord_e}"
            if linha_valida_coordenada(nova_linha):
                linhas_limpas.append(nova_linha)

    return "\n".join(linhas_limpas)
def parece_tabela_coordenadas(texto: str) -> bool:
    low = texto.lower()

    pistas = [
        "vertice",
        "vértice",
        "coord_e",
        "coord_n",
        "coordenadas",
        "azimute",
        "distancia",
        "distância",
    ]
    return any(p in low for p in pistas)


def identificar_tipo_tabela(texto: str, primeira_tabela_coord_encontrada: bool) -> str:
    """
    Classifica a tabela por heurística:
    - Se tiver marcadores de ARL -> ARL
    - Se tiver marcadores de ATP -> ATP
    - Se for a primeira tabela de coordenadas -> ATP
    - Demais tabelas de coordenadas -> ARL
    - Caso não pareça tabela de coordenadas -> DESCONHECIDA
    """
    low = texto.lower()

    if not parece_tabela_coordenadas(texto):
        return "DESCONHECIDA"

    marcadores_atp = [
        "lista de coordenadas da atp",
        "lista de co0ordenadas da atp",
        "lista de c0ordenadas da atp",
        "lista de co00rdenadas da atp",
        " atp ",
        "área total da propriedade",
        "area total da propriedade",
    ]

    marcadores_arl = [
        "lista de coordenadas da arl",
        "lista de co0ordenadas da arl",
        "lista de c0ordenadas da arl",
        "lista de co00rdenadas da arl",
        " arl ",
        "reserva legal",
    ]

    if any(m in low for m in marcadores_arl):
        return "ARL"

    if any(m in low for m in marcadores_atp):
        return "ATP"

    if not primeira_tabela_coord_encontrada:
        return "ATP"

    return "ARL"


# =========================
# FUNÇÃO LLM VIA OLLAMA
# =========================
def validar_com_llm(texto: str, nome_tabela: str) -> str:
    """
    Envia a tabela OCR para o Ollama e retorna somente as linhas
    de coordenadas, sem cabeçalho, já prontas para compor uma tabela contínua.

    Ordem correta exigida pelo PDF:
    Vertice Coord_N Coord_E

    Exemplo:
    1 7793364,08 367497,28
    2 7793366,53 367513,63
    """

    prompt = f"""

Sua tarefa é TRANSCRVER TODAS AS LINHAS DE COORDENADAS presentes no texto OCR.

Objetivo:
- recuperar o maior número possível de vértices
- manter uma linha para cada vértice
- não resumir
- não omitir linhas
- não agrupar múltiplos vértices em uma mesma linha

Para cada linha identificada, extraia:
Vertice Coord_N Coord_E

Se o texto estiver imperfeito, reconstrua a linha usando o padrão das demais.

Você é um validador de coordenadas UTM extraídas por OCR a partir de um PDF de imóvel rural.

A tabela abaixo corresponde à tabela {nome_tabela}.

ATENÇÃO:
NO PDF, A COLUNA Coord_N VEM ANTES DA COLUNA Coord_E.

A saída final deve seguir EXATAMENTE esta ordem:

Vertice Coord_N Coord_E

FORMATO EXATO OBRIGATÓRIO DA SAÍDA:

Vertice Coord_N Coord_E
1 0000000,00 000000,00
2 0000000,00 000000,00
3 0000000,00 000000,00

REGRAS OBRIGATÓRIAS:

1. Retorne APENAS a tabela final.
2. Não escreva explicações.
3. Não escreva títulos como "TABELA 1", "TABELA 8" ou semelhantes.
4. Não escreva texto antes ou depois da tabela.
5. A primeira linha deve ser EXATAMENTE:
   Vertice Coord_N Coord_E

6. NÃO OMITA NENHUMA LINHA DE COORDENADA PRESENTE NO TEXTO OCR.
7. NÃO RESUMA A TABELA.
8. NÃO RETORNE APENAS PARTE DOS VÉRTICES.
9. RECUPERE O MAIOR NÚMERO POSSÍVEL DE LINHAS.
10. Se houver dúvidas em uma linha, ainda assim tente reconstruí-la com base no padrão da tabela.
11. Preserve todos os vértices identificáveis no texto.

12. Cada linha abaixo do cabeçalho deve conter EXATAMENTE 3 valores:
   - Vertice
   - Coord_N
   - Coord_E

13. O vértice deve ser inteiro.
14. Coord_N e Coord_E devem:
   - conter apenas números e uma vírgula decimal
   - ter exatamente duas casas decimais
   - NÃO conter ponto como separador de milhar
   - manter a vírgula como separador decimal

15. É PROIBIDO usar ponto em números.
   Exemplos inválidos:
   7.793.364,08
   367.497,28
   7793364.08

   Exemplos válidos:
   7793364,08
   367497,28

16. Preserve a correspondência correta entre as colunas do PDF:
   - a primeira coordenada numérica da linha é Coord_N
   - a segunda coordenada numérica da linha é Coord_E

17. NÃO inverter para Coord_E Coord_N.
18. Ignore colunas como Azimute, Distância, Área e Perímetro.
19. Corrija erros de OCR de pontos, vírgulas, espaços, dígitos e linhas quebradas.
20. Considere que o primeiro conjunto de coordenadas pertence à ATP, mesmo se o título estiver ausente ou ilegível.
21. Essa saída será salva em .txt para uso posterior no QGIS no padrão SIRGAS 2000.

Texto OCR bruto da tabela:
{texto}
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()

    coordenadas = response.json().get("response", "").strip()

    linhas = coordenadas.split("\n")

    linhas_limpas = []
    for linha in linhas:
        low = linha.lower().strip()

        if not low:
            continue

        if "vertice" in low and "coord" in low:
            continue

        linhas_limpas.append(linha.strip())

    coordenadas = response.json().get("response", "").strip()
    return limpar_resposta_llm(coordenadas)

def remover_pontos(texto: str) -> str:
    """
    Remove todos os pontos do texto.
    Mantém vírgulas.
    Exemplo:
    7.793.364,08 -> 7793364,08
    """
    return texto.replace(".", "")

# =========================
# PROCESSAMENTO DO PDF
# =========================
doc = layout(pdf_path)

resultado_bruto = []
atp_resultados = []
arl_resultados = []
desconhecidas_resultados = []

primeira_tabela_coord_encontrada = False

for i, table in enumerate(doc._.tables, start=1):
    nome_tabela = f"TABELA {i}"
    df_texto = table._.data.to_string(index=False)

    # Salvar bruto
    resultado_bruto.append(f"===== {nome_tabela} =====\n")
    resultado_bruto.append(f"Start token: {table.start}\n")
    resultado_bruto.append(f"End token: {table.end}\n\n")
    resultado_bruto.append(df_texto)
    resultado_bruto.append("\n\n")

    try:
        tipo = identificar_tipo_tabela(df_texto, primeira_tabela_coord_encontrada)

        if tipo in {"ATP", "ARL"}:
            coordenadas = validar_com_llm(df_texto, tipo)
            coordenadas = remover_pontos(coordenadas)

            bloco_saida = coordenadas
            if tipo == "ATP":
                atp_resultados.append(bloco_saida)
                if parece_tabela_coordenadas(df_texto):
                    primeira_tabela_coord_encontrada = True

            elif tipo == "ARL":
                arl_resultados.append(bloco_saida)
                if parece_tabela_coordenadas(df_texto) and not primeira_tabela_coord_encontrada:
                    primeira_tabela_coord_encontrada = True

        else:
            desconhecidas_resultados.append(
                f"===== {nome_tabela} =====\n"
                f"{df_texto}\n\n"
            )

    except Exception as e:
        desconhecidas_resultados.append(
            f"===== {nome_tabela} =====\n"
            f"ERRO AO PROCESSAR COM LLM: {str(e)}\n\n"
            f"{df_texto}\n\n"
        )

# =========================
# SALVAR ARQUIVOS
# =========================
with open(output_path_tabelas, "w", encoding="utf-8") as f:
    f.write("".join(resultado_bruto))

with open(output_path_atp, "w", encoding="utf-8") as f:
    if atp_resultados:
        f.write("Vertice Coord_N Coord_E\n")
        f.write("\n".join(atp_resultados) + "\n")
    else:
        f.write("Nenhuma tabela ATP identificada.\n")

with open(output_path_arl, "w", encoding="utf-8") as f:
    if arl_resultados:
        f.write("Vertice Coord_N Coord_E\n")
        f.write("\n".join(arl_resultados) + "\n")
    else:
        f.write("Nenhuma tabela ARL identificada.\n")

with open(output_path_desconhecidas, "w", encoding="utf-8") as f:
    if desconhecidas_resultados:
        f.write("".join(desconhecidas_resultados))
    else:
        f.write("Nenhuma tabela desconhecida.\n")

# =========================
# LOG FINAL
# =========================
print(f"Arquivo bruto salvo em: {output_path_tabelas}")
print(f"Tabelas ATP limpas salvas em: {output_path_atp}")
print(f"Tabelas ARL limpas salvas em: {output_path_arl}")
print(f"Tabelas desconhecidas salvas em: {output_path_desconhecidas}")