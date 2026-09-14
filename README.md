<p align="center">
  <img src="assets/banner.svg" alt="curupira-llm — um GPT construído do zero em PyTorch" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-1d5a41?style=flat-square" alt="Python 3.12">
  <img src="https://img.shields.io/badge/pytorch-2.14-ee4c2c?style=flat-square" alt="PyTorch 2.14">
  <img src="https://img.shields.io/badge/roda%20em-CPU-1d5a41?style=flat-square" alt="Roda em CPU">
  <img src="https://img.shields.io/badge/sem-HuggingFace-ff4d1a?style=flat-square" alt="Sem HuggingFace">
  <img src="https://img.shields.io/badge/tipado-pyright%200%20erros-2a78d6?style=flat-square" alt="Tipado">
  <img src="https://img.shields.io/badge/licen%C3%A7a-MIT-898781?style=flat-square" alt="Licença MIT">
</p>

<h3 align="center">Um LLM estilo GPT construído do zero, peça por peça, para entender como funciona.</h3>

<p align="center">
  O Curupira tem os pés virados para trás.<br>
  O nosso modelo é <b>causal</b>: só olha para trás para prever o próximo token.
</p>

---

## A ideia

Um GPT faz uma coisa só: **recebe um pedaço de texto e chuta o próximo caractere**. Repetindo isso, ele escreve. Este repositório constrói esse mecanismo do zero, sem atalhos:

- **só `torch`, `numpy` e `matplotlib`**;
- **nada de HuggingFace, `transformers` ou `tiktoken`**;
- tokenizer, atenção, função de perda e otimizador **implementados à mão**;
- **roda em CPU**, com modelos pequenos e um corpus de poucos MB;
- **tudo tipado**, verificado com pyright;
- cada tensor tem o **shape comentado** em cada passo:

```python
logits = self.table[idx]  # row lookup: (B, T) -> (B, T, V)
```

O corpus são oito romances de **Machado de Assis** em domínio público, na ortografia original: o modelo aprende a escrever "elle", "commigo" e "Braz".

## Como funciona o caminho dos dados

```mermaid
flowchart LR
    A["Livros<br/>Gutenberg"] --> B["prepare_data.py<br/>limpeza + split"]
    B --> C["tokenizer.py<br/>letra ↔ número"]
    C --> D["dataset.py<br/>get_batch: x, y"]
    D --> E["modelo<br/>logits (B,T,V)"]
    E --> F["ops.py<br/>cross-entropy"]
    F --> G["gradiente<br/>ajusta os pesos"]
    G --> E
```

## Resultados até agora

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/phase2_loss_dark.png">
  <img src="assets/phase2_loss_light.png" alt="Curva de loss do bigram: cai de 4,75 para 2,37 em 3000 passos" width="100%">
</picture>

A **loss** mede a surpresa do modelo diante da letra certa: quanto menor, melhor.

| Modelo | Parâmetros | Loss treino | Loss validação |
|---|---:|---:|---:|
| Chute uniforme (não sabe nada) | 0 | 4,754 | 4,754 |
| Bigram por contagem (o melhor possível) | — | 2,345 | **2,367** |
| Bigram treinado com SGD | 13.456 | 2,351 | 2,372 |
| **Uma cabeça de self-attention** | 35.444 | 2,296 | **2,323** |

O treino, sozinho, **redescobriu as estatísticas do livro**: chegou ao mesmo número que se obtém contando pares de letras. A tabela aprendida faz sentido — depois de `q` vem `u` com 100%, e depois de uma quebra de linha vem `-` em 48% dos casos, porque os diálogos de Machado começam com `--`.

<details>
<summary><b>Texto gerado pelo bigram</b> (clique para abrir)</summary>

```
-misosmosizeuem ea gr enço, eve, laca ccase Sia. eme Pomide frra to laeraeru
ado nom ceuveres po ssoma a uss, de erdasaredave ara, quistres s os menda pr
```

Tem cara de português, mas não diz nada: sem memória além da letra anterior, não se formam palavras. **O número a bater nas próximas fases é 2,37.**

</details>

### A atenção, olhando só para trás

Cada posição distribui 100% da sua atenção entre as posições **anteriores** — o triângulo de cima é sempre zero. Depois de treinar, a cabeça aprende a procurar o que importa:

```
              D    o    m    _    C    a    s    m    u    r    r    o   <- lido
      D |  100    .    .    .    .    .    .    .    .    .    .    .
      o |   49   51    .    .    .    .    .    .    .    .    .    .
      m |   90    9    1    .    .    .    .    .    .    .    .    .
      _ |    0    0   99    1    .    .    .    .    .    .    .    .
      C |    1    0    0    2   97    .    .    .    .    .    .    .
      u |    1    3    4    0    1    1    4   62   24    .    .    .
      r |    1    0    0    2    0    0    1    0    0   74   20    .
```

<details>
<summary><b>Texto gerado com uma cabeça de atenção</b></summary>

```
---Nãe sium que a gre va puevetre. Te cas oria. a e ses fifre das la.
---ma de num ceuver e petisom, apussaide vressareda porra, quu daçã saponue
```

Ainda é conversa fiada, mas já aparecem "Não", "que a" e sílabas mais longas.

</details>

## Roteiro

| Fase | Conteúdo | Status |
|---|---|:---:|
| 1 | Corpus, tokenizer char-level, split e `get_batch` | ✅ |
| 2 | Baseline bigram, cross-entropy e SGD à mão | ✅ |
| 3 | Self-attention: uma cabeça, passo a passo | ✅ |
| 4 | Bloco Transformer: multi-head, MLP, residual, LayerNorm | 🚧 |
| 5 | Treino de verdade: AdamW, warmup + cosine, checkpoints | ⏳ |
| 6 | Geração: temperature e top-k | ⏳ |
| 7 | Upgrades modernos: BPE, RoPE, RMSNorm, SwiGLU, KV-cache | ⏳ |

## Começando

```bash
python -m venv .venv
.venv/bin/pip install torch numpy matplotlib

.venv/bin/python prepare_data.py          # baixa e limpa o corpus (uma vez)
.venv/bin/python phase1.py --device cpu   # inspeciona dados e tokenizer
.venv/bin/python phase2.py --device cpu   # treina o bigram (~1 min em CPU)
.venv/bin/python phase3.py --device cpu   # self-attention passo a passo (~2 min em CPU)
```

Use `--device cpu` para forçar a CPU; sem a flag, o código usa a GPU se houver.

## Os arquivos

| Arquivo | O que faz |
|---|---|
| `prepare_data.py` | Baixa os livros, remove a licença do Gutenberg e a diagramação, e separa os 10% finais de **cada** livro para validação |
| `tokenizer.py` | `CharTokenizer`: cada caractere vira um inteiro (vocabulário de 116 símbolos) |
| `dataset.py` | `load_data`, `get_batch` (janelas `x: (B, T)` e alvos `y: (B, T)`) e `pick_device` |
| `ops.py` | `cross_entropy` e o otimizador `SGD` (com momento), ambos escritos à mão |
| `bigram.py` | `BigramLM`: uma tabela `(V, V)`, com `forward` e `generate` |
| `attention.py` | `Head`: uma cabeça causal (query/key/value, máscara, escala `1/sqrt(d)`, softmax) e `AttentionLM` |
| `phase1.py` … `phase3.py` | Um script por fase, que demonstra e mede o que foi construído |

Código, nomes e comentários em inglês; documentação e explicações em português. `data/` e `checkpoints/` não são versionados.

## Verificando os tipos

```bash
npx --yes pyright@latest --pythonpath .venv/bin/python .
```

## Licença

MIT — veja [LICENSE](LICENSE).
