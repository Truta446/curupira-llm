# curupira-llm

Um LLM estilo GPT construído **do zero** em Python + PyTorch, com o objetivo de
entender cada peça por dentro.

O Curupira tem os pés virados para trás. O nosso modelo é **causal**: só olha
para trás para prever o próximo token.

Regras do projeto: só `torch`, `numpy` e `matplotlib`. Tokenizer, atenção e
otimizador são implementados à mão, sem HuggingFace, `transformers` ou
`tiktoken`. Tudo roda em CPU (usa GPU se houver).

## Status

| Fase | Conteúdo | Status |
|---|---|---|
| 1 | Dados + tokenizer char-level + `get_batch` | ✅ |
| 2 | Baseline bigram | ✅ |
| 3 | Self-attention (uma cabeça, passo a passo) | ⏳ |
| 4 | Bloco Transformer (multi-head, MLP, residual, LayerNorm) | ⏳ |
| 5 | Loop de treino (AdamW, warmup + cosine, checkpoints, curvas) | ⏳ |
| 6 | Geração (temperature, top-k) | ⏳ |
| 7 | Upgrades modernos (BPE, RoPE, RMSNorm, SwiGLU, KV-cache) | ⏳ |

## O que já existe

### Fase 1: dados e tokenizer

- **Corpus**: oito romances de Machado de Assis do Project Gutenberg, em
  domínio público e na ortografia original ("elle", "commigo", "Braz"). São
  ~2,8 MB de texto.
- **`prepare_data.py`**: baixa os livros para `data/raw/` e limpa o texto.
  Remove a licença do Gutenberg, a marcação `_itálico_` e os separadores
  `* * *`, e junta as linhas quebradas de cada parágrafo. Depois separa
  **os 10% finais de cada livro** para validação (`data/val.txt`); o resto vai
  para `data/train.txt`. Assim treino e validação têm o mesmo estilo, e o corte
  cai numa fronteira de parágrafo.
- **`tokenizer.py`**: `CharTokenizer`, em que cada caractere distinto vira um
  inteiro. Tem `encode`, `decode`, `save` e `load`.
- **`dataset.py`**: `load_data()` transforma os textos em tensores `(N,)`.
  `get_batch()` sorteia janelas `x: (B, T)` e os alvos `y: (B, T)`, que são `x`
  deslocado um token à frente. `pick_device()` usa cuda se houver, senão cpu.
- **`phase1.py`**: script de inspeção. Mostra o vocabulário, as frequências, um
  encode/decode de ida e volta, um batch e os pares contexto → próximo token.

```bash
.venv/bin/python prepare_data.py   # baixa e limpa (uma vez)
.venv/bin/python phase1.py         # inspeção (use --device cpu para forçar CPU)
```

### Fase 2: baseline bigram

O modelo mais simples possível: prevê o próximo caractere olhando **só o
caractere atual**. O modelo inteiro é uma tabela 116×116 (13.456 parâmetros).

- **`ops.py`**: `cross_entropy` escrita à mão, sem `F.cross_entropy`.
- **`bigram.py`**: `BigramLM`, com a tabela, o `forward` (que devolve logits
  `(B, T, V)` e a loss) e o `generate` (que amostra um caractere por vez).
- **`phase2.py`**: calcula o bigram "perfeito" só **contando** pares de
  caracteres. Depois treina a tabela com SGD escrito à mão e mostra que chega
  ao mesmo número. Por fim, espia a tabela aprendida (depois de `q` vem `u` com
  100% de chance) e gera texto.

| Modelo | Loss treino | Loss validação |
|---|---|---|
| Chute uniforme | 4,754 | 4,754 |
| Bigram por contagem | 2,345 | 2,367 |
| Bigram treinado (3000 passos, SGD lr=50) | 2,351 | 2,372 |

**Número a bater nas próximas fases: loss de validação ≈ 2,37.**

```bash
.venv/bin/python phase2.py --device cpu   # ~1 min
```

Código e comentários estão em inglês; a documentação, em português.
`data/` e `checkpoints/` não são versionados.

## Licença

MIT. Veja [LICENSE](LICENSE).
