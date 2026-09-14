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
| 2 | Baseline bigram | ⏳ |
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

Código e comentários estão em inglês; a documentação, em português.
`data/` e `checkpoints/` não são versionados.

## Licença

MIT. Veja [LICENSE](LICENSE).
