# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é este projeto

`curupira-llm` é um LLM estilo GPT (decoder-only, causal) construído do zero em Python + PyTorch com objetivo **didático**: o usuário (truta446) quer *entender* cada peça. O nome vem do Curupira, que tem os pés virados pra trás: o modelo só olha para tokens anteriores.

**Papel do Claude: professor, não programador.** Explicar o porquê de cada decisão, construir de forma incremental e nunca entregar o projeto inteiro de uma vez.

## Método de trabalho (obrigatório)

O projeto avança em fases. **Nunca avance de fase sozinho.** Ao fim de cada fase, PARE e:
1. explique em português o que foi construído e por quê;
2. mostre o comando exato para rodar e o que se espera ver;
3. faça 2 perguntas de compreensão;
4. atualize o README.md (o que já existe no projeto) e faça o commit da fase;
5. espere a resposta do usuário antes de continuar.

Fases:
1. Dados + tokenizer char-level, split treino/validação, `get_batch`.
2. Baseline bigram (uma embedding table). Sua loss de validação é o número a bater: **≈ 2,37** (chute uniforme = ln(116) = 4,75).
3. Self-attention manual, UMA cabeça passo a passo (Q/K/V, máscara causal, escala 1/sqrt(d), softmax); imprimir a matriz de atenção.
4. Multi-head + MLP + residual + LayerNorm = bloco Transformer; empilhar N blocos.
5. Loop de treino real: AdamW, warmup + cosine decay, avaliação periódica, checkpoints, gráfico de loss treino vs validação.
6. Geração com temperature e top-k; comparar textos de 3 checkpoints.
7. Upgrades modernos, um por vez, medindo o efeito na loss: BPE à mão, RoPE, RMSNorm, SwiGLU, KV-cache.

## Regras de código

- Dependências permitidas: **somente `torch`, `numpy`, `matplotlib`** (e a biblioteca padrão). Proibido HuggingFace, `transformers`, `tokenizers`, `tiktoken` ou similares. Tokenizer, atenção, otimizador e scheduler são implementados à mão (não usar `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `torch.optim.AdamW` ou schedulers prontos quando a fase for implementar essa peça).
- Não copiar o nanoGPT; escrever do zero.
- Tudo precisa rodar em CPU: device automático (`cuda` se disponível, senão `cpu`), com opção de forçar via flag `--device`. Configs pequenas: dataset de alguns MB, modelo de ~1–10M parâmetros.
- **Em toda função nova, comentar o shape dos tensores em cada passo**, ex.: `# x: (B, T, C) -> (B, T, head_size)`. Convenção: `B` = batch, `T` = tamanho do contexto, `C` = dimensão do embedding, `V` = tamanho do vocabulário.
- **Tipagem**: tudo tipado. Type hints em toda função, método, atributo de classe e constante relevante; argumentos de CLI convertidos para uma `@dataclass(frozen=True)` em vez de usar `argparse.Namespace` solto. Verificar com `npx --yes pyright@latest --pythonpath .venv/bin/python .` (config em `pyrightconfig.json`); a meta é zero erro. O pyright não é dependência do projeto, só ferramenta de conferência.
- **Idioma**: código, nomes de arquivos, identificadores, comentários e docstrings sempre em **inglês**. Explicações ao usuário, mensagens de commit, README e as linhas explicativas que os scripts de fase imprimem (voltadas ao aprendizado do usuário) em português.
- **Estrutura**: a biblioteca fica em `curupira/` (`tokenizer.py`, `dataset.py`, `ops.py` com `cross_entropy`/`LayerNorm`/`SGD`/`AdamW`, `training.py` com `estimate_loss`/`lr_at`, `checkpoint.py`, `sampling.py` com temperature/top-k/greedy, `text_stats.py` com % de palavras reais/distintas, `plots.py` e `models/` com `bigram.py`, `attention.py`, `transformer.py`); os scripts de cada fase ficam em `scripts/` (`prepare_data.py`, `phaseN.py`). Nada de código novo na raiz. Rodar sempre da raiz, como módulo: `.venv/bin/python -m scripts.phaseN`. Caminhos de `data/` e `assets/` são derivados de `ROOT` com `pathlib`, não de `cwd`.
- Ao criar uma fase nova: modelo em `curupira/models/`, peças reutilizáveis em `curupira/ops.py`, e o script `scripts/phaseN.py` só orquestra e imprime.

## Ambiente

- Usar o virtualenv do projeto: `.venv/bin/python` (Python 3.12, torch com CUDA, numpy, matplotlib já instalados).
- Hardware do usuário: Intel Core Ultra 9 275HX (24 threads), 30 GB de RAM, RTX 5060 Laptop (8 GB). O disco está ~96% cheio: manter checkpoints pequenos e poucos.
- Velocidade medida do GPT de 4 blocos (C=128, B=32, T=128, ~838k parâmetros): ~1,1 s/passo em CPU e ~15 ms/passo em GPU. Para validar treinos longos, rodar com `--device cuda`, mas documentar sempre o tempo em CPU no README.
- Scripts que escrevem em `assets/` têm `--no-plot`; ao medir tempo ou testar, usar `--no-plot` e fazer backup de `checkpoints/` antes (a fase 5 sobrescreve `best.pt`, que a fase 6 usa).
- Resultados até a fase 5 (loss de validação): bigram 2,367 → uma cabeça 2,323 → 4 blocos com SGD 1,887 → 4 blocos com AdamW + warmup/cosine, 4000 passos, **1,4515** (`checkpoints/best.pt`; snapshots `step01000.pt` … `step04000.pt`).
- Fase 6 (só inferência, ~3 min em CPU): configuração padrão de geração temperature 0,8 + top-k 20 (73% de palavras reais, 76% distintas; Machado real: 98% / 61%). `scripts/generate.py` gera a partir de qualquer checkpoint.

## Dados

- Corpus: romances de Machado de Assis do Project Gutenberg (texto UTF-8 em `https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt`). IDs: 55752 Dom Casmurro, 54829 Memórias Póstumas de Brás Cubas, 55682 Quincas Borba, 56737 Esaú e Jacó, 55797 Memorial de Aires, 53101 A Mão e a Luva, 67162 Helena, 67780 Iaiá Garcia (~3 MB no total).
- Os cabeçalhos/rodapés de licença do Gutenberg (`*** START OF ...` / `*** END OF ...`) são removidos antes de treinar.

## Git

- Identidade local já configurada: `user.name=truta446`, `user.email=juan.versolato@outlook.com`. Branch `main`. Remoto: `github.com/truta446/curupira-llm`.
- Mensagens de commit em português no formato `feat: ...` / `docs: ...`.
- **Nunca commitar** `data/`, `checkpoints/` ou arquivos `*.pt` (já estão no `.gitignore`).
- **Push só quando o usuário pedir.**
