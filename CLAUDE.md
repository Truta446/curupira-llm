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
2. Baseline bigram (uma embedding table). Sua loss de validação é o número a bater.
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
- **Idioma**: código, nomes de arquivos, identificadores, comentários e saídas dos scripts sempre em **inglês**. Explicações ao usuário, mensagens de commit e README em português.
- Módulos atuais: `prepare_data.py` (download + limpeza + split por livro), `tokenizer.py` (`CharTokenizer`), `dataset.py` (`load_data`, `get_batch`, `pick_device`), `phaseN.py` (script de demonstração/inspeção de cada fase). Rodar sempre a partir da raiz do repo (caminhos `data/...` são relativos).

## Ambiente

- Usar o virtualenv do projeto: `.venv/bin/python` (Python 3.12, torch com CUDA, numpy, matplotlib já instalados).
- Hardware do usuário: Intel Core Ultra 9 275HX (24 threads), 30 GB de RAM, RTX 5060 Laptop (8 GB). O disco está ~96% cheio: manter checkpoints pequenos e poucos.

## Dados

- Corpus: romances de Machado de Assis do Project Gutenberg (texto UTF-8 em `https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt`). IDs: 55752 Dom Casmurro, 54829 Memórias Póstumas de Brás Cubas, 55682 Quincas Borba, 56737 Esaú e Jacó, 55797 Memorial de Aires, 53101 A Mão e a Luva, 67162 Helena, 67780 Iaiá Garcia (~3 MB no total).
- Os cabeçalhos/rodapés de licença do Gutenberg (`*** START OF ...` / `*** END OF ...`) são removidos antes de treinar.

## Git

- Identidade local já configurada: `user.name=truta446`, `user.email=juan.versolato@outlook.com`. Branch `main`. Remoto: `github.com/truta446/curupira-llm`.
- Mensagens de commit em português no formato `feat: ...` / `docs: ...`.
- **Nunca commitar** `data/`, `checkpoints/` ou arquivos `*.pt` (já estão no `.gitignore`).
- **Push só quando o usuário pedir.**
