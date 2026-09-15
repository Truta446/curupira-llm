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
7. Upgrades modernos, um por vez, medindo o efeito na loss: BPE à mão, RoPE, RMSNorm, SwiGLU, KV-cache. Dividida em sub-rodadas (7a BPE, 7b RoPE, 7c RMSNorm, 7d SwiGLU, 7e KV-cache), cada uma com script próprio `scripts/phase7X_nome.py`, e com a mesma parada (explicação, comando, 2 perguntas, commit, esperar) ao fim de cada sub-rodada.

## Regras de código

- Dependências permitidas: **somente `torch`, `numpy`, `matplotlib`** (e a biblioteca padrão). Proibido HuggingFace, `transformers`, `tokenizers`, `tiktoken` ou similares. Tokenizer, atenção, otimizador e scheduler são implementados à mão (não usar `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `torch.optim.AdamW` ou schedulers prontos quando a fase for implementar essa peça).
- Não copiar o nanoGPT; escrever do zero.
- Tudo precisa rodar em CPU: device automático (`cuda` se disponível, senão `cpu`), com opção de forçar via flag `--device`. Configs pequenas: dataset de alguns MB, modelo de ~1–10M parâmetros.
- **Em toda função nova, comentar o shape dos tensores em cada passo**, ex.: `# x: (B, T, C) -> (B, T, head_size)`. Convenção: `B` = batch, `T` = tamanho do contexto, `C` = dimensão do embedding, `V` = tamanho do vocabulário.
- **Tipagem**: tudo tipado. Type hints em toda função, método, atributo de classe e constante relevante; argumentos de CLI convertidos para uma `@dataclass(frozen=True)` em vez de usar `argparse.Namespace` solto. Verificar com `npx --yes pyright@latest --pythonpath .venv/bin/python .` (config em `pyrightconfig.json`); a meta é zero erro. O pyright não é dependência do projeto, só ferramenta de conferência.
- **Idioma**: código, nomes de arquivos, identificadores, comentários e docstrings sempre em **inglês**. Explicações ao usuário, mensagens de commit, README e as linhas explicativas que os scripts de fase imprimem (voltadas ao aprendizado do usuário) em português.
- **Comparações justas**: entre tokenizers diferentes, comparar sempre **loss por caractere** (loss por token ÷ caracteres por token), nunca loss por token. Todo número citado no README ou em constantes de script precisa ter sido **medido** (dizer onde e como); nunca estimar um valor e apresentá-lo como medição.
- **Estrutura**: a biblioteca fica em `curupira/` (`tokenizer.py` com o `Protocol` `Tokenizer` e `CharTokenizer`, `bpe.py` com `BPETokenizer`, `dataset.py` com `load_texts`/`encode_splits`, `ops.py` com `cross_entropy`/`LayerNorm`/`SGD`/`AdamW`, `training.py` com `estimate_loss`/`lr_at`/`TrainConfig`/`train_model` (a receita da fase 5, reutilizada na fase 7), `checkpoint.py`, `sampling.py` com temperature/top-k/greedy, `text_stats.py` com % de palavras reais/distintas, `plots.py` e `models/` com `bigram.py`, `attention.py`, `transformer.py`); os scripts de cada fase ficam em `scripts/` (`prepare_data.py`, `phaseN.py`). Nada de código novo na raiz. Rodar sempre da raiz, como módulo: `.venv/bin/python -m scripts.phaseN`. Caminhos de `data/` e `assets/` são derivados de `ROOT` com `pathlib`, não de `cwd`.
- Ao criar uma fase nova: modelo em `curupira/models/`, peças reutilizáveis em `curupira/ops.py`, e o script `scripts/phaseN.py` só orquestra e imprime.

## Ambiente

- Usar o virtualenv do projeto: `.venv/bin/python` (Python 3.12, torch com CUDA, numpy, matplotlib já instalados).
- Hardware do usuário: Intel Core Ultra 9 275HX (24 threads), 30 GB de RAM, RTX 5060 Laptop (8 GB). O disco está ~96% cheio: manter checkpoints pequenos e poucos.
- Velocidade medida do GPT de 4 blocos (C=128, B=32, T=128, ~838k parâmetros): ~1,1 s/passo em CPU e ~15 ms/passo em GPU. Para validar treinos longos, rodar com `--device cuda`, mas documentar sempre o tempo em CPU no README.
- **Medir tempo**: antes de medir tempo em CPU, conferir a carga com `uptime`/`ps` (o usuário roda outros trabalhos pesados na máquina) e intercalar as variantes comparadas; se a carga estiver alta ou as rodadas variarem muito, não publicar o número — dizer que não foi medido. O tempo de RoPE em CPU (fase 7b) ficou pendente por isso.
- Scripts que escrevem em `assets/` têm `--no-plot`; ao medir tempo ou testar, usar `--no-plot` e fazer backup de `checkpoints/` antes (a fase 5 sobrescreve `best.pt`, que a fase 6 usa).
- Resultados até a fase 5 (loss de validação): bigram 2,367 → uma cabeça 2,323 → 4 blocos com SGD 1,887 → 4 blocos com AdamW + warmup/cosine, 4000 passos, **1,4515** (`checkpoints/best.pt`; snapshots `step01000.pt` … `step04000.pt`).
- Fase 6 (só inferência, ~3 min em CPU): configuração padrão de geração temperature 0,8 + top-k 20 (73% de palavras reais, 76% distintas; Machado real: 98% / 61%). `scripts/generate.py` gera a partir de qualquer checkpoint.
- Fase 7a (BPE, vocabulário 1024, merges em cache em `data/bpe_2048.json`, 2,65 caracteres/token na validação): mesmo modelo e receita da fase 5 → val 3,4727 por token = **1,3086 por caractere** (-9,8% vs. letras), `checkpoints/bpe1024_best.pt`. Distância treino→val por caractere dobrou (0,039 → 0,075): ~18 épocas sobre 920k tokens. Geração, 8×400 caracteres: BPE T 1,0 = 78,1% reais / 76,6% distintas (letras: 59,7% / 80,4%). **Linha de base para 7b–7d: 1,3086 por caractere com BPE 1024.**
- Fase 7b (RoPE, `GPT(position="rope")`, 2 sementes 1337/2024, ~5 min em GPU): posição aprendida 1,3086/1,3030 (média 1,3058) vs. RoPE 1,2388/1,2529 (média **1,2458**, -4,6%), variação entre sementes 0,014; treino→val por caractere 0,074 → 0,167 e val da semente 2024 subiu no fim (1,2529 → 1,2633): começa a decorar. ~25% mais lento por passo em GPU. `checkpoints/bpe1024_rope_best.pt`. **Linha de base para 7c–7d: RoPE, média 1,2458 por caractere.**
- Fase 7c (RMSNorm, `GPT(norm="rmsnorm")`, com RoPE nas duas variantes): LayerNorm 1,2388/1,2529 (média 1,2458, reproduziu a 7b exatamente) vs. RMSNorm 1,2427/1,2540 (média 1,2484, +0,2%): **empate**, diferença 0,0025 contra variação entre sementes 0,014. Ganho real é custo: treino em GPU ~73 s vs. 79–83 s. `checkpoints/bpe1024_rope_rmsnorm_best.pt`. **Linha de base para 7d: RoPE + LayerNorm, média 1,2458** (RMSNorm empatou, então a 7d segue com o LayerNorm já medido; decidir com o usuário se quiser mudar).
- Fase 7d (SwiGLU, `GPT(mlp="swiglu")`, largura 8C/3 = 344 para igualar parâmetros, +0,15%; base RoPE + RMSNorm, escolhida pelo Claude por recomendação, usuário não objetou): ReLU 1,2427/1,2540 (média 1,2484, reproduziu a 7c) vs. SwiGLU 1,2466/1,2442 (média 1,2454, -0,2%): **sinal muda entre sementes, acaso**. Mas nos passos 1000 e 2000 a SwiGLU estava à frente nas duas sementes (~0,03 no passo 1000); decorou antes (treino→val 0,188 vs. 0,168; semente 1337 com melhor checkpoint no passo 3000). Neurônios mortos da ReLU no modelo da 7c: 0 de 2048 (hipótese refutada). `checkpoints/bpe1024_rope_rmsnorm_swiglu_best.pt`. **Modelo moderno completo para a 7e: RoPE + RMSNorm + SwiGLU.**
- Fase 7e (KV-cache): `Head.forward_cached`/`Block.forward_cached`/`GPT.forward_cached` e `GPT.generate_cached`; a conta da atenção fica em `Head._attend`, compartilhada com o `forward` de treino (regressão conferida: loss e logits idênticos ao código anterior). O cache só vale até `block_size` tokens no total (depois disso `generate` desliza a janela e as posições mudam); `generate_cached` levanta `ValueError` além disso. Resultados: loss idêntica com e sem cache (7d: 2,985283; fase 5: 1,313808), logits até 2e-5, texto idêntico (greedy e sorteado). Velocidade em GPU: **1,0×** em 119 a 1000 tokens (~300 tok/s, oscilação 1–3%): o overhead de disparar operações (16 cabeças em loop Python) domina num modelo de 1M parâmetros. CPU: medição contaminada (carga 18 antes de começar, VM qemu rodando, rodadas até 133×), não publicada; repetir com a máquina livre se o usuário pedir. Cache medido: 512 KiB com 128 tokens. **Com a 7e, as 7 fases do plano estão concluídas.**
- Lição de medição: o load average inclui o próprio benchmark (em CPU ele ocupa todos os núcleos), então "carga alta" só indica contenção quando passa do número de núcleos; o sinal mais direto de medição inválida é a oscilação entre rodadas da mesma medição (`reliability_note` no script da 7e).
- Método de ablação da fase 7 (código em `curupira/ablation.py`: `Variant`, `bpe_config`, `run_ablation`, `print_scoreboard`, `verdict`, `generation_stats`): cada upgrade vira um argumento do `GPT`/`ModelConfig` com padrão igual ao modelo anterior (checkpoints antigos continuam carregando), e cada comparação treina as variantes com **as mesmas sementes** (padrão 1337 e 2024) e compara a diferença média com a variação entre sementes.

## Dados

- Corpus: romances de Machado de Assis do Project Gutenberg (texto UTF-8 em `https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt`). IDs: 55752 Dom Casmurro, 54829 Memórias Póstumas de Brás Cubas, 55682 Quincas Borba, 56737 Esaú e Jacó, 55797 Memorial de Aires, 53101 A Mão e a Luva, 67162 Helena, 67780 Iaiá Garcia (~3 MB no total).
- Os cabeçalhos/rodapés de licença do Gutenberg (`*** START OF ...` / `*** END OF ...`) são removidos antes de treinar.

## Git

- Identidade local já configurada: `user.name=truta446`, `user.email=juan.versolato@outlook.com`. Branch `main`. Remoto: `github.com/truta446/curupira-llm`.
- Mensagens de commit em português no formato `feat: ...` / `docs: ...`.
- **Nunca commitar** `data/`, `checkpoints/` ou arquivos `*.pt` (já estão no `.gitignore`).
- **Push só quando o usuário pedir.**
