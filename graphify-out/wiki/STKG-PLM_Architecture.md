# STKG-PLM Architecture

> 20 nodes · cohesion 0.13

## Key Concepts

- **Adaptive Fusion (PLM + Transformer + STKG)** (5 connections) — `photos/paperstkgplm.png`
- **Next POI Prediction (Mask Filling)** (5 connections) — `photos/paperstkgplm.png`
- **Spatial-Temporal Trajectories Prompt** (4 connections) — `photos/paperstkgplm.png`
- **Dual Augmented Views G'_V_u and G''_V_u** (3 connections) — `photos/paperstkgplm.png`
- **Pre-trained Language Model (PLM)** (3 connections) — `photos/paperstkgplm.png`
- **Spatial-Temporal Transformer** (3 connections) — `photos/paperstkgplm.png`
- **STKG Augmented Graph Construction** (3 connections) — `photos/paperstkgplm.png`
- **STKG Contrastive Learning Layer** (3 connections) — `photos/paperstkgplm.png`
- **Users' Trajectories (Multi-User t1–t4 Timeline)** (3 connections) — `photos/paperstkgplm.png`
- **Contrastive Loss + Recommendation Loss** (2 connections) — `photos/paperstkgplm.png`
- **Knowledge Graph Attention** (2 connections) — `photos/paperstkgplm.png`
- **Mean Pooling, Concatenate, Attention Network** (2 connections) — `photos/paperstkgplm.png`
- **PLM User Representation (H_u^l)** (2 connections) — `photos/paperstkgplm.png`
- **Softmax POI Probability Distribution Output** (2 connections) — `photos/paperstkgplm.png`
- **Special Tokens [CLS] [TRA] [CHE] [STR] [SHT] [MID] [SEP] [Mask]** (2 connections) — `photos/paperstkgplm.png`
- **STKG-PLM Architecture Diagram** (2 connections) — `photos/paperstkgplm.png`
- **User and POI Graph Embeddings (H_u^l, H_p^l)** (2 connections) — `photos/paperstkgplm.png`
- **Natural-Language Visit Sequences with Time and Distance** (1 connections) — `photos/paperstkgplm.png`
- **POI Recommendation Research Figure** (1 connections) — `photos/paperstkgplm.png`
- **Spatial-Temporal Knowledge Graph** (1 connections) — `photos/paperstkgplm.png`

## Relationships

- [STKGRec Architecture](STKGRec_Architecture.md) (1 shared connections)

## Source Files

- `photos/paperstkgplm.png`

## Audit Trail

- EXTRACTED: 22 (85%)
- INFERRED: 4 (15%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*