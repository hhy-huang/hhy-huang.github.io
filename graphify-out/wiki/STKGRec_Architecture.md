# STKGRec Architecture

> 27 nodes · cohesion 0.13

## Key Concepts

- **Spatio-Temporal Knowledge Graph (STKG)** (7 connections) — `photos/paperstkgrec.png`
- **STKG Embedding Module** (6 connections) — `photos/paperstkgrec.png`
- **GRU Encoder** (5 connections) — `photos/paperstkgrec.png`
- **Multi-Component Prediction Scores {yn+, yn, yn*, yn'}** (5 connections) — `photos/paperstkgrec.png`
- **Short-Term Preference Modeling** (5 connections) — `photos/paperstkgrec.png`
- **Current Trajectory (single vertical chain)** (4 connections) — `photos/paperstkgrec.png`
- **PTransE Encoder** (4 connections) — `photos/paperstkgrec.png`
- **Short-term Relation Attention** (4 connections) — `photos/paperstkgrec.png`
- **Time-Geo-dilated GRU** (4 connections) — `photos/paperstkgrec.png`
- **Historical Trajectories (stacked sub-graphs)** (3 connections) — `photos/paperstkgrec.png`
- **Historical Trajectory Encoding Module** (3 connections) — `photos/paperstkgrec.png`
- **Long-Term Preference Modeling** (3 connections) — `photos/paperstkgrec.png`
- **Next POI Recommendation Task** (3 connections) — `photos/paperstkgrec.png`
- **Nonlocal Operation** (3 connections) — `photos/paperstkgrec.png`
- **POI / Location Nodes (pn, green circles)** (3 connections) — `photos/paperstkgrec.png`
- **STKG Building (trajectory → graph)** (3 connections) — `photos/paperstkgrec.png`
- **STKGRec End-to-End Architecture Figure** (3 connections) — `photos/paperstkgrec.png`
- **User–POI Interaction Edges (blue dashed lines)** (3 connections) — `photos/paperstkgrec.png`
- **Users' Trajectories (u1–u3 over POIs p1–p10, timeline T1–T8)** (3 connections) — `photos/paperstkgrec.png`
- **Average Operation** (2 connections) — `photos/paperstkgrec.png`
- **Sequential POI Transition Edges (brown solid arrows)** (2 connections) — `photos/paperstkgrec.png`
- **User Nodes (un, blue circles)** (2 connections) — `photos/paperstkgrec.png`
- **GRU Score yn** (2 connections) — `photos/paperstkgrec.png`
- **Long-Term Score yn+** (2 connections) — `photos/paperstkgrec.png`
- **Time-Geo-dilated GRU Score yn'** (2 connections) — `photos/paperstkgrec.png`
- *... and 2 more nodes in this community*

## Relationships

- [STKG-PLM Architecture](STKG-PLM_Architecture.md) (1 shared connections)

## Source Files

- `photos/paperstkgrec.png`

## Audit Trail

- EXTRACTED: 37 (82%)
- INFERRED: 8 (18%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*