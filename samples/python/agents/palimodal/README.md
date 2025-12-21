# Palimodal - Agente de Física con ColPali + MUVERA

Este agente es una versión avanzada del agente `multimodal` que utiliza **ColPali** para embeddings multi-vector y **MUVERA** para búsqueda eficiente.

## Características

### ColPali (Multi-Vector Embeddings)
- Usa el modelo `vidore/colqwen2-v1.0` para generar representaciones multi-vector
- Cada documento/imagen se representa con ~1000+ vectores de 128 dimensiones
- Captura información de layouts, tablas, gráficos, ecuaciones y texto
- Mejor precisión para documentos visualmente ricos

### MUVERA (Búsqueda Eficiente)
- Comprime multi-vectors a un vector de dimensión fija (FDE - Fixed Dimension Embedding)
- Permite búsqueda rápida inicial usando índices HNSW tradicionales
- Re-ranking con MaxSim usando los multi-vectors originales
- Balance óptimo entre velocidad y precisión

## Arquitectura

```
┌─────────────────┐     ┌──────────────────┐
│   Documento     │────▶│   ColQwen2       │
│   (PDF/Imagen)  │     │   Model          │
└─────────────────┘     └────────┬─────────┘
                                 │
                    Multi-Vector Embeddings
                    (~1030 × 128 dim)
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
           ┌───────────────┐         ┌────────────────┐
           │    MUVERA     │         │   Qdrant       │
           │  Compression  │         │   MultiVector  │
           └───────┬───────┘         └────────────────┘
                   │
           FDE (2048 dim)
                   │
                   ▼
           ┌───────────────┐
           │    Qdrant     │
           │   Standard    │
           └───────────────┘


Búsqueda:
1. Query → ColQwen2 → Multi-Vector
2. Multi-Vector → MUVERA → FDE
3. FDE → Qdrant (Top-100 candidatos)
4. Multi-Vector → MaxSim Re-ranking → Top-5 resultados
```

## Colecciones Qdrant

| Colección | Tipo | Dimensión | Propósito |
|-----------|------|-----------|-----------|
| `palimodal_muvera` | Dense Vector | 2048 | Búsqueda rápida inicial |
| `palimodal_mv` | Multi-Vector | 128 × N | Re-ranking con MaxSim |

## Requisitos

### Hardware
- **GPU**: NVIDIA con al menos 6-8GB VRAM (recomendado)
- **CPU**: Funciona pero muy lento para procesamiento de documentos

### Software
```bash
pip install colpali-engine>=0.3.13
pip install qdrant-client>=1.15.1
```

### Variables de Entorno
```bash
GOOGLE_API_KEY=xxx     # Para Gemini
QDRANT_URL=xxx         # URL de Qdrant
QDRANT_KEY=xxx         # API Key de Qdrant
PDF_DIR=/path/to/pdfs  # Directorio con PDFs (opcional)
```

## Uso

### Iniciar el servidor
```bash
cd samples/python/agents/palimodal
python -m app --host localhost --port 10004
```

### Opciones
- `--host`: Host del servidor (default: localhost)
- `--port`: Puerto del servidor (default: 10004)
- `--pdf-dir`: Directorio con PDFs a procesar

## Diferencias con Multimodal

| Aspecto | Multimodal (CLIP) | Palimodal (ColPali) |
|---------|-------------------|---------------------|
| Modelo | CLIP ViT-B/32 | ColQwen2 |
| Tipo Embedding | Single Vector (512) | Multi-Vector (~1030 × 128) |
| Búsqueda | Cosine Similarity | MUVERA + MaxSim |
| VRAM | ~2GB | ~7GB |
| Precisión Doc Visual | Buena | Excelente |
| Velocidad Indexación | Rápida | Moderada |

## Referencias

- [ColPali Paper](https://arxiv.org/abs/2407.01449)
- [MUVERA by Google Research](https://research.google/blog/muvera-multi-vector-retrieval-via-fixed-dimensional-encodings/)
- [Qdrant Multi-Vector](https://qdrant.tech/documentation/)
