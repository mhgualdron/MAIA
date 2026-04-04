# Fine-Tuning — Adaptación de LLMs

> 🚧 **Módulo en construcción** (WIP)

Experimentos de fine-tuning eficiente de modelos de lenguaje usando técnicas de PEFT (Parameter-Efficient Fine-Tuning).

## Técnicas a explorar

- **LoRA** (Low-Rank Adaptation): adapta matrices de atención con factorización de bajo rango
- **QLoRA**: LoRA con cuantización 4-bit para reducir uso de VRAM
- **Instruction Tuning**: fine-tuning con datasets de instrucciones

## Estado

Los experimentos están en progreso. Este módulo se actualizará con:
- Notebooks de entrenamiento
- Scripts de evaluación
- Comparativas de métricas (BLEU, ROUGE, perplexity)

## Referencias

- [HuggingFace PEFT](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
