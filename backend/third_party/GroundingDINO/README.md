# Grounding DINO Tiny model files

This directory contains the minimal pretrained model and processor files used
by SmartUnwrapper. The inference implementation is provided by the installed
`transformers` package; the Grounding DINO repository source is not vendored.

- Upstream model: [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- Upstream snapshot: `a2bb814dd30d776dcf7e30523b00659f4f141c71`
- Model and configuration license: Apache-2.0; see [`LICENSE`](LICENSE)

Included artifacts are `model.safetensors`, `config.json`, processor and
tokenizer configuration, and tokenizer data. The model is loaded from this
directory with local-only access.
