# Third-party notices

This repository redistributes the following third-party material. Everything else is released
under the OpenMDW-1.1 license in `LICENSE`.

## Ettin encoder (base model)

The model weights are a fine-tune of [`jhu-clsp/ettin-encoder-68m`](https://huggingface.co/jhu-clsp/ettin-encoder-68m)
(revision `ac19ae4bc51093b31c475665ac872a936d056cc2`), released under the MIT License by the
[Ettin project](https://github.com/jhu-clsp/ettin-encoder-vs-decoder):

```text
MIT License

Copyright (c) 2025 Johns Hopkins University Center for Language and Speech Processing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ModernBERT tokenizer

`tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json` are the tokenizer of
[`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) (revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`), licensed under the Apache License, Version 2.0. In
`tokenizer_config.json`, `model_max_length` was set to this model's operating context (2,048);
`tokenizer.json` serializes the same vocabulary and merges.

```text
{{apache_license}}
```
