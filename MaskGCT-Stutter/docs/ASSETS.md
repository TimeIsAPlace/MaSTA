# External assets

The following paths are relative to the repository root. None are committed.

```text
MaskGCT/
  MaskGCT_model/
    t2s_model/model.safetensors
    semantic_codec/model.safetensors
    acoustic_codec/model.safetensors
    acoustic_codec/model_1.safetensors
    s2a_model/s2a_model_1layer/model.safetensors
    s2a_model/s2a_model_full/model.safetensors
    w2v_bert/                   # full facebook/w2v-bert-2.0 snapshot
  ckpt/wav2vec2bert_stats.pt
  g2p/sources/                  # official MaskGCT G2P resources
    chinese_lexicon.txt
    bpmf_2_pinyin.txt
    pinyin_2_bpmf.txt
    g2p_chinese_model/
      poly_bert_model.onnx
      config.json
      vocab.txt
      polychar.txt
      polydict.json
      polydict_r.json
```

Run `scripts/setup_assets.py --amphion-root PATH` to prepare these assets, or
copy your existing compatible assets to these locations. Do not copy over the
release's G2P Python files or modified `g2p/g2p/vocab.json`.
`--skip-download` copies only G2P resources and semantic statistics; you must
provide model snapshots yourself. The setup script downloads upstream defaults;
record actual upstream revisions in your experiment metadata for reproducibility.

English G2P can also require NLTK corpora (`cmudict` and the appropriate averaged
perceptron tagger for your NLTK version). Install them in the same environment.
An installed package is not proof its binary dependencies import successfully.
