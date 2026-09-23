# Third-party notices

The model, codec, G2P frontend, and configuration utilities derive from
[Amphion](https://github.com/open-mmlab/Amphion), under its
[MIT license](https://github.com/open-mmlab/Amphion/blob/main/LICENSE).
Original file-level notices have been retained. Local changes include event tags,
cache construction, mask-based training/inference, and portable launch utilities.
The copy was extracted from the local research tree, not a pristine upstream
commit; no exact upstream revision is asserted.

External dependencies are installed separately and retain their own licenses.
G2P lexicons/ONNX files, pretrained checkpoints, and normalization statistics are
not redistributed here. Obtain them from their authorized upstream sources.
The locally modified phoneme vocabulary is retained because checkpoint token IDs
depend on it; do not replace it with a different upstream vocabulary.

[MaskGCT weights](https://huggingface.co/amphion/MaskGCT) are marked CC BY-NC 4.0.
The MIT code license does not override that restriction or grant rights to
speaker recordings, AISHELL-1, AS70, or third-party alignments.

Before publishing separately trained weights or audio, verify permissions and
the license obligations of all training data and inherited models.
