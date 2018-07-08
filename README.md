# PNASNet.pytorch

PyTorch implementation of [PNASNet-5](https://arxiv.org/1712.00559). Specifically, PyTorch code from [this repository](https://github.com/quark0/darts) is adapted to completely match both [my implemetation](https://github.com/chenxi116/PNASNet.TF) and the [official implementation](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/pnasnet.py) of PNASNet-5, both written in TensorFlow. This complete match allows the pretrained TF model to be exactly converted to PyTorch: see `convert.py`.

If you use the code, please cite:
```bash
@inproceedings{liu2018progressive,
  author    = {Chenxi Liu and
               Barret Zoph and
               Maxim Neumann and
               Jonathon Shlens and
               Wei Hua and
               Li{-}Jia Li and
               Li Fei{-}Fei and
               Alan L. Yuille and
               Jonathan Huang and
               Kevin Murphy},
  title     = {Progressive Neural Architecture Search},
  booktitle = {European Conference on Computer Vision},
  year      = {2018}
}
```

## Requirements

- TensorFlow 1.8.0 (for image preprocessing)
- PyTorch 0.4.0
- torchvision 0.2.1

## Data and Model Preparation

- Download the ImageNet validation set and move images to labeled subfolders. To do the latter, you can use [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). Make sure the folder `val` is under `data/`.
- Download [PNASNet.TF](https://github.com/chenxi116/PNASNet.TF) and follow its README to download the `PNASNet-5_Large_331` pretrained model.
- Convert TensorFlow model to PyTorch model:
```bash
python convert.py
```

## Notes on Model Conversion

- In both TensorFlow implementations, `net[0]` means `prev` and `net[1]` means `prev_prev`. However, in the [PyTorch implementation](https://github.com/quark0/darts), `states[0]` means `prev_prev` and `states[1]` means `prev`. I followed the PyTorch implemetation in this repository. This is why the 0 and 1 in PNASCell specification are reversed.
- The default value of `eps` in BatchNorm layers is `1e-3` in TensorFlow and `1e-5` in PyTorch. I changed all BatchNorm `eps` values to `1e-3` (see `operations.py`) to exactly match the TensorFlow pretrained model.
- The TensorFlow pretrained model uses `tf.image.resize_bilinear` to resize the image (see `utils.py`). I cannot find a python function that exactly matches this function's behavior (also see [this thread](https://github.com/tensorflow/tensorflow/issues/6720) and [this post](https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35) on this topic), so currently in `main.py` I call TensorFlow to do the image preprocessing, in order to guarantee both models have the identical input.
- When converting the model from TensorFlow to PyTorch (i.e. `convert.py`), I use input image size of 323 instead of 331. This is because the 'SAME' padding in TensorFlow may differ from padding in PyTorch in some layers (see [this link](https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t); basically TF may only pad 1 right and bottom, whereas PyTorch always pads 1 for all four margins). However, they behave exactly the same when image size is 323: `conv0` does not have padding, so feature size becomes 161, then 81, 41, etc.
- The exact conversion when image size is 323 is also corroborated by the following table:

Image Size | Official TensorFlow Model | Converted PyTorch Model
--- | --- | ---
(331, 331) | (0.829, 0.962) | (0.828, 0.961)
(323, 323) | (0.827, 0.961) | (0.827, 0.961)


## Usage

```bash
python main.py
```

The last printed line should read:
```bash
Test: [50000/50000]	Prec@1 0.828	Prec@5 0.961
```
