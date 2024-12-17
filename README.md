# Introduction
This is a full pipeline for AMT(Automatic Music Transcription) Training using the mt3 model with some improvements using perceiver transformer and pitch shift augmentations. It builds on the works of both [Magenta](https://archives.ismir.net/ismir2021/paper/000030.pdf) and [Sungkyun Chang and Co authors](https://arxiv.org/abs/2407.04822)

# Dataset preparation | Training

## Workflow
Begin by running `src/install_dataset.py` and select the dataset opt you want. The available arguments for this file are:  
* `data_home` which denotes the Path to data home directory.
* `nodown` which stores the argument as true inorder to control downloading. If set, no downloading will occur

Example of this file in use :
> windows
```sh
python install_dataset.py src/datasets
```
> unix | mac
```sh
python3 install_dataset.py src/datasets
```
This file will carry out the following processes:
1. Create the `data_home` directory.
2. Prompt the user to select a singular or list of dataset(s) to install. The list of choices can be selected using comma separated values i.e. `1, 2, 4`. 
   > **Note :** Some of the datasets in the selection require a Zenodo access token. Be sure to request access before attemting to download these datasets 
   >* 10 RWC-Pop (Bass and Full) (**Restricted Access**),
   >* 9 CMedia (**Restricted Access**),
   >* 8 MIR-ST500 (**Restricted Access**)

3. Extract then preprocess the downloaded data using the processing strategy specified for the specific dataset. This varies across the datasets since the data may contain differential multitrack information. The strategy may stem the tracks , Extract note or note_event and metadata from midi or split the data into train/test/validation splits depending on the type of dataset specified.

## Training
---
The training file consists of an extensive number of inline argument that control audio spectrogram generation, model configuration data augmentation and so much more.
The arguments for the file `train.py` can be broken down int the following major stages:

#### General experiment setup
___

* `exp_id` **:** Which is a unique identifier for the experiment is used to resume training. YPTF.MoE+Multi (PS) With Pich Shift Augmentation has the identifier; `mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2`
&nbsp;
* `-p or --project` **:** Denotes the project name Can be any arbitrary string with "2024"
&nbsp;
* `-d or --data-preset` **:** This is the multiconfig dataset for training. The config file is located in the `src/config/dataset_presets.py` directory.
By default it uses the single `musicnet_thickstun_ext_em` dataset preset however YPTF.MoE+Multi (PS) was trained with the multi-config `all_cross_final` dataset preset.
#### Model configs
---
* `-epe or --encoder-position-encoding-type` **:** This denotes the positional encoding type of encoder. By default, pre-defined PE for T5 or Perceiver-TF encoder in `src/configs/config.py`.

>>##### T5 (Text-to-Text Transfer Transformer)
>>Options: `{'sinusoidal', 'trainable'}`
>>
>>- **sinusoidal**: Uses sine and cosine functions to encode position information. >>This method doesn't require learning and can generalize to sequence lengths not >>-seen during training.
>>- **trainable**: Learnable position embeddings that are updated during the >>-training process. These can potentially capture more nuanced position >>-information but may not generalize as well to unseen sequence lengths.
>>
>>##### Conformer (Convolution-augmented Transformer)
>>Options: `{'rotary', 'trainable'}`
>>
>>- **rotary**: Refers to Rotary Position Embedding (RoPE). This method encodes >>relative position information using rotation matrices, allowing the model to >>better capture relative distances between tokens.
>>- **trainable**: Same as in T5, these are learned during training.
>>
>>##### Perceiver-TF (Perceiver Transformer)
>>Options: `{'trainable', 'rope', 'alibi', 'alibit', 'None', '0', 'none', 'tkd', >>'td', 'tk', 'kdt'}`
>>
>>- **trainable**: Learned position embeddings, as described above.
>>- **rope**: Rotary Position Embedding, same as in Conformer.
>>- **alibi**: ALiBi (Attention with Linear Biases). This method adds a bias term >>to attention scores based on relative positions, allowing for better >>extrapolation to longer sequences.
>>- **alibit**: Likely a variant of ALiBi, possibly with some modifications.
>>- **None/0/none**: No position embedding used.
>>- **tkd**: Possibly refers to "Time, Key, and Dimension" - a custom position >>embedding method.
>>- **td**: Likely "Time and Dimension" - another custom method.
>>- **tk**: Possibly "Time and Key" - another variation.
>>- **kdt**: Could be "Key, Dimension, and Time" - yet another custom approach.


>**Note**: The choice of position embedding method may depend on the specific encoder type used in the model. YPTF.MoE+Multi (PS) for example was trained using `rope`. (This utility depends on the encoder type)
&nbsp;
* `-rp or --rope-partial-pe` **:**  This is a boolean that checks whether to apply Rotary Positional Encodings to partial positions (default=None). If None, use config. (Depends on the encoder type)
&nbsp;
* `-enc or --encoder-type` **:**  Encoder type. 't5' or 'perceiver-tf' or 'conformer'. Default is 't5', following `src/configs/config.py`. YPTF.MoE+Multi (PS) was trained on perceiver-tf encoder.
&nbsp;
* `-sqr or --sca-use-query-residual`: This is a string boolean where Streaming Cross-attention Alignment sca uses query residual flag. Default follows `src/config/config.py`
&nbsp;
* `-ff or --ff-layer-type` **:** Denotes the Feed forward layer type (default=None). Available types : `{"Multilayer Perceptron (mlp)", "Mixture of Experts (moe)", "Gated Multilayer Perceptron  (mlp)"}`.
If None, default value defined in `src/config/config.py` will be used.

- `wf 4 or --ff-widening-factor 4`**:** This sets the feed-forward layer widening factor for MLP/MoE/gMLP to 4. It affects the size of the feed-forward layers in the model.
- `nmoe 8 or --moe-num-experts 8`**:** This sets the number of experts for Mixture of Experts (MoE) to 8. MoE is a technique where multiple "expert" networks specialize in different inputs.
- `kmoe 2 or --moe-topk 2`**:** This sets the top-k value for MoE to 2, determining how many experts are consulted for each input.
- `act silu or --hidden-act silu`**:** This sets the hidden activation function to SiLU (Sigmoid Linear Unit), also known as swish.
- `dec multi-t5 or --decoder-type multi-t5`**:** This sets the decoder type to 'multi-t5', a variant of the T5 decoder.
- `nl 26 or --num-latents 26`**:** This sets the number of latents for the Perceiver model to 26. This is ignored for T5 models.
- `edr 0.05 or --encoder-dropout-rate 0.05`**:** This sets the encoder dropout rate to 0.05 (5%).
- `ddr 0.05 or --decoder-dropout-rate 0.05`**:** This sets the decoder dropout rate to 0.05 (5%).
- `atc 1 or --attention-to-channel 1`**:** This enables the attention-to-channel flag for Perceiver-TF. It's ignored for T5 models.


#### Trainer configs
---
* `-it or --max-steps` **:**  This denotes the number of max steps (default is -1, disabled). This overrides the number of total steps defined in config.
&nbsp;
* `-vit or --val-interval` **:** This denotes  the validation interval (default=None). If None, use the check_val_every_n_epoch defined in `src/config/config.py`.
>**Note :** Each validation takes 0.5~1 hour. Avoid frequent validations due to the time-consuming nature of auto-regressive inference and evaluation metrics. 
&nbsp;
- `bsz 10 10 or --train-batch-size 10 10` **:** This sets the training batch size to 10 for both sub and local batches per GPU.
&nbsp;
- `sb 1 or --sync-batchnorm 1` **:** This enables synchronized batch normalization across devices.
&nbsp;
- `st ddp or --strategy ddp` **:** This sets the training strategy to DistributedDataParallel (DDP).
&nbsp;
- `wb online or --wandb-mode online` **:** This sets the Weights & Biases logging mode to 'online', enabling real-time logging.

#### Audio Processing Configurations
___
- `ac spec or --audio-codec spec` **:** This sets the audio codec to 'spec' (spectrogram). It determines how the audio input is processed. Options are "spec" for spectrogram or "melspec" for mel-spectrogram.
- `hop 300 or --hop-length 300` **:** This sets the hop length in frames to 300. The hop length is the number of frames between successive FFT windows. 300 is typically used for PerceiverTF models, while 128 is used for MT3 models.

### Model-Specific Training Arguments
---

To train each individual model in the given options, the arguments need to match specific configurations. Below are the explanations for each model:

#### YMT3+

This model uses the basic configuration with only the project and precision specified:

`-p [project] -pr [precision]`

No specific encoder, decoder, or audio processing arguments are required.

#### YPTF+Single (noPS)

This model requires the following specific arguments:

`-enc perceiver-tf -ac spec -hop 300 -atc 1 -pr [precision]`

- `-enc perceiver-tf`: Sets the encoder type to Perceiver Transformer
- `-ac spec`: Uses spectrogram as the audio codec
- `-hop 300`: Sets the hop length to 300 frames
- `-atc 1`: Enables attention to channel

#### YPTF+Multi (PS)

This model requires more specific arguments:

`-tk mc13_full_plus_256 -dec multi-t5 -nl 26 -enc perceiver-tf -ac spec -hop 300 -atc 1 -pr [precision]`

- `-tk mc13_full_plus_256`: Sets the task to a specific configuration
- `-dec multi-t5`: Uses a multi-T5 decoder
- `-nl 26`: Sets the number of latents to 26
- Other arguments are similar to YPTF+Single

#### YPTF.MoE+Multi (noPS and PS)

Both these models use the same set of arguments, with the only difference being the checkpoint:

`-tk mc13_full_plus_256 -dec multi-t5 -nl 26 -enc perceiver-tf -sqr 1 -ff moe -wf 4 -nmoe 8 -kmoe 2 -act silu -epe rope -rp 1 -ac spec -hop 300 -atc 1 -pr [precision]`

- `-sqr 1`: Enables SCA use query residual
- `-ff moe`: Sets the feed-forward layer type to Mixture of Experts
- `-wf 4`: Sets the feed-forward widening factor to 4
- `-nmoe 8`: Uses 8 experts in the Mixture of Experts
- `-kmoe 2`: Sets the top-k value for MoE to 2
- `-act silu`: Uses the SiLU activation function
- `-epe rope`: Sets the encoder position encoding type to Rotary Position Embedding
- `-rp 1`: Enables RoPE partial position encoding

These specific configurations ensure that each model is trained with its unique architecture and hyperparameters. When training a particular model, make sure to use the corresponding set of arguments to match the intended architecture and behavior.

## Notes
- Since this code was ported from Unix, PyTorch ^2.0 will not work since it relies on triton for the new ignite backend. You must manually compile the binaries from [here](https://github.com/woct0rdho/triton-windows). Or use a unix based machine to avoid compilation errors
- This project requires at least 16GB of VRAM i.e. High Compute usage.
- Some Datasets are very large See Dataset Sizes [here](https://zenodo.org/records/10009959)

