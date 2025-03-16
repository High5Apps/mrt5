<div align="center">

# MrT5

**[MrT5: Dynamic Token Merging for Efficient Byte-level Language Models](https://arxiv.org/pdf/2410.20771)**\
(Kallini et al., 2024)
</div>

![](/icons/MrT5.png)

**MrT5** (**M**e**r**ge**T5**) is a more efficient variant of ByT5 that integrates a token deletion mechanism in its encoder to *dynamically* shorten the input sequence length. After processing through a fixed number of encoder layers, a learnt *delete gate* determines which tokens are to be removed and which are to be retained for subsequent layers. By effectively "merging" critical information from deleted tokens into a more compact sequence, MrT5 presents a solution to the practical limitations of existing byte-level models.

This repository includes the code to replicate every experiment in our paper and train/fine-tune your own MrT5 models.

## Citation

If you use this repo, please cite the MrT5 paper:

```bibtex
@inproceedings{
    kallini2025mrt,
    title={MrT5: Dynamic Token Merging for Efficient Byte-level Language Models},
    author={Julie Kallini and Shikhar Murty and Christopher D Manning and Christopher Potts and R{\'o}bert Csord{\'a}s},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=VYWBMq1L7H}
}
```

Also cite the ByT5 paper:

```bibtex
@article{xue-etal-2022-byt5,
    title = "{B}y{T}5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models",
    author = "Xue, Linting  and
      Barua, Aditya  and
      Constant, Noah  and
      Al-Rfou, Rami  and
      Narang, Sharan  and
      Kale, Mihir  and
      Roberts, Adam  and
      Raffel, Colin",
    editor = "Roark, Brian  and
      Nenkova, Ani",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "10",
    year = "2022",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2022.tacl-1.17",
    doi = "10.1162/tacl_a_00461",
    pages = "291--306",
}
```


## Getting Started

First, clone the MrT5 repo and install the required dependencies:

```
git clone https://github.com/jkallini/mrt5.git
cd mrt5
pip install -r requirements.txt
```

Next, locate the `BASE_PATH` macro in `utils.py`, and redefine it to point
to the path of your project. This is where model checkpoints and datasets
will be written.

## Dataset Creation

The `\data` directory contains data collators and scripts for generating each dataset in the paper. The generated datasets are used by the training
and eval scripts described in the next sections. We do not pre-tokenize the datasets for the XNLI and QA tasks. 

### Span Corruption Datasets

The script that generates span corruption data is `preprocess_lm_dataset.py`.
It uses data from [multilingual C4](https://huggingface.co/datasets/allenai/c4) (mC4).
The script's default behavior is to generate monolingual train, validation, and test splits for each of the 15 languages in the continued pre-training section of our paper (English, French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili, and Urdu).

In the paper, we use a multilingual dataset mixture for the continued pre-training experiments. To generate a multilingual training corpus that contains a random mixture of data in the 15 languages, use the `multilingual` flag:

```
python3 preprocess_lm_dataset.py --multilingual
```

To create test sets across the 15 languages from our paper for multilingual evaluation, indicate that you would only like to generate the `test` split:

```
python3 preprocess_lm_dataset.py --split test
```

To create just an English span corruption dataset with train, validation, and test splits, run the following command:

```
python3 preprocess_lm_dataset.py --en_only
```

To support more languages, update the `SUBSET_LANGUAGES` dictionary in
`utils.py` with any languages that are part of mC4. The full list is provided
in the `ALL_LANGUAGES` dictionary in `utils.py`.

One additional detail is that there is not enough validation data in mC4 to have equal-sized validation and test splits for all languages, so we only use English data for validation and save the other languages' data for testing (described in Appendix E.2 of the paper).

### Diagnostic Datasets

Below are example inputs and targets for each of our three diagnostic tasks.

| Task                       | Input                                  | Target                                |
|----------------------------|----------------------------------------|---------------------------------------|
| Simple Vowel Removal       | z<span style="color:darkorange">E</span>KRr<span style="color:darkorange">e</span>JcBxG<span style="color:darkorange">U</span>JQbZS<span style="color:darkorange">Io</span>s                   | zKRrJcBxGJQbZSs                       |
| Contextual Vowel Removal   | <span style="color:green">EOu</span>bXg<span style="color:darkorange">a</span>YVb<span style="color:darkorange">i</span><span style="color:green">O</span>g<span style="color:darkorange">i</span><span style="color:green">I</span>r<span style="color:darkorange">E</span>nld                   | <span style="color:green">EOu</span>bXgYVb</span><span style="color:green">O</span>g</span><span style="color:green">I</span>rnld                      |
| Sequence Merge             | KjAxIp<span style="color:magenta">ABC</span>ZCxBcni<span style="color:magenta">ABC</span>s                   |  KjAxIp<span style="color:magenta">D</span>ZCxBcni<span style="color:magenta">D</span>s                      |


The script that generates data for the diagnostic tasks is `preprocess_diagnostic_dataset.py`.
Here is an example usage of the script to generate the train, dev, and test splits for the
simple vowel removal task:

```
python3 preprocess_diagnostic_dataset.py vowel_removal --train_n 6400000 --eval_n 32000
```

### Downstream Task Datasets

The script that preprocesses the datasets for the character-level tasks
is `preprocess_char_dataset.py`.

First, download the data from the [char-iit](https://github.com/explanare/char-iit) github repository and place it
in a directory at `BASE_PATH + 'finetune_datasets/char_iit_data/'`. The
script assumes that the data is located at this path.

Right now, we support the contextual *Spelling Correction with Context* and *Word Search* tasks from the char-iit repo. Below is an example command to preprocess the data for the contextual spelling correction task:

```
python3 preprocess_char_dataset.py spelling_correction_contextual
```

## Training

All model training code is located in the `\training` directory, and model architectures are located in the `\models` directory. Our codebase supports training all model architectures described in the paper:
1. MrT5 ([modeling_mrt5.py](./models/modeling_mrt5.py))
2. ByT5 ([modeling_byt5.py](./models/modeling_byt5.py))
3. Random baseline ([modeling_mrt5.py](./models/modeling_mrt5.py))
4. Fixed baseline ([modeling_mrt5.py](./models/modeling_mrt5.py))
5. Boundary Predictor (BP) baseline ([modeling_bpt5.py](./models/modeling_bpt5.py))
6. Convolutional Pooling (CP) baseline ([modeling_canine.py](./models/modeling_canine.py))

To view the full list of training arguments:

```
 python3 train.py --help
```


Here is an example usage of our `train.py` script to fine-tune a pre-trained ByT5 Small on the multilingual span corruption task (with $\mathrm{softmax}_1$).

```
python3 train.py span_corruption_multilingual \
  --warmup_steps 0 \
  --logging_steps 10 \
  --eval_steps 50 \
  --effective_batch_size 1024 \
  --per_device_train_batch_size 8 \
  --run_name t5_span_corruption \
  --random_seed 28 \
  --max_steps 5000 \
  --use_softmax1
```

To train MrT5 models, set the `model_type` parameter to `MrT5`. This will
train MrT5's delete gate on top of a pre-trained ByT5 Small, as described in
Section 5 of the paper (the continued pre-training experiments).

```
python3 train.py span_corruption_multilingual \
  --warmup_steps 0 \
  --logging_steps 10 \
  --eval_steps 50 \
  --effective_batch_size 1024 \
  --per_device_train_batch_size 8 \
  --run_name mrt5_span_corruption \
  --random_seed 28 \
  --max_steps 5000 \
  --use_softmax1 \
  --model_type MrT5
```

### MrT5-specific Training Arguments

> [!IMPORTANT]  
> When training your own MrT5 models, we **highly recommend** using a
PI-controller to target a specific deletion rate (described in Section 3.2 of the paper).
For example, you can set
a hyperparameter that will steer MrT5 to delete about 40% of tokens.
This will help avoid parameter sweeps across $\alpha$ values and generally
allow for more stable training.

Our training script supports several MrT5-specific training arguments:
- `delete_gate_loss_coeff` is the $\alpha$ hyperparameter of the delete gate regularizer (defaults to 0.0).
  When using a PI-controller, which dynamically sets $\alpha$, this argument sets the starting $\alpha_0$.
- `sigmoid_mask_scale` is the scale $k$ of the sigmoid activation in MrT5's delete gate (defaults to -30.0).
- `regularizer_delay` is the number of steps before applying delete gate regularizer (defaults to 0). For the diagnostic tasks, we set this to 10k steps.
- `delete_gate_layer` is the layer after which the delete gate is placed (defaults to 3, or the 3rd layer).
- `target_deletion_rate` is the desired sequence length reduction $\delta$ when using a PI-controller
  (defaults to None, i.e. no PI-controller is used).
- `controller_p` is the proportional gain $k_p$ when using a PI-controller (defaults to 0.5). This parameter is only used if `target_deletion_rate` is not None.
- `controller_i` is the integral gain $k_i$ when using a PI-controller (defaults to 1e-5). This parameter is only used if `target_deletion_rate` is not None.


Below is an example of a continued pre-training run for a MrT5 model on the span corruption task with a target
deletion rate of 40\%. You can set the PI-controller training arguments as follows:

```
python3 train.py span_corruption_multilingual \
  --warmup_steps 0 \
  --logging_steps 10 \
  --eval_steps 50 \
  --effective_batch_size 1024 \
  --per_device_train_batch_size 8 \
  --run_name mrt5_span_corruption_40% \
  --random_seed 28 \
  --max_steps 5000 \
  --use_softmax1 \
  --model_type MrT5 \
  --target_deletion_rate 0.4 \
  --controller_p 0.5 \
  --controller_i 0.00005
```

This is equivalent to $\delta = 0.4$, $k_p = 0.5$, and $k_i = 1\text{e-}5$ as described in Section 3.2 of the paper.

### Training from Scratch

By default, our script runs *fine-tuning* on top of a pre-trained ByT5
model (which corresponds to *continued pre-training* when training on the
span corruption task, as described above). However, we also support training a model from scratch with custom architecture configurations. This is how we trained models from scratch for the diagnostic task experiments.

Below is an example command to train a tiny T5 model with 3 encoder layers, 3 decoder layers, $d_{\text{ff}} = 1024$, and $d_{\text{model}} = 512$ from scratch on the vowel removal diagnostic task:

```
python3 train.py vowel_removal \
  --random_seed 59 \
  --run_name t5_vowel_removal \
  --train_from_scratch \
  --max_steps 30000 \
  --effective_batch_size 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --use_softmax1 \
  --d_ff 1024 \
  --d_model 512 \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --num_heads 4
```

When training MrT5 from scratch, we encourage enabling an additional *attention score regularizer* to prevent attention scores from inflating, as described in Appendix D. This can be set using the `scores_loss_coeff` parameter, referred to as $\beta$ in the paper. A value of $\beta=0.5$ worked well in practice.
