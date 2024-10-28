<div align="center">

# MrT5

**[MrT5: Dynamic Token Merging for Efficient Byte-level Language Models]()**\
(Kallini et al., 2024)
</div>

**MrT5** (**M**e**r**ge**T5**) is a more efficient variant of ByT5 that integrates a token deletion mechanism in its encoder to *dynamically* shorten the input sequence length. After processing through a fixed number of encoder layers, a learnt *delete gate* determines which tokens are to be removed and which are to be retained for subsequent layers. By effectively "merging" critical information from deleted tokens into a more compact sequence, MrT5 presents a solution to the practical limitations of existing byte-level models.

This repository includes the code to replicate every experiment in our paper and train/fine-tune your own MrT5 models.

## Usage

First, clone the MrT5 repo and install the required dependencies:

```
git clone https://github.com/jkallini/mrt5.git
cd mrt5
pip install -r requirements.txt
```

### Training MrT5 Models

Our codebase supports training ByT5 or MrT5 architectures, as well as the
random and fixed deletion baselines described in the paper. All model training
code is located in the `\training` directory, and model architectures are
located in the `\models` directory.

Here is an example usage of our `train.py` script to continue pre-training a (pre-trained) ByT5 Small on the span corruption task (with $\mathrm{softmax}_1$), as described in the paper:

```
python3 train.py span_corruption \
  --warmup_steps 0 \
  --logging_steps 10 \
  --eval_steps 50 \
  --effective_batch_size 1024 \
  --per_device_train_batch_size 8 \
  --run_name t5_span_corruption \
  --random_seed 28 \
  --max_steps 3000 \
  --use_softmax1
```

To train MrT5 models, set the `model_type` parameter to `MrT5`. This will
train MrT5's delete gate on top of a pre-trained ByT5 Small, as described in
Section 5 of the paper (the continued pre-training experiments).

```
python3 train.py span_corruption \
  --warmup_steps 0 \
  --logging_steps 10 \
  --eval_steps 50 \
  --effective_batch_size 1024 \
  --per_device_train_batch_size 8 \
  --run_name mrt5_span_corruption \
  --random_seed 28 \
  --max_steps 3000 \
  --use_softmax1 \
  --model_type MrT5
```

Our training script supports several MrT5-specific training arguments:
- `delete_gate_loss_coeff` is the $\alpha$ hyperparameter of the delete gate regularizer (defaults to 0.0).
- `sigmoid_mask_scale` is the scale $k$ of the sigmoid activation in MrT5's delete gate (defaults to -30.0).
- `regularizer_delay` is the number of steps before applying delete gate regularizer.
- `delete_gate_layer` is the layer after which the delete gate is placed (defaults to 2, or the 3rd layer).

> [!IMPORTANT]  
> When training your own MrT5 models, we **highly recommend** using a P-controller to target a specific deletion rate. For example, you can set a hyperparameter that will steer MrT5 to delete about 40% of tokens. This will help avoid parameter sweeps across $\alpha$ values and generally allow for more stable training.

To continue pre-training a MrT5 model on the span corruption task with a target
deletion rate of 40\%, you can set the P-controller training arguments as follows:

```
python3 train.py span_corruption \
  --warmup_steps 0 \
  --logging_steps 10 \
  --eval_steps 50 \
  --effective_batch_size 1024 \
  --per_device_train_batch_size 8 \
  --run_name mrt5_span_corruption_pctrl40% \
  --random_seed 28 \
  --max_steps 3000 \
  --use_softmax1 \
  --model_type MrT5 \
  --target_deletion_rate 0.4 \
  --p_controller_value 0.000001
```

This is equivalent to $\hat{\delta} = 0.4$ and $k_p = 10^{-6}$, as described in Section 3.2 of the paper.
