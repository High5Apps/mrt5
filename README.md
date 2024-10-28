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

Our codebase supports training ByT5 or MrT5 architectures.