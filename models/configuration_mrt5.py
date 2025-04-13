from transformers.models.t5.configuration_t5 import T5Config

class MrT5Config(T5Config):
    model_type = "mrt5"
    def __init__(
        self,
        *args,
        sigmoid_mask_scale=-10.0,
        gate_layer_norm=True,
        deletion_threshold=None,
        delete_gate_layer=2,
        use_softmax1=False,
        deletion_type=None,
        random_deletion_probability=0.5,
        fixed_deletion_amount=0.5,
        train_language="en",
        eval_language="en",
        use_gumbel_noise=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.deletion_threshold = deletion_threshold
        self.sigmoid_mask_scale = sigmoid_mask_scale
        self.gate_layer_norm = gate_layer_norm
        self.use_softmax1 = use_softmax1
        self.deletion_type = deletion_type
        self.random_deletion_probability = random_deletion_probability
        self.fixed_deletion_amount = fixed_deletion_amount
        self.train_language = train_language
        self.eval_language = eval_language
        self.delete_gate_layer = delete_gate_layer
        self.use_gumbel_noise = use_gumbel_noise