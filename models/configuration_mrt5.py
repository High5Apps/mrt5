from transformers.models.t5.configuration_t5 import T5Config

class MrT5Config(T5Config):
    model_type = "mrt5"
    def __init__(
        self,
        *args,
        sigmoid_mask_scale=-30.0,
        deletion_threshold=-15.0,
        delete_gate_layer=3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.deletion_threshold = deletion_threshold
        self.sigmoid_mask_scale = sigmoid_mask_scale
        self.delete_gate_layer = delete_gate_layer
