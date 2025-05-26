import torch
import torch.nn as nn
from transformers import GPT2Model

#PFA: Pre-trained Fine-tuned Adapter
class PFA(nn.Module):
    def __init__(self, device="cpu", gpt_layers=6, U=1):        
        super(PFA, self).__init__()
        
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        # gpt_layers = int(gpt_layers)
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.device = torch.device(device)
        self.U = U
        # self.device = device        


        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if int(layer_index) < int(gpt_layers) - int(self.U):    # Freeze the first gpt_layers - U layers
                    if "ln" in name or "wpe" in name: # ln=layer norm and wpe=positional embedding, trainable
                        param.requires_grad = True
                    else:
                        param.requires_grad = False # Freeze some layers fully
                else:
                    if "mlp" in name:
                        param.requires_grad = False # Freeze MLP layers, mlp=multi-layer perceptron
                    else:
                        param.requires_grad = True

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state #gpt2 skips tokenization, so we can directly use inputs_embeds
        #returns the last hidden state of the model, which is the output of the last layer
