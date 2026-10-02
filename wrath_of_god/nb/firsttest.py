import torch
import torch.nn as nn
import esm

device = "cuda" if torch.cuda.is_available() else "mps"
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
#print(alphabet.all_toks)
AS_list = alphabet.all_toks[4:-7]
#print(len(AS_list))


model.eval().to(device)
Sequenz = [("pepgadiola", "VERTIDLLLLLS")]
_, _,Sequenz_tokens = batch_converter(Sequenz)
print(Sequenz_tokens.shape, len("VERTIDLLLLLS"))
output = model(Sequenz_tokens.to(device), repr_layers = [model.num_layers])
reps = output["representations"][model.num_layers]
print(reps, reps.shape)
reps_shaped = reps[:,1:-1,:]
print(reps_shaped.shape)
#Embedding =torch.no_grad(model(Sequenz))
 