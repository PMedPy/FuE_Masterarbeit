01-10-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Code

### Helperfunctions?
Für was sind die Helperfunction gut? Würde die jetzt nicht brauchen
- denke eher an Helpferfunctions, die die Shapes direkt printen können

### Normalization:

Verstehe nicht, warum hier die formel   (img + 6) / (4) - 1 return 4 * t - 2 verwendet wird? Hätte hier eher standardnormalverteilt (x-$\mu$)/ $\sigma$ .
- In welches .py kommen solche Helferfunktionen?

### Klassen

class WeightStandardizedConv2d(nn.Conv1d):. veerbt wird von nn.Conv1d
- wird mit reduce die shapes von o auf o , 1, 1 geändert?
- Wo ist hier die __innit__?
- Was ist hie die Shape?

Layernorm: Den Forwardprozess verstehe ich nicht

#### SinusoidalPosEMB:
Warum gibts hier kein @notorchautograd oder so ähnlich? Brauch es hier Parameter?
Für was ist der Schritt: 
emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

emb = x[:, None] * emb[None, :]
- Was ist hier eigentlich dim gemeint? 

### für was ist der Gaussian Diffusion trainer? 
z.B def cosine_beta_schedule(timesteps, s=0.008)


### GaussianDiffusion1D(nn.Module)
- 1D gemeint, dass ein Vektor durch das Netzwerk geht?
- Warum hat man hier keine Klassenmethoden gebaut:
  ```Python 
  if beta_schedule == 'linear':

betas = linear_beta_schedule(timesteps)

elif beta_schedule == 'cosine':

betas = cosine_beta_schedule(timesteps)

else:

raise ValueError(f'unknown beta schedule {beta_schedule}')
  ```
- cumprod: kumulatives produkt in der dimension dim
- Werden alle Objekte, die durch register_buffer gehen als nicht parameter gesetzt und auf torch.float32?: 
  ```Python 
  register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
  ```
- Wie kommt man auf den Posterior?
  ```Python
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

  ```

- Was machen hier die extract()-funtkionen bei z.B q_posterior?
- Verstehe diesen Code nicht:
```Python 
if design_len is not None:

# Padding Mask

B, T = x.size(0), x.size(1)

# Create a tensor of zeros with size [B, T]

padding_mask = torch.zeros((B, T), dtype=torch.bool, device = x.device)

# Set values to one after the design_len+1 position, but leave the last position as one

padding_mask[:, design_len+1:-1] = 1
#x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
#print(sum(padding_mask))

#print(x)
```

- Sag mal wird einfach hier nur der noise ausgegeben?:
```Python
if self.objective == 'pred_noise':

pred_noise = model_output

x_start = self.predict_start_from_noise(x, t, pred_noise)

x_start = maybe_clip(x_start)

  

elif self.objective == 'pred_x0':

x_start = model_output

x_start = maybe_clip(x_start)

pred_noise = self.predict_noise_from_start(x, t, x_start)

  

elif self.objective == 'pred_v':

v = model_output

x_start = self.predict_start_from_v(x, t, v)

x_start = maybe_clip(x_start)

pred_noise = self.predict_noise_from_start(x, t, x_start)

return ModelPrediction(pred_noise, x_start)
```
- Allgemeine Frage: wo bleibt das torch.to(device)? Also der system agnostik code?
- Bei Modellprediction: Für was hier die Padding-Mask? ist das schon die Einstellung für die Länge des AMPs, die generiert werden soll?
- with torch.no_grad(): Eigentlich mittlerweile "with torch.inferencemode()"
- Sollte hier nicht eigentlich x als return stehen und kein Loss?
```Python
def forward(self, img, padding_mask, *args, **kwargs):

b, device = len(img), self.device

t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

img = self.normalize(img)

return self.p_losses(img, t, padding_mask = padding_mask, *args, **kwargs)
```
#### class Denoise_Transformer(nn.Module)
- Hier Embedding dimension: 1280? doch großes modell?
- Warum hier SinussoidalPosEMb?:
```Python
self.time_mlp = nn.Sequential(

SinusoidalPosEmb(embed_dim),

nn.Linear(embed_dim, time_dim),

nn.GELU(),

nn.Linear(time_dim, embed_dim * pep_max_len * 2),

)
```
ESM-2 hat doch schon RoPE?
- Was bedeutet Time-Embedding bei dem forwardpass?
- Wird hier mehrmals die Attentionlayer des ESM-2 verwendet?: 
```Python
for layer_idx, layer in enumerate(self.esm_layers):
	x, attn = layer(
	x,
	self_attn_padding_mask=padding_mask,
	)
```

# Aus dem Paper selbst

Grundkonzept:
Training: ESM-2 Embeddings aus Datensatz AMPs encoden.
Encodeter Datensatz schrittweise mit Rauschen beaufschlagen (Diffusionsmodell lernt wie aus Rauschen AMPs ESM-2 Embeddings generiert werden)
INferenz: Input entweder AMP größe mit padding oder random. Aufbau eines verrrauschten Embeddings -> Denoising mit dem Diffusionsmodell, decode mit Multiattentionhead!
Frage: Könnte man nicht einen VAE mit Transformer für das Diffmodell bauen? Also:
ESM-2 - Encode- Latent Space (Diffusionsmodell generiert latentes embedding)- decode zu ESM-2 Embedding - Decode zu AS-Sequenz. 

- AMP Diffusionsmodell nach DDPM framework: 
- EMA für Retraining? Was ist damit gemeint?
- Könnte man für ein guided Diffusion modell nicht für alle Trainingsdaten APEX anschmeißen, der generiert MICS gegen bestimmte Pathogene und diese Werte können zum Training verwendet werden?