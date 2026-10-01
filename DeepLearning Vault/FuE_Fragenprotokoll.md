01-10-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Topic

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
- Allgemeine Frage: wo bleibt das torch.to(device)? Also der agnostik code?
- 