01-10-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Topic

#### LayerNorm(nn.Module)
forward macht aus x, dim  -> x, dim

### class PreNorm(nn.Module):
- selbiges wie bei LayerNorm

#### class WeightStandardizedConv2d(nn.Conv1d):
- Shapes weiß ich nicht
- 
#### class RandomOrLearnedSinusoidalPosEmb(nn.Module):
shapes:
self.weights (0.5xdim,)
x = (b,)  (wobei b alles sein kann: 23,4, 5 z.B)
freqs (b, 0.5xdim)
fouriered (b, dim)