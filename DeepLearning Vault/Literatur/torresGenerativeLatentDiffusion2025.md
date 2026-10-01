---
title: "Generative latent diffusion language modeling yields anti-infective synthetic peptides"
authors: Marcelo D. T. Torres, Leo Tianlai Chen, Fangping Wan, Pranam Chatterjee, Cesar de la Fuente-Nunez
year: 2025
citekey: torresGenerativeLatentDiffusion2025
tags: [literatur]
---

# Generative latent diffusion language modeling yields anti-infective synthetic peptides

**Autor(en):** Marcelo D. T. Torres, Leo Tianlai Chen, Fangping Wan, Pranam Chatterjee, Cesar de la Fuente-Nunez
**Jahr:** 2025
**Citekey:** [[@torresGenerativeLatentDiffusion2025]]



## Annotationen

%% begin annotations %%
### Importiert am 2026-09-25 10:19


> due to vast sequence space, complex structure-activity relationships, and the need to balance antimicrobial potency with low toxicity. [page. 2](zotero://open-pdf/library/items/HMVNQVXZ?page=2&annotation=NGGN4UCJ)

**Notiz:** Wichtiger Punkt

---

> In this work, we show that AMP-Diffusion, a pre-trained latent diffusion model, generates functionally potent AMP sequences by applying Gaussian noise over the pre-trained ESM-2-650M embedding space [page. 3](zotero://open-pdf/library/items/HMVNQVXZ?page=3&annotation=48W5TLEH)

**Notiz:** Was ist der ESM-2-650M Embeddingspace?

---

> By reusing ESM-2’s attention-layer weights in our denoiser, AMP-Diffusion is better constrained to generate sequences that are not only statistically likely but also adhere to the complex, underlying biological rules learned by ESM-2. [page. 3](zotero://open-pdf/library/items/HMVNQVXZ?page=3&annotation=6DH6IG3P)

**Notiz:** Hier wurden Weights und Transformerblöcke eines anderen Models verwendet. Wie geht das?

---

> The model generates functional AMPs by applying a diffusion process in the latent space of ESM-2 embeddings, progressively adding and then removing noise to generate peptide sequences. [page. 3](zotero://open-pdf/library/items/HMVNQVXZ?page=3&annotation=66F6UB5F)

---

> AMP-Diffusion was trained on a compiled dataset of AMPs from DRAMP 3.0,30 APD3,31 and DBAASP32 databases [page. 3](zotero://open-pdf/library/items/HMVNQVXZ?page=3&annotation=S42VRZS2)

**Notiz:** Also hier nicht UniProt

---

> From an initial set of 50,000 AMP candidates (Dataset S2), we selected 46 peptides for experimental validation based on three criteria: high predicted antimicrobial activity, low similarity to existing AMPs, and broad sequence diversity [page. 3](zotero://open-pdf/library/items/HMVNQVXZ?page=3&annotation=46MP8BC6)

---

> The minimum inhibitory concentration (MIC) distribution of our generated sequences was comparable to the training set, indicating the model’s ability to recapitulate the MIC profile of training AMPs [page. 3](zotero://open-pdf/library/items/HMVNQVXZ?page=3&annotation=PYY39E7A)

---

> To assess the naturalness of the generated peptides, we employed ProGen2, a pLM trained on millions of natural protein sequences. Naturalness was quantified using perplexity scores, where lower values indicate greater naturalness. [page. 3](zotero://open-pdf/library/items/HMVNQVXZ?page=3&annotation=UK6NW5IQ)

---

> Interestingly, the filtered subset of generated sequences exhibited a distinct pattern, with notably higher frequencies of lysine (K), leucine (L), and arginine (R). These residues are known to be important for initial electrostatic interactions (K and R) with negatively charged bacterial membranes and for amphiphilicity (K, L, and R), which facilitates effective lipid membrane interactions. [page. 3](zotero://open-pdf/library/items/HMVNQVXZ?page=3&annotation=EKSZUBFD)

---

> hus, enrichment in cationic and aliphatic residues directly influences the physicochemical features of the AMP-Diffusion-predicted peptides (Figures 1E and S2), making the filtered set slightly more amphiphilic and highly charged, with enhanced structure tendencies and increased antimicrobial activity [page. 3](zotero://open-pdf/library/items/HMVNQVXZ?page=3&annotation=YV2WFYHJ)

---

> Consistent with their high lysine (K), leucine (L), and arginine (R) content and in line with known AMPs, the peptides underwent a helix-coil transition. [page. 5](zotero://open-pdf/library/items/HMVNQVXZ?page=5&annotation=VW4LNEBJ)

---

> These results indicate that the peptides are more effective at permeabilizing bacterial membranes than many previously reported AMPs and encrypted peptides (EPs) derived from human proteins, extinct organisms, or bacterial metagenomes. [page. 5](zotero://open-pdf/library/items/HMVNQVXZ?page=5&annotation=DP5AJGBJ)

**Notiz:** Besser als die Evolution?

---

> All 46 synthesized peptides were tested for cytotoxic activity against human embryonic kidney (HEK293T) [page. 5](zotero://open-pdf/library/items/HMVNQVXZ?page=5&annotation=MNGPJP4A)

---

> n addition, AMP-Diffusion operates as an unconditional diffusion model, lacking conditional guidance mechanisms. Future work may explore training dedicated classifiers for noised sequences to guide generation based on key therapeutic properties (e.g., inhibitory activity and toxicity), or, alternatively, using classifierfree guidance with pathogen-specific and therapeutic property embeddings to produce peptides tailored for particular bacterial strains with high safety [page. 7](zotero://open-pdf/library/items/HMVNQVXZ?page=7&annotation=EYYW4YWZ)

**Notiz:** Was bedeutet in dem zusammenhang therapeutic embeddings?

---

> We believe that discrete diffusion architectures, such as masked discrete diffusion (MDLM) and discrete denoising posterior prediction (DDPP) [page. 7](zotero://open-pdf/library/items/HMVNQVXZ?page=7&annotation=M3TZGV9B)

**Notiz:** Was ist discrete denoising posterior prediction?

---

> this could be effectively addressed through strategic subsampling to create more balanced and focused da [page. 7](zotero://open-pdf/library/items/HMVNQVXZ?page=7&annotation=SC6JTETD)

**Notiz:** meyne aufgabe

---

> The model then trains a denoiser to reconstruct the original latent embeddings from the noised inputs, minimizing the l2 loss between predicted and original latent embeddings. [page. 8](zotero://open-pdf/library/items/HMVNQVXZ?page=8&annotation=AE8LG85Q)

---

> The denoising architecture employs pre-trained ESM-2-8M attention blocks integrated with positional time embeddings and a multilayer perceptron (MLP) for final processing [page. 8](zotero://open-pdf/library/items/HMVNQVXZ?page=8&annotation=JACJ6SCW)

---

> exponential moving average. [page. 8](zotero://open-pdf/library/items/HMVNQVXZ?page=8&annotation=FYJBZDMU)

**Notiz:** Was ist das? Jüngere Daten bekommen mehr gewicht als ältere (also auf x-achse weiter weg)

---

> The update rule is as follows: Θema;t = decay ∗ Θema;t 1 + (1 decay)Θcurrent, where Θema;t is the model’s EMA weights at training step t, Θcurrent is the model’s weights after the optimizer update at step t and decay is a hyperparameter set to a value very close to 1 [page. 8](zotero://open-pdf/library/items/HMVNQVXZ?page=8&annotation=ARF36KII)

---

> We utilized APEX,49 a bacterial strain-specific antimicrobial activity predictor for peptide sequences to rank and select the generated peptides (https://gitlab.com/machine-biologygroup-public/apex-pathogen). As APEX predicts antimicrobial activities against eleven pathogen strains, we used the mean prediction MIC to sort the peptides obtained by AMPDiffusion. [page. 9](zotero://open-pdf/library/items/HMVNQVXZ?page=9&annotation=J39AWZ2J)

**Notiz:** Das APEX model

---

> To ensure that the selected sequences for validation were (a) showing high antimicrobial activities, (b) sufficiently distinct from known AMPs, and (c) covered a diverse sequence space, we applied three filtering criteria to the generated peptide sequences [page. 9](zotero://open-pdf/library/items/HMVNQVXZ?page=9&annotation=6H2EYQ2K)

---

> The models use a standard transformer decoder with left-to-right causal masking and incorporate rotary positional encodings. [page. 9](zotero://open-pdf/library/items/HMVNQVXZ?page=9&annotation=ADJ6TYCG)

**Notiz:** Was könnte ein Transformer decoder sein?

---

> Perplexity measures how well a probabilistic model predicts a sample and is commonly used in language models, including pLMs, to evaluate sequence quality. [page. 9](zotero://open-pdf/library/items/HMVNQVXZ?page=9&annotation=C8AHQZS3)

---

> The perplexity is defined as the exponential of the average negative log-likelihood per token. [page. 9](zotero://open-pdf/library/items/HMVNQVXZ?page=9&annotation=J6SPGHCE)

---
%% end annotations %%

%% Import Date: 2026-09-25T10:19:26.311+02:00 %%
