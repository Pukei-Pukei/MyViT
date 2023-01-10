# Introduction
MyViT is simplified version of `timm/models/vision_transformer`  

This project aim to make easy to review code and the paper *<An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale>*


# Equations

## Transformer Encoder
$(H, W)$ = the resolution of the original image\
$C$ = the number of channels\
$(P, P)$ = the resolution of each image patch\
$D$ = latent vector size\
$N' = H \cdot W / P^2$ = the number of patches\
$N = N' + 1$ = the Transformer’s sequence length\
\
$\mathrm{LN}$ = LayerNorm\
\
**Input**\
$\mathbf{x}_{p} \in \mathbb{R}^{N' \times (P^2 \cdot C)}$\
\
**Learnable**\
$\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$\
$\mathbf{E}_{pos} \in \mathbb{R}^{N \times D}$\
$\mathbf{x}_{class} \in \mathbb{R}^{1 \times D}$\
\
$\mathbf{z}_{0} = [\mathbf{x}_{class}\ ;\ \mathbf{x}_{p}\mathbf{E}] \qquad\qquad \mathbf{z}_{0} \in \mathbb{R}^{N \times D}$\
\
$\mathbf{z'}_{l} = \mathrm{MSA}(~\mathrm{LN}(\mathbf{z}_{l-1})~)~+~\mathbf{z}_{l-1}$\
$\mathbf{z}_{l} = \mathrm{MLP}(~\mathrm{LN}(\mathbf{z'}_{l})~)~+~\mathbf{z'}_{l}$\
\
**Output**\
$\mathbf{y} = \mathrm{LN}(\mathbf{z}^{0}_{L})$

## MSA (Multihead Self Attention)
$h$ = number of heads\
$d = D / h$\
\
**Input**\
$\mathbf{z} \in \mathbb{R}^{N \times D}$\
\
**Learnable**\
$\mathbf{U}_{qkv} \in \mathbb{R}^{D \times (3 \cdot d)}$\
$\mathbf{U}_{msa} \in \mathbb{R}^{D \times D}$\
\
$[\mathbf{q, k, v}] = \mathbf{zU}_{qkv} \qquad\qquad \mathbf{q, k, v} \in \mathbb{R}^{N \times d}$\
\
$A = \mathrm{softmax}(\ \mathbf{qk}^{\top}\ /\ \sqrt{d}\ ) \qquad\qquad A \in \mathbb{R}^{N \times N}$\
\
$\mathrm{SA}(\mathbf{z}) = A\mathbf{v} \qquad\qquad \mathrm{SA}(\mathbf{z}) \in \mathbb{R}^{N \times d}$\
\
**Output**\
$\mathrm{MSA}(\mathbf{z}) = [\mathrm{SA}_{1}(\mathbf{z}) ; \mathrm{SA}_{2}(\mathbf{z}) ; \cdots ; \mathrm{SA}_{h}(\mathbf{z})] \mathbf{U}_{msa} \qquad\qquad \mathrm{MSA}(\mathbf{z}) \in \mathbb{R}^{N \times D}$

## MLP(Mulilayer Perceptron)
$D_{hidden}$ = hidden layer size\
\
**Input**\
$\mathbf{z} \in \mathbb{R}^{N \times D}$\
\
**Learnable**\
$\mathbf{L}_{hidden} \in \mathbb{R}^{D \times D_{hidden}}$\
$\mathbf{L}_{out} \in \mathbb{R}^{D_{hidden} \times D}$\
\
**Output**\
$\mathrm{MLP}(\mathbf{z}) = \mathrm{GELU}(~\mathbf{zL}_{hidden}~)\mathbf{L}_{out} \qquad\qquad \mathrm{MLP}(\mathbf{z}) \in \mathbb{R}^{N \times D}$