# Introduction
MyViT is simplified version of [`rwightman/pytorch-image-models/timm/models/vision_transformer`](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py)  

This project aim to make easy to review code and the paper [*<An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale>*](https://arxiv.org/pdf/2010.11929.pdf)

# Equations

## Transformer Encoder

$$\begin{aligned}
(H, W) &= \text{the resolution of the original image}\\
C &= \text{the number of channels}\\
(P, P) &= \text{the resolution of each image patch}\\
D &= \text{latent vector size}\\
N' &= H \cdot W / P^2 = \text{the number of patches}\\
N &= N' + 1 = \text{the Transformer’s sequence length}\\
\\
\mathrm{LN} &= \text{LayerNorm}\\
\\
&\textbf{Input}\\
\mathbf{x}_{p} &\in \mathbb{R}^{N' \times (P^2 \cdot C)}\\
\\
&\textbf{Learnable}\\
\mathbf{E} &\in \mathbb{R}^{(P^2 \cdot C) \times D}\\
\mathbf{E}_{pos} &\in \mathbb{R}^{N \times D}\\
\mathbf{x}_{class} &\in \mathbb{R}^{1 \times D}\\
\\
\mathbf{z}_{0} &= [\mathbf{x}_{class}\ ;\ \mathbf{x}_{p}\mathbf{E}]~+~\mathbf{E}_{pos} &\mathbf{z}_{0} &\in \mathbb{R}^{N \times D}\\
\\
\mathbf{z'}_{l} &= \mathrm{MSA}(~\mathrm{LN}(\mathbf{z}_{l-1})~)~+~\mathbf{z}_{l-1} &\mathbf{z'}_{l} &\in \mathbb{R}^{N \times D}\\
\mathbf{z}_{l} &= \mathrm{MLP}(~\mathrm{LN}(\mathbf{z'}_{l})~)~+~\mathbf{z'}_{l} &\mathbf{z}_{l} &\in \mathbb{R}^{N \times D}\\
&\text{,where} \quad l = 1 \ldots L\\
\\
&\textbf{Output}&\\
\mathbf{y} &= \mathrm{LN}(\mathbf{z}^{0}_{L}) &\mathbf{y} &\in \mathbb{R}^{D}\\
\end{aligned}$$

## MSA (Multihead Self Attention)

$$\begin{aligned}
h &= \text{number of heads}\\
d &= D / h\\
\\
&\textbf{Input}\\
\mathbf{z} &\in \mathbb{R}^{N \times D}\\
\\
&\textbf{Learnable}\\
\mathbf{U}_{qkv} &\in \mathbb{R}^{D \times (3 \cdot d)}\\
\mathbf{U}_{msa} &\in \mathbb{R}^{D \times D}\\
\\
[\mathbf{q, k, v}] &= \mathbf{zU}_{qkv} &\mathbf{q, k, v} &\in \mathbb{R}^{N \times d}\\
\\
A &= \mathrm{softmax}(\ \mathbf{qk}^{\top}\ /\ \sqrt{d}\ ) &A &\in \mathbb{R}^{N \times N}\\
\\
\mathrm{SA}(\mathbf{z}) &= A\mathbf{v} &\mathrm{SA}(\mathbf{z}) &\in \mathbb{R}^{N \times d}\\
\\
&\textbf{Output}\\
\mathrm{MSA}(\mathbf{z}) &= [\mathrm{SA}_{1}(\mathbf{z}) ; \mathrm{SA}_{2}(\mathbf{z}) ; \cdots ; \mathrm{SA}_{h}(\mathbf{z})] \mathbf{U}_{msa} &\mathrm{MSA}(\mathbf{z}) &\in \mathbb{R}^{N \times D}
\end{aligned}$$

## MLP(Mulilayer Perceptron)
$$\begin{aligned}
D_{hidden} &= \text{hidden layer size}\\
\\
&\textbf{Input}\\
\mathbf{z} &\in \mathbb{R}^{N \times D}\\
\\
&\textbf{Learnable}\\
\mathbf{L}_{hidden} &\in \mathbb{R}^{D \times D_{hidden}}\\
\mathbf{L}_{out} &\in \mathbb{R}^{D_{hidden} \times D}\\
\\
&\textbf{Output}\\
\mathrm{MLP}(\mathbf{z}) &= \mathrm{GELU}(~\mathbf{zL}_{hidden}~)\mathbf{L}_{out}  &\mathrm{MLP}(\mathbf{z}) &\in \mathbb{R}^{N \times D}
\end{aligned}$$