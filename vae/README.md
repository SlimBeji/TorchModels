# Variational Autoencoder (VAE) Cheat Sheet

## Core Definitions

-   **Posterior**: $p(z|x)$ (True latent distribution given data - encoder's target)
-   **Likelihood**: $p(x|z)$ (Data generation model - decoder's role)
-   **Prior**: $p(z)$ (Assumed latent distribution, typically $\mathcal{N}(0,I)$)
-   **Evidence**: $p(x)$ (Intractable marginal likelihood)
-   **Approximate Posterior**: $q_\phi(z|x) \approx \mathcal{N}(\mu,\sigma)$ (Encoder's output, learns $\mu,\sigma$)

## Objective: Minimize KL Divergence

**Goal**: Make $q_\phi(z|x)$ close to true posterior $p(z|x)$

We minimize the KL divergence between these distributions:

$$\min_\phi \text{KL}(q_\phi(z|x) \parallel p(z|x)) = \mathbb{E}_{q_\phi(z|x)} \left[ \ln \frac{q_\phi(z|x)}{p(z|x)} \right]$$

Key points:

1. $\phi$ are the encoder parameters we optimize
2. The expectation is taken over $q_\phi(z|x)$
3. Direct computation is intractable (requires $p(z|x)$ which depends on $p(x)$)

## Bayes' Rule

The fundamental relationships between the distributions:

1. **Joint Probability Symmetry**:
   $$p(x \cap z) = p(z \cap x)$$

2. **Conditional Probability Definition**:
   $$p(x|z)p(z) = p(z|x)p(x)$$

3. **Bayes' Theorem** (Posterior Calculation):
   $$p(z|x) = \frac{p(x|z)p(z)}{p(x)}$$

## Derivation Steps

1. **Initial KL Objective**:
   $$\text{KL}(q_\phi(z|x) \parallel p(z|x)) = \mathbb{E}_{q_\phi(z|x)} \left[ \ln \frac{q_\phi(z|x)}{p(z|x)} \right]$$

2. **Apply Bayes' Rule**:
   $$= \mathbb{E}_{q_\phi(z|x)} \left[ \ln \frac{q_\phi(z|x)p(x)}{p(x|z)p(z)} \right]$$

3. **Logarithm Expansion**:
   $$= \mathbb{E}_{q_\phi(z|x)} \left[ \ln \frac{q_\phi(z|x)}{p(z)} + \ln p(x) - \ln p(x|z) \right]$$
   $$= \underbrace{\mathbb{E}_{q_\phi(z|x)} \left[ \ln \frac{q_\phi(z|x)}{p(z)} \right]}_{(1)} + \underbrace{\mathbb{E}_{q_\phi(z|x)} [\ln p(x)]}_{(2)} - \underbrace{\mathbb{E}_{q_\phi(z|x)} [\ln p(x|z)]}_{(3)}$$

4. **Term Analysis**:

    - **(1)**: $\text{KL}(q_\phi(z|x) \parallel p(z))$ (Latent Loss)
    - **(2)**: $\ln p(x)$ (Constant, can be ignored for optimization)
    - **(3)**: $\mathbb{E}[\ln p(x|z)]$ (Reconstruction)

5. **Final Loss Components**:
   $$\text{Minimization Target} = \underbrace{-\mathbb{E}_{q_\phi(z|x)} [\ln p(x|z)]}_{\text{Reconstruction Loss}} + \underbrace{\text{KL}(q_\phi(z|x) \parallel p(z))}_{\text{Latent Loss}}$$

## Practical Implementation (PyTorch)

### Latent Loss (KL Divergence)

**Closed-form solution for Gaussians** (using [Gaussian KL Identity](https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians)):

For $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$, $p(z) = \mathcal{N}(0, I)$:
$$\text{KL} = \frac{1}{2} \sum (\mu^2 + \sigma^2 - \ln \sigma^2 - 1)$$

```python
kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
```

### Reconstruction Loss (Negative Log-Likelihood)

For image data, we model the decoder output as a Gaussian distribution. The negative log-likelihood simplifies to **Mean Squared Error** (MSE):

$$-\mathbb{E}_{q_\phi(z|x)} [\ln p_\theta(x|z)] \approx \|x - \text{decoder}(z)\|^2$$

PyTorch implementation:

```python
recon_loss = F.mse_loss(decoder_output, input, reduction='sum')
```
