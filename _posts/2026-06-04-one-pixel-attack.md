---
title: "The One-Pixel Attack: Fooling a Neural Network by Changing One Pixel"
date: 2026-06-04 00:00:00 -0400
categories: [rl-blogs]
tags: [Adversarial ML]
description: "A detailed explanation of the one-pixel attack, why it works, how it is formulated, and what it teaches us about neural network robustness."
math: true
mermaid: false
pin: false
---

## 1. Motivation

One of the most surprising lessons from adversarial machine learning is that a neural network can be highly accurate on clean test data and still be extremely fragile under carefully chosen perturbations.

The **one-pixel attack** is one of the cleanest examples of this fragility.

The idea sounds almost absurd:

> Can we fool an image classifier by changing only one pixel?

Surprisingly, the answer is often yes.

This attack was introduced by Jiawei Su, Danilo Vasconcellos Vargas, and Kouichi Sakurai in the paper **“One Pixel Attack for Fooling Deep Neural Networks.”** The paper studies an extremely restricted adversarial setting where the attacker is allowed to modify only one pixel of the input image. Despite this severe restriction, the authors showed that deep neural networks can still be fooled in many cases.

This is fascinating because the attack is not changing the entire image. It is not adding noise everywhere. It is not using a large visible patch. It changes only one spatial location.

And yet, the classifier may completely change its prediction.

That is the core beauty of the one-pixel attack.

---

## 2. The image-classification setup

Let an image classifier be a function

$$
f_\theta : [0,1]^{H \times W \times C} \to \Delta^K,
$$

where:

- $$H$$ is the image height,
- $$W$$ is the image width,
- $$C$$ is the number of channels,
- $$K$$ is the number of classes,
- $$\Delta^K$$ is the probability simplex over $$K$$ classes.

For an image $$x$$, the model outputs

$$
f_\theta(x)
=
\bigl(
p_1(x), p_2(x), \ldots, p_K(x)
\bigr),
$$

where $$p_k(x)$$ is the model's predicted probability for class $$k$$.

The predicted class is

$$
\widehat{y}(x)
=
\arg\max_{k \in \{1,\ldots,K\}} p_k(x).
$$

For example, if the model is trained on CIFAR-10, then the classes may be:

$$
\{\text{airplane}, \text{automobile}, \text{bird}, \text{cat}, \text{deer}, \text{dog}, \text{frog}, \text{horse}, \text{ship}, \text{truck}\}.
$$

Suppose the true image is a dog and the classifier correctly predicts dog. An adversarial attack tries to construct a perturbed image $$x_{\text{adv}}$$ such that

$$
\widehat{y}(x_{\text{adv}}) \neq \widehat{y}(x).
$$

In a targeted attack, the goal is even stronger. The attacker chooses a target class $$t$$ and tries to force

$$
\widehat{y}(x_{\text{adv}}) = t.
$$

The one-pixel attack studies this question under an extremely sparse perturbation constraint.

---

## 3. What does it mean to change one pixel?

For an RGB image, a pixel is not just one scalar value. A pixel has three color channels:

$$
(R,G,B).
$$

So changing one pixel means choosing one spatial coordinate

$$
(i,j)
$$

and replacing the RGB value at that location.

A one-pixel modification can therefore be represented by five numbers:

$$
z = (i,j,r,g,b).
$$

Here:

- $$i$$ is the row coordinate,
- $$j$$ is the column coordinate,
- $$r$$ is the new red-channel value,
- $$g$$ is the new green-channel value,
- $$b$$ is the new blue-channel value.

If the image is normalized, then

$$
r,g,b \in [0,1].
$$

If the image is stored in 8-bit form, then

$$
r,g,b \in \{0,1,\ldots,255\}.
$$

Given a candidate perturbation

$$
z=(i,j,r,g,b),
$$

we define the modified image as

$$
x^z.
$$

This image is identical to $$x$$ everywhere except at location $$(i,j)$$, where we set

$$
x^z_{i,j,:} = (r,g,b).
$$

So the one-pixel attack is a very low-dimensional search problem. Instead of optimizing over all image coordinates, we only optimize over five variables.

That is the first key insight.

---

## 4. The adversarial objective

There are two common versions of the attack.

### 4.1 Untargeted one-pixel attack

In an untargeted attack, the attacker simply wants the classifier to make a mistake.

If the true label is $$y$$, the attacker wants

$$
\widehat{y}(x^z) \neq y.
$$

A natural optimization objective is to minimize the model's confidence in the true class:

$$
\min_z p_y(x^z).
$$

Equivalently, we can maximize

$$
1 - p_y(x^z).
$$

The constraint is that $$z$$ must represent only one changed pixel.

Thus, the untargeted one-pixel attack can be written as

$$
\max_{i,j,r,g,b}
\left[
1 - p_y(x^{(i,j,r,g,b)})
\right].
$$

### 4.2 Targeted one-pixel attack

In a targeted attack, the attacker wants the classifier to predict a particular target class $$t$$.

For example, the image may be a dog, but the attacker wants the model to classify it as a bird.

The objective is

$$
\max_z p_t(x^z).
$$

Writing the pixel variables explicitly, we get

$$
\max_{i,j,r,g,b}
p_t
\left(
x^{(i,j,r,g,b)}
\right).
$$

The attack succeeds if

$$
\arg\max_k p_k(x^z)=t.
$$

This is a compact and elegant optimization problem:

> Search over one pixel location and one RGB value so that the model's prediction changes.

---

## 5. The sparse-perturbation view

Most adversarial attacks are described using norm constraints.

For example, an $$\ell_\infty$$-bounded attack may solve

$$
\max_{\|\delta\|_\infty \leq \epsilon}
L(f_\theta(x+\delta),y),
$$

where $$L$$ is the loss function.

This kind of attack changes many pixels by a small amount.

The one-pixel attack is different. It changes very few pixels, but the changed pixel may change by a large amount.

So the natural constraint is closer to an $$\ell_0$$ constraint.

Let $$\delta$$ be the perturbation. Then a one-pixel perturbation satisfies

$$
\|\delta\|_{0,\text{pixel}} \leq 1.
$$

Here $$\|\cdot\|_{0,\text{pixel}}$$ counts the number of spatial pixel locations whose RGB vector is changed.

So the targeted version can be written as

$$
\max_\delta p_t(x+\delta)
$$

subject to

$$
\|\delta\|_{0,\text{pixel}} \leq 1,
$$

and

$$
x+\delta \in [0,1]^{H \times W \times C}.
$$

This is important because the attack is not necessarily small in every norm.

For example, if a black pixel is turned into a bright red pixel, then the change at that pixel can be large in $$\ell_\infty$$. But the change is still extremely sparse.

So the one-pixel attack is best understood as a **sparse adversarial attack**, not necessarily an imperceptible attack.

---

## 6. Why this is a black-box attack

Many adversarial attacks require access to gradients.

For example, the Fast Gradient Sign Method uses

$$
x_{\text{adv}}
=
x + \epsilon \cdot \operatorname{sign}
\left(
\nabla_x L(f_\theta(x),y)
\right).
$$

This requires knowledge of the model and the ability to compute the gradient with respect to the input.

The one-pixel attack does not need this.

The attacker only needs to query the model. For a candidate pixel modification $$z$$, the attacker computes

$$
f_\theta(x^z)
=
(p_1(x^z),\ldots,p_K(x^z)).
$$

Then the attacker uses the output probability as a score.

For a targeted attack, the score is

$$
S(z)=p_t(x^z).
$$

For an untargeted attack, the score can be

$$
S(z)=1-p_y(x^z).
$$

This makes the attack a **black-box optimization problem**.

The attacker does not need to know the model architecture.  
The attacker does not need to know the weights.  
The attacker does not need to compute gradients.

The attacker only needs model queries.

This is why the original paper used **differential evolution**, a population-based gradient-free optimization method.

---

## 7. Differential evolution: the search engine

The one-pixel attack needs to solve

$$
\max_{i,j,r,g,b}
S(i,j,r,g,b),
$$

where $$S$$ is the attack score.

This objective is difficult for several reasons:

1. The model is highly non-linear.
2. The objective is non-convex.
3. Pixel coordinates are discrete.
4. Color values may be continuous or discrete.
5. Gradients may not be available.

Differential evolution is a natural choice because it does not require gradients.

It keeps a population of candidate solutions:

$$
z_1,z_2,\ldots,z_N.
$$

Each candidate is a possible one-pixel attack:

$$
z_m=(i_m,j_m,r_m,g_m,b_m).
$$

The algorithm repeatedly improves this population using three operations:

1. Mutation
2. Crossover
3. Selection

---

## 8. Mutation

For each candidate $$z_m$$, differential evolution chooses three other candidates from the population:

$$
z_a, z_b, z_c.
$$

Then it creates a mutant vector

$$
v_m = z_a + F(z_b-z_c),
$$

where $$F>0$$ is a scaling parameter.

This operation says:

> Take one candidate and move it in a direction suggested by the difference between two other candidates.

The difference vector $$z_b-z_c$$ gives a direction in the search space. This is useful because the population gradually learns which pixel locations and color values are promising.

---

## 9. Crossover

After mutation, the algorithm combines the original candidate $$z_m$$ and the mutant candidate $$v_m$$.

This produces a trial candidate $$u_m$$.

Coordinate-wise, the trial candidate may take some entries from $$z_m$$ and some entries from $$v_m$$.

For example,

$$
z_m=(i_m,j_m,r_m,g_m,b_m),
$$

$$
v_m=(i'_m,j'_m,r'_m,g'_m,b'_m).
$$

A crossover candidate could be

$$
u_m=(i'_m,j_m,r'_m,g_m,b'_m).
$$

The goal is to preserve useful parts of the current candidate while also exploring new possibilities.

---

## 10. Selection

Now the algorithm compares the old candidate $$z_m$$ and the trial candidate $$u_m$$.

If the trial candidate has a better score, it replaces the old candidate.

For a targeted attack, this means

$$
z_m^{\text{new}}
=
\begin{cases}
u_m, & \text{if } p_t(x^{u_m}) > p_t(x^{z_m}),\\
z_m, & \text{otherwise.}
\end{cases}
$$

For an untargeted attack, the selection rule becomes

$$
z_m^{\text{new}}
=
\begin{cases}
u_m, & \text{if } 1-p_y(x^{u_m}) > 1-p_y(x^{z_m}),\\
z_m, & \text{otherwise.}
\end{cases}
$$

The population is repeatedly updated until either the attack succeeds or the query budget is exhausted.

---

## 11. Pseudocode

A targeted one-pixel attack can be summarized as follows.

```text
Input:
    image x
    classifier f
    target class t
    population size N
    number of generations G
    mutation scale F
    crossover probability CR

Initialize:
    Randomly sample N candidates:
        z_m = (i_m, j_m, r_m, g_m, b_m)

For generation = 1, 2, ..., G:

    For each candidate z_m:

        1. Choose three distinct candidates z_a, z_b, z_c.

        2. Mutation:
               v_m = z_a + F * (z_b - z_c)

        3. Crossover:
               u_m = crossover(z_m, v_m, CR)

        4. Clip u_m to valid image coordinates and color range.

        5. Evaluate:
               old_score = p_t(x modified by z_m)
               new_score = p_t(x modified by u_m)

        6. Selection:
               if new_score > old_score:
                   z_m = u_m

    If any candidate causes the model to predict target class t:
        return successful adversarial image

Return the best candidate found
```

The untargeted version is identical except that the score is changed from target-class confidence to true-class confidence reduction.

---

## 12. A small implementation sketch

Below is a simple implementation-oriented sketch. This is not a full optimized attack implementation. The goal is to show how the one-pixel modification is represented.

```python
import torch

def apply_one_pixel(image, candidate):
    """
    image: Tensor of shape (C, H, W), values in [0, 1]
    candidate: (i, j, r, g, b)
    """
    adv = image.clone()

    i, j, r, g, b = candidate

    _, H, W = adv.shape

    i = int(round(i))
    j = int(round(j))

    i = max(0, min(H - 1, i))
    j = max(0, min(W - 1, j))

    adv[0, i, j] = float(r)
    adv[1, i, j] = float(g)
    adv[2, i, j] = float(b)

    adv = torch.clamp(adv, 0.0, 1.0)

    return adv
```

For a targeted attack, the objective can be written as:

```python
def targeted_score(model, image, candidate, target_class):
    """
    Returns the model's probability for the target class
    after applying the one-pixel perturbation.
    """
    adv = apply_one_pixel(image, candidate)

    with torch.no_grad():
        logits = model(adv.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)

    return probs[0, target_class].item()
```

For an untargeted attack, we can use:

```python
def untargeted_score(model, image, candidate, true_class):
    """
    Returns one minus the model's probability for the true class.
    Larger score means the model is less confident in the correct class.
    """
    adv = apply_one_pixel(image, candidate)

    with torch.no_grad():
        logits = model(adv.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)

    return 1.0 - probs[0, true_class].item()
```

The differential evolution optimizer repeatedly proposes candidates and keeps the ones with better scores.

---

## 13. Why can one pixel matter?

At first, it feels impossible that one pixel should matter so much.

But there are several reasons why it can.

### 13.1 Neural networks do not see images like humans

Humans recognize objects using semantic structure. If we see a dog, we use global information: shape, texture, pose, background, and context.

A neural network uses learned feature representations. These representations may be useful for classification, but they are not guaranteed to align perfectly with human perception.

So a tiny input change may cause a large change in the network's internal activations.

### 13.2 Decision boundaries can be close to natural images

A classifier divides the image space into decision regions.

Define

$$
\mathcal{R}_k
=
\{x : \widehat{y}(x)=k\}.
$$

If the model predicts class $$k$$ for image $$x$$, then

$$
x \in \mathcal{R}_k.
$$

An adversarial example exists when a small perturbation moves $$x$$ from one region to another.

The one-pixel attack succeeds if there exists a one-pixel modification $$z$$ such that

$$
x^z \notin \mathcal{R}_y.
$$

For a targeted attack, it succeeds if

$$
x^z \in \mathcal{R}_t.
$$

The shocking part is that the set of all one-pixel modifications can sometimes intersect the wrong decision region.

In other words, even this tiny sparse modification set may cross the decision boundary.

### 13.3 One pixel can propagate through the network

In a convolutional neural network, early layers apply filters to local neighborhoods.

Changing one pixel affects the local convolutional responses around that pixel. As the signal propagates through deeper layers, the receptive field grows. A local change can eventually influence larger parts of the representation.

So although the perturbation is local in the input image, its effect does not necessarily remain local inside the network.

### 13.4 Softmax decisions can be unstable near ties

The final prediction is determined by the largest probability:

$$
\widehat{y}(x)=\arg\max_k p_k(x).
$$

Suppose the top two class probabilities are close:

$$
p_{\text{dog}}(x)=0.41,
$$

$$
p_{\text{cat}}(x)=0.39.
$$

The model predicts dog. But a small perturbation may change the probabilities to

$$
p_{\text{dog}}(x^z)=0.37,
$$

$$
p_{\text{cat}}(x^z)=0.43.
$$

Now the model predicts cat.

The image may still look like a dog to us, but the classifier's decision has changed.

---

## 14. Geometry of the one-pixel attack

The image space is high-dimensional.

For CIFAR-10, an image has size

$$
32 \times 32 \times 3 = 3072.
$$

So each image is a point in

$$
\mathbb{R}^{3072}.
$$

The classifier partitions this space into decision regions:

$$
\mathbb{R}^{3072}
=
\mathcal{R}_1
\cup
\mathcal{R}_2
\cup
\cdots
\cup
\mathcal{R}_K.
$$

The one-pixel attack does not search the entire space. It searches the much smaller set

$$
\mathcal{A}_1(x)
=
\{
x^{(i,j,r,g,b)}
:
(i,j)\in [H]\times[W],
(r,g,b)\in [0,1]^3
\}.
$$

This is the set of all images obtainable from $$x$$ by changing one pixel.

A targeted one-pixel attack succeeds if

$$
\mathcal{A}_1(x)\cap \mathcal{R}_t \neq \emptyset.
$$

An untargeted one-pixel attack succeeds if

$$
\mathcal{A}_1(x)\cap \mathcal{R}_y^c \neq \emptyset.
$$

This gives a clean geometric interpretation:

> The one-pixel attack asks whether the one-pixel modification set intersects a wrong decision region.

---

## 15. One-pixel versus many-pixel attacks

It is useful to compare the one-pixel attack with more standard adversarial attacks.

### 15.1 FGSM

The Fast Gradient Sign Method changes many pixels slightly:

$$
x_{\text{adv}}
=
x+\epsilon \operatorname{sign}
\left(
\nabla_x L(f_\theta(x),y)
\right).
$$

This is an $$\ell_\infty$$-bounded attack.

### 15.2 PGD

Projected Gradient Descent repeatedly applies gradient steps and projects the perturbation back into a constraint set:

$$
x_{k+1}
=
\Pi_{\mathcal{B}(x,\epsilon)}
\left[
x_k+\alpha\operatorname{sign}
\left(
\nabla_x L(f_\theta(x_k),y)
\right)
\right].
$$

PGD is usually a white-box attack.

### 15.3 One-pixel attack

The one-pixel attack changes very few pixels, often just one.

It is:

- sparse,
- black-box,
- gradient-free,
- low-dimensional,
- search-based.

It is not trying to spread small noise everywhere. It is trying to find one extremely influential pixel.

This makes it conceptually different from attacks such as FGSM and PGD.

---

## 16. Is the one-pixel attack invisible?

Not always.

This is an important point.

Changing only one pixel sounds imperceptible, but that depends on the image resolution and the amount of color change.

For a high-resolution image, one changed pixel may be almost impossible to notice.

But for a small image such as CIFAR-10, which has only

$$
32 \times 32 = 1024
$$

spatial pixels, one pixel can be visible if its color changes drastically.

Therefore, the one-pixel attack should not be understood only as an imperceptible attack.

A better interpretation is:

> The one-pixel attack is an extremely sparse attack.

Its importance comes from the fact that the classifier can be fooled by changing a tiny number of spatial locations.

---

## 17. Experimental message of the original paper

The original paper showed that deep networks trained on image datasets could be fooled by one-pixel modifications in a nontrivial number of cases.

One of the striking observations was that the attack could be performed in a black-box manner using differential evolution. The attacker did not need model gradients. It only needed to query the classifier and observe output probabilities.

The paper reported that many natural images from CIFAR-10 and ImageNet-style settings could be perturbed to at least one target class by modifying just one pixel.

The precise success rate depends on the dataset, model architecture, target class, attack budget, and evaluation protocol. So the main lesson is not a single number.

The main lesson is this:

> Neural networks can have decision boundaries that are surprisingly close to natural images, even along extremely sparse directions.

---

## 18. Robustness lessons

The one-pixel attack teaches several important lessons.

### 18.1 Accuracy is not robustness

A model can achieve high clean accuracy and still be vulnerable to adversarial perturbations.

Clean accuracy measures average-case performance on natural test data.

Robustness asks a different question:

> Does the prediction remain stable under meaningful perturbations?

The one-pixel attack shows that the answer can be no.

### 18.2 Human similarity and model similarity are different

To a human, two images may look almost identical.

To a neural network, those two images may lie on opposite sides of a decision boundary.

This means that the model's learned geometry is not necessarily aligned with human perceptual geometry.

### 18.3 Sparse perturbations matter

Many attacks focus on small dense perturbations. The one-pixel attack shows that sparse perturbations can also be dangerous.

This is especially relevant in settings where localized sensor corruption is possible.

Examples include:

- camera artifacts,
- defective pixels,
- sensor attacks,
- compression artifacts,
- small physical marks,
- local occlusions.

### 18.4 Black-box attacks are realistic

White-box attacks are useful for analysis, but black-box attacks are often more realistic.

If an attacker can query a model and observe probabilities, then gradient-free search may be enough to find adversarial examples.

The one-pixel attack is a simple example of this broader principle.

---

## 19. Possible defenses

There is no perfect defense against all adversarial attacks, but the one-pixel attack suggests several directions.

### 19.1 Adversarial training

A robust training objective can be written as

$$
\min_\theta
\mathbb{E}_{(x,y)}
\left[
\max_{\delta \in \mathcal{S}}
L(f_\theta(x+\delta),y)
\right],
$$

where $$\mathcal{S}$$ is the perturbation set.

For one-pixel robustness, we could define

$$
\mathcal{S}
=
\{\delta : \|\delta\|_{0,\text{pixel}}\leq 1\}.
$$

Then the model is trained against the worst one-pixel perturbation.

This is conceptually clean but computationally expensive, because we need to search over many possible pixel modifications during training.

### 19.2 Randomized preprocessing

One can try to reduce the effect of a carefully chosen pixel using randomized transformations, such as:

- random cropping,
- random resizing,
- random padding,
- random bit-depth reduction,
- random pixel dropout.

However, such defenses must be evaluated carefully. If the attacker knows the defense, they may adapt the attack.

### 19.3 Median filtering

Since the one-pixel attack may create an isolated abnormal pixel, median filtering can sometimes remove the perturbation.

For a pixel location, median filtering replaces the pixel value using the median of nearby values.

This can suppress isolated artifacts.

However, median filtering is not a complete defense. It can also remove useful image details, and adaptive attacks may find perturbations that survive filtering.

### 19.4 Certified sparse robustness

A stronger goal is to prove that the classifier is invariant to all one-pixel changes:

$$
\widehat{y}(x+\delta)=\widehat{y}(x)
\quad
\text{for all }
\delta
\text{ such that }
\|\delta\|_{0,\text{pixel}}\leq 1.
$$

This would be a certified robustness guarantee.

Such guarantees are difficult, but they are much more meaningful than testing against only one attack algorithm.

---

## 20. Why this attack is conceptually beautiful

The one-pixel attack is beautiful because it is minimal.

It removes almost everything unnecessary from the adversarial-example story.

No large perturbation.  
No full-image noise.  
No gradient access.  
No complicated white-box optimization.

Just one question:

> Is there one pixel that can change the model's mind?

That question is powerful because it exposes the geometry of the classifier.

If one pixel can flip a prediction, then the model's decision boundary may be dangerously close to the original image along some sparse direction.

This reveals a mismatch between what humans consider stable and what the model considers stable.

A human sees the same object.

The model sees a different class.

That is the essence of adversarial fragility.

---

## 21. Summary

The one-pixel attack is a sparse black-box adversarial attack on image classifiers.

The attack represents a perturbation as

$$
z=(i,j,r,g,b),
$$

where $$(i,j)$$ is a pixel location and $$(r,g,b)$$ is the new color.

For a targeted attack, it solves

$$
\max_{i,j,r,g,b}
p_t(x^{(i,j,r,g,b)}).
$$

For an untargeted attack, it solves

$$
\max_{i,j,r,g,b}
\left[
1-p_y(x^{(i,j,r,g,b)})
\right].
$$

The original method uses differential evolution, a gradient-free population-based optimization algorithm.

The key lessons are:

- one pixel can sometimes change a neural network's prediction,
- sparse perturbations can be powerful,
- black-box attacks can be effective,
- clean accuracy does not imply robustness,
- model similarity and human perceptual similarity are different.

The one-pixel attack is therefore not just a trick. It is a compact demonstration of a deep issue in modern machine learning:

> A classifier can be accurate, confident, and still extremely fragile.

---

## References

1. Jiawei Su, Danilo Vasconcellos Vargas, and Kouichi Sakurai, **“One Pixel Attack for Fooling Deep Neural Networks,”** arXiv:1710.08864, 2017.  
   [https://arxiv.org/abs/1710.08864](https://arxiv.org/abs/1710.08864)

2. Jiawei Su, Danilo Vasconcellos Vargas, and Kouichi Sakurai, **“One Pixel Attack for Fooling Deep Neural Networks,”** *IEEE Transactions on Evolutionary Computation*, 23(5):828--841, 2019.  
   [https://ieeexplore.ieee.org/document/8601309](https://ieeexplore.ieee.org/document/8601309)

3. Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy, **“Explaining and Harnessing Adversarial Examples,”** ICLR, 2015.  
   [https://arxiv.org/abs/1412.6572](https://arxiv.org/abs/1412.6572)

4. Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu, **“Towards Deep Learning Models Resistant to Adversarial Attacks,”** ICLR, 2018.  
   [https://arxiv.org/abs/1706.06083](https://arxiv.org/abs/1706.06083)
