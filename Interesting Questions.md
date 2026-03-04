# We’ll understand the difference between:

1. Batch Gradient Descent (BGD)
2. Stochastic Gradient Descent (SGD)
3. Mini-Batch Gradient Descent

Gradient Descent methods differ in **how much data is used to compute the gradient before updating the model parameters (θ).**

---

# 1. Batch Gradient Descent (BGD)

**Explanation (one line):**  
Uses the **entire dataset** to compute the gradient before updating the parameters.

### Example
Dataset: 1000 house prices.
```
Iteration 1 → use all 1000 samples → update θ
Iteration 2 → use all 1000 samples → update θ
Iteration 3 → use all 1000 samples → update θ
```

### Pros
- Stable updates
- Smooth convergence

### Cons
- Slow for large datasets
- Requires loading full dataset

---

# 2. Stochastic Gradient Descent (SGD)

**Explanation (one line):**  
Uses **one training example at a time** to update the parameters.

### Example
Dataset: 1000 house prices.
```
Sample 1 → update θ
Sample 2 → update θ
Sample 3 → update θ
...
Sample 1000 → update θ
```


### Pros
- Very fast
- Works well for huge datasets
- Can learn online (streaming data)

### Cons
- Updates are noisy
- Loss fluctuates

---

# 3. Mini-Batch Gradient Descent

**Explanation (one line):**  
Uses a **small subset (batch) of data** to compute the gradient.

### Example
Dataset: 1000 house prices  
Batch size: 100

```
Samples 1–100 → update θ
Samples 101–200 → update θ
Samples 201–300 → update θ
...
```


### Pros
- Faster than batch GD
- More stable than SGD
- Efficient for GPU training

### Cons
- Still slightly noisy updates

---

# Summary Table

| Method | Data Used per Update | Speed | Stability |
|------|------|------|------|
| Batch GD | Entire dataset | Slow | Very stable |
| Stochastic GD | 1 sample | Very fast | Noisy |
| Mini-Batch GD | Small batch (32–256) | Fast | Balanced |

---


# Why there is multiple iteration in Batch method? we are using full dataset at once, why then multiple iteration?

## 1️⃣ What happens in Batch Gradient Descent

In **Batch Gradient Descent**, we:

- Use the **entire dataset**
- Compute the **gradient (slope of the cost function)**
- Update **θ (parameters)**
- Repeat until the **cost is minimized**

### Update Rule

θ = θ − η * ∇J(θ)

Where:

- **θ** = model parameters  
- **η (eta)** = learning rate  
- **∇J(θ)** = gradient of the cost function  



## 2️⃣ Why One Iteration Is Not Enough

Imagine you're standing on a **hill** and trying to reach the **lowest point**.

You:

1. Check the **slope**
2. Take **one step downward**
3. Check the **slope again**
4. Take **another step**

You **cannot jump directly to the bottom**, because you don't know exactly where it is.

Machine Learning works **the same way**. 🚶‍♂️📉

The algorithm must **repeatedly adjust the parameters step by step** until it reaches the **minimum cost**.

---
# Tell me how does Stochastic and Mini Batch works then?


## Epoch vs Iteration in Gradient Descent

### What is an Epoch?
An **epoch** means:

> One complete pass through the entire training dataset.

If the dataset has **1000 samples**, one epoch means the model has processed **all 1000 samples once**.

---

## Batch Gradient Descent

Uses the **entire dataset at once** to compute the gradient and update parameters.

- One update uses **all samples**
- Therefore:
```
1 iteration = 1 epoch
```

Example:
```
Epoch 1 → use all data → update θ
Epoch 2 → use all data → update θ
Epoch 3 → use all data → update θ
```


---

## Stochastic Gradient Descent (SGD)

Updates the model **after every single training example**.

If dataset = **1000 samples**
```
1000 iterations = 1 epoch
```

Example:

```
Update with sample 1
Update with sample 2
...
Update with sample 1000
```

After sample 1000 → **1 epoch completed**

---

## Mini-Batch Gradient Descent

Updates the model using **small batches of data**.

Example:
```
Dataset size = 1000
Batch size = 100

1000 / 100 = 10 batches

10 iterations = 1 epoch
```


---

## Summary

| Method | Iterations per Epoch |
|------|------|
| Batch Gradient Descent | 1 |
| Stochastic Gradient Descent | Number of samples |
| Mini-Batch Gradient Descent | Number of batches |

---

## Key Idea

> Epoch = one full pass through dataset

> Iteration = one parameter update


# Why need to pass target y in 1d using y.ravel() and why not X with ravel as well?

Answer is Scikit learn expects the target in 1D hence ravel() and expects the X with some features, if we pass the X in 1D then it wont know how many features are there.

## Learning Schedule in Mini-Batch Gradient Descent

### Idea
A **learning schedule** gradually **reduces the learning rate (η)** during training.

Large learning rate at the start → **fast learning**  
Small learning rate later → **stable convergence**

---

## Code
```python
t0, t1 = 200, 1000

def learning_schedule(t):
    return t0 / (t + t1)
```


# How does matplotlib.pyplot.plt() works in terms of plotting the visuals?

> If Single array passed to params then it will interally take the X as len(y)

> If Two array passed the it will treat them as X and y.