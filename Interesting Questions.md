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

