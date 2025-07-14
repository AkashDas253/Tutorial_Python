## **Random Number Generation in NumPy**  

NumPy provides functions to generate random numbers, random sampling, and perform statistical simulations through the `numpy.random` module.

---

### **Basic Random Number Generation**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `rand()` | Generates random numbers in the range [0, 1). | `result = np.random.rand(shape)` |
| `randn()` | Generates random numbers from a standard normal distribution. | `result = np.random.randn(shape)` |
| `random()` | Generates random numbers in the range [0, 1). | `result = np.random.random(shape)` |
| `randint()` | Generates random integers in a specified range. | `result = np.random.randint(low, high, size)` |
| `choice()` | Selects random elements from an array. | `result = np.random.choice(arr, size, replace)` |

---

### **Random Distribution Functions**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `normal()` | Draws samples from a normal distribution. | `result = np.random.normal(mean, std, size)` |
| `uniform()` | Draws samples from a uniform distribution. | `result = np.random.uniform(low, high, size)` |
| `binomial()` | Draws samples from a binomial distribution. | `result = np.random.binomial(n, p, size)` |
| `poisson()` | Draws samples from a Poisson distribution. | `result = np.random.poisson(lam, size)` |
| `exponential()` | Draws samples from an exponential distribution. | `result = np.random.exponential(scale, size)` |

---

### **Shuffling and Permutations**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `shuffle()` | Shuffles an array in-place. | `np.random.shuffle(arr)` |
| `permutation()` | Returns a randomly permuted sequence. | `result = np.random.permutation(arr)` |

---

### **Setting the Random Seed**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `seed()` | Sets the random seed for reproducibility. | `np.random.seed(value)` |

---

### **Summary**  
- **Basic random numbers**: `rand()`, `randn()`, `random()`, `randint()`, `choice()`.  
- **Distributions**: `normal()`, `uniform()`, `binomial()`, `poisson()`, `exponential()`.  
- **Shuffling & permutations**: `shuffle()`, `permutation()`.  
- **Reproducibility**: `seed()`.