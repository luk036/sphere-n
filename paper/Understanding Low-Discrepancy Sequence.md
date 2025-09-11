# Understanding Low-Discrepancy Sequences: Uniform Sampling and Higher Dimensions

This essay explores the concept of low-discrepancy sequences, their applications, advantages, drawbacks, and the methodologies for their generation and mapping, particularly in the context of uniform sampling on various geometric shapes, from unit disks to higher-dimensional spheres. The discussion builds upon recent work on uniform sampling on a unit disk and revisits the fundamental aspects of these sequences.

## Revisiting Low-Discrepancy Sequences

Low-discrepancy sequences are a specific type of sequence used for sampling, characterized by their ability to provide a more uniform distribution of points within a given space compared to other random or pseudo-random sampling methods. While they possess numerous applications, it is important to acknowledge their limitations.

### Key Advantages

The sources highlight three primary advantages that set low-discrepancy sequences apart from other sampling methods:

1.  **Uniformity**: Low-discrepancy sequences inherently aim for a **uniform distribution** of points. This means that the points are spread out evenly across the sampling space, minimizing areas of high concentration or sparseness.
2.  **Determinism**: Unlike truly random sampling methods, low-discrepancy sequences are **deterministic**. This implies that for a given starting point and parameters, the sequence of points generated will always be the same, making results reproducible.
3.  **Incrementality**: This is described as the **most important advantage** and a defining characteristic. Incrementality means that the sequences can **grow progressively**.
    *   **Dynamic Distribution**: Every time a new point is added to the sequence, the **overall distribution remains relatively uniform**. This is crucial for applications where the required number of samples is unknown beforehand.
    *   **Flexibility in Applications**: In fields such as robotics, one might not know precisely how many samples are needed, potentially stopping after a few points or switching strategies. Low-discrepancy sequences accommodate this by **maintaining uniform distribution** with each new point.
    *   **Illustrative Example**: To demonstrate incrementality, the sources suggest an animation where points are added one by one. Lacking this, an alternative approach involves using two different colors. For instance, the **first ten points might be orange**, and the **next ten points might be purple**. When viewed separately, both the orange points and the purple points are quite uniform. Crucially, when these two sets are combined, the **overall distribution continues to remain uniform**. This characteristic is generally not achievable by other sampling methods.

### Drawbacks

Despite their advantages, low-discrepancy sequences are not without drawbacks. The primary disadvantage noted is that as the **number of points increases, the process of generating them becomes slower**.

## Applications and Extensions

Low-discrepancy sequences are versatile and can be applied in various contexts beyond their typical definition on the interval.

### Uniform Sampling on a Unit Disk and Circle

The discussion begins with recent work on performing **uniform sampling on a unit disk**. While typically defined on the interval \(\), the sources suggest that applying low-discrepancy sequences to a **circle** can lead to **even better uniformity**. The sequence functions equally well in this context. The principle of incrementality holds true here as well, with initial points and subsequent points each being uniform, and their combination maintaining uniformity.

### Higher Dimensions

The utility of low-discrepancy sequences extends to **higher dimensions**.

*   **Halton Sequence**: The **Halton sequence** is specifically mentioned as being applicable for higher dimensions. It often uses **Cylindrical mapping** with a special transformation to ensure uniformity in these contexts.
*   **Computer Graphics Applications**: In 1997, the author co-authored a paper focusing on **computer graphics applications** of these sequences, a paper that has been widely cited since.
*   **Hopf Fibration**: For higher dimensions, some researchers have employed the **Hopf fibration method for mapping**. This method was implemented for comparison purposes, indicating its relevance in assessing uniformity.

## Generation and Mapping Methods

The generation of low-discrepancy sequences typically involves creating the sequence itself and then applying a **simple mapping**. For two dimensions, two sequences are used. This approach can be extended to higher dimensions.

### Uniform Sampling on a Unit Disk: A Simple Solution

A recent realization regarding uniform sampling on a unit disk revealed a remarkably simple solution. To achieve uniform distribution, one must examine the **surface element** and understand its properties.

For a unit disk, the surface element is straightforward:
$$
dA = r \, dr \, d\theta \tag{1} \label{eq:1}
$$

When this surface element is integrated, the result related to the radial component becomes \(r^2\).
$$
\int r \, dr = \frac{1}{2} r^2 \tag{2} \label{eq:2}
$$

To achieve a uniform distribution, an **inverse function** must be applied. In this specific case, the inverse function is simply taking the **square root**. Thus, where \(r^2\) was present, one would instead use \(\sqrt{r}\). This method has been confirmed to work and achieve the desired uniform effect, applying to both the circumference and the interior of the circle, making it easier to understand. The implementation of this is described as quite straightforward.

### Higher Dimensions (S³ and n-dimensional Spheres)

The complexity significantly increases when dealing with higher dimensions, such as **S³** (a 3-sphere).

*   **Complex Surface Elements**: For S³, the **surface element becomes more complex**, and consequently, the **inverse function is not as obvious**. This is because the surface element itself has a more complicated form.
*   **Recursive Approach**: To tackle this, a **recursive approach** is employed. The strategy is to understand lower-dimensional spheres first and then build up to higher ones:
    *   **S⁰ (a point)**: This case is **trivial**.
    *   **S¹ (the circle)**: This case is **already solved** (as discussed with the unit disk).
    *   **π case (hemisphere)**: This case is also **handled**.
    *   All these lower-dimensional cases involve inverse functions.
*   **S³ Case and Beyond**: For the S³ case, understanding the integral is necessary, which can be expressed recursively. However, a significant challenge arises: the **inverse function for S³ cannot be expressed in a closed form**.
*   **Numerical Table Lookup**: Due to the lack of a closed-form solution for the inverse function in higher dimensions, a **numerical table lookup** method is used instead. The code implementation reflects this approach, making the solution less obvious compared to lower dimensions.
*   **Generalization to n-dimensional Spheres**: The same recursive approach is applied to **n-dimensional spheres**. The function definition itself is recursive, and its implementation relies on **lookup tables for higher dimensions**.
*   **Closed-Form Limitations**: It is explicitly stated that **only for dimensions 0 and 1 can closed-form inverse functions be found**. For all higher dimensions, **table lookup is necessary**.

This recursive structure for handling higher dimensions can be visualized conceptually:

```mermaid
graph TD
    A[Handle S^n] --> B{n = 0 or 1?};
    B -- Yes --> C[Closed-Form Inverse Function];
    B -- No --> D[Understand Surface Element];
    D --> E[Integrate Recursively];
    E --> F[Inverse Function (No Closed Form)];
    F --> G[Numerical Table Lookup];
    G --> H[Code Implementation];
```

## Conclusion

Low-discrepancy sequences offer significant advantages in uniform sampling due to their uniformity, determinism, and crucial incrementality. While they do have the drawback of slowing down with more points, their ability to maintain uniform distribution as points are progressively added makes them invaluable for dynamic applications like robotics. The methods for generating these sequences involve simple mappings, with straightforward solutions for lower dimensions like the unit disk using inverse square root functions. However, extending to higher dimensions like S³ necessitates a recursive approach and the use of numerical table lookups for inverse functions, as closed-form solutions are not available beyond dimensions 0 and 1. Further experimental comparisons are mentioned to provide more insight into their performance.

---
