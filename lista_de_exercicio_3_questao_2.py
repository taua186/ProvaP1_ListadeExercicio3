import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond, eig

def wilkinson_bidiagonal_matrix(n):
    """
    Constructs an n x n Wilkinson bidiagonal matrix.
    """
    A = np.diag(np.arange(n, 0, -1)) + np.diag(n * np.ones(n - 1), 1)
    return A

# Part (b): Compute and graph condition numbers
condition_numbers = []
orders = range(1, 16)
for n in orders:
    A = wilkinson_bidiagonal_matrix(n)
    condition_numbers.append(cond(A))

# Plot the condition numbers
plt.figure(figsize=(8, 5))
plt.plot(orders, condition_numbers, marker='o', label='Condition Number')
plt.xlabel('Matrix Order (n)')
plt.ylabel('Condition Number')
plt.title('Condition Number of Wilkinson Bidiagonal Matrices')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# Part (c): Eigenvalue computation and perturbation analysis
n = 20
A = wilkinson_bidiagonal_matrix(n)
eigenvalues_original = np.sort(np.real(eig(A)[0]))

# Perturb A by 10^-10 at position (20, 1)
A[19, 0] += 1e-10
eigenvalues_perturbed = np.sort(np.real(eig(A)[0]))

# Display results
print("Original Eigenvalues:")
print(eigenvalues_original)
print("\nPerturbed Eigenvalues:")
print(eigenvalues_perturbed)

# Differences between eigenvalues
print("\nDifferences between Original and Perturbed Eigenvalues:")
differences = np.abs(eigenvalues_original - eigenvalues_perturbed)
print(differences)
