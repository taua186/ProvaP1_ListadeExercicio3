import numpy as np
import matplotlib.pyplot as plt

def wilkinson_bidiagonal_matrix(n):
    """
    Constructs an n x n Wilkinson bidiagonal matrix.
    """
    diagonal = np.arange(n, 0, -1)
    off_diagonal = np.arange(n-1, 0, -1)
    A = np.diag(diagonal) + np.diag(off_diagonal, k=-1)
    return A

# Part (b): Compute and graph condition numbers
condition_numbers = []
orders = range(1, 16)
for n in orders:
    A = wilkinson_bidiagonal_matrix(n)
    cond_number = np.linalg.cond(A)
    condition_numbers.append(cond_number)

# Plot the condition numbers
plt.figure(figsize=(8, 6))
plt.plot(orders, condition_numbers, marker='o', label='Condition Number')
plt.xlabel('Matrix Order (n)')
plt.ylabel('Condition Number')
plt.title('Condition Number of Wilkinson Bidiagonal Matrices')
plt.grid()
plt.legend()
plt.show()

# Part (c): Eigenvalue computation and perturbation analysis
n = 20
A = wilkinson_bidiagonal_matrix(n)
eigenvalues_original = np.linalg.eigvals(A)

# Perturb A by 10^-10 at position (20, 1)
perturbation = np.zeros_like(A)
perturbation[19, 0] = 1e-10
A_perturbed = A + perturbation
eigenvalues_perturbed = np.linalg.eigvals(A_perturbed)

# Compare eigenvalues
print("Original Eigenvalues:")
print(np.sort(eigenvalues_original))
print("\nPerturbed Eigenvalues:")
print(np.sort(eigenvalues_perturbed))

# Comment on results
differences = np.abs(np.sort(eigenvalues_original) - np.sort(eigenvalues_perturbed))
print("\nDifferences between Original and Perturbed Eigenvalues:")
print(differences)
