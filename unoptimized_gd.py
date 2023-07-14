import numpy as np
import matplotlib.pyplot as plt

def lennard_jones_potential(r):
  epsilon = 0.997 # kJ/mol
  sigma = 3.20 # Angstroms
  return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

r_values = np.linspace(3, 6, 500)
potential_values = lennard_jones_potential(r_values)

plt.figure(figsize=(6, 6))
plt.plot(r_values, potential_values)
plt.xlabel('Distance')
plt.ylabel('Potential (kJ/mol)')
plt.grid(True)
plt.show()

def lennard_jones_gradient(r):
  epsilon = 0.997 # kJ/mol
  sigma = 3.20 # Angstroms
  return 4 * epsilon * (-12 / r * (sigma / r)**12 + 6 / r * (sigma / r)**6)

r_values = np.linspace(3, 6, 500)
potential_values = lennard_jones_potential(r_values)
gradient_values = lennard_jones_gradient(r_values)

plt.figure(figsize=(6, 6))
plt.plot(r_values, potential_values, label='Potential')
plt.plot(r_values, gradient_values, label='Gradient')
plt.xlabel('Distance')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.show()

def gradient_descent(initial_r, learning_rate, max_iterations, tolerance):
  r = initial_r
  iteration = 0
  while iteration < max_iterations:
      gradient = lennard_jones_gradient(r)
      r_new = r - learning_rate * gradient
      if np.linalg.norm(lennard_jones_gradient(r_new) < tolerance):
        break
      r = r_new
      iteration += 1

  return r, lennard_jones_potential(r), iteration

initial_r = 5.0
learning_rate = 0.01
max_iterations = 1000
tolerance = 1e-6
result_r, result_potential, num_iterations = gradient_descent(initial_r, learning_rate, max_iterations, tolerance)
print("Gradient Descent Results:")
print("Minimum r value:", result_r)
print("Corresponding potential value:", result_potential)
print("Number of iterations:", num_iterations)

def gradient_descent(initial_r, learning_rate, max_iterations, tolerance):
  r = initial_r
  iteration = 0
  r_history = [r]
  potential_history = [lennard_jones_potential(r)]
  while iteration < max_iterations:
    gradient = lennard_jones_gradient(r)
    r_new = r - learning_rate * gradient
    if np.linalg.norm(lennard_jones_gradient(r_new)) < tolerance:
      break;
    r = r_new
    iteration += 1
    r_history.append(r)
    potential_history.append(lennard_jones_potential(r))
  return r_history, potential_history, iteration

initial_r = 3.0
learning_rate = 0.001
max_iterations = 1000
tolerance = 1e-5
r_history, potential_history, num_iterations = gradient_descent(initial_r, learning_rate, max_iterations, tolerance)
r_values = np.linspace(3, 6, 500)
potential_values = lennard_jones_potential(r_values)

plt.figure(figsize=(6, 6))
plt.plot(r_values, potential_values, label='Potential')
plt.scatter(r_history, potential_history, color='red', label='Optimization History')
plt.xlabel('Distance')
plt.ylabel('Potential (kJ/mol)')
plt.grid(True)
plt.legend()
plt.show()
iteration_values = np.arange(num_iterations + 1)

plt.figure(figsize=(6, 6))
plt.plot(iteration_values, r_history, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.grid(True)
plt.show()

print("Gradient Descent Results:")
print("Learning rate:", learning_rate)
print("Number of iterations:", num_iterations)
print("Minimum r value:", r_history[-1])
print("Corresponding potential value:", potential_history[-1])

initial_r = 6.0
learning_rate = 0.01
max_iterations = 1000
tolerance = 1e-5

r_history, potential_history, num_iterations = gradient_descent(initial_r, learning_rate, max_iterations, tolerance)
r_values = np.linspace(3, 6, 500)
potential_values = lennard_jones_potential(r_values)

plt.figure(figsize=(6, 6))
plt.plot(r_values, potential_values, label='Potential')
plt.scatter(r_history, potential_history, color='red', label='Optimization History')
plt.xlabel('Distance')
plt.ylabel('Potential (kJ/mol)')
plt.grid(True)
plt.legend()
plt.show()
iteration_values = np.arange(num_iterations + 1)

plt.figure(figsize=(6, 6))
plt.plot(iteration_values, r_history, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.grid(True)
plt.show()

print("Gradient Descent Results:")
print("Learning rate:", learning_rate)
print("Number of iterations:", num_iterations)
print("Minimum r value:", r_history[-1])
print("Corresponding potential value:", potential_history[-1])

np.random.seed(1)
n_inits = 5

solutions = []
num_iterations = []
convergence_histories = []
learning_rate = 0.01
max_iterations = 1000
tolerance = 1e-6

for _ in range(n_inits):
  x0 = 3*np.random.random() + 3
  r_history, potential_history, iterations = gradient_descent(x0, learning_rate, max_iterations, tolerance)
  solutions.append(r_history[-1])
  num_iterations.append(iterations)
  convergence_histories.append(potential_history)

print("Gradient Descent Results:")
print("Learning rate:", learning_rate)
print("Average number of iterations:", np.mean(num_iterations))
print("Minimum r value:", np.min(solutions))
print("Maximum r value:", np.max(solutions))

plt.figure(figsize=(8, 6))
for history in convergence_histories:
  plt.plot(history)
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.title('GD History')
plt.grid(True)
plt.show()

def gramacy_lee_function(x):
  return np.sin(10 * np.pi * x) / (2 * x) + (x - 1)**4

x_values = np.linspace(-0.5, 2.5, 500)
y_values = gramacy_lee_function(x_values)

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gramacy-lee Function')
plt.grid(True)
plt.show()

def gramacy_lee_gradient(x):
  return (-np.sin(10 * np.pi * x) / (2 * x**2)) + (4 * (x - 1)**3) + ((5 * np.pi * np.cos(10 * np.pi * x)) / x)

def gradient_descent(initial_x, max_iterations, tolerance, learning_rate):
  x = initial_x
  iteration = 0
  x_history = [x]
  function_history = [gramacy_lee_function(x)]
  while iteration < max_iterations:
    gradient = gramacy_lee_gradient(x)
    x_new = x - learning_rate * gradient
    if np.linalg.norm(gramacy_lee_gradient(x_new)) < tolerance:
      break
    x = x_new
    iteration += 1
    x_history.append(x)
    function_history.append(gramacy_lee_function(x))
  return x_history, function_history, iteration

initial_x = 0.05
max_iterations = 100
learning_rate = 0.0001
tolerance = 1e-4

x_history, function_history, num_iterations = gradient_descent(initial_x, max_iterations, tolerance, learning_rate)

x_values = np.linspace(-0.5, 2.5, 500)
y_values = gramacy_lee_function(x_values)

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values)
plt.scatter(x_history, function_history, color='red')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent on Gramacy-Lee Function')
plt.grid(True)
plt.show()

print("Gradient Descent Results:")
print("Number of iterations:", num_iterations)
print("Minimum x value:", x_history[-1])
print("Corresponding function value:", function_history[-1])

np.random.seed(2)

solutions = []
num_iterations_list = []
convergence_histories = []

n_inits = 5

for _ in range(n_inits):
    x0 = 3 * np.random.random() - 0.5
    max_iterations = 1000
    learning_rate = 0.001
    tolerance = 1e-5
    x_history, function_history, num_iterations = gradient_descent(x0, max_iterations, tolerance, learning_rate)
    solutions.append(x_history[-1])
    num_iterations_list.append(num_iterations)
    convergence_histories.append(function_history)

print("Gradient Descent Results:")
print("Number of initializations:", n_inits)
print("Average number of iterations:", np.mean(num_iterations_list))
print("Solutions:", solutions)

plt.figure(figsize=(8, 6))
for history in convergence_histories:
    plt.plot(history)
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.title('GD History')
plt.grid(True)
plt.show()