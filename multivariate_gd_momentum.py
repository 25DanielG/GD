import numpy as np
import matplotlib.pyplot as plt

def himmelblau_f(x):
  func = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
  return func

def himmelblau_g(x):
  grad = np.zeros(2)
  grad[0] = 4.0 * x[0]**3 + 4.0 * x[0] * x[1] - 44.0 * x[0] + 2.0 * x[0] + 2.0 * x[1]**2 - 14.0
  grad[1] = 2.0 * x[0]**2 + 2.0 * x[1] - 22.0 + 4.0 * x[0] * x[1] + 4.0 * x[1]**3 - 28.0 * x[1]
  return grad

def rosenbrock_f(x):
  func = (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2
  return func

def rosenbrock_g(x):
  grad = np.zeros(2)
  grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * (x[1] - x[0]**2) * x[0]
  grad[1] = 200.0 * (x[1] - x[0]**2)
  return grad

def secret_f(x):
  m = np.linspace(0, 0.1, 25)
  g = 9.8
  k = 100
  func = 0
  for i in range(25):
    if i == 0:
      func += (0.0 - x[i + 1])**2 + (0.0 - x[i + 26])**2
    elif i == 24:
      func += (x[i] - 1.0)**2 + (x[i + 25] - 0.0)**2
    else:
      func += (x[i] - x[i+1])**2 + (x[i + 25] - x[i + 26])**2
  func *= 0.5 * k
  func += np.sum(m * g * x[25::])
  return func

def secret_g(x):
  grad = np.zeros(50)
  m = np.linspace(0, 0.1, 25)
  g = 9.8
  k = 100
  for i in range(25):
    if i == 0:
      grad[i] = k * (-0.0 + 2 * x[i] - x[i + 1])
    elif i == 24:
      grad[i] = k * (-x[i - 1] + 2 * x[i] - 1.0)
    else:
      grad[i] = k * (-x[i - 1] + 2 * x[i] - x[i + 1])
    if i == 0:
      grad[i + 25] = m[i] * g + k * (-0.0 + 2 * x[i + 25] - x[i + 1 + 25])
    elif i == 24:
      grad[i + 25] = m[i] * g + k * (-x[i - 1 + 25] + 2 * x[i + 25] - 0.0)
    else:
      grad[i + 25] = m[i] * g + k * (-x[i - 1 + 25] + 2 * x[i + 25] - x[i + 1 + 25])
  return grad

def himmelblau_init():
  return np.clip(np.random.normal(size=2), -3.0, 3.0)

def rosenbrock_init():
  return np.clip(np.random.normal(size=2), -3.0, 3.0)

def secret_init():
  return np.random.random(50)

def gradient_descent(f, g, x0, learning_rate, iterations, tolerance = 1e-6):
  x = np.copy(x0)
  x_history = [x]
  i = 0
  while True:
    x_next = x - learning_rate * g(x)
    x_history.append(x_next)
    if (np.linalg.norm(x_next - x) < tolerance) or i > iterations:
      return np.array(x_history)
    x = x_next
    i += 1

np.random.seed(1)

n_inits = 10
tolerance = 1e-4
max_iterations = 1000
learning_rate = 0.001

himmelblau_history = []
rosenbrock_history = []
secret_history = []

for _ in range(n_inits):
  x0 = himmelblau_init()
  history = gradient_descent(himmelblau_f, himmelblau_g, x0, learning_rate, max_iterations, tolerance)
  himmelblau_history.append(history)

  x0 = rosenbrock_init()
  history = gradient_descent(rosenbrock_f, rosenbrock_g, x0, learning_rate, max_iterations, tolerance)
  rosenbrock_history.append(history)

  x0 = secret_init()
  history = gradient_descent(secret_f, secret_g, x0, learning_rate, max_iterations, tolerance)
  secret_history.append(history)

print("Himmelblau function:")
for history in himmelblau_history:
  print(himmelblau_f(history[-1]))

print("Rosenbrock function:")
for history in rosenbrock_history:
  print(rosenbrock_f(history[-1]))

print("Secret function:")
for history in secret_history:
  print(secret_f(history[-1]))

x1_axis = np.arange(-5, 5, 0.1)
x2_axis = np.arange(-5, 5, 0.1)
x1_mesh, x2_mesh = np.meshgrid(x1_axis, x2_axis)
y_mesh = himmelblau_f([x1_mesh, x2_mesh])
plt.contour(x1_mesh, x2_mesh, y_mesh, 20)
for i in [0,1,2]:
  plt.scatter(himmelblau_history[i][:,0], himmelblau_history[i][:,1], marker='x', label='history: {}'.format(i))
plt.title("Himmelblau GD")
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

x1_axis = np.arange(-1, 1, 0.01)
x2_axis = np.arange(-1.5, 1.5, 0.01)
x1_mesh, x2_mesh = np.meshgrid(x1_axis, x2_axis)
y_mesh = rosenbrock_f([x1_mesh,x2_mesh])
plt.contour(x1_mesh,x2_mesh,y_mesh,40)
for i in [0,1,2]:
  plt.scatter(rosenbrock_history[i][:,0], rosenbrock_history[i][:,1], marker='x', label='history: {}'.format(i))
plt.title("Rosenbrock GD")
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

def gradient_descent_momentum(f, g, x0, learning_rate, iterations, tolerance = 1e-3, rho = 0.9):
  x = np.array(x0)
  x_history = [x]
  velocity = 0
  i = 0
  while True:
    new_velocity = rho * velocity + g(x)
    x_next = x - learning_rate * new_velocity
    x = x_next
    velocity = new_velocity
    x_history.append(x_next)
    if (abs(f(x_next) - 0.0) < tolerance ) or i > iterations:
      return np.array(x_history)
    i += 1

np.random.seed(1)

n_inits = 10
tolerance = 1e-4
max_iterations = 1000
learning_rate = 0.001
rho = 0.9

himmelblau_history = []
rosenbrock_history = []
secret_history = []

for _ in range(n_inits):
  x0 = himmelblau_init()
  history = gradient_descent_momentum(himmelblau_f, himmelblau_g, x0, learning_rate, max_iterations, tolerance, rho)
  himmelblau_history.append(history)

  x0 = rosenbrock_init()
  history = gradient_descent_momentum(rosenbrock_f, rosenbrock_g, x0, learning_rate, max_iterations, tolerance, rho)
  rosenbrock_history.append(history)

  x0 = secret_init()
  history = gradient_descent_momentum(secret_f, secret_g, x0, learning_rate, max_iterations, tolerance, rho)
  secret_history.append(history)

print("Himmelblau function:")
for history in himmelblau_history:
  print(himmelblau_f(history[-1]))

print("Rosenbrock function:")
for history in rosenbrock_history:
  print(rosenbrock_f(history[-1]))

print("Secret function:")
for history in secret_history:
  print(secret_f(history[-1]))

x1_axis = np.arange(-5, 5, 0.1)
x2_axis = np.arange(-5, 5, 0.1)
x1_mesh, x2_mesh = np.meshgrid(x1_axis, x2_axis)
y_mesh = himmelblau_f([x1_mesh, x2_mesh])
plt.contour(x1_mesh, x2_mesh, y_mesh, 20)
for i in [0,1,2]:
  plt.scatter(himmelblau_history[i][:,0], himmelblau_history[i][:,1], marker='x', label='history: {}'.format(i))
plt.title("Himmelblau GD")
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

x1_axis = np.arange(-1, 1, 0.01)
x2_axis = np.arange(-1.5, 1.5, 0.01)
x1_mesh, x2_mesh = np.meshgrid(x1_axis, x2_axis)
y_mesh = rosenbrock_f([x1_mesh,x2_mesh])
plt.contour(x1_mesh,x2_mesh,y_mesh,40)
for i in [0,1,2]:
  plt.scatter(rosenbrock_history[i][:,0], rosenbrock_history[i][:,1], marker='x', label='history: {}'.format(i))
plt.title("Rosenbrock GD")
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()