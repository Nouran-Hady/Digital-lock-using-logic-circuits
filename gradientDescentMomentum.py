# Gradient descent algorithm with momentum acceleration
# A solution to the problem of zigzagging around the minimum

import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

def gradient_descent(f, df, x0, alpha, beta, num_iters):
    x = x0
    x_history = [x]
    fx_history = [f(*x)]
    
    for i in range(num_iters):
        dx = df(*x)
        # Momentum acceleration (beta * dx_prev) + (1 - beta) * dx
        if i > 0:
            dx[0] = beta * dx_prev[0] + (1 - beta) * dx[0]
            dx[1] = beta * dx_prev[1] + (1 - beta) * dx[1]
        dx_prev = dx
        x = (x[0] - alpha * dx[0], x[1] - alpha * dx[1])
        x_history.append(x)
        fx_history.append(f(*x))
        
    return x_history, fx_history

def f(x1, x2):
    return 2 * x1**2 + 2 * x2**2 + 2 * x1 * x2 + 1

x0 = (0, 2) # initial value
alpha = 0.25 # learning rate 
beta = 0 # momentum acceleration [0, 0.2 (Best in this case), 0.5, 0.7]
num_iters = 10 # number of iterations

x1, x2 = sp.symbols('x1 x2') # create a symbolic variable
dfx1 = sp.diff(f(x1, x2), x1) # compute the partial derivative of f with respect to x1
dfx2 = sp.diff(f(x1, x2), x2) # compute the partial derivative of f with respect to x2
df_lambda = sp.lambdify((x1, x2), [dfx1, dfx2]) # create a lambda function for the gradient of f

x_history, fx_history = gradient_descent(f, df_lambda, x0, alpha, beta, num_iters)

print('x_history =', x_history)
print('fx_history =', fx_history)
############################################################################################
# Create a figure and axis
fig, ax = plt.subplots()

# Set the x and y limits
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Maximize the window
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

# Set title
ax.set_title('First and Last Values of x and y')

# Add the text to the plot
ax.text(5, 5, str(x_history[0])+" - "+str(x_history[len(x_history)-1]), fontsize=24, ha='center', va='center')

# Show the plot
plt.show()
############################################################################################
# Create a figure and axis
fig, ax = plt.subplots()

# Set the x and y limits
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Maximize the window
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

# Set title
ax.set_title('First and Last Values of f(x, y)')

# Add the text to the plot
ax.text(5, 5, str(fx_history[0])+" - "+str(fx_history[len(x_history)-1]), fontsize=24, ha='center', va='center')

# Show the plot
plt.show()
############################################################################################
# Show the value of the function at each iteration

plt.plot(fx_history, 'o-')

plt.xlabel('Iteration')

plt.ylabel('f(x1, x2)')

# Maximize the window
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

plt.show()

# Plot the contour of the function and show the points of f(x1, x2) at each iteration

x1 = np.linspace(-5, 5, 100)

x2 = np.linspace(-5, 5, 100)

X1, X2 = np.meshgrid(x1, x2)

Y = f(X1, X2)

plt.contour(X1, X2, Y, 100)

plt.plot([x[0] for x in x_history], [x[1] for x in x_history], 'o-')

plt.xlabel('x1')

plt.ylabel('x2')

# Maximize the window
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

plt.show()