import numpy as np

def target_func(x, y):
    return 2*x**2 + y**2 - 12*x - 8*y + 10

def restrict_func(x, y):
    return -6*y - x**2 + 10*x

def grad_func(x, y):
    df_dx = 4*x - 12
    df_dy = 2*y - 8
    return np.array([df_dx, df_dy])

def grad_g(x, y):
    dg_dx = -2*x + 10
    dg_dy = -6
    return np.array([dg_dx, dg_dy])

def projection_gradient_method(x, y, alpha=0.1, epsilon=0.001, max_iter=1000):
    for i in range(max_iter):
        gradient = grad_func(x, y)
        constraint_gradient = grad_g(x, y)

        projection = gradient - (np.dot(gradient, constraint_gradient) / np.dot(constraint_gradient, constraint_gradient)) * constraint_gradient
        if np.linalg.norm(projection) < epsilon:
            break

        x -= alpha * projection[0]
        y -= alpha * projection[1]

        if restrict_func(x, y) < 0:
            correction = grad_g(x, y)
            x += alpha * correction[0]
            y += alpha * correction[1]

        print(f"Iteration {i+1}: x={x:.4f}, y={y:.4f}, f(x,y)={target_func(x, y):.4f}, g(x,y)={restrict_func(x, y):.4f}")

    return x, y

def penalty_method(x, y, r=5, C=1.5, epsilon=0.001, alpha=0.01, max_iter=300):
    for i in range(max_iter):
        def penalty_func(x, y):
            penalty = max(0, -restrict_func(x, y)) ** 2
            return target_func(x, y) + r * penalty

        def grad_penalty_func(x, y):
            penalty = max(0, -restrict_func(x, y))
            grad_penalty = -2 * penalty * grad_g(x, y)
            return grad_func(x, y) + r * grad_penalty

        gradient = grad_penalty_func(x, y)
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < epsilon:
            break

        x -= alpha * gradient[0] / grad_norm
        y -= alpha * gradient[1] / grad_norm

        # Збільшуємо штраф рідше і не вище певного максимуму
        if (i+1) % 25 == 0:
            r *= C
            if r > 1e6:
                r = 1e6

        print(f"Iteration {i+1}: x={x:.4f}, y={y:.4f}, f(x,y)={target_func(x, y):.4f}, penalty={penalty_func(x, y):.4f}, g(x,y)={restrict_func(x, y):.4f}")

    return x, y

print("Метод проекції градієнта:")
proj_x, proj_y = projection_gradient_method(5, 1)
print(f"Проекція градієнта: x={proj_x:.4f}, y={proj_y:.4f}, f(x,y)={target_func(proj_x, proj_y):.4f}")

print("\nМетод штрафних функцій:")
penalty_x, penalty_y = penalty_method(5, 1)
print(f"Штрафні функції: x={penalty_x:.4f}, y={penalty_y:.4f}, f(x,y)={target_func(penalty_x, penalty_y):.4f}")