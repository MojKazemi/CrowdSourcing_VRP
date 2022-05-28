import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem


class optValueFunctionEval(ElementwiseProblem):
    def __init__(self):
        xl = np.zeros(2)
        xu = np.ones(2)
        xu[0] = 100
        xu[1] = 100
        super().__init__(n_var=2, n_obj=1, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # OTHER EXAMPLE:
        out["F"] = x[0] + x[1]
        # out["F"] = (x[0]-0.5)**2 + (x[1] - 0.3)**2


problem = optValueFunctionEval()

algorithm = PSO(max_velocity_rate=0.025)

res = minimize(
    problem,
    algorithm,
    seed=1,
    save_history=True,
    verbose=False
)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

