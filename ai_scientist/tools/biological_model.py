from typing import Dict, Any, Optional
import json
import os
import itertools
import numpy as np
import warnings
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.pathing import resolve_output_path

mpl_config_dir = os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
cache_dir = os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
os.makedirs(mpl_config_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

try:
    from scipy.integrate import solve_ivp  # type: ignore
    from scipy.optimize import fsolve, minimize_scalar  # type: ignore
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

    def solve_ivp(func, t_span, y0, t_eval, method="RK45", **kwargs):  # type: ignore
        """Minimal RK4 fallback when SciPy is unavailable."""
        y0_arr = np.array(y0, dtype=float)
        t_eval_arr = np.array(t_eval, dtype=float)
        ys = np.zeros((len(y0_arr), len(t_eval_arr)))
        ys[:, 0] = y0_arr

        for i in range(len(t_eval_arr) - 1):
            t = t_eval_arr[i]
            dt = t_eval_arr[i + 1] - t
            y_curr = ys[:, i]

            k1 = np.array(func(t, y_curr, **kwargs))
            k2 = np.array(func(t + dt / 2, y_curr + dt * k1 / 2, **kwargs))
            k3 = np.array(func(t + dt / 2, y_curr + dt * k2 / 2, **kwargs))
            k4 = np.array(func(t + dt, y_curr + dt * k3, **kwargs))

            ys[:, i + 1] = y_curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return SimpleNamespace(success=True, t=t_eval_arr, y=ys)

    def fsolve(*args, **kwargs):  # type: ignore
        raise ImportError("SciPy is required for fsolve; install scipy to enable equilibrium solving.")

    def minimize_scalar(*args, **kwargs):  # type: ignore
        raise ImportError("SciPy is required for minimize_scalar; install scipy to enable optimization.")

warnings.filterwarnings('ignore')


class BiologicalModel:
    """Base class for biological mathematical models"""

    def __init__(self):
        self.parameters = {}
        self.variables = {}
        self.equations = []
        self.initial_conditions = {}

    def set_parameters(self, params_dict):
        """Set model parameters"""
        self.parameters.update(params_dict)

    def set_initial_conditions(self, ic_dict):
        """Set initial conditions"""
        self.initial_conditions.update(ic_dict)

    def validate_parameters(self):
        """Validate parameter values"""
        return True

    def solve(self, time_points, method='RK45'):
        """Solve the model equations"""
        raise NotImplementedError("Subclasses must implement solve method")


class DifferentialEquationModel(BiologicalModel):
    """Class for solving systems of differential equations in biology"""

    def __init__(self):
        super().__init__()
        self.equations_defined = False

    def define_equations(self, equations_func):
        """Define the system of differential equations"""
        self.equations_func = equations_func
        self.equations_defined = True

    def predator_prey_equations(self, t, y, alpha, beta, gamma, delta):
        """Classic Lotka-Volterra predator-prey equations"""
        x, y_prey = y  # y_prey to avoid name conflict
        dxdt = alpha * x - beta * x * y_prey
        dydt = -gamma * y_prey + delta * x * y_prey
        return [dxdt, dydt]

    def epidemiology_sir(self, t, y, beta, gamma):
        """SIR epidemiological model"""
        S, I_state, R = y
        dSdt = -beta * S * I_state
        dIdt = beta * S * I_state - gamma * I_state
        dRdt = gamma * I_state
        return [dSdt, dIdt, dRdt]

    def evolutionary_game_theory(self, t, y, b, c):
        """Replicator dynamics for evolution of cooperation"""
        x_coop, x_defect = y

        # Payoff matrix for Prisoner's Dilemma
        # Cooperate: (b-c, b-c), Defect: (0, b)
        # Average fitnesses
        w_coop = x_coop * (b - c) + x_defect * b
        w_defect = x_coop * 0 + x_defect * b

        # Replicator equations
        dx_coop = x_coop * (w_coop - (x_coop * w_coop + x_defect * w_defect))
        dx_defect = x_defect * (w_defect - (x_coop * w_coop + x_defect * w_defect))

        return [dx_coop, dx_defect]

    def chemical_reaction_network(self, t, y, k1, k2, k3):
        """Simple enzyme-catalyzed reaction network"""
        S, E, ES, P = y  # Substrate, Enzyme, Enzyme-Substrate complex, Product

        dSdt = -k1 * S * E + k2 * ES
        dEdt = -k1 * S * E + (k2 + k3) * ES
        dESdt = k1 * S * E - (k2 + k3) * ES
        dPdt = k3 * ES

        return [dSdt, dEdt, dESdt, dPdt]

    def solve(self, time_points, method='RK45', **solver_kwargs):
        """Solve the system of differential equations"""
        if not self.equations_defined:
            raise ValueError("Equations must be defined before solving")

        if not self.initial_conditions:
            raise ValueError("Initial conditions must be set")

        if not self.parameters:
            raise ValueError("Parameters must be set")

        # Extract parameters for the equations function
        y0 = list(self.initial_conditions.values())

        def wrapped_equations(t, y):
            return self.equations_func(t, y, **self.parameters)

        # Solve using scipy's solve_ivp (or fallback)
        sol = solve_ivp(
            wrapped_equations,
            (time_points[0], time_points[-1]),
            y0,
            t_eval=time_points,
            method=method,
            **solver_kwargs,
        )

        if sol.success:
            return {
                'time': sol.t,
                'solutions': sol.y.T,  # Transpose to get (time, variables)
                'variables': list(self.initial_conditions.keys()),
                'success': True
            }
        else:
            return {
                'success': False,
                'message': sol.message
            }

    def find_equilibria(self, guess=None):
        """Find equilibrium points of the system"""
        if not self.equations_defined:
            raise ValueError("Equations must be defined before finding equilibria")

        if guess is None:
            guess = list(self.initial_conditions.values())

        # Function that returns derivatives (should be zero at equilibria)
        def equilibrium_func(y):
            # Use t=0 for equilibrium calculation
            return self.equations_func(0, y, **self.parameters)

        try:
            equilibrium = fsolve(equilibrium_func, guess)
            return equilibrium
        except Exception as e:
            print(f"Failed to find equilibrium: {e}")
            return None

    def compute_jacobian(self, y, t=0):
        """Compute Jacobian matrix for stability analysis"""
        if not self.equations_defined:
            raise ValueError("Equations must be defined")

        eps = 1e-8
        n = len(y)
        jacobian = np.zeros((n, n))

        for i in range(n):
            y_plus = y.copy()
            y_minus = y.copy()
            y_plus[i] += eps
            y_minus[i] -= eps

            f_plus = np.array(self.equations_func(t, y_plus, **self.parameters))
            f_minus = np.array(self.equations_func(t, y_minus, **self.parameters))

            jacobian[:, i] = (f_plus - f_minus) / (2 * eps)

        return jacobian

    def assess_stability(self, equilibrium_point):
        """Assess stability of an equilibrium point"""
        jacobian = self.compute_jacobian(equilibrium_point)
        eigenvals = np.linalg.eigvals(jacobian)

        real_parts = np.real(eigenvals)
        max_real_part = np.max(real_parts)

        if max_real_part < 0:
            stability = "stable"
        elif max_real_part > 0:
            stability = "unstable"
        else:
            stability = "marginally stable"

        return {
            'eigenvalues': eigenvals,
            'real_parts': real_parts,
            'max_real_part': max_real_part,
            'stability': stability,
            'jacobian': jacobian
        }


class AgentBasedModel(BiologicalModel):
    """Class for agent-based modeling in biology"""

    def __init__(self):
        super().__init__()
        self.agents = []
        self.time_step = 0
        self.history = []

    def initialize_agents(self, num_agents, agent_types=None, **agent_params):
        """Initialize agents in the model"""
        self.agents = []
        if agent_types is None:
            agent_types = ['default'] * num_agents

        for i in range(num_agents):
            agent = {
                'id': i,
                'type': agent_types[i],
                'fitness': 0.0,
                **agent_params
            }
            self.agents.append(agent)

    def define_agent_behavior(self, behavior_func):
        """Define behavior rules for agents"""
        self.agent_behavior = behavior_func

    def define_interaction_rules(self, interaction_func):
        """Define how agents interact"""
        self.interaction_rules = interaction_func

    def define_reproduction_rules(self, reproduction_func):
        """Define reproduction/mutation rules"""
        self.reproduction_rules = reproduction_func

    def step(self):
        """Execute one time step of the agent-based model"""
        if not hasattr(self, 'agent_behavior'):
            raise ValueError("Agent behavior must be defined")

        # Reset fitness for this round
        for agent in self.agents:
            agent['fitness'] = 0.0

        # Execute agent behaviors and interactions
        self.agent_behavior(self.agents, self.parameters)

        # Interactions between agents
        if hasattr(self, 'interaction_rules'):
            self.interaction_rules(self.agents, self.parameters)

        # Record current state
        current_state = {
            'time': self.time_step,
            'agent_states': [agent.copy() for agent in self.agents],
            'population_stats': self.get_population_stats()
        }
        self.history.append(current_state)

        # Reproduction/evolution
        if hasattr(self, 'reproduction_rules'):
            self.agents = self.reproduction_rules(self.agents, self.parameters)

        self.time_step += 1

    def get_population_stats(self):
        """Get statistics about the current population"""
        agent_types = {}
        for agent in self.agents:
            agent_type = agent.get('type', 'unknown')
            if agent_type not in agent_types:
                agent_types[agent_type] = 0
            agent_types[agent_type] += 1

        total_fitness = sum(agent.get('fitness', 0) for agent in self.agents)
        avg_fitness = total_fitness / len(self.agents) if self.agents else 0

        return {
            'total_agents': len(self.agents),
            'agent_type_counts': agent_types,
            'average_fitness': avg_fitness,
            'total_fitness': total_fitness
        }

    def run_simulation(self, num_steps):
        """Run the simulation for specified number of steps"""
        for _ in range(num_steps):
            self.step()

        return self.history

    def analyze_evolution(self):
        """Analyze evolutionary dynamics from simulation history"""
        if not self.history:
            return None

        analysis = {
            'time_points': [h['time'] for h in self.history],
            'population_sizes': [h['population_stats']['total_agents'] for h in self.history],
            'average_fitnesses': [h['population_stats']['average_fitness'] for h in self.history]
        }

        # Track agent type frequencies over time
        agent_types = set()
        for h in self.history:
            agent_types.update(h['population_stats']['agent_type_counts'].keys())

        for agent_type in agent_types:
            freq_series = []
            for h in self.history:
                freq_series.append(h['population_stats']['agent_type_counts'].get(agent_type, 0))
            analysis[f'{agent_type}_frequency'] = freq_series

        return analysis


class GameTheoryModel(BiologicalModel):
    """Models for evolutionary game theory in biology"""

    def __init__(self):
        super().__init__()

    def define_payoff_matrix(self, payoff_matrix):
        """Define payoff matrix for the game"""
        self.payoff_matrix = np.array(payoff_matrix)

    def compute_fitness(self, strategy_frequencies, strategy_i, strategy_j=None):
        """Compute fitness of strategy i against all opponents"""
        if strategy_j is not None:
            # Fitness against specific opponent
            return sum(self.payoff_matrix[strategy_i, j] * strategy_frequencies[j]
                      for j in range(len(strategy_frequencies)))

        # Average fitness against the population
        return sum(self.payoff_matrix[strategy_i, j] * strategy_frequencies[j]
                  for j in range(len(strategy_frequencies)))

    def replicator_dynamics(self, strategy_frequencies, time_points):
        """Compute evolutionary dynamics using replicator equations"""
        def replicator_eq(t, frequencies):
            avg_fitness = sum(self.compute_fitness(frequencies, i) * frequencies[i]
                            for i in range(len(frequencies)))

            dy = np.zeros_like(frequencies)
            for i in range(len(frequencies)):
                fitness_i = self.compute_fitness(frequencies, i)
                dy[i] = frequencies[i] * (fitness_i - avg_fitness)

            return dy

        initial_freq = np.array(strategy_frequencies)
        sol = solve_ivp(replicator_eq, (time_points[0], time_points[-1]),
                       initial_freq, t_eval=time_points)

        return {
            'time': sol.t,
            'frequencies': sol.y.T,
            'strategies': [f'Strategy {i}' for i in range(len(strategy_frequencies))],
            'success': sol.success
        }

    def find_evolutionary_stable_strategies(self):
        """Find Evolutionarily Stable Strategies (ESS)"""
        ess_candidates = []

        for i in range(len(self.payoff_matrix)):
            is_ess = True
            strategy_i = np.zeros(len(self.payoff_matrix))
            strategy_i[i] = 1.0

            for j in range(len(self.payoff_matrix)):
                if i == j:
                    continue

                # Check against mutant strategies
                fitness_i = self.compute_fitness(strategy_i, i, i)
                fitness_j_against_i = self.compute_fitness(strategy_i, j, i)

                # Condition 1: own strategy better against itself
                if fitness_i < fitness_j_against_i:
                    is_ess = False
                    break

                # Condition 2: when rare, mutant doesn't invade
                if fitness_i == fitness_j_against_i:
                    # Check if mutant would be selected against
                    pass

            if is_ess:
                ess_candidates.append(i)

        return ess_candidates


# Pre-built model templates for common biological scenarios
class EvolutionaryModels:
    """Collection of ready-to-use evolutionary models"""

    @staticmethod
    def cooperation_model(benefit=3, cost=1, mutation_rate=0.01):
        """Prisoner's Dilemma model for evolution of cooperation"""
        model = GameTheoryModel()

        # Payoff matrix for Prisoner's Dilemma
        # Row: focal strategy, Column: opponent strategy
        payoff_matrix = [
            [benefit - cost, -cost],  # Cooperate vs [Cooperate, Defect]
            [benefit, 0]              # Defect vs [Cooperate, Defect]
        ]

        model.define_payoff_matrix(payoff_matrix)
        model.set_parameters({
            'benefit': benefit,
            'cost': cost,
            'mutation_rate': mutation_rate
        })

        return model

    @staticmethod
    def predator_prey_model(alpha=1.0, beta=0.1, gamma=1.5, delta=0.075):
        """Lotka-Volterra predator-prey model"""
        model = DifferentialEquationModel()

        model.set_parameters({
            'alpha': alpha,  # prey reproduction
            'beta': beta,    # predation rate
            'gamma': gamma,  # predator death rate
            'delta': delta   # predator reproduction from food
        })

        model.set_initial_conditions({
            'prey': 10.0,
            'predator': 5.0
        })

        model.define_equations(model.predator_prey_equations)
        return model

    @staticmethod
    def epidemiology_sir_model(beta=0.3, gamma=0.1):
        """SIR epidemiological model"""
        model = DifferentialEquationModel()

        model.set_parameters({
            'beta': beta,   # infection rate
            'gamma': gamma  # recovery rate
        })

        model.set_initial_conditions({
            'susceptible': 0.99,  # S
            'infected': 0.01,     # I
            'recovered': 0.0      # R
        })

        model.define_equations(model.epidemiology_sir)
        return model


def solve_biological_model(model, time_points, **kwargs):
    """
    Convenience function to solve biological models

    Args:
        model: BiologicalModel instance
        time_points: Array of time points
        **kwargs: Additional arguments for solver

    Returns:
        Dictionary with solution results
    """
    if hasattr(model, 'solve'):
        return model.solve(time_points, **kwargs)
    else:
        raise ValueError("Model does not have a solve method")


def create_sample_models():
    """Create example models for testing"""
    models = {}

    # Evolutionary game theory example
    coop_model = EvolutionaryModels.cooperation_model()
    models['cooperation_evolution'] = coop_model

    # Predator-prey dynamics
    pp_model = EvolutionaryModels.predator_prey_model()
    models['predator_prey'] = pp_model

    # Epidemiology
    epi_model = EvolutionaryModels.epidemiology_sir_model()
    models['epidemiology_sir'] = epi_model

    return models


class RunBiologicalModelTool(BaseTool):
    """
    Run a small library biological model (ODE/replicator) and store outputs.
    """

    def __init__(
        self,
        name: str = "RunBiologicalModel",
        description: str = (
            "Run a built-in biological model (e.g., cooperation_evolution, predator_prey, epidemiology) "
            "and save the solution to JSON under output_dir."
        ),
    ):
        parameters = [
            {
                "name": "model_key",
                "type": "str",
                "description": "Model to run: cooperation_evolution | predator_prey | epidemiology",
            },
            {
                "name": "time_end",
                "type": "float",
                "description": "End time for simulation (default 20.0)",
            },
            {
                "name": "num_points",
                "type": "int",
                "description": "Number of time points (default 200)",
            },
            {
                "name": "output_dir",
                "type": "str",
                "description": "Directory to store results JSON (default experiment_results)",
            },
            {
                "name": "sweep_params",
                "type": "dict",
                "description": "Optional dictionary of parameters to sweep (e.g. {'beta': [0.1, 0.5, 1.0]})",
            },
        ]
        super().__init__(name, description, parameters)

    def use_tool(
        self,
        model_key: str = "cooperation_evolution",
        time_end: float = 20.0,
        num_points: int = 200,
        output_dir: str = "experiment_results",
        sweep_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        models = create_sample_models()
        if model_key not in models:
            raise ValueError(f"Unknown model_key '{model_key}'. Available: {list(models.keys())}")

        base_model = models[model_key]
        t = np.linspace(0, time_end, num_points)
        
        out_dir = BaseTool.resolve_output_dir(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path, _, _ = resolve_output_path(
            subdir=None,
            name=f"{model_key}_solution.json",
            run_root=out_dir,
            allow_quarantine=True,
            unique=True,
        )

        results = []
        
        if sweep_params:
            keys = list(sweep_params.keys())
            values = list(sweep_params.values())
            # Ensure all values are lists
            values = [[v] if not isinstance(v, list) else v for v in values]
            
            for combination in itertools.product(*values):
                current_params = dict(zip(keys, combination))
                
                # Clone/reset model for this run
                # (Simple reassignment since our models are stateful containers of params)
                base_model.set_parameters(current_params)
                
                if hasattr(base_model, "replicator_dynamics"):
                    sol = base_model.replicator_dynamics([0.01, 0.99], t)
                else:
                    sol = solve_biological_model(base_model, t)
                
                res_entry = {
                    "parameters": current_params,
                    "success": sol.get("success", False),
                    "message": sol.get("message", ""),
                    # Potentially summarize or store full time series? 
                    # For sweeps, full time series can get huge. 
                    # Let's store full for now, but user might want to limit this later.
                    "time": sol.get("time", []),
                    "variables": sol.get("variables", []),
                    "solutions": sol.get("solutions", []),
                }
                
                # Helper to convert numpy types within the loop
                def _rec_to_native(x):
                    if isinstance(x, np.ndarray):
                        return x.tolist()
                    if isinstance(x, np.floating):
                        return float(x)
                    if isinstance(x, np.integer):
                        return int(x)
                    if isinstance(x, list):
                        return [_rec_to_native(i) for i in x]
                    if isinstance(x, dict):
                        return {k: _rec_to_native(v) for k, v in x.items()}
                    return x

                results.append(_rec_to_native(res_entry))
                
            payload = {
                "model": model_key,
                "sweep_params": sweep_params,
                "results": results,
                "success": all(r["success"] for r in results)
            }
            
        else:
            # Single run behavior (legacy)
            if hasattr(base_model, "replicator_dynamics"):
                sol = base_model.replicator_dynamics([0.01, 0.99], t)
            else:
                sol = solve_biological_model(base_model, t)

            payload: Dict[str, Any] = {
                "model": model_key,
                "time": sol.get("time", []),
                "variables": sol.get("variables", []),
                "solutions": sol.get("solutions", []),
                "success": sol.get("success", False),
                "message": sol.get("message", ""),
            }

        # Convert numpy types to native for JSON (final pass for single run or top level)
        def _to_native(x):
            if hasattr(x, "tolist"):
                return x.tolist()
            return x

        # For sweep, payload["results"] is already converted, but top-level fields might need it
        payload = {k: _to_native(v) for k, v in payload.items()}

        with out_path.open("w") as f:
            json.dump(payload, f, indent=2)

        return {"output_json": str(out_path), "success": payload["success"]}
