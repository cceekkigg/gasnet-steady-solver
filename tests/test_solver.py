import importlib.util
import unittest

import numpy as np

from gas_sss import (Control, IdealGas, LocalPipePhysics, Network, NewtonOptions, Node, Pipe,
                     START_METHODS, SteadyStateSolver, solve_newton)

HAS_REAL_GAS = all(importlib.util.find_spec(m) for m in ("CoolProp", "pyaga8"))


def looped_network(load=1.0):
    """Two supplies, a loop, a compressor, an open and a closed valve, node heights."""
    nodes = [Node("S", pressure=60e5, height=10.0), Node("S2", pressure=55e5),
             Node("a", injection=0.0, height=40.0), Node("b", injection=-8 * load),
             Node("c", injection=0.0), Node("d", injection=-6 * load, height=-20.0),
             Node("e", injection=-4 * load), Node("f", injection=2.0),
             Node("g", injection=0.0), Node("h", injection=-3 * load)]
    pipes = [Pipe("p1", "S", "a", 20e3, .6, 2e-5), Pipe("p2", "a", "b", 15e3, .5, 2e-5),
             Pipe("p3", "b", "c", 10e3, .4, 2e-5), Pipe("p4", "a", "c", 25e3, .4, 2e-5),
             Pipe("p5", "d", "e", 12e3, .3, 2e-5), Pipe("p6", "e", "f", 8e3, .3, 2e-5),
             Pipe("p7", "S2", "f", 30e3, .4, 2e-5), Pipe("p8", "g", "h", 5e3, .3, 2e-5),
             Pipe("p9", "c", "g", 7e3, .3, 2e-5), Pipe("p10", "h", "e", 9e3, .25, 2e-5)]
    controls = [Control("cs", "c", "d", "CS", setpoint=62e5),
                Control("va_open", "f", "g", "VA", is_open=True),
                Control("va_closed", "b", "h", "VA", is_open=False)]
    return Network(nodes, pipes + controls)


class SinglePipeClosedForm(unittest.TestCase):
    def test_ideal_gas_fixed_friction(self):
        """Ideal gas + constant Darcy factor: p_in^2 - p_out^2 = f R T L q|q| / (D A^2)."""
        gas, f, q = IdealGas(), 0.015, 3.0
        pipe = Pipe("p", "in", "out", 10e3, 0.15, 1e-5, 293.15)
        network = Network([Node("in", pressure=60e5), Node("out", injection=-q)], [pipe])
        result = SteadyStateSolver(network, LocalPipePhysics(gas, fixed_friction=f)).solve()
        self.assertTrue(result.success, result.numerical.message)
        exact = np.sqrt(60e5**2 - f * gas.gas_constant * pipe.temperature * pipe.length * q * q
                        / (pipe.diameter * pipe.area**2))
        self.assertAlmostEqual(result.pressures[1] / exact, 1.0, delta=1e-8)
        self.assertAlmostEqual(result.flows[0], q, delta=1e-9)


class NetworkSolve(unittest.TestCase):
    def setUp(self):
        self.network = looped_network()
        self.solver = SteadyStateSolver(self.network, LocalPipePhysics(IdealGas()))

    def test_analytical_jacobian_matches_finite_differences(self):
        y = self.solver.initial_guess("linearized")
        analytical = self.solver.jacobian(y).toarray()
        numerical = np.empty_like(analytical)
        for k in range(y.size):
            h = 1e-6 * max(abs(y[k]), 1.0)
            step = np.zeros_like(y)
            step[k] = h
            numerical[:, k] = (self.solver.residual(y + step) - self.solver.residual(y - step)) / (2 * h)
        scale = np.maximum(np.abs(analytical), 1e-8 * np.abs(analytical).max())
        self.assertLess(np.max(np.abs(analytical - numerical) / scale), 1e-4)

    def test_converged_solution_satisfies_the_equations(self):
        result = self.solver.solve()
        self.assertTrue(result.success, result.numerical.message)
        scaled = self.solver.residual(result.numerical.x) / self.solver.residual_scale
        self.assertLessEqual(np.max(np.abs(scaled)), NewtonOptions().residual_tolerance)
        self.assertEqual(result.numerical.residual_norm, result.numerical.history[-1])
        # Mass balance at every node, including recovered supply injections.
        self.assertLess(np.max(np.abs(self.network.incidence @ result.flows + result.injections)), 1e-6)
        p, q = result.pressure_by_node, result.flow_by_branch
        self.assertAlmostEqual(p["d"], 62e5, delta=1e-2)    # compressor outlet at setpoint
        self.assertAlmostEqual(p["f"], p["g"], delta=1e-2)  # open valve
        self.assertEqual(q["va_closed"], 0.0)               # closed valve

    def test_every_start_converges_to_the_same_solution(self):
        reference = self.solver.solve().numerical.x
        for method in START_METHODS:
            with self.subTest(method=method):
                result = self.solver.solve(method)
                self.assertTrue(result.success, result.numerical.message)
                # Agreement is limited by the residual tolerance (1e-8) times conditioning.
                scale = self.solver.variable_scale
                self.assertLess(np.max(np.abs(result.numerical.x - reference) / scale), 1e-5)

    def test_accepted_steps_are_logged(self):
        numerical = self.solver.solve("uniform").numerical
        self.assertEqual(len(numerical.step_lengths), numerical.iterations)
        self.assertTrue(all(0 < a <= 1 for a in numerical.step_lengths))


class NewtonCore(unittest.TestCase):
    def test_backtracking_globalizes_newton(self):
        """Full Newton steps diverge for arctan(x) from x0 = 10; backtracking converges."""
        result = solve_newton(np.arctan, lambda x: np.diag(1 / (1 + x**2)), [10.0])
        self.assertTrue(result.success, result.message)
        self.assertGreater(result.backtracks, 0)
        self.assertLess(abs(result.x[0]), 1e-8)

    def test_singular_jacobian_is_reported(self):
        result = solve_newton(lambda x: x**2 + 1, lambda x: np.zeros((1, 1)), [0.0])
        self.assertFalse(result.success)
        self.assertIn("Singular", result.message)

    def test_default_options_are_the_experiment_settings(self):
        options = NewtonOptions()
        self.assertEqual((options.max_iterations, options.residual_tolerance, options.max_backtracks),
                         (80, 1e-8, 35))
        physics = LocalPipePhysics(IdealGas())
        self.assertEqual((physics.integration_rtol, physics.integration_atol), (1e-10, 1e-4))


class Validation(unittest.TestCase):
    def test_component_without_pressure_boundary(self):
        with self.assertRaises(ValueError):
            Network([Node("a", injection=1.0), Node("b", injection=-1.0)],
                    [Pipe("p", "a", "b", 1e3, .1)])

    def test_conflicting_pressure_constraints(self):
        with self.assertRaises(ValueError):
            Network([Node("s", pressure=50e5), Node("t", pressure=40e5)],
                    [Control("cs", "s", "t", "CS", setpoint=60e5)])


@unittest.skipUnless(HAS_REAL_GAS, "requires the real-gas extra (CoolProp, pyaga8)")
class RealGas(unittest.TestCase):
    def test_presets_converge(self):
        from gas_sss import gas_preset
        for fluid, law, load in (("CH4", "gerg2008", 1.0), ("CH4", "papay", 1.0),
                                 ("H2", "gerg2008", 0.3), ("H2", "heos", 0.3)):
            with self.subTest(fluid=fluid, law=law):
                solver = SteadyStateSolver(looped_network(load), LocalPipePhysics(gas_preset(fluid, law)))
                result = solver.solve("linearized")
                self.assertTrue(result.success, result.numerical.message)

    def test_samples_reproduce_thesis_and_agree_with_references(self):
        from gas_sss.samples import compare, list_samples, run_sample
        self.assertEqual(len(list_samples()), 10)
        for sample_id in list_samples():
            with self.subTest(sample=sample_id):
                sample, result = run_sample(sample_id)
                self.assertTrue(result.success, result.numerical.message)
                self.assertEqual(result.numerical.iterations, sample["references"]["thesis"]["iterations"])
                c = compare(sample, result)
                self.assertLess(c["thesis"]["max_dp_Pa"], 1e-3)       # stored to 12 digits
                self.assertLess(c["thesis"]["max_dq_kg_s"], 1e-9)
                self.assertLess(c["mynts"]["max_dp_Pa"], 0.5e5)       # external simulators
                self.assertLess(c["simone"]["max_dp_Pa"], 0.5e5)


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "requires matplotlib")
class Plots(unittest.TestCase):
    def test_plots_render(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from gas_sss.visualization import plot_convergence, plot_network, plot_parity
        result = SteadyStateSolver(looped_network(), LocalPipePhysics(IdealGas())).solve()
        fig, axes = plt.subplots(1, 3)
        plot_network(result, ax=axes[0])
        plot_convergence([result, result], labels=["a", "b"], ax=axes[1])
        plot_parity({"self": (result.pressures, result.pressures)}, ax=axes[2])
        fig.canvas.draw()
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
