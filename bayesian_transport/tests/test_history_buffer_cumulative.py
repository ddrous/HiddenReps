"""Focused checks without executing the experiment's notebook-style training cells.

Run with the project's JAX environment:
    JAX_PLATFORMS=cpu python -m unittest discover -s bayesian_transport/tests -v
"""
import ast
from pathlib import Path
import sys
import types
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'bayes_transport_two_moons_history_buffer.py'


def load_definitions():
    module = types.ModuleType('history_buffer_under_test')
    sys.modules[module.__name__] = module
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
            continue
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), module.__dict__)
        if isinstance(node, ast.ClassDef) and node.name == 'Config':
            module.CFG = module.replace(module.Config(), hidden_dim=8, heads=2,
                likelihood_hidden_dim=8, likelihood_heads=2, posterior_depth=1,
                likelihood_depth=1, observation_sequence_depth=1, mlp_ratio=2,
                likelihood_mlp_ratio=2, max_training_observations=3, max_training_particles=4,
                sequence_length=3, posterior_grid_size=32, evaluation_sequences=2,
                radial_std=0.05, attention_dropout_rate=0.0)
            module.Array = jax.Array
            module.PRIOR_CENTER = 0.0
            module.PRIOR_STD = 2 / np.sqrt(12.0)
            module.X_OBS = np.zeros(2, dtype=np.float32)
            module._ENERGY_NORM_EPS = 1e-12
    return module


m = load_definitions()


def nonidentity_model(conditioning='adaln'):
    cfg = m.replace(m.CFG, posterior_conditioning=conditioning)
    model = m.ConditionalParticleTransport(cfg, key=jax.random.key(0))
    # Zero displacement at initialization would make a causality test vacuous.
    model = eqx.tree_at(lambda x: x.displacement_head.weight, model,
                       0.02 * jax.random.normal(jax.random.key(1), model.displacement_head.weight.shape))
    if conditioning == 'adaln':
        # AdaLN modulation is also zero-initialized; activate it to test the observation path.
        model = eqx.tree_at(lambda x: x.blocks[0].modulation.weight, model,
                           0.1 * jax.random.normal(jax.random.key(9), model.blocks[0].modulation.weight.shape))
    return model


class PrefixTests(unittest.TestCase):
    def test_causality_truncation_single_observation_and_particle_equivariance(self):
        prior = jax.random.normal(jax.random.key(2), (4, 2)) * 0.2
        observations = jnp.array([[0.1, 0.2], [-0.4, 0.3], [0.8, -0.5]])
        for conditioning in ('adaln', 'cross_attention'):
            with self.subTest(conditioning=conditioning):
                model = nonidentity_model(conditioning)
                predict = eqx.filter_jit(lambda model, p, x: model.predict_prefixes(p, x, inference=True))
                baseline = predict(model, prior, observations)
                changed = predict(model, prior, observations.at[2].set(jnp.array([90., -70.])))
                np.testing.assert_allclose(baseline[:2], changed[:2], atol=2e-6)
                self.assertGreater(float(jnp.max(jnp.abs(baseline[-1] - changed[-1]))), 1e-6)
                for length in (1, 2, 3):
                    short = predict(model, prior, observations[:length])
                    np.testing.assert_allclose(baseline[:length], short, atol=2e-6)
                single = eqx.filter_jit(lambda model, p, x: model(p, x, inference=True))
                np.testing.assert_allclose(baseline[0], single(model, prior, observations[0]), atol=2e-6)
                order = jnp.array([2, 0, 3, 1])
                permuted = predict(model, prior[order], observations)
                np.testing.assert_allclose(permuted, baseline[:, order], atol=2e-6)

    def test_prefix_loss_padding_and_gradient_flow(self):
        model = nonidentity_model()
        prior = jax.random.normal(jax.random.key(3), (2, 3, 4, 2)) * 0.2
        observations = jax.random.normal(jax.random.key(4), (2, 3, 2))
        targets = jnp.array([[0.1, 0.2], [-0.2, -0.3]])
        weights, counts = jnp.array([1., 2.]), jnp.array([3, 1])
        objective = eqx.filter_jit(m.transport_objective)
        loss, (metrics, clouds) = objective(model, prior, observations, targets, weights, jax.random.key(5), counts)
        changed_obs = observations.at[1, 1:].set(80.0)
        changed_loss, _ = objective(model, prior, changed_obs, targets, weights, jax.random.key(5), counts)
        np.testing.assert_allclose(loss, changed_loss, atol=1e-6)
        scores = jax.vmap(jax.vmap(m.energy_score_terms, in_axes=(0, None)))(clouds, targets)[0]
        expected = (scores[0].mean() + 2 * scores[1, 0]) / 2
        np.testing.assert_allclose(loss, expected, atol=1e-6)
        np.testing.assert_array_equal(metrics['valid_rows_by_prefix'], [2, 1, 1])
        gradient = eqx.filter_jit(eqx.filter_grad(lambda model: m.transport_objective(
            model, prior, observations, targets, weights, jax.random.key(5), counts)[0]))(model)
        for component in (gradient.observation_embedder, gradient.observation_sequence_embedder,
                          gradient.blocks, gradient.displacement_head):
            arrays = [np.asarray(x) for x in jax.tree_util.tree_leaves(component) if eqx.is_array(x)]
            self.assertTrue(all(np.all(np.isfinite(x)) for x in arrays))
            self.assertGreater(sum(float(np.sum(np.abs(x))) for x in arrays), 0.0)

    def test_replay_keeps_prefixes_separate(self):
        buffer = m.SimulationBuffer(2, 4, ('buffered',))
        indices = buffer.add_batch(np.zeros((2, 2)), np.zeros((2, 3, 2)), np.ones(2), np.array([3, 1]))
        clouds = np.broadcast_to(np.arange(3)[None, :, None, None], (2, 3, 4, 2)).copy()
        buffer.update_posteriors('buffered', indices, clouds)
        replay = buffer.posterior_batch('buffered', indices, 3, np.random.default_rng(0))
        np.testing.assert_array_equal(replay[0, :, 0, 0], [0, 1, 2])
        np.testing.assert_array_equal(replay[1, 1:], 0)
        cfg = m.replace(m.CFG, prior_interpolation_probability=0., historical_output_prior_probability=1.)
        incoming, info = m.make_training_prior_batch_np(np.random.default_rng(1), np.random.default_rng(2),
            buffer, indices, 'buffered', 4, cfg)
        np.testing.assert_array_equal(incoming[0], clouds[0])
        np.testing.assert_array_equal(info['buffer_used'], 1)


class EndpointTests(unittest.TestCase):
    def scenarios(self):
        return tuple(m.Scenario(name, forced_reset_time=2 if name == 'relocation' else None)
                     for name in m.CFG.evaluation_transitions) + (m.Scenario('independent', reset_probability=1.),)

    def test_transition_adjoint(self):
        rng = np.random.default_rng(10)
        axis = m.diagnostic_grid_axis()
        mass = m.normalize_grid_mass(rng.uniform(size=(len(axis), len(axis))))
        values = rng.uniform(size=mass.shape)
        for scenario in self.scenarios() + (m.Scenario('brownian', reset_probability=0.25),):
            for t in (1, 2):
                rho = m.reset_probability_at(scenario, t)
                forward = (1 - rho) * m.predict_grid_mass(mass, axis, t, scenario) + rho / mass.size
                pulled = m.transition_grid_pullback(values, axis, t, scenario)
                np.testing.assert_allclose(np.sum(forward * values), np.sum(mass * pulled), atol=1e-10)

    def test_shared_endpoints_observations_and_reference_selection(self):
        scenarios = self.scenarios()
        axis, endpoints, common, design = m.matched_endpoint_design(scenarios, np.random.default_rng(12))
        endpoint, final_x, current_reference = None, None, None
        for index, scenario in enumerate(scenarios):
            entry = design[scenario.name]
            marginals, selection = entry['marginals'], entry['selection_weights']
            np.testing.assert_allclose(marginals[-1] * selection[-1], common, atol=1e-12)
            for t in range(m.CFG.sequence_length):
                np.testing.assert_allclose(np.sum(marginals[t] * selection[t]), 1., atol=1e-10)
            theta, observations = m.simulate_sequence_np(np.random.default_rng(index), scenario,
                endpoint_index=int(endpoints[0]), axis=axis, marginals=marginals, observation_seed=77)
            _, single, history, _ = m.filtering_grid_references(observations, scenario, selection_weights=selection)
            if endpoint is None:
                endpoint, final_x, current_reference = theta[-1], observations[-1], single[-1]
            np.testing.assert_array_equal(theta[-1], endpoint)
            np.testing.assert_array_equal(observations[-1], final_x)
            np.testing.assert_allclose(single[-1], current_reference, atol=1e-10)
            self.assertTrue(np.all(theta >= m.CFG.prior_low) and np.all(theta <= m.CFG.prior_high))
            if scenario.name == 'identity':
                np.testing.assert_array_equal(theta, np.broadcast_to(endpoint, theta.shape))
            if scenario.name == 'independent':
                np.testing.assert_allclose(single, history, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
