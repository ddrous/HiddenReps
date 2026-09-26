"""Transport-mode regressions; no notebook training or plotting cells are executed."""
import unittest
from unittest.mock import patch
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from test_history_buffer_cumulative import m


def model_for(mode, **kwargs):
    cfg = m.replace(m.CFG, training_mode=mode, ode_steps=3, ode_max_steps=16, **kwargs)
    model = m.ConditionalParticleTransport(cfg, key=jax.random.key(101))
    return eqx.tree_at(lambda x: x.displacement_head.weight, model,
                       0.03 * jax.random.normal(jax.random.key(102), model.displacement_head.weight.shape))


class TransportModeTests(unittest.TestCase):
    def setUp(self):
        self.prior = jnp.array([[-0.4, 0.2], [0.1, -0.2], [0.2, 0.4], [0.5, -0.6]])
        self.obs = jnp.array([[0.1, -0.2], [0.3, 0.2]])

    def test_default_and_config_validation(self):
        cfg = m.Config()
        self.assertEqual(cfg.training_mode, 'one_step')
        self.assertEqual(cfg.ode_solver, 'euler')
        self.assertFalse(cfg.ode_adaptive)
        self.assertIs(m.prepare_transport_config(cfg), cfg)
        for changes in ({'training_mode': 'bad'}, {'ode_solver': 'bad'}, {'ode_steps': 0},
                        {'ode_steps': True}, {'ode_max_steps': 1}, {'ode_rtol': float('nan')},
                        {'ode_adaptive': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.validate_transport_config(m.replace(cfg, **changes))
        for mode in ('ode', 'flow_map'):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                prepared = m.prepare_transport_config(m.replace(cfg, training_mode=mode))
            self.assertTrue(any('disable interpolation' in str(w.message) for w in caught))
            self.assertEqual(prepared.prior_interpolation_probability, 0)
            self.assertEqual(prepared.historical_output_prior_probability, 0)

    def test_continuous_modes_need_no_posterior_storage(self):
        buffer = m.SimulationBuffer(1, 4, ('buffered',), store_posteriors=False)
        ids = buffer.add_batch(np.zeros((1, 2)), np.zeros((1, 3, 2)), np.ones(1), np.array([3]))
        self.assertEqual(buffer.posteriors, {})
        self.assertEqual(buffer.posterior_counts, {})
        for mode in ('ode', 'flow_map'):
            # Even unprepared configs must not interpolate or access posterior storage.
            prior, info = m.make_training_prior_batch_np(np.random.default_rng(1), np.random.default_rng(2),
                buffer, ids, 'buffered', 4, m.replace(m.CFG, training_mode=mode))
            self.assertEqual(prior.shape, (1, 3, 4, 2))
            np.testing.assert_array_equal(prior[:, 0], prior[:, 2])
            np.testing.assert_array_equal(info['exact_prior_used'], 1)
            np.testing.assert_array_equal(info['buffer_used'], 0)
            np.testing.assert_array_equal(info['interpolation_used'], 0)

    def test_euler_matches_explicit_updates_with_fixed_context(self):
        model = model_for('ode')
        context = model.observation_contexts(self.obs, inference=True)[-1]
        expected = self.prior
        for i in range(model.ode_steps):
            expected = expected + model.prior_std / model.ode_steps * model._residual(
                expected, context, i / model.ode_steps, inference=True)
        actual = eqx.filter_jit(lambda model, p, x: model(p, x, inference=True))(model, self.prior, self.obs)
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        self.assertGreater(float(jnp.max(jnp.abs(actual - self.prior))), 1e-4)

    def test_solver_choices_constant_velocity_and_identity_initialization(self):
        for mode in ('one_step', 'ode', 'flow_map'):
            model = m.ConditionalParticleTransport(m.replace(m.CFG, training_mode=mode, ode_steps=2),
                                                  key=jax.random.key(0))
            actual = eqx.filter_jit(lambda model, p, x: model(p, x, inference=True))(model, self.prior, self.obs)
            np.testing.assert_allclose(actual, self.prior, atol=1e-7)
        for solver, adaptive in (('euler', False), ('heun', False), ('midpoint', False),
                                 ('tsit5', False), ('dopri5', False), ('tsit5', True), ('dopri5', True)):
            with self.subTest(solver=solver, adaptive=adaptive):
                model = model_for('ode', ode_solver=solver, ode_adaptive=adaptive)
                model = eqx.tree_at(lambda m: m.displacement_head.weight, model,
                                    jnp.zeros_like(model.displacement_head.weight))
                bias = jnp.array([0.03, -0.05])
                model = eqx.tree_at(lambda m: m.displacement_head.bias, model, bias)
                actual = eqx.filter_jit(lambda model, p, x: model(p, x, inference=True))(model, self.prior, self.obs)
                expected = self.prior + model.prior_std * model.max_displacement * jnp.tanh(bias)
                np.testing.assert_allclose(actual, expected, atol=2e-6)

    def test_flow_map_endpoints_and_full_time_derivative(self):
        model = model_for('flow_map')
        np.testing.assert_array_equal(model.flow_at_time(self.prior, self.obs, 0., inference=True), self.prior)
        np.testing.assert_allclose(model.flow_at_time(self.prior, self.obs, 1., inference=True),
                                   model(self.prior, self.obs, inference=True), atol=1e-7)
        t, eps = 0.4, 0.001
        finite_difference = (model.flow_at_time(self.prior, self.obs, t + eps, inference=True)
                             - model.flow_at_time(self.prior, self.obs, t - eps, inference=True)) / (2 * eps)
        velocity = model.flow_velocity(self.prior, self.obs, t, inference=True)
        np.testing.assert_allclose(velocity, finite_difference, atol=5e-5, rtol=2e-3)
        context = model.observation_contexts(self.obs, inference=True)[-1]
        residual = model.prior_std * model._residual(self.prior, context, t, inference=True)
        self.assertGreater(float(jnp.max(jnp.abs(velocity - residual))), 1e-5)

    def test_flow_training_integrates_derived_velocity_from_zero_to_one(self):
        model = model_for('flow_map')
        context = model.observation_contexts(self.obs, inference=False)[-1]
        expected = self.prior
        for i in range(model.ode_steps):
            time = jnp.asarray(i / model.ode_steps, dtype=self.prior.dtype)
            velocity = jax.jvp(lambda t: model._flow_map(self.prior, context, t),
                               (time,), (jnp.ones_like(time),))[1]
            expected = expected + velocity / model.ode_steps
        predict = eqx.filter_jit(lambda model, p, x: model(p, x, inference=False))
        integrated = predict(model, self.prior, self.obs)
        np.testing.assert_allclose(integrated, expected, atol=2e-6)
        direct = model(self.prior, self.obs, inference=True)
        # A coarse Euler quadrature is deliberately distinguishable from calling Phi(1).
        self.assertGreater(float(jnp.max(jnp.abs(integrated - direct))), 1e-5)
        accurate = model_for('flow_map', ode_solver='tsit5', ode_adaptive=True,
                             ode_rtol=1e-6, ode_atol=1e-7)
        integrated_accurate = predict(accurate, self.prior, self.obs)
        np.testing.assert_allclose(integrated_accurate, direct, atol=2e-6, rtol=2e-5)

    def test_terminal_energy_score_gradients_and_prefix_causality(self):
        priors = jnp.broadcast_to(self.prior, (1, 2, 4, 2))
        obs = self.obs[None]
        target, weights, counts = jnp.array([[0.15, -0.25]]), jnp.array([1.7]), jnp.array([2])
        key = jax.random.key(110)
        for mode in ('ode', 'flow_map'):
            for conditioning in ('adaln', 'cross_attention'):
                with self.subTest(mode=mode, conditioning=conditioning):
                    model = model_for(mode, posterior_conditioning=conditioning, attention_dropout_rate=0.1)
                    if conditioning == 'adaln':
                        model = eqx.tree_at(lambda x: x.blocks[0].modulation.weight, model,
                            0.1 * jax.random.normal(key, model.blocks[0].modulation.weight.shape))
                    objective = eqx.filter_jit(eqx.filter_value_and_grad(m.transport_objective, has_aux=True))
                    (loss, (_, clouds)), grad = objective(model, priors, obs, target, weights, key, counts)
                    scores = jax.vmap(m.energy_score_terms, in_axes=(0, None))(clouds[0], target[0])[0]
                    np.testing.assert_allclose(loss, 1.7 * scores.mean(), atol=1e-6)
                    for component in (grad.observation_embedder, grad.observation_sequence_embedder,
                                      grad.blocks, grad.displacement_head, grad.time_projection):
                        arrays = [np.asarray(x) for x in jax.tree_util.tree_leaves(component) if eqx.is_array(x)]
                        self.assertTrue(all(np.isfinite(a).all() for a in arrays))
                        self.assertGreater(sum(float(np.abs(a).sum()) for a in arrays), 0.)
                    predict = eqx.filter_jit(lambda model, p, x: model.predict_prefixes(p, x, inference=True))
                    original = predict(model, self.prior, self.obs)
                    changed = predict(model, self.prior, self.obs.at[1].set(50.))
                    np.testing.assert_allclose(original[0], changed[0], atol=2e-6)
                    self.assertGreater(float(jnp.max(jnp.abs(original[1] - changed[1]))), 1e-6)

    def test_parallel_control_is_one_block_call_from_matched_prior(self):
        calls = []
        def evaluate(model, prior, x):
            calls.append((np.array(prior), np.array(x)))
            return np.asarray(prior) + np.asarray(x).reshape(-1, 2).sum(axis=0)
        models = dict.fromkeys(('single_observation', 'no_replay', 'buffered'), object())
        for length in (1, 3):
            calls.clear()
            observations = np.arange(length * 2, dtype=np.float32).reshape(length, 2) * 0.1
            with patch.object(m, 'models', models, create=True), patch.object(m, 'evaluate_bt', evaluate), \
                 patch.object(m, 'CFG', m.replace(m.CFG, sequence_length=length)), \
                 patch.object(m, 'categorical_proposal_from_posterior_particles_np',
                              lambda rng, cloud, n, cfg: (cloud, np.ones(n))):
                final, _ = m.evaluate_sequence(observations, m.Scenario('identity'), 90)
            blocks = [(p, x) for p, x in calls if x.ndim == 2]
            self.assertEqual(len(blocks), 1)
            prior, block = blocks[0]
            np.testing.assert_array_equal(block, np.repeat(observations[-1:], length, axis=0))
            np.testing.assert_array_equal(prior, calls[2][0])
            np.testing.assert_allclose(final['parallel_xT'], prior + block.sum(axis=0))
            if length == 1:
                np.testing.assert_array_equal(final['parallel_xT'], final['buffer_xT'])
                np.testing.assert_array_equal(final['parallel_xT'], final['refine_xT'])


if __name__ == '__main__':
    unittest.main()
