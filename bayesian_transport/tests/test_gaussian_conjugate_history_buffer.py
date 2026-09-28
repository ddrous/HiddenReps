"""Numerical and lifecycle checks without executing full-size notebook training cells.

JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m unittest discover \
    -s bayesian_transport/tests -p 'test_gaussian_conjugate_history_buffer.py' -v
"""
import ast
import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import types
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtri

SOURCE = Path(__file__).resolve().parents[1] / "bayes_transport_gaussian_conjugate_history_buffer.py"


def load_definitions():
    module = types.ModuleType("gaussian_conjugate_under_test")
    sys.modules[module.__name__] = module
    module.Array = jax.Array
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), module.__dict__)
    return module


m = load_definitions()


def tiny_config(**overrides):
    cfg = m.Config(hidden_dim=8, heads=2, mlp_ratio=2, posterior_depth=1,
                   likelihood_hidden_dim=8, likelihood_heads=2, likelihood_mlp_ratio=2,
                   likelihood_depth=1, observation_sequence_depth=1, max_training_particles=4,
                   max_training_observations=3, simulation_budget=5, batch_size=2, replay_epochs=1,
                   training_seeds=(7,), categorical_proposal_warmup_steps=0,
                   categorical_proposal_reference_particles=4, categorical_proposal_candidate_particles=8,
                   categorical_proposal_knn=2, categorical_proposal_refresh_every=1)
    return replace(cfg, **overrides)


def activated_model(kind):
    model = m.ConditionalParticleTransport(tiny_config(), kind, key=jax.random.key(1))
    model = eqx.tree_at(lambda x: x.displacement_head.weight, model,
                       .02 * jax.random.normal(jax.random.key(2), model.displacement_head.weight.shape))
    if kind == "adaln":
        model = eqx.tree_at(lambda x: x.blocks[0].modulation.weight, model,
                           .1 * jax.random.normal(jax.random.key(3), model.blocks[0].modulation.weight.shape))
    return model


class GaussianMathTests(unittest.TestCase):
    def test_conjugacy_associativity_order_and_empty(self):
        cfg = tiny_config(prior_mean=.4, prior_std=1.7, noise_std=.3)
        x = np.array([[.1, -.3, .8], [.4, .1, -.2]])
        mean, variance = m.exact_posterior(x, cfg)
        mu1, var1 = m.exact_posterior(x[:, :1], cfg)
        mu2, var2 = m.exact_posterior(x[:, 1:], cfg, prior_mean=mu1, prior_variance=var1)
        np.testing.assert_allclose(mu2, mean, rtol=1e-14)
        np.testing.assert_allclose(var2, variance, rtol=1e-14)
        np.testing.assert_allclose(m.exact_posterior(x[:, ::-1], cfg)[0], mean)
        empty_mean, empty_var = m.exact_posterior(x[:, :0], cfg)
        np.testing.assert_allclose(empty_mean, cfg.prior_mean)
        self.assertAlmostEqual(empty_var, cfg.prior_std**2)
        prefix_mean, prefix_var = m.exact_prefixes(x, cfg)
        np.testing.assert_allclose(prefix_mean[:, -1], mean)
        np.testing.assert_allclose(prefix_var[:, -1], variance)

    def test_exact_empirical_w2_against_quadrature(self):
        particles = np.array([-1.2, .2, .8, 2.])
        mean, variance = .3, 1.7
        expected = sum(quad(lambda u: (value-mean-np.sqrt(variance)*ndtri(u))**2,
                            i/4, (i+1)/4, epsabs=1e-10)[0] for i, value in enumerate(particles))
        self.assertAlmostEqual(float(m.empirical_normal_w2(particles, mean, variance))**2, expected, places=9)
        self.assertEqual(float(m.cloud_w2(particles, particles[::-1])), 0)
        # A degenerate point mass has W2^2 = squared bias + reference variance.
        self.assertAlmostEqual(float(m.empirical_normal_w2(np.ones(4), mean, variance))**2,
                               (1-mean)**2+variance, places=12)

    def test_crps_fast_formula_and_analytic_bayes_risk(self):
        cfg = tiny_config()
        theta, x, prior, z = m.heldout_data(11, 30000, 10, 4, cfg)
        mean, var = m.exact_prefixes(x, cfg)
        risk = np.mean((mean-theta[:, None])**2, axis=0)
        np.testing.assert_allclose(risk, var[0], rtol=.035)
        cloud = np.array([[[1., -.1, .4, 2.]]])
        values = m.cloud_metrics(cloud, np.array([.2]), np.array([[.3]]), np.array([[.8]]), .95)
        exact = np.mean(np.abs(cloud-.2))-.5*np.mean(np.abs(cloud[..., :, None]-cloud[..., None, :]))
        self.assertAlmostEqual(values["crps"][0,0], exact)


class TransformerAndReplayTests(unittest.TestCase):
    def test_causality_permutation_and_gradients(self):
        p = jax.random.normal(jax.random.key(4), (4, 1))
        x = jnp.array([[.1], [-.3], [.9]])
        for kind in ("adaln", "cross_attention"):
            with self.subTest(kind=kind):
                model = activated_model(kind)
                predict = eqx.filter_jit(lambda model, p, x: model.predict_prefixes(p, x, inference=True))
                clouds = predict(model, p, x)
                changed = predict(model, p, x.at[2].set(50.))
                np.testing.assert_allclose(clouds[:2], changed[:2], atol=2e-6)
                self.assertGreater(float(jnp.max(jnp.abs(clouds[2]-changed[2]))), 1e-6)
                np.testing.assert_allclose(clouds[:2], predict(model, p, x[:2]), atol=2e-6)
                perm = jnp.array([3, 0, 2, 1])
                np.testing.assert_allclose(predict(model, p[perm], x), clouds[:, perm], atol=2e-6)
                incoming = jnp.broadcast_to(p, (2, 3, 4, 1))
                observations = jnp.stack([x, x])
                theta, weights, counts = jnp.array([[.2], [.4]]), jnp.array([1., 2.]), jnp.array([3, 1])
                key = jax.random.key(10)
                loss_fn = eqx.filter_jit(m.transport_objective)
                loss, (out, _) = loss_fn(model, incoming, observations, theta, weights, counts, key)
                changed_loss, _ = loss_fn(model, incoming, observations.at[1, 1:].set(100.), theta, weights, counts, key)
                np.testing.assert_allclose(loss, changed_loss, atol=1e-6)
                scores = jax.vmap(jax.vmap(m.energy_score, in_axes=(0, None)))(out, theta)
                np.testing.assert_allclose(loss, (scores[0].mean()+2*scores[1,0])/2, atol=1e-6)
                grads = eqx.filter_jit(eqx.filter_grad(lambda net: m.transport_objective(
                    net, incoming, observations, theta, weights, counts, key)[0]))(model)
                for component in (grads.observation_embedder, grads.observation_sequence_embedder, grads.blocks):
                    leaves = [np.asarray(a) for a in jax.tree_util.tree_leaves(component) if eqx.is_array(a)]
                    self.assertTrue(all(np.all(np.isfinite(a)) for a in leaves))
                    self.assertGreater(sum(np.sum(np.abs(a)) for a in leaves), 0)

    def test_prefix_replay_and_paired_inputs(self):
        cfg = tiny_config(prior_interpolation_probability=0., historical_output_prior_probability=1.)
        names = tuple(m.model_specs())
        buffer = m.SimulationBuffer(cfg, names)
        ids = buffer.add(np.zeros((2,1)), np.zeros((2,3,1)), np.ones(2), np.array([3,2]))
        a, b = "adaln__buffered", "cross_attention__buffered"
        clouds = np.broadcast_to(np.arange(3)[None,:,None,None], (2,3,4,1)).copy()
        for name in (a,b):
            buffer.update(name, ids, clouds)
        inputs_a, info = m.training_inputs(buffer, ids, a, "buffered", 3, cfg, 88)
        inputs_b, _ = m.training_inputs(buffer, ids, b, "buffered", 3, cfg, 88)
        np.testing.assert_array_equal(inputs_a, inputs_b)
        np.testing.assert_array_equal(inputs_a[0,:,0,0], [0,1,2])
        np.testing.assert_array_equal(inputs_a[1,2], 0)
        self.assertEqual(info["buffer_fraction"], 1)
        cfg = replace(cfg, prior_interpolation_probability=.5, historical_output_prior_probability=.25)
        for variant in ("fresh", "no_replay", "buffered"):
            left, _ = m.training_inputs(buffer, ids, f"adaln__{variant}", variant, 4, cfg, 90)
            right, _ = m.training_inputs(buffer, ids, f"cross_attention__{variant}", variant, 4, cfg, 90)
            np.testing.assert_array_equal(left, right)


class LifecycleTests(unittest.TestCase):
    def test_multi_seed_checkpoint_budget_matching(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = tiny_config(training_seeds=(1, 2))
            for seed, steps in ((1, (1, 2)), (2, (1,))):
                for step in steps:
                    folder = root / f"seed_{seed}" / f"checkpoint_{step:09d}"
                    folder.mkdir(parents=True)
                    for file in [f"{n}.eqx" for n in m.model_specs()] + ["training_history.csv", "run_config.json", "buffer.npz"]:
                        (folder/file).touch()
                    (folder/"complete.json").write_text(json.dumps({"step": step, "models": list(m.model_specs())}))
            selected = m.matched_checkpoint_locations(root, cfg)
            self.assertTrue(all(p.name == "checkpoint_000000001" for p in selected.values()))

    def test_train_reload_and_all_experiments(self):
        with tempfile.TemporaryDirectory() as temporary:
            requested = tiny_config(output_dir=str(Path(temporary)/"gaussian_smoke"))
            cfg, out = m.setup_run(requested, SOURCE)
            models, rows, paths = m.train_or_load(cfg, out)
            self.assertEqual(len(rows), 12)  # 6 models x (one acquisition + one replay update).
            root = out / "seed_7"
            incomplete = root / "checkpoint_999999999"
            incomplete.mkdir()
            folder = m.checkpoint_location(root, m.model_specs())
            self.assertEqual(folder.name, "checkpoint_000000002")
            saved_buffer = np.load(folder/"buffer.npz")
            self.assertEqual(saved_buffer["counts"].sum(), 5)
            loaded_cfg, loaded_out = m.setup_run(replace(requested, train=False, checkpoint_dir=str(out)), SOURCE)
            loaded, loaded_rows, loaded_paths = m.train_or_load(loaded_cfg, loaded_out)
            self.assertEqual(len(loaded_rows), len(rows))
            for name in models[7]:
                for a,b in zip(jax.tree_util.tree_leaves(models[7][name]), jax.tree_util.tree_leaves(loaded[7][name])):
                    if eqx.is_array(a):
                        np.testing.assert_array_equal(a,b)
            m.CFG, m.OUT, m.MODELS, m.CHECKPOINT_PATHS = loaded_cfg, loaded_out, loaded, loaded_paths
            m.plt.show = lambda: m.plt.close("all")
            m.TRAINING_HISTORY = loaded_rows
            # Execute the actual notebook loss cell after reloading, not a duplicate helper.
            source_lines = SOURCE.read_text().splitlines()
            start = next(i+1 for i,line in enumerate(source_lines) if line.startswith("#%% 7)"))
            end = next(i+1 for i,line in enumerate(source_lines) if line.startswith("#%% 8)"))
            nodes = [node for node in ast.parse(SOURCE.read_text()).body if start < node.lineno < end]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), m.__dict__)
            self.assertTrue(any((out/"experiments").glob("loss_*/prefix_loss.png")))
            common = dict(seed=100, trajectories=3, particles=4, batch_size=2,
                          variants=("buffered",), credible_mass=.95, bootstrap=10)
            sequential = m.run_sequential(dict(common, observations=3, fit_min_observations=1,
                                               fit_max_observations=3, snapshot_counts=(1,3), save_clouds=True))
            self.assertTrue((sequential/"convergence_slopes.csv").is_file())
            m.run_chunking(dict(common, total_observations=4, observations_per_call=(1,2,4)))
            m.run_composition(dict(common, block_observations=1, history_observations=2))
            m.run_particles(dict(common, observations=2, particle_counts=(2,4)))
            m.run_refinement(dict(common, repetitions=3))
            self.assertEqual(len(list((out/"experiments").glob("*/settings.json"))), 6)


if __name__ == "__main__":
    unittest.main()
