"""Checkpoint recovery and evaluation-input regressions without running the full experiment."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np

from test_history_buffer_cumulative import m, nonidentity_model


class PersistenceTests(unittest.TestCase):
    def test_schedule_has_ten_distinct_steps_and_includes_last(self):
        for total in (10, 11, 23, 100, 200020):
            schedule = m.checkpoint_schedule(total)
            self.assertEqual(len(set(schedule)), 10)
            self.assertEqual(schedule[-1], total)
            self.assertTrue(all(0 < b - a <= (total + 9) // 10
                                for a, b in zip((0,) + schedule[:-1], schedule)))
        self.assertEqual(m.checkpoint_schedule(3), (1, 2, 3))
        with self.assertRaises(ValueError):
            m.checkpoint_schedule(0)

    def test_checkpoint_roundtrip_recovery_configuration_and_script_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / 'source.py'
            script.write_text('# preserved source\n')
            out = root / 'run'
            cfg, resolved = m.setup_run(m.replace(m.CFG, output_dir=str(out)), script)
            self.assertEqual(resolved, out)
            self.assertEqual((out / script.name).read_bytes(), script.read_bytes())
            model = nonidentity_model()
            names = ('single_observation', 'no_replay', 'buffered')
            models = dict.fromkeys(names, model)
            configs = dict.fromkeys(names, m.replace(cfg, posterior_conditioning='adaln'))
            buffer = m.SimulationBuffer(1, 4, names)
            rows = [{'model': 'buffered', 'step': 3, 'energy_score': 0.5}]
            m.save_training_checkpoint(out, 3, models, configs, rows, buffer, {'step': 3})
            # Neither an incomplete newer set nor unpublished temporary set is loadable.
            incomplete = out / 'checkpoint_000000009'
            incomplete.mkdir()
            (incomplete / 'buffered.eqx').touch()
            (out / '.checkpoint_000000010.tmp').mkdir()
            folder, step = m.checkpoint_location(out, names)
            self.assertEqual(step, 3)
            loaded_cfg, loaded_out = m.setup_run(m.replace(m.CFG, train=False), out / script.name)
            self.assertEqual(loaded_out, out)
            self.assertFalse(loaded_cfg.train)
            self.assertEqual(loaded_cfg.posterior_conditioning, 'adaln')
            restored = m.load_model(folder / 'buffered.eqx', loaded_cfg)
            for expected, actual in zip(jax.tree_util.tree_leaves(model), jax.tree_util.tree_leaves(restored)):
                np.testing.assert_array_equal(expected, actual)
            self.assertEqual(m.read_training_rows(out / 'training_history.csv', 3), rows)
            with self.assertRaises(FileExistsError):
                m.setup_run(cfg, script)

    def test_optional_metrics_missing_corrupt_and_filtered_to_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'training_history.csv'
            with self.assertWarnsRegex(UserWarning, 'unavailable'):
                self.assertEqual(m.read_training_rows(path), [])
            path.write_text('model,step,energy_score\nbuffered,1,0.2\nbuffered,4,0.1\n')
            self.assertEqual(len(m.read_training_rows(path, 2)), 1)
            path.write_text('model,step\nbuffered,broken\n')
            with self.assertWarns(UserWarning):
                self.assertEqual(m.read_training_rows(path), [])

    def test_evaluation_cache_roundtrip_and_staleness(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'evaluation_cache.npz'
            examples = {'identity': {'theta': np.ones((3, 2)), 'paths': {'history_raw': np.zeros((3, 4, 2))}}}
            sweep = {'identity__gaussian__1': {'history_raw': np.ones((3, 4, 2))}}
            metrics = [{'method': 'buffer_xT', 'energy_score': 0.5}]
            m.save_evaluation_cache(path, 'signature', metrics, metrics, examples, sweep)
            rows, sweep_rows, restored, restored_sweep = m.load_evaluation_cache(path, 'signature')
            self.assertEqual(rows, metrics)
            self.assertEqual(sweep_rows, metrics)
            np.testing.assert_array_equal(restored['identity']['theta'], examples['identity']['theta'])
            np.testing.assert_array_equal(restored_sweep['identity__gaussian__1']['history_raw'],
                                          sweep['identity__gaussian__1']['history_raw'])
            with self.assertWarnsRegex(UserWarning, 'differ'):
                self.assertIsNone(m.load_evaluation_cache(path, 'changed'))


class EvaluationPriorTests(unittest.TestCase):
    def test_tau_endpoints_and_gaussian_moments(self):
        cloud = m.sample_evaluation_prior(np.random.default_rng(1), 20000, 'gaussian')
        np.testing.assert_allclose(cloud.mean(axis=0), m.PRIOR_CENTER, atol=0.02)
        np.testing.assert_allclose(cloud.std(axis=0), m.PRIOR_STD, atol=0.02)
        self.assertGreater(np.mean(np.abs(cloud) > 1), 0)
        anchor = np.array([[0.2, -0.3]])
        self.assertIs(m.interpolate_evaluation_prior(cloud, anchor, 1), cloud)
        np.testing.assert_allclose(m.interpolate_evaluation_prior(cloud, anchor, 0),
                                   np.broadcast_to(anchor, cloud.shape))
        np.testing.assert_allclose(m.interpolate_evaluation_prior(cloud, anchor, 0.5),
                                   (cloud + anchor) / 2, atol=1e-7)

    def test_sweep_anchor_and_history_do_not_use_future_observations(self):
        def evaluate(model, prior, observations):
            return np.asarray(prior) + np.asarray(observations).reshape(-1, 2).mean(axis=0)
        observations = np.array([[0.1, 0.2], [0.3, 0.1], [-0.1, 0.2]])
        changed = observations.copy()
        changed[-1] += 0.5
        models = dict.fromkeys(('single_observation', 'no_replay', 'buffered'), object())
        with patch.object(m, 'models', models, create=True), patch.object(m, 'evaluate_bt', evaluate):
            for family in ('uniform', 'gaussian'):
                for tau in (0.25, 1.0):
                    _, paths = m.evaluate_sequence(observations, m.Scenario('identity'), 9,
                                                   prior_family=family, tau=tau)
                    _, altered = m.evaluate_sequence(changed, m.Scenario('identity'), 9,
                                                     prior_family=family, tau=tau)
                    for name in ('history_raw', 'history_defensive', 'no_replay_history'):
                        np.testing.assert_array_equal(paths[name][:-1], altered[name][:-1])


if __name__ == '__main__':
    unittest.main()
