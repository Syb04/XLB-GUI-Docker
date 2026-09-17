import unittest

import numpy as np

from workbench.forcing import corrected_equilibrium, guo_source


def _d3q27():
    velocities = []
    weights = []
    for z in (-1, 0, 1):
        for y in (-1, 0, 1):
            for x in (-1, 0, 1):
                velocities.append((x, y, z))
                shell = abs(x) + abs(y) + abs(z)
                weights.append({0: 8 / 27, 1: 2 / 27, 2: 1 / 54, 3: 1 / 216}[shell])
    return np.asarray(velocities, dtype=np.float32).T, np.asarray(weights, dtype=np.float32)


class GuoForcingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c, cls.w = _d3q27()

    def _equilibrium(self, rho, velocity):
        rho_scalar = rho[0] if rho.ndim == velocity.ndim else rho
        cu = np.einsum("iq,i...->q...", self.c, velocity)
        u2 = np.sum(velocity * velocity, axis=0)
        weights = self.w.reshape((27,) + (1,) * (velocity.ndim - 1))
        return (
            rho_scalar[None, ...]
            * weights
            * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
        ).astype(np.float32)

    def test_source_has_zero_mass_and_rho_times_acceleration_momentum(self):
        rho = np.asarray(1.0 + np.arange(6, dtype=np.float32).reshape(2, 3) * 0.1)[None, ...]
        velocity = np.asarray(
            [
                [[0.03, 0.02, 0.01], [0.00, -0.01, 0.02]],
                [[-0.02, 0.01, 0.03], [0.01, 0.00, -0.02]],
                [[0.01, -0.02, 0.00], [0.02, 0.01, -0.01]],
            ],
            dtype=np.float32,
        )
        acceleration = np.asarray(
            [
                [[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]],
                [[-0.002, -0.001, 0.000], [0.001, 0.002, 0.003]],
                [[0.003, 0.002, 0.001], [0.000, -0.001, -0.002]],
            ],
            dtype=np.float32,
        )

        source = guo_source(
            rho,
            velocity,
            acceleration,
            jnp=np,
            c=self.c,
            w=self.w,
        )

        self.assertEqual(source.shape, (27, 2, 3))
        self.assertEqual(source.dtype, np.float32)
        np.testing.assert_allclose(source.sum(axis=0), 0.0, rtol=0.0, atol=2.0e-7)
        first_moment = np.einsum("iq,q...->i...", self.c, source)
        np.testing.assert_allclose(
            first_moment,
            rho[0] * acceleration,
            rtol=2.0e-6,
            atol=2.0e-7,
        )

    def test_zero_acceleration_is_exactly_zero(self):
        rho = np.ones((1, 4), dtype=np.float32)
        velocity = np.asarray(
            [[0.01, 0.02, 0.03, 0.04], [0.0, 0.01, 0.0, -0.01], [0.02, 0.0, -0.01, 0.01]],
            dtype=np.float32,
        )
        source = guo_source(
            rho,
            velocity,
            np.zeros_like(velocity),
            jnp=np,
            c=self.c,
            w=self.w,
        )
        np.testing.assert_array_equal(source, np.zeros((27, 4), dtype=np.float32))

    def test_half_force_corrected_equilibrium_and_one_step_velocity(self):
        rho = np.ones((1, 5), dtype=np.float32)
        u_raw = np.asarray(
            [[0.020, 0.015, 0.010, 0.005, 0.000], [0.000, 0.005, 0.010, 0.015, 0.020], [0.010] * 5],
            dtype=np.float32,
        )
        acceleration = np.asarray(
            [[0.001] * 5, [-0.0005] * 5, [0.00025] * 5],
            dtype=np.float32,
        )
        u_phys = u_raw + np.float32(0.5) * acceleration
        omega = np.float32(0.8)
        f_raw = self._equilibrium(rho, u_raw)
        f_eq_phys = self._equilibrium(rho, u_phys)
        source = guo_source(
            rho,
            u_phys,
            acceleration,
            jnp=np,
            c=self.c,
            w=self.w,
        )
        f_post = f_raw + omega * (f_eq_phys - f_raw) + (1.0 - omega / 2.0) * source
        velocity_after = np.einsum("iq,q...->i...", self.c, f_post) / rho[0]
        np.testing.assert_allclose(velocity_after, u_raw + acceleration, rtol=2.0e-6, atol=2.0e-7)

        corrected = corrected_equilibrium(
            self._equilibrium,
            rho,
            u_phys,
            acceleration,
            jnp=np,
            c=self.c,
            w=self.w,
        )
        np.testing.assert_allclose(
            corrected,
            f_eq_phys - np.float32(0.5) * source,
            rtol=0.0,
            atol=2.0e-7,
        )
        self.assertEqual(corrected.dtype, np.float32)

    def test_layout_validation_is_explicit(self):
        rho = np.ones((1, 2), dtype=np.float32)
        velocity = np.zeros((3, 2), dtype=np.float32)
        acceleration = np.zeros_like(velocity)
        with self.assertRaisesRegex(ValueError, "c must have shape"):
            guo_source(rho, velocity, acceleration, jnp=np, c=self.c.T, w=self.w)
        with self.assertRaisesRegex(ValueError, "same shape"):
            guo_source(
                rho,
                velocity,
                np.zeros((3, 1), dtype=np.float32),
                jnp=np,
                c=self.c,
                w=self.w,
            )


if __name__ == "__main__":
    unittest.main()
