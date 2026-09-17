"""Gravity and reduced-pressure thermal buoyancy in an incompressible LBM."""
from __future__ import annotations

import math
import numpy as np

from .fields_jax import _property_spec
from .schema import normalise_gravity


class GravityModel:
    def __init__(self, physics, material, dt, dx, shape=None):
        self.config = normalise_gravity(physics.get('gravity'))
        self.enabled = bool(physics.get('flow', True) and self.config['enabled'] and any(self.config['vector']))
        self.mode = self.config['mode']
        self.dt, self.dx = float(dt), float(dx)
        self.extent = np.asarray(shape, dtype=float) * self.dx if shape is not None else None
        self.spec = _property_spec(material.get('density'), 1000.0, 'density')
        self.reference_density = float(self.spec.constant if self.spec.kind == 'constant' else
                                       np.interp(self.config['reference_temperature'], self.spec.temperatures, self.spec.values))
        if not math.isfinite(self.reference_density) or self.reference_density <= 0:
            raise ValueError('gravity reference density must be positive and finite')

    def make_operator(self, xp):
        """Dynamic temperature/mask operands; no domain arrays in JIT constants."""
        vector = xp.asarray(self.config['vector'], dtype=xp.float64).reshape(3, 1, 1, 1)
        temperatures = xp.asarray(self.spec.temperatures, dtype=xp.float64) if self.spec.kind == 'table' else None
        values = xp.asarray(self.spec.values, dtype=xp.float64) if self.spec.kind == 'table' else None

        def evaluate(temperature, fluid):
            if self.mode == 'buoyancy':
                density = xp.full_like(temperature, self.spec.constant, dtype=xp.float64) if self.spec.kind == 'constant' else xp.interp(temperature, temperatures, values)
                fraction = density / self.reference_density - 1.0
            else:
                fraction = xp.ones_like(temperature, dtype=xp.float64)
            fraction = xp.where(fluid, fraction, 0.0)
            acceleration = vector * fraction[None, ...] * (self.dt**2 / self.dx)
            return acceleration.astype(xp.float32), xp.max(xp.abs(fraction))

        return evaluate

    def check(self, relative_magnitude):
        relative_magnitude = float(relative_magnitude)
        if not math.isfinite(relative_magnitude):
            raise ValueError('gravity/buoyancy acceleration became non-finite')
        if self.mode == 'buoyancy' and relative_magnitude > 0.1:
            raise ValueError('buoyancy density variation exceeds 10% of the reference density; the incompressible buoyancy approximation is not supported for this temperature range')
        acceleration = float(np.linalg.norm(self.config['vector'])) * relative_magnitude
        if acceleration * self.dt**2 / self.dx > 0.05:
            maximum = math.sqrt(0.05 * self.dx / acceleration)
            raise ValueError(f'gravity acceleration is too large for study.dt; reduce dt to <= {maximum:.9g} s (lattice velocity increment limit 0.05 per step)')
        if self.extent is not None:
            head = float(np.dot(np.abs(self.config['vector']), self.extent)) * relative_magnitude
            if 3.0 * head * (self.dt / self.dx)**2 > 0.1:
                maximum = self.dx * math.sqrt(0.1 / (3.0 * head))
                raise ValueError(f'gravity pressure head is too large for weakly-compressible LBM; reduce dt to <= {maximum:.9g} s (estimated lattice density span limit 10%)')

    def diagnostics(self):
        return dict(self.config, active=self.enabled, reference_density_kg_m3=self.reference_density,
                    force_scheme='Guo D3Q27 with half-step velocity correction' if self.enabled else 'disabled')
