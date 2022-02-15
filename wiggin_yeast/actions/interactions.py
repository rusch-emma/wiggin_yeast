import logging
from dataclasses import dataclass
from typing import Any

import polychrom
import polychrom.forces
from wiggin.core import SimAction

logging.basicConfig(level=logging.INFO)


@dataclass
class ParticleBond(SimAction):
    bonds: Any = list[(int, int)]
    wiggle_dist: float = 0.05
    bond_length: float = 1.0

    def run_init(self, sim):
        sim.add_force(
            polychrom.forces.harmonic_bonds(
                sim_object=sim,
                bonds=self.bonds,
                bondWiggleDistance=self.wiggle_dist,
                bondLength=self.bond_length,
                name='particle_bonds',
                override_checks=True
            )
        )
