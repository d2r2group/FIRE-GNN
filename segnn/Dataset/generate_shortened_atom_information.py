# This file is for generating atom_init.json as the one from CGCNN is broken
import numpy as np
from pymatgen.core.periodic_table import Element
import math
import json

BLOCK_TO_NUM = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3
}

MAX_ATOMIC_NUM = 100 # No real use of elements above this number as they are usually unstable and radioactive.

def stop_nan(num: float) -> float:
    """
    Turns NaNs into 0.0

    Arg:
        num: The number to stop a NaN value for

    Returns:
        A float that is not NaN
    """
    if math.isnan(num):
        return 0.0
    else:
        return num

def main():
    atom_init = {}
    for z in range(1, MAX_ATOMIC_NUM + 1):
        elem = Element.from_Z(z)
        # Group or "column"
        group = np.zeros(18)
        group[elem.group - 1] = 1
        # Period or "row"
        period = np.zeros(7)
        period[elem.row - 1] = 1
        # s, p, d, f
        block = np.zeros(4)
        block[BLOCK_TO_NUM[elem.block]] = 1
        # Pauling electronegativity
        electronegativity = stop_nan(elem.X)
        # Radius of atom
        radius = stop_nan(elem.van_der_waals_radius)
        # Valence electrons
        if z == 24 or z == 42 or z == 78 or z == 92:
            num_valence_electrons = 6
        elif z == 41:
            num_valence_electrons = 5
        elif z == 44:
            num_valence_electrons = 8
        elif z == 45:
            num_valence_electrons = 9
        elif z == 58:
            num_valence_electrons = 4
        elif z == 64:
            num_valence_electrons = 10
        elif z == 91:
            num_valence_electrons = 2
        elif z == 93:
            num_valence_electrons = 7
        elif z == 96:
            num_valence_electrons = 3
        else:
            num_valence_electrons = elem.valence[1]
        # First ionization energy (only 1)
        ionization_energy_log10 = stop_nan(math.log10(elem.ionization_energy))
        # Electron affinity
        electron_affinity = stop_nan(elem.electron_affinity)
        # Atomic mass
        atomic_mass = stop_nan(math.log10(elem.atomic_mass))
        # Atomic number (one hot)
        atomic_num = np.zeros(MAX_ATOMIC_NUM)
        atomic_num[z - 1] = 1
        # Stack all one hot encodings together
        atom_init[z] = np.hstack([
            atomic_num,
            group, 
            period
        ]).tolist()
    return atom_init

def verify_uniqueness(atom_init):
    """
    Verifiies every atom has their own unique embedding
    in the atom_init
    """
    data = atom_init.values()
    return len(data) == len(np.unique(list(data), axis=0))

def no_nan(atom_init):
    """
    Verifies that there are no NaN values in the atom_init
    """
    data = list(atom_init.values())
    return not np.isnan(np.array(data, dtype=float)).any()

if __name__ == "__main__": 
    atom_init = main()
    assert verify_uniqueness(atom_init)
    assert no_nan(atom_init)
    with open("short_atom_init.json", "wt+") as f:
        json.dump(atom_init, f)