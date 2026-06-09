"""
naming.py  -  Auto-generate output filenames from simulation parameters.

Encoding rules (consistent with simulation.py / config.py integer command format):
    Three-digit integer +/-XYZ:
        hundreds digit X (1-8): initial phase -> [pi/4, pi/2, pi*3/4, pi,
                                                   pi*5/4, pi*3/2, pi*7/4, 2*pi]
        tens digit     Y (1-6): amplitude     -> [pi/12, pi/6, pi/4, pi/3, pi*5/12, pi/2]
        units digit    Z (1-9): frequency     -> [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5] Hz
        0 in any digit means "not matched in table" or "inconsistent across robots"

Public exports (importable by sweep.py and other modules):
    _INITIAL_DICT : initial phase list (rad), length 8
    _AMPLI_DICT   : amplitude list (rad), length 6
    _FREQ_DICT    : frequency list (Hz), length 9
"""

import math

# =============================================================================
# Lookup tables  (1-based: list index 0 corresponds to slot 1)
# =============================================================================

_INITIAL_DICT = [
    math.pi/4, math.pi/2, math.pi*3/4, math.pi,
    math.pi*5/4, math.pi*3/2, math.pi*7/4, 2*math.pi,
]  # slots 1-8

_AMPLI_DICT = [
    math.pi/12, math.pi/6, math.pi/4,
    math.pi/3,  math.pi*5/12, math.pi/2,
]  # slots 1-6

_FREQ_DICT = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]  # slots 1-9

_TOL = 1e-4   # tolerance for floating-point comparisons


# =============================================================================
# Internal helpers
# =============================================================================

def _match(value: float, table: list) -> int:
    """Return the 1-based index of the closest matching entry in table, or 0 if not found."""
    for idx, v in enumerate(table):
        if abs(value - v) < _TOL:
            return idx + 1
    return 0


def _encode_phase(phase_rad: float) -> int:
    """Encode a phase value (rad) as a 1-8 slot index; return 0 if not matched."""
    return _match(phase_rad, _INITIAL_DICT)


def _encode_omega(omega_rad_per_s: float) -> int:
    """Convert angular frequency (rad/s) to Hz and look up _FREQ_DICT; return 0 if not matched."""
    freq = omega_rad_per_s / (2 * math.pi)
    return _match(freq, _FREQ_DICT)


def _encode_ampli(ampli_deg: float) -> int:
    """Convert amplitude (degrees) to radians and look up _AMPLI_DICT; return 0 if not matched."""
    return _match(math.radians(ampli_deg), _AMPLI_DICT)


def _all_same(values: list) -> bool:
    """Return True if all values in the list are equal within floating-point tolerance."""
    if not values:
        return True
    return all(abs(v - values[0]) < _TOL for v in values)


# =============================================================================
# Public interface
# =============================================================================

def generate_trial_name(
    n_robots: int,
    init_phases: list,   # [(j1_phase, j2_phase), ...], length = n_robots, in radians
    omega: tuple,        # (omega1, omega2) in rad/s, applied globally
    amplitude: tuple,    # (A_deg1, A_deg2) in degrees, applied globally
    prefix: str = "trial",
) -> str:
    """
    Generate a filename string from robot count, initial phases, frequency, and amplitude.

    Parameters
    ----------
    n_robots    : number of robots
    init_phases : list of (joint1_phase, joint2_phase) tuples per robot, in radians
    omega       : (omega1, omega2) angular frequency for both joints, in rad/s
    amplitude   : (A_deg1, A_deg2) amplitude for both joints, in degrees
    prefix      : filename prefix, default 'trial'

    Returns
    -------
    Naming string, e.g. 'trial_N17_J1p5_J2p5_W1f2_W2f2_A1a6_A2a6'
        J1p5  -> joint1 initial phase slot 5 (pi*5/4)
        W1f2  -> joint1 frequency slot 2 (1.0 Hz)
        A1a6  -> joint1 amplitude slot 6 (pi/2 = 90 deg)
        *0    -> not matched in table (inconsistent across robots, or random phase)

    Examples
    --------
    >>> init_phases = [(math.pi*5/4, math.pi*5/4)] * 17
    >>> omega       = (2*math.pi, 2*math.pi)    # 1 Hz
    >>> amplitude   = (90.0, 90.0)              # pi/2
    >>> generate_trial_name(17, init_phases, omega, amplitude)
    'trial_N17_J1p5_J2p5_W1f2_W2f2_A1a6_A2a6'
    """
    assert len(init_phases) == n_robots, (
        f"init_phases length {len(init_phases)} does not match n_robots {n_robots}")

    j1_phases = [ph[0] for ph in init_phases]
    j2_phases = [ph[1] for ph in init_phases]

    # Encode as 0 when phases differ across robots (inconsistent -> not representable)
    j1_code = _encode_phase(j1_phases[0]) if _all_same(j1_phases) else 0
    j2_code = _encode_phase(j2_phases[0]) if _all_same(j2_phases) else 0

    w1_code = _encode_omega(omega[0])
    w2_code = _encode_omega(omega[1])

    a1_code = _encode_ampli(amplitude[0])
    a2_code = _encode_ampli(amplitude[1])

    return (f"{prefix}_N{n_robots}"
            f"_J1p{j1_code}_J2p{j2_code}"
            f"_W1f{w1_code}_W2f{w2_code}"
            f"_A1a{a1_code}_A2a{a2_code}")


# =============================================================================
# Quick self-test
# =============================================================================

if __name__ == "__main__":
    N = 17

    phases_uniform = [(math.pi*5/4, math.pi*5/4) for _ in range(N)]
    phases_mixed   = [(math.pi/2 if i % 2 == 0 else math.pi, math.pi) for i in range(N)]

    omega_same = (2 * math.pi, 2 * math.pi)   # both joints at 1 Hz
    omega_diff = (2 * math.pi, 4 * math.pi)   # joint1=1 Hz, joint2=2 Hz
    ampli_same = (90.0, 90.0)                  # both joints at pi/2 (slot 6)
    ampli_diff = (30.0, 60.0)                  # joint1=pi/6 (slot 2), joint2=pi/3 (slot 4)

    print(generate_trial_name(N, phases_uniform, omega_same, ampli_same))
    # -> trial_N17_J1p5_J2p5_W1f2_W2f2_A1a6_A2a6

    print(generate_trial_name(N, phases_mixed, omega_diff, ampli_diff))
    # -> trial_N17_J1p0_J2p4_W1f2_W2f4_A1a2_A2a4

    print(generate_trial_name(N, phases_uniform, omega_same, ampli_same, prefix="sim"))
    # -> sim_N17_J1p5_J2p5_W1f2_W2f2_A1a6_A2a6
