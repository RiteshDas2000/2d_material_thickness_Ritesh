"""
optics.py
============
Physics part of the app.

Computes the optical reflectance of an N-layer 2D material.

Used in dataset.py

Stack (top -> bottom):
    air (n=1)  /  2D material (thickness = N * d_layer)  /  SiO2 (285 nm)  /  Si

References
----------
- Transfer-matrix method: Born & Wolf, *Principles of Optics*.
- Analytic CIE 1931 CMF fit: Wyman, Sloan, Shirley,
  "Simple Analytic Approximations to the CIE XYZ Colour Matching Functions",
  Journal of Computer Graphics Techniques (2013).
- 285 nm SiO2 is the standard thickness that maximises single-layer graphene
  visibility (Blake et al., APL 2007).
"""

from __future__ import annotations
import numpy as np

# numpy>=2.0 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# --------------------------------------------------------------------------- #
#  CIE 1931 colour-matching functions (analytic multi-Gaussian approximation)  #
# --------------------------------------------------------------------------- #
def _piecewise_gauss(x, mu, sigma_lo, sigma_hi):
    """Gaussian with different width on each side of the peak (Wyman et al.)."""
    sigma = np.where(x < mu, sigma_lo, sigma_hi)
    t = (x - mu) * sigma
    return np.exp(-0.5 * t * t)


def cie_xyz_bar(lam_nm):
    """Return (x_bar, y_bar, z_bar) at the given wavelengths (nm)."""
    g = _piecewise_gauss
    x = (1.056 * g(lam_nm, 599.8, 0.0264, 0.0323)
         + 0.362 * g(lam_nm, 442.0, 0.0624, 0.0374)
         - 0.065 * g(lam_nm, 501.1, 0.0490, 0.0382))
    y = (0.821 * g(lam_nm, 568.8, 0.0213, 0.0247)
         + 0.286 * g(lam_nm, 530.9, 0.0613, 0.0322))
    z = (1.217 * g(lam_nm, 437.0, 0.0845, 0.0278)
         + 0.681 * g(lam_nm, 459.0, 0.0385, 0.0725))
    return x, y, z


# --------------------------------------------------------------------------- #
#  Refractive indices                                                          #
# --------------------------------------------------------------------------- #
# Silicon complex index across the visible (n - i k), interpolated from anchors.
_SI_LAM = np.array([400., 450., 500., 550., 600., 650., 700.])
_SI_N   = np.array([5.57, 4.68, 4.30, 4.08, 3.94, 3.85, 3.78])
_SI_K   = np.array([0.39, 0.15, 0.073, 0.031, 0.023, 0.016, 0.012])


def silicon_index(lam_nm):
    n = np.interp(lam_nm, _SI_LAM, _SI_N)
    k = np.interp(lam_nm, _SI_LAM, _SI_K)
    return n - 1j * k


# Preset optical constants (complex index, per-layer thickness in nm).
# Values are standard first-order approximations used for optical-contrast work.
MATERIALS = {
    "graphene": {"index": 2.6 - 1.3j, "d_layer": 0.335},
    "hBN":      {"index": 2.2 - 0.0j, "d_layer": 0.333},
    "MoS2":     {"index": 5.0 - 1.0j, "d_layer": 0.650},
    "WSe2":     {"index": 4.6 - 1.1j, "d_layer": 0.650},
}


# --------------------------------------------------------------------------- #
#  Transfer-matrix reflectance                                                 #
# --------------------------------------------------------------------------- #
def _characteristic_matrix(n, thickness_nm, lam_nm):
    """2x2 optical characteristic matrix of a single homogeneous layer."""
    delta = 2.0 * np.pi * n * thickness_nm / lam_nm
    c, s = np.cos(delta), np.sin(delta)
    return np.array([[c, 1j * s / n],
                     [1j * n * s, c]], dtype=complex)


def reflectance_single(lam_nm, n_layers, mat_index, sio2_nm, d_layer):
    """Reflectance R(lambda) for one wavelength and an integer layer count."""
    n_air = 1.0 + 0j
    n_sub = silicon_index(lam_nm)

    M = np.eye(2, dtype=complex)
    if n_layers > 0:
        M = M @ _characteristic_matrix(mat_index, n_layers * d_layer, lam_nm)
    M = M @ _characteristic_matrix(1.46 + 0j, sio2_nm, lam_nm)  # SiO2

    B = M[0, 0] + M[0, 1] * n_sub
    C = M[1, 0] + M[1, 1] * n_sub
    r = (n_air * B - C) / (n_air * B + C)
    return float(np.abs(r) ** 2)


def layer_reflectance_spectrum(n_layers, material="graphene",
                               sio2_nm=285.0, lam_nm=None):
    """Full reflectance spectrum R(lambda) for an N-layer flake."""
    if lam_nm is None:
        lam_nm = np.arange(380.0, 721.0, 5.0)
    m = MATERIALS[material]
    R = np.array([reflectance_single(l, n_layers, m["index"], sio2_nm, m["d_layer"])
                  for l in lam_nm])
    return lam_nm, R


def _xyz_to_srgb(X, Y, Z):
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [0.0557, -0.2040,  1.0570]])
    rgb = M @ np.array([X, Y, Z])
    rgb = np.clip(rgb, 0.0, None)
    # sRGB gamma
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0.0, 1.0)


def layer_color(n_layers, material="graphene", sio2_nm=285.0):
    """sRGB colour (3 floats in [0,1]) of an N-layer flake under equal-energy light."""
    lam, R = layer_reflectance_spectrum(n_layers, material, sio2_nm)
    xb, yb, zb = cie_xyz_bar(lam)
    X = _trapz(R * xb, lam)
    Y = _trapz(R * yb, lam)
    Z = _trapz(R * zb, lam)
    norm = _trapz(yb, lam)            # normalise so a perfect reflector -> white
    return _xyz_to_srgb(X / norm, Y / norm, Z / norm)


def color_table(max_layers, material="graphene", sio2_nm=285.0):
    """Array of shape (max_layers+1, 3): colour of substrate (0) and 1..N layers."""
    return np.stack([layer_color(n, material, sio2_nm)
                     for n in range(max_layers + 1)], axis=0)


def optical_contrast(n_layers, material="graphene", sio2_nm=285.0):
    """
    Green-channel optical contrast C = (R_sub - R_flake) / R_sub, a standard
    experimental figure of merit for how visible an N-layer flake is.
    """
    lam, R_f = layer_reflectance_spectrum(n_layers, material, sio2_nm)
    _,   R_s = layer_reflectance_spectrum(0, material, sio2_nm)
    g = (lam > 520) & (lam < 560)      # ~green band
    return float((R_s[g].mean() - R_f[g].mean()) / R_s[g].mean())


if __name__ == "__main__":
    # Quick sanity print: colours and contrasts should change monotonically-ish.
    for mat in ("graphene", "MoS2"):
        print(f"\n{mat} on 285 nm SiO2/Si")
        for n in range(0, 5):
            rgb = layer_color(n, mat)
            c = optical_contrast(n, mat) if n else 0.0
            print(f"  N={n}: RGB=({rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f})  "
                  f"contrast={c:+.3f}  thickness={n*MATERIALS[mat]['d_layer']:.3f} nm")
