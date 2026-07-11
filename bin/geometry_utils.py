"""
Shared geometry helpers for the SORAT pipeline.

Provides a single source of truth for safely resampling an intensity image into
a segmentation mask's reference grid.  This is robust to the coordinate-space
mismatches that arise when a segmentation model writes output with a different
physical origin / direction than the original image (e.g. VSA-3L writes
origin 0 + identity direction; M&Ms images have a non-zero origin and oblique
acquisition direction).

The canonical hazard: a physical-space SimpleITK resample of an oblique /
offset image into an identity-geometry mask grid samples every voxel outside
the image field, yielding all-zero intensities inside the mask.  This silently
zeroes out every intensity-dependent feature (PyRadiomics firstorder / GLCM)
and breaks intensity-aware postprocessing (LV -> MYO relabel).

The safe helper below mirrors the proven pattern already used in
``generate_segmentation_preview.py``: when the image and mask share the same
voxel-grid size, their index spaces coincide (true for every current model --
VSA-3L resizes slices back to the original shape; nnFormer and CineMA write at
the original size), so the image array is returned directly, wrapped in the
mask's geometry so downstream physical-space-aware code (PyRadiomics) sees a
consistent image / mask pair.  When sizes genuinely differ (cropped models),
a real physical-space resample is performed.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk


def _spatial_direction_3d(direction: Tuple[float, ...]) -> Tuple[float, ...]:
    """Extract a flat 9-element 3x3 spatial direction from 3D or 4D cosines.

    For a 4D image SimpleITK stores a 4x4 (16-element) direction matrix; the
    spatial 3x3 block is rows / columns 0..2 (indices 0,1,2,4,5,6,8,9,10).
    Returns the identity direction when the matrix is degenerate.
    """
    if len(direction) == 9:
        return tuple(float(x) for x in direction)

    dim = int(round(len(direction) ** 0.5))
    if dim * dim != len(direction) or dim < 3:
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    try:
        import numpy as np

        mat = np.asarray(direction, dtype=np.float64).reshape(dim, dim)
        spatial = mat[:3, :3]
        if np.linalg.matrix_rank(spatial) < 3:
            return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        return tuple(float(x) for x in spatial.reshape(-1))
    except Exception:
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _orthonormalize_direction_3d(direction: np.ndarray) -> Tuple[float, ...]:
    """Return the closest orthonormal 3D direction matrix in row-major order."""
    try:
        u, _singular_values, vt = np.linalg.svd(direction)
        orthonormal = u @ vt
        if abs(float(np.linalg.det(orthonormal))) < 1e-8:
            raise ValueError("degenerate direction matrix")
        return tuple(float(x) for x in orthonormal.reshape(-1))
    except Exception:
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def nifti_affine_lps_geometry(nib_image) -> dict:
    """Derive ITK-compatible LPS geometry from a nibabel NIfTI image.

    nibabel exposes the NIfTI affine in RAS coordinates.  SimpleITK/ITK use
    LPS coordinates and require orthonormal direction cosines.  Some M&Ms-2
    files provide only a sheared sform, which nibabel can read but ITK rejects.
    For those files we preserve origin and spacing and use the closest
    orthonormal direction matrix representable by ITK.
    """
    affine = np.asarray(nib_image.affine, dtype=np.float64)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    linear_lps = ras_to_lps @ affine[:3, :3]

    spacing = np.asarray(nib_image.header.get_zooms()[:3], dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        spacing = np.linalg.norm(linear_lps, axis=0)
    spacing = np.where(spacing > 0, spacing, 1.0)

    direction = linear_lps / spacing[np.newaxis, :]
    origin = ras_to_lps @ affine[:3, 3]

    return {
        "size": [int(x) for x in nib_image.shape[:3]],
        "spacing": [float(x) for x in spacing],
        "origin": [float(x) for x in origin],
        "direction": list(_orthonormalize_direction_3d(direction)),
    }


def _direction_4d_from_3d(direction_3d: Tuple[float, ...]) -> Tuple[float, ...]:
    direction = np.eye(4, dtype=np.float64)
    direction[:3, :3] = np.asarray(direction_3d, dtype=np.float64).reshape(3, 3)
    return tuple(float(x) for x in direction.reshape(-1))


def nibabel_to_sitk_image(nib_image, dtype=None) -> sitk.Image:
    """Build a SimpleITK image from nibabel data and ITK-compatible geometry."""
    data = np.asanyarray(nib_image.dataobj)
    if dtype is not None:
        data = data.astype(dtype)

    if data.ndim == 4:
        image = sitk.GetImageFromArray(np.transpose(data, (3, 2, 1, 0)), isVector=False)
    elif data.ndim == 3:
        image = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)), isVector=False)
    else:
        raise ValueError(f"Expected 3D or 4D NIfTI image, got shape {data.shape}")

    geometry = nifti_affine_lps_geometry(nib_image)
    spatial_spacing = geometry["spacing"]
    spatial_origin = geometry["origin"]
    spatial_direction = tuple(geometry["direction"])

    if data.ndim == 4:
        zooms = nib_image.header.get_zooms()
        time_spacing = float(zooms[3]) if len(zooms) > 3 and zooms[3] > 0 else 1.0
        image.SetSpacing(tuple(spatial_spacing) + (time_spacing,))
        image.SetOrigin(tuple(spatial_origin) + (0.0,))
        image.SetDirection(_direction_4d_from_3d(spatial_direction))
    else:
        image.SetSpacing(tuple(spatial_spacing))
        image.SetOrigin(tuple(spatial_origin))
        image.SetDirection(spatial_direction)

    return image


def read_nifti_with_sitk_fallback(path, dtype=None) -> sitk.Image:
    """Read a NIfTI with SimpleITK, falling back for non-orthonormal sforms."""
    try:
        return sitk.ReadImage(str(path))
    except RuntimeError as exc:
        if "orthonormal direction cosines" not in str(exc):
            raise

        import nibabel as nib

        print(
            "WARNING: SimpleITK rejected non-orthonormal NIfTI geometry for "
            f"{Path(path)}; using nibabel fallback with orthonormalized direction.",
            file=sys.stderr,
        )
        return nibabel_to_sitk_image(nib.load(str(path)), dtype=dtype)


def resample_image_to_reference_safe(
    image: sitk.Image,
    reference: sitk.Image,
    interpolator=sitk.sitkLinear,
    default_pixel_value: float = 0.0,
) -> sitk.Image:
    """Resample ``image`` into the ``reference`` grid, robust to geometry gaps.

    When ``image`` and ``reference`` have the **same voxel-grid size**, their
    index spaces coincide (every current SORAT model writes segmentations at
    the original image size, possibly with different origin / direction
    metadata).  In that case a physical-space resample can map the image
    entirely outside the mask grid (e.g. M&Ms oblique images vs an
    identity-direction segmentation), producing all-zero intensities inside
    the mask.  To avoid that, the image array is returned directly, wrapped in
    the reference's geometry so downstream physical-space-aware code
    (PyRadiomics) sees a consistent image / mask pair.

    When the sizes **differ** (genuinely different grids, e.g. a cropped
    model), a real physical-space resample is performed -- this is the
    correct behaviour for regridding an image into a cropped mask grid.

    Parameters
    ----------
    image : sitk.Image
        Intensity image to resample.
    reference : sitk.Image
        Image defining the target grid (typically a segmentation mask).
    interpolator : int
        SimpleITK interpolator constant (default linear, suitable for
        intensities; callers resampling labels should pass sitkNearestNeighbor).
    default_pixel_value : float
        Fill value for voxels outside the image field during a real resample.

    Returns
    -------
    sitk.Image
        Image aligned with the reference grid.
    """
    if image.GetSize() == reference.GetSize():
        out = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))
        out.CopyInformation(reference)
        return out

    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(reference)
    rs.SetInterpolator(interpolator)
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(default_pixel_value)
    return rs.Execute(image)


def resample_label_to_reference_safe(
    label: sitk.Image,
    reference: sitk.Image,
) -> sitk.Image:
    """Resample a label image into the reference grid using nearest-neighbor.

    Same geometry-robustness guarantee as
    :func:`resample_image_to_reference_safe` but with nearest-neighbor
    interpolation so label values are preserved.
    """
    return resample_image_to_reference_safe(
        label, reference, interpolator=sitk.sitkNearestNeighbor, default_pixel_value=0
    )
