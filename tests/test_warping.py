"""
Integration test for map warping functionality.

This test verifies that the warping function correctly transforms radar data
from Transverse Mercator to geographic coordinates using config parameters.
"""

import numpy as np

from nwc_webapp.geo.warping import warp_map


def test_warp_map_basic():
    """Test basic warping functionality."""
    # Create test data (1400x1200 array)
    nlines, ncols = 1400, 1200
    test_data = np.random.rand(nlines, ncols).astype(np.float32) * 100

    # Apply warping
    warped = warp_map(test_data)

    # Verify output shape matches expected destination grid
    assert warped.shape == (1400, 1200), f"Expected shape (1400, 1200), got {warped.shape}"

    # Verify data type is preserved
    assert warped.dtype == test_data.dtype, f"Expected dtype {test_data.dtype}, got {warped.dtype}"

    # Verify there's valid data (not all NaN)
    valid_pixels = np.count_nonzero(~np.isnan(warped))
    assert valid_pixels > 1000000, f"Expected >1M valid pixels, got {valid_pixels}"

    # Verify data is in reasonable range
    valid_data = warped[~np.isnan(warped)]
    assert valid_data.min() >= 0, f"Min value should be >= 0, got {valid_data.min()}"
    assert valid_data.max() <= 100, f"Max value should be <= 100, got {valid_data.max()}"

    print(f"✓ Warping test passed:")
    print(f"  Output shape: {warped.shape}")
    print(f"  Valid pixels: {valid_pixels} / {warped.size}")
    print(f"  Value range: [{valid_data.min():.2f}, {valid_data.max():.2f}]")


def test_warp_map_with_pattern():
    """Test warping with a known pattern."""
    # Create test data with a simple gradient
    nlines, ncols = 1400, 1200
    y_coords, x_coords = np.ogrid[0:nlines, 0:ncols]
    test_data = (x_coords / ncols * 50 + y_coords / nlines * 50).astype(np.float32)

    # Apply warping
    warped = warp_map(test_data)

    # Verify output characteristics
    assert warped.shape == (1400, 1200)
    valid_pixels = np.count_nonzero(~np.isnan(warped))
    assert valid_pixels > 1000000

    # Verify the warped data has similar statistical properties
    valid_data = warped[~np.isnan(warped)]
    assert 0 <= valid_data.mean() <= 100

    print(f"✓ Pattern warping test passed:")
    print(f"  Mean value: {valid_data.mean():.2f}")
    print(f"  Std dev: {valid_data.std():.2f}")


if __name__ == "__main__":
    test_warp_map_basic()
    test_warp_map_with_pattern()
    print("\n✓ All warping tests passed!")
