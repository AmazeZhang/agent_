"""Focused offline checks derived from the SWE-bench pydicom-1458 test patch."""

from __future__ import annotations

import numpy as np

from pydicom.dataset import Dataset
from pydicom.pixel_data_handlers.numpy_handler import get_pixeldata
from pydicom.uid import ExplicitVRLittleEndian


def base_dataset(bits: int, samples: int = 1) -> Dataset:
    dataset = Dataset()
    dataset.file_meta = Dataset()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.BitsAllocated = bits
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.SamplesPerPixel = samples
    dataset.PhotometricInterpretation = "MONOCHROME2" if samples == 1 else "RGB"
    return dataset


def expect_missing(dataset: Dataset, keyword: str) -> None:
    try:
        get_pixeldata(dataset)
    except AttributeError as exc:
        if keyword not in str(exc):
            raise AssertionError(f"error did not mention {keyword}: {exc}") from exc
    else:
        raise AssertionError(f"missing {keyword} did not raise AttributeError")


def main() -> None:
    float_dataset = base_dataset(32)
    float_dataset.FloatPixelData = np.arange(4, dtype="<f4").tobytes()
    assert get_pixeldata(float_dataset).dtype == np.dtype("float32")

    double_dataset = base_dataset(64)
    double_dataset.DoubleFloatPixelData = np.arange(4, dtype="<f8").tobytes()
    assert get_pixeldata(double_dataset).dtype == np.dtype("float64")

    missing_bits_stored = base_dataset(16)
    missing_bits_stored.PixelRepresentation = 0
    missing_bits_stored.PixelData = np.arange(4, dtype="<u2").tobytes()
    expect_missing(missing_bits_stored, "BitsStored")

    missing_planar_configuration = base_dataset(8, samples=3)
    missing_planar_configuration.BitsStored = 8
    missing_planar_configuration.PixelRepresentation = 0
    missing_planar_configuration.PixelData = bytes(range(12))
    expect_missing(missing_planar_configuration, "PlanarConfiguration")

    print("4 focused checks passed")


if __name__ == "__main__":
    main()
