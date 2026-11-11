"""Images, sounds and videos that stimuli can carry, and helpers for making them."""

from __future__ import annotations

import hashlib
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import torch

MEDIA_KINDS = ("image", "video", "audio")

# the reference white of sRGB, D65, and the matrix from CIE XYZ to linear sRGB
_WHITE = np.array([0.95047, 1.0, 1.08883])
_XYZ_TO_RGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570]
])
# CIELAB's inverse companding switches from a cube to a line below this value
_LAB_EPSILON = 6 / 29


class Media:
    """Something a subject sees or hears, loaded only when a subject needs it.

    A medium is built from a file path or from data already in memory. A path is
    read when a subject first needs the content, so a task can list thousands of
    files without loading them. Every medium has a digest of its content, which is
    what results and run keys record in place of the content itself.

    Attributes:
        kind: `"image"`, `"video"` or `"audio"`.
        source: The path or in-memory data the medium was built from.
    """

    kind: str = ""

    def __init__(self, source: Any) -> None:
        if isinstance(source, Media):
            raise TypeError(f"{type(source).__name__} is already a medium. Pass it directly.")
        self.source = Path(source) if isinstance(source, str) else source
        if isinstance(self.source, Path) and not self.source.exists():
            raise FileNotFoundError(f"No {self.kind} file at {self.source}.")

    @cached_property
    def digest(self) -> str:
        """A SHA-256 hash of the content, which identifies the medium in results."""
        hasher = hashlib.sha256(self.kind.encode())
        if isinstance(self.source, Path):
            hasher.update(self.source.read_bytes())
        else:
            for part in self._content_parts():
                hasher.update(part)
        return hasher.hexdigest()

    def describe(self) -> str:
        """Names the medium briefly, for previews and error messages.

        Returns:
            The kind and the start of the digest, as in `<image 3f2a9c1b>`.
        """
        return f"<{self.kind} {self.digest[:8]}>"

    def _content_parts(self) -> list[bytes]:
        array = _as_array(self.source)
        return [str(array.dtype).encode(), str(array.shape).encode(), np.ascontiguousarray(array).tobytes()]

    def __repr__(self) -> str:
        source = str(self.source) if isinstance(self.source, Path) else type(self.source).__name__
        return f"{type(self).__name__}({source}, digest={self.digest[:8]})"


class Image(Media):
    """A still image.

    Args:
        source: A path to an image file, a PIL image, an array of shape
            (height, width) or (height, width, channels), or a tensor of shape
            (channels, height, width), as torchvision gives. Arrays of floats are
            read as values from 0 to 1, and arrays of integers as values from 0 to 255.

    Raises:
        FileNotFoundError: If `source` is a path that doesn't exist.
        ValueError: If an array doesn't have the shape of an image.
    """

    kind = "image"

    def __init__(self, source: Any) -> None:
        super().__init__(source)
        if not isinstance(self.source, Path) and not _is_pil(self.source):
            _image_array(self.source)

    def load(self) -> Any:
        """Reads the image as an RGB PIL image.

        Returns:
            The image.

        Raises:
            ImportError: If Pillow isn't installed, which `pip install cognarium[vision]`
                provides.
        """
        pil = _import_pil()
        if isinstance(self.source, Path):
            with pil.open(self.source) as opened:
                return opened.convert("RGB")
        if _is_pil(self.source):
            return self.source.convert("RGB")
        return pil.fromarray(_image_array(self.source))

    def _content_parts(self) -> list[bytes]:
        if _is_pil(self.source):
            return [self.source.mode.encode(), str(self.source.size).encode(), self.source.tobytes()]
        return super()._content_parts()


class Audio(Media):
    """A sound, kept as one mono waveform.

    Args:
        source: A path to an audio file, or a waveform of shape (samples,) or
            (channels, samples), which is averaged to mono.
        sampling_rate: The samples per second of an in-memory waveform. A file
            states its own.

    Raises:
        FileNotFoundError: If `source` is a path that doesn't exist.
        ValueError: If an in-memory waveform comes without a sampling rate, or has
            more than two dimensions.
    """

    kind = "audio"

    def __init__(self, source: Any, sampling_rate: int | None = None) -> None:
        super().__init__(source)
        if not isinstance(self.source, Path):
            if sampling_rate is None:
                raise ValueError("An in-memory waveform needs its sampling_rate.")
            if _as_array(self.source).ndim not in (1, 2):
                raise ValueError(
                    f"A waveform has shape (samples,) or (channels, samples), got {_as_array(self.source).shape}."
                )
        self.sampling_rate = sampling_rate

    def load(self, sampling_rate: int | None = None) -> np.ndarray:
        """Reads the waveform, resampled if a rate is asked for.

        Args:
            sampling_rate: The rate a subject expects. None keeps the original.

        Returns:
            A mono float32 waveform of shape (samples,).

        Raises:
            ImportError: If reading a file or resampling needs a package that isn't
                installed, which `pip install cognarium[audio]` provides.
        """
        if isinstance(self.source, Path):
            try:
                import soundfile
            except ImportError as error:
                raise ImportError("Reading audio files needs soundfile: pip install cognarium[audio].") from error
            waveform, rate = soundfile.read(self.source, dtype="float32", always_2d=True)
            waveform = waveform.T
        else:
            waveform, rate = _as_array(self.source).astype(np.float32), self.sampling_rate
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        if sampling_rate is None or sampling_rate == rate:
            return waveform
        try:
            from scipy.signal import resample_poly
        except ImportError as error:
            raise ImportError("Resampling audio needs scipy: pip install cognarium[audio].") from error
        divisor = np.gcd(int(sampling_rate), int(rate))
        return resample_poly(waveform, sampling_rate // divisor, rate // divisor).astype(np.float32)

    def _content_parts(self) -> list[bytes]:
        return [*super()._content_parts(), str(self.sampling_rate).encode()]


class Video(Media):
    """A sequence of frames.

    Args:
        source: A path to a video file, an array of shape (frames, height, width,
            channels), or a list of images in any form `Image` takes.
        fps: The frames per second of in-memory frames, which some models use to
            place frames in time. A file states its own.

    Raises:
        FileNotFoundError: If `source` is a path that doesn't exist.
        ValueError: If in-memory frames don't have the shape of a video.
    """

    kind = "video"

    def __init__(self, source: Any, fps: float | None = None) -> None:
        if isinstance(source, (list, tuple)):
            source = np.stack([_image_array(frame) for frame in source])
        super().__init__(source)
        if not isinstance(self.source, Path):
            frames = _as_array(self.source)
            if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
                raise ValueError(f"Frames need shape (frames, height, width, channels), got {frames.shape}.")
        self.fps = fps

    def load(self) -> tuple[np.ndarray, float | None]:
        """Reads every frame.

        Returns:
            The frames as uint8 RGB, shape (frames, height, width, 3), and the frames
            per second if known.

        Raises:
            ImportError: If reading a file needs PyAV and it isn't installed, which
                `pip install cognarium[video]` provides.
        """
        if not isinstance(self.source, Path):
            return np.stack([_image_array(frame) for frame in _as_array(self.source)]), self.fps
        try:
            import av
        except ImportError as error:
            raise ImportError("Reading video files needs PyAV: pip install cognarium[video].") from error
        with av.open(str(self.source)) as container:
            stream = container.streams.video[0]
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
            fps = float(stream.average_rate) if stream.average_rate else None
        return np.stack(frames), fps

    def _content_parts(self) -> list[bytes]:
        return [*super()._content_parts(), str(self.fps).encode()]


def as_media(kind: str, value: Any) -> Media:
    """Turns what a stimulus field was given into a medium of that kind.

    Args:
        kind: `"image"`, `"video"` or `"audio"`.
        value: A medium, a path, or in-memory data. Audio in memory comes as a
            `(waveform, sampling_rate)` pair.

    Returns:
        The medium.

    Raises:
        TypeError: If `value` is a medium of another kind, or audio in memory comes
            without its sampling rate.
    """
    if isinstance(value, Media):
        if value.kind != kind:
            raise TypeError(f"The {kind} field got {value.describe()}.")
        return value
    if kind == "image":
        return Image(value)
    if kind == "video":
        return Video(value)
    if isinstance(value, (str, Path)):
        return Audio(value)
    if isinstance(value, tuple) and len(value) == 2:
        return Audio(*value)
    raise TypeError("Audio in memory needs its rate: pass cg.media.Audio(waveform, sampling_rate) or (waveform, rate).")


def lab_to_srgb(lab: Any) -> np.ndarray:
    """Converts CIELAB colours to sRGB under the D65 white point.

    Colours outside the sRGB gamut are clipped to it.

    Args:
        lab: Colours of shape (..., 3), as L*, a* and b*.

    Returns:
        The sRGB values from 0 to 1, with the same shape.
    """
    lab = np.asarray(lab, dtype=float)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(f > _LAB_EPSILON, f**3, 3 * _LAB_EPSILON**2 * (f - 4 / 29)) * _WHITE
    linear = xyz @ _XYZ_TO_RGB.T
    rgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.abs(linear) ** (1 / 2.4) - 0.055)
    return rgb.clip(0, 1)


def swatch(rgb: Any, size: int = 32) -> Image:
    """Makes a square of one flat colour, such as a colour chip.

    Args:
        rgb: The colour as red, green and blue values from 0 to 1.
        size: The side of the square in pixels. Many vision models resize small
            images up to a minimum, so a smaller square mainly saves memory.

    Returns:
        The swatch.

    Raises:
        ValueError: If `rgb` isn't three values from 0 to 1, or `size` is below 1.
    """
    values = np.asarray(rgb, dtype=float)
    if values.shape != (3,) or values.min() < 0 or values.max() > 1:
        raise ValueError(f"A swatch needs three values from 0 to 1, got {values.tolist()}.")
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}.")
    pixel = np.round(values * 255).astype(np.uint8)
    return Image(np.broadcast_to(pixel, (size, size, 3)).copy())


def _as_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_array(value: Any) -> np.ndarray:
    """Gives an image as uint8 RGB of shape (height, width, 3)."""
    if _is_pil(value):
        return np.asarray(value.convert("RGB"))
    array = _as_array(value)
    if isinstance(value, torch.Tensor) and array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = array.transpose(1, 2, 0)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"An image needs shape (height, width) or (height, width, channels), got {array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        array = np.round(np.clip(array, 0, 1) * 255)
    array = array.astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array[..., :3]


def _is_pil(value: Any) -> bool:
    return type(value).__module__.startswith("PIL.")


def _import_pil() -> Any:
    try:
        from PIL import Image as pil
    except ImportError as error:
        raise ImportError("Images need Pillow: pip install cognarium[vision].") from error
    return pil
