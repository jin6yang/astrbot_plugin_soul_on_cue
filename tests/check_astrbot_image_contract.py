"""Check the public AstrBot image preparation entry, with Pillow installed.

Only media resolution is stubbed to local files. The input limit, actual image
encoding and preview-file publication execute the supplied AstrBot source.
"""

import argparse
import ast
import asyncio
import base64
import errno
import io
import logging
import math
import os
import struct
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageOps


class LocalMediaResolver:
    """A local-only resolver; the real public image policy remains under test."""

    def __init__(self, image_ref, *, media_type):
        assert media_type == "image"
        self.path = Path(image_ref)

    @asynccontextmanager
    async def as_path(self):
        yield SimpleNamespace(path=self.path, read_bytes=self.path.read_bytes, detach=lambda: None)


def load_helpers(source):
    path = source / "astrbot/core/utils/media_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {
        "_encode_image_frame_bytes", "_prepare_model_image_sync",
        "_even_frame_indices", "_extract_animation_montage_sync",
        "prepare_model_image", "is_recoverable_image_error",
    }
    constants = {
        "MODEL_IMAGE_MAX_INPUT_BYTES", "MODEL_IMAGE_MAX_BYTES",
        "ANIMATED_MONTAGE_GRID", "ANIMATED_MONTAGE_FRAME_COUNT",
    }
    selected = [
        node for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions)
        or (isinstance(node, ast.ClassDef) and node.name == "ImageInputTooLargeError")
        or (isinstance(node, ast.Assign) and any(isinstance(item, ast.Name) and item.id in constants for item in node.targets))
    ]
    assert {node.name for node in selected if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))} == functions
    module = ModuleType("astrbot.core.utils.media_utils")
    module.__dict__.update(
        PILImage=Image, ImageOps=ImageOps, io=io, math=math, os=os,
        errno=errno, struct=struct, asyncio=asyncio, tempfile=tempfile, Path=Path,
        MediaResolver=LocalMediaResolver, ClientError=type("ClientError", (Exception,), {}),
        DownloadFileHTTPError=type("DownloadFileHTTPError", (Exception,), {}),
        logger=logging.getLogger(__name__),
    )
    wrapped = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(wrapped), str(path), "exec"), module.__dict__)
    return module


async def check(source):
    from test_timing import plugin_module

    encode = plugin_module.ImageInputPreparer.prepare.__globals__["encode_cached_image"]
    helper = load_helpers(source)
    prepared = []
    public_prepare = helper.prepare_model_image

    async def track_public_entry(image_ref, *, max_size, output_dir):
        result = await public_prepare(image_ref, max_size=max_size, output_dir=output_dir)
        prepared.append((Path(image_ref), Path(output_dir), result))
        if result is not None:
            assert Path(result[0]).is_file(), "Native preview must exist before plugin reads it"
        return result

    helper.prepare_model_image = track_public_entry
    with tempfile.TemporaryDirectory(prefix="oncue-native-image-contract-") as directory:
        root = Path(directory)
        still = root / "still.png"
        with Image.new("RGB", (1600, 400), "red") as image:
            image.save(still)
        compliant = root / "compliant.png"
        with Image.new("RGB", (32, 16), "blue") as image:
            image.save(compliant)
        animation = root / "animated.gif"
        frames = [Image.new("RGB", (120, 60), (index * 20, 30, 100)) for index in range(10)]
        try:
            frames[0].save(animation, save_all=True, append_images=frames[1:], duration=100, loop=0)
        finally:
            for frame in frames:
                frame.close()
        cache = plugin_module.SessionImageCache(root / "owned-cache")
        try:
            with patch.dict(sys.modules, {helper.__name__: helper}):
                for path, animated in ((still, False), (animation, True), (compliant, False)):
                    original = path.read_bytes()
                    ref = plugin_module.ImageReference(path.name, source=str(path))
                    async with cache.acquire("chat", ref) as resource:
                        result = await encode(resource, max_size=128)
                        assert result is not None, "Real preview helper failed"
                        assert result.is_montage is animated
                        data = base64.b64decode(result.data_uri.split(",", 1)[1])
                        assert len(data) < helper.MODEL_IMAGE_MAX_BYTES
                        with Image.open(io.BytesIO(data)) as image:
                            assert max(image.size) <= 128
                        assert resource.path.read_bytes() == original
                        assert prepared[-1][0] == resource.path, "Plugin must use the public entry"
                        _, output_dir, native_result = prepared[-1]
                        preview, _, needs_cleanup, native_source = native_result
                        assert Path(native_source) == resource.path
                        if needs_cleanup:
                            assert not Path(preview).exists(), "Owned native preview was not cleaned"
                        else:
                            assert Path(preview) == resource.path, "Compliant source should be reused"
                            assert Path(preview).exists(), "Reused cache source must not be deleted"
                        assert not output_dir.exists() or not any(output_dir.iterdir()), "Preview directory leaked files"
                    assert path.read_bytes() == original, "Platform source changed"
                assert len(prepared) == 3
                assert any(result[2] for _, _, result in prepared), "Must exercise converted previews"
                assert any(not result[2] for _, _, result in prepared), "Must exercise compliant passthrough"

                oversized = root / "over-native-limit.bin"
                # Sparse/extended local file: no large decoded image allocation.
                with oversized.open("wb") as output:
                    output.truncate(helper.MODEL_IMAGE_MAX_INPUT_BYTES + 1)
                with patch.object(helper, "_prepare_model_image_sync", side_effect=AssertionError("Oversized input reached encoder")) as internal:
                    with patch.object(helper, "prepare_model_image", wraps=track_public_entry) as public:
                        result = await encode(SimpleNamespace(path=oversized), max_size=128)
                        assert result is None, "Native oversize rejection must be preserved"
                        public.assert_awaited_once()
                        internal.assert_not_called()
                assert oversized.stat().st_size == helper.MODEL_IMAGE_MAX_INPUT_BYTES + 1
            assert len(list(cache.root.iterdir())) == 3, "Preview must not leave unaccounted cache files"
        finally:
            await cache.close()
        assert still.exists() and animation.exists() and compliant.exists()
    print(
        "AstrBot image contract passed: public still/GIF preparation and passthrough; "
        "owned previews cleaned, sources unchanged; native input limit rejects before encoding."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("astrbot_source", type=Path)
    args = parser.parse_args()
    asyncio.run(check(args.astrbot_source.resolve()))


if __name__ == "__main__":
    main()
