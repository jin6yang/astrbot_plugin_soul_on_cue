"""Check the real AstrBot preview helper used by the plugin, with Pillow installed."""

import argparse
import ast
import asyncio
import base64
import io
import math
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from PIL import Image, ImageOps


def load_helpers(source):
    path = source / "astrbot/core/utils/media_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {
        "_encode_image_frame_bytes", "_prepare_model_image_sync",
        "_even_frame_indices", "_extract_animation_montage_sync",
    }
    constants = {"MODEL_IMAGE_MAX_BYTES", "ANIMATED_MONTAGE_GRID", "ANIMATED_MONTAGE_FRAME_COUNT"}
    selected = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (isinstance(node, ast.Assign) and any(isinstance(item, ast.Name) and item.id in constants for item in node.targets))
    ]
    module = ModuleType("astrbot.core.utils.media_utils")
    module.__dict__.update(PILImage=Image, ImageOps=ImageOps, io=io, math=math)
    wrapped = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(wrapped), str(path), "exec"), module.__dict__)
    return module


async def check(source):
    from test_timing import plugin_module

    encode = plugin_module.ImageInputPreparer.prepare.__globals__["encode_cached_image"]
    helper = load_helpers(source)
    with tempfile.TemporaryDirectory(prefix="oncue-native-image-contract-") as directory:
        root = Path(directory)
        still = root / "still.png"
        Image.new("RGB", (1600, 400), "red").save(still)
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
                for path, animated in ((still, False), (animation, True)):
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
                    assert path.read_bytes() == original, "Platform source changed"
            assert len(list(cache.root.iterdir())) == 2, "Preview must not create unaccounted files"
        finally:
            await cache.close()
        assert still.exists() and animation.exists()
    print("AstrBot image contract passed: still/GIF previews fit byte and edge limits; sources unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("astrbot_source", type=Path)
    args = parser.parse_args()
    asyncio.run(check(args.astrbot_source.resolve()))


if __name__ == "__main__":
    main()
