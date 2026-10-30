"""Bounded reads of the image resource supplied by a message platform."""

import asyncio
import base64
import binascii
import errno
import os
import stat
from collections.abc import Callable
from contextlib import AsyncExitStack
from pathlib import Path
from urllib.parse import unquote_to_bytes, urlsplit
from urllib.request import url2pathname


_CHUNK_BYTES = 64 * 1024


class ImageTooLarge(Exception):
    """The image exceeds the configured limit, with its full size if known."""

    def __init__(self, size_bytes: int | None = None):
        self.size_bytes = size_bytes
        super().__init__("图片超过单张大小限制")


class ImageFetchError(Exception):
    """A safe failure reason without a URL, local path, or encoded payload."""

    def __init__(self, reason: str = "获取失败"):
        self.reason = reason
        super().__init__(reason)


async def _read_base64(source: str, offset: int, write: Callable[[bytes], None], *, data_uri: bool) -> None:
    """Decode bounded pieces, accepting whitespace and omitted final padding."""
    pending = b""
    finished = False
    while offset < len(source):
        await asyncio.sleep(0)
        end = min(offset + _CHUNK_BYTES, len(source))
        # A data URI may percent-encode base64 characters. Keep each escape whole.
        if data_uri and end < len(source):
            if source[end - 1] == "%":
                end -= 1
            elif source[end - 2] == "%":
                end -= 2
        raw = source[offset:end]
        offset = end
        try:
            encoded = unquote_to_bytes(raw) if data_uri else raw.encode("ascii")
        except (UnicodeError, ValueError):
            raise ImageFetchError() from None
        encoded = b"".join(encoded.split())
        if finished and encoded:
            raise ImageFetchError()
        pending += encoded
        complete = len(pending) // 4 * 4
        if not complete:
            continue
        encoded, pending = pending[:complete], pending[complete:]
        if b"=" in encoded:
            if pending:
                raise ImageFetchError()
            finished = True
        try:
            chunk = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise ImageFetchError() from None
        if chunk:
            write(chunk)
    if pending:
        await asyncio.sleep(0)
        try:
            chunk = base64.b64decode(pending + b"=" * (-len(pending) % 4), validate=True)
        except (binascii.Error, ValueError):
            raise ImageFetchError() from None
        if chunk:
            write(chunk)


async def fetch_image(source: str, write: Callable[[bytes], None], max_bytes: int) -> int:
    """Copy the platform's resource without decoding or transforming its image.

    Args:
        source: Local path, file URI, HTTP(S) URL, prefixed or legacy bare base64,
            or data URI. An existing local file takes precedence over bare base64.
        write: Synchronous sink, called only after checking the image size limit.
            Its exceptions propagate so the cache can enforce its own quota.
        max_bytes: Maximum number of bytes allowed for this image.

    Returns:
        The number of resource bytes copied to the sink.

    Raises:
        ImageTooLarge: A reported or observed resource size exceeds the limit.
            ``size_bytes`` is None when the complete resource size is unknown.
        ImageFetchError: The resource cannot be read, or contains no bytes.
        ValueError: The size limit is negative.

    Cancellation closes open resources. No background tasks or worker threads
    write to the sink. The caller owns cleanup of any partially written copy;
    this function never deletes or modifies the source file.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes must not be negative")
    if not isinstance(source, str) or not source:
        raise ImageFetchError()
    total = 0
    known_size = None

    def emit(chunk: bytes) -> None:
        nonlocal total
        if total + len(chunk) > max_bytes:
            raise ImageTooLarge(known_size)
        write(chunk)
        total += len(chunk)

    prefix = source[:10].lower()
    if prefix.startswith("base64://"):
        await _read_base64(source, 9, emit, data_uri=False)
    elif prefix.startswith("data:"):
        separator = source.find(",")
        if separator < 0 or "base64" not in source[5:separator].lower().split(";")[1:]:
            raise ImageFetchError()
        await _read_base64(source, separator + 1, emit, data_uri=True)
    elif prefix.startswith(("http://", "https://")):
        try:
            import aiohttp
        except ImportError:
            raise ImageFetchError() from None
        async with AsyncExitStack() as stack:
            try:
                session = await stack.enter_async_context(aiohttp.ClientSession(trust_env=True))
            except (aiohttp.ClientError, OSError, ValueError):
                raise ImageFetchError() from None
            try:
                response = await session.get(source)
            except (aiohttp.ClientError, TimeoutError, ValueError):
                raise ImageFetchError() from None
            try:
                # A generic status code does not establish platform cache cleanup.
                if response.status != 200:
                    raise ImageFetchError()
                known_size = response.content_length
                if known_size is not None and known_size > max_bytes:
                    raise ImageTooLarge(known_size)
                while True:
                    try:
                        chunk = await response.content.read(_CHUNK_BYTES)
                    except (aiohttp.ClientError, TimeoutError):
                        raise ImageFetchError() from None
                    if not chunk:
                        break
                    if known_size is not None and total + len(chunk) > known_size:
                        known_size = None
                    emit(chunk)
                    await asyncio.sleep(0)
            finally:
                response.release()
    else:
        if prefix.startswith("file:"):
            try:
                parsed = urlsplit(source)
                host = parsed.netloc
                path = parsed.path
                if host and host.lower() != "localhost":
                    path = host + path if len(host) == 2 and host[1] == ":" else "//" + host + path
                source = url2pathname(path)
                if len(source) >= 4 and source[0] in "/\\" and source[1].isalpha() and source[2] == ":":
                    source = source[1:]
            except (UnicodeError, ValueError, OSError):
                raise ImageFetchError() from None
        file = None
        try:
            file = Path(source).open("rb")
        except OSError as exc:
            # Match AstrBot's legacy input support without reinterpreting an
            # existing but unreadable file, or an explicitly requested file URI.
            if prefix.startswith("file:") or exc.errno not in {
                errno.ENOENT, errno.ENOTDIR, errno.ENAMETOOLONG, errno.EINVAL,
            }:
                raise ImageFetchError() from None
        except ValueError:
            raise ImageFetchError() from None
        # Leave the path-error handler before decoding so a later size/sink
        # failure cannot chain the raw base64 payload from the failed open.
        if file is None:
            await _read_base64(source, 0, emit, data_uri=False)
        else:
            try:
                try:
                    metadata = os.fstat(file.fileno())
                except OSError:
                    raise ImageFetchError() from None
                if not stat.S_ISREG(metadata.st_mode):
                    raise ImageFetchError()
                known_size = metadata.st_size
                if known_size > max_bytes:
                    raise ImageTooLarge(known_size)
                while True:
                    await asyncio.sleep(0)
                    try:
                        chunk = file.read(_CHUNK_BYTES)
                    except OSError:
                        raise ImageFetchError() from None
                    if not chunk:
                        break
                    # A changed source file no longer has a known complete size.
                    if known_size is not None and total + len(chunk) > known_size:
                        known_size = None
                    emit(chunk)
            finally:
                file.close()
    if not total:
        raise ImageFetchError()
    return total
