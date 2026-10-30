"""图片资源读取：原始字节、写入前限额和取消；HTTP 全部使用内存桩。"""

import asyncio
import base64
import importlib
import os
import sys
import tempfile
import traceback
import unittest
from collections import deque
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from test_timing import plugin_module


with patch.dict(sys.modules, {plugin_module.__package__: plugin_module}):
    resources = importlib.import_module(f"{plugin_module.__package__}.image_resources")


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    "AQUBAScY42YAAAAASUVORK5CYII="
)
HTTP_SOURCE = "https://image.invalid/picture?token=sensitive-image-token"


class FakeResponse:
    def __init__(self, chunks=(), *, content_length=None, status=200):
        self.chunks = deque(chunks)
        self.content_length = content_length
        self.status = status
        self.content = self
        self.read_count = 0
        self.released = False

    async def read(self, size):
        self.read_count += 1
        await asyncio.sleep(0)
        return self.chunks.popleft() if self.chunks else b""

    def release(self):
        self.released = True


class ImageResourceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.source_file = Path(directory.name) / "原始图片 with space.png"
        self.source_file.write_bytes(PNG)
        self.response = FakeResponse([PNG], content_length=len(PNG))
        self.requests = []
        self.sessions = []
        self.network_error = None
        self.session_error = None
        test = self

        class ClientError(Exception):
            pass

        class ClientSession:
            def __init__(self, **options):
                if test.session_error is not None:
                    raise test.session_error
                self.options = options
                self.closed = False
                test.sessions.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                self.closed = True

            async def get(self, source, **options):
                test.requests.append((source, options))
                if test.network_error is not None:
                    raise test.network_error
                return test.response

        fake = ModuleType("aiohttp")
        fake.ClientError = ClientError
        fake.ClientSession = ClientSession
        self.client_error = ClientError
        http = patch.dict(sys.modules, {"aiohttp": fake})
        http.start()
        self.addCleanup(http.stop)

    async def test_local_path_and_file_uri_copy_original_bytes_without_changing_source(self):
        for source in (str(self.source_file), self.source_file.as_uri()):
            with self.subTest(source=source):
                chunks = []
                size = await resources.fetch_image(source, chunks.append, len(PNG))
                self.assertEqual(size, len(PNG))
                self.assertEqual(b"".join(chunks), PNG)
                self.assertEqual(self.source_file.read_bytes(), PNG)
        self.assertEqual(self.requests, [])

    @unittest.skipUnless(os.name == "nt", "Windows drive URI forms require Windows")
    async def test_windows_legacy_drive_and_localhost_file_uris_read_the_real_file(self):
        normal = self.source_file.as_uri()
        for source in (normal.replace("file:///", "file://", 1), normal.replace("file:///", "file://localhost/", 1)):
            with self.subTest(source=source):
                chunks = []
                size = await resources.fetch_image(source, chunks.append, len(PNG))
                self.assertEqual(size, len(PNG))
                self.assertEqual(b"".join(chunks), PNG)
        self.assertEqual(self.source_file.read_bytes(), PNG)

    async def test_base64_and_data_uris_preserve_bytes_with_legacy_padding_and_escapes(self):
        payload = PNG + bytes(range(256)) * 600 + b"end"
        encoded = base64.b64encode(payload).decode()
        sources = (
            "base64://" + encoded,
            "base64:// \n" + encoded.rstrip("=") + "\n",
            encoded,
            "\n" + encoded.rstrip("=") + "\n",
            "data:image/png;base64," + encoded,
            "data:image/png;base64," + encoded.replace("+", "%2B").replace("/", "%2F"),
        )
        for source in sources:
            with self.subTest(kind=source[:25]):
                chunks = []
                size = await resources.fetch_image(source, chunks.append, len(payload))
                self.assertEqual(size, len(payload))
                self.assertEqual(b"".join(chunks), payload)
        self.assertEqual(self.requests, [])

    async def test_long_bare_base64_is_read_even_when_it_exceeds_filesystem_path_limits(self):
        payload = b"\x00" * 150000
        source = base64.b64encode(payload).decode()
        chunks = []
        size = await resources.fetch_image(source, chunks.append, len(payload))
        self.assertEqual(size, len(payload))
        self.assertEqual(b"".join(chunks), payload)
        self.assertEqual(self.requests, [])

    async def test_existing_base64_looking_filename_keeps_file_precedence_even_when_unreadable(self):
        path = self.source_file.with_name("YWJj")
        path.write_bytes(PNG)
        previous = Path.cwd()
        try:
            os.chdir(path.parent)
            chunks = []
            size = await resources.fetch_image(path.name, chunks.append, len(PNG))
            self.assertEqual(size, len(PNG))
            self.assertEqual(b"".join(chunks), PNG)
            with patch.object(Path, "open", side_effect=PermissionError(13, "unreadable file")):
                with self.assertRaises(resources.ImageFetchError):
                    await resources.fetch_image(path.name, chunks.append, len(PNG))
        finally:
            os.chdir(previous)
        self.assertEqual(path.read_bytes(), PNG)

    async def test_unreadable_empty_or_invalid_sources_fail_without_exposing_the_source(self):
        missing = self.source_file.with_name("sensitive-missing-image.png")
        for source in ("", str(missing), "base64://invalid!sensitive-payload", "base64://", "data:image/png,not-base64"):
            with self.subTest(kind=source[:15]):
                chunks = []
                with self.assertRaises(resources.ImageFetchError) as caught:
                    await resources.fetch_image(source, chunks.append, 1024)
                self.assertEqual(caught.exception.reason, "获取失败")
                self.assertEqual(str(caught.exception), "获取失败")
        self.source_file.write_bytes(b"")
        with self.assertRaises(resources.ImageFetchError):
            await resources.fetch_image(str(self.source_file), lambda chunk: None, 1024)
        self.assertTrue(self.source_file.exists())

    async def test_oversized_local_file_is_rejected_before_any_write_and_keeps_the_source(self):
        chunks = []
        with self.assertRaises(resources.ImageTooLarge) as caught:
            await resources.fetch_image(str(self.source_file), chunks.append, len(PNG) - 1)
        self.assertEqual(caught.exception.size_bytes, len(PNG))
        self.assertEqual(chunks, [])
        self.assertEqual(self.source_file.read_bytes(), PNG)

    async def test_encoded_resource_never_writes_a_chunk_past_the_limit_or_invents_full_size(self):
        encoded = base64.b64encode(PNG).decode()
        for source in ("base64://" + encoded, encoded):
            with self.subTest(prefixed=source.startswith("base64://")):
                chunks = []
                with self.assertRaises(resources.ImageTooLarge) as caught:
                    await resources.fetch_image(source, chunks.append, len(PNG) - 1)
                self.assertIsNone(caught.exception.size_bytes)
                self.assertLessEqual(sum(map(len, chunks)), len(PNG) - 1)
                self.assertNotIn(encoded, "".join(traceback.format_exception(caught.exception)))

    async def test_http_reads_original_bytes_and_releases_the_response(self):
        self.response = FakeResponse([PNG[:10], PNG[10:]], content_length=len(PNG))
        chunks = []
        size = await resources.fetch_image(HTTP_SOURCE, chunks.append, len(PNG))
        self.assertEqual(size, len(PNG))
        self.assertEqual(b"".join(chunks), PNG)
        self.assertTrue(self.response.released)
        self.assertTrue(self.sessions[0].closed)

    async def test_content_length_rejects_before_reading_or_writing_the_http_body(self):
        chunks = []
        with self.assertRaises(resources.ImageTooLarge) as caught:
            await resources.fetch_image(HTTP_SOURCE, chunks.append, len(PNG) - 1)
        self.assertEqual(caught.exception.size_bytes, len(PNG))
        self.assertEqual(self.response.read_count, 0)
        self.assertEqual(chunks, [])
        self.assertTrue(self.response.released)
        self.assertTrue(self.sessions[0].closed)

    async def test_unknown_http_size_stops_before_the_oversized_chunk_without_inventing_size(self):
        self.response = FakeResponse([b"first", b"second"])
        chunks = []
        with self.assertRaises(resources.ImageTooLarge) as caught:
            await resources.fetch_image(HTTP_SOURCE, chunks.append, 8)
        self.assertIsNone(caught.exception.size_bytes)
        self.assertEqual(chunks, [b"first"])
        self.assertTrue(self.response.released)
        self.assertTrue(self.sessions[0].closed)

    async def test_http_status_codes_do_not_claim_that_cloud_cache_was_cleaned(self):
        for status in (403, 404, 410, 500):
            with self.subTest(status=status):
                self.response = FakeResponse([PNG], status=status)
                chunks = []
                with self.assertRaises(resources.ImageFetchError) as caught:
                    await resources.fetch_image(HTTP_SOURCE, chunks.append, 1024)
                self.assertEqual(caught.exception.reason, "获取失败")
                self.assertEqual(chunks, [])
                self.assertEqual(self.response.read_count, 0)
                self.assertTrue(self.response.released)

    async def test_network_failure_keeps_default_tls_verification_and_does_not_leak_urls(self):
        self.network_error = self.client_error(HTTP_SOURCE)
        with self.assertRaises(resources.ImageFetchError) as caught:
            await resources.fetch_image(HTTP_SOURCE, lambda chunk: None, 1024)
        self.assertEqual(caught.exception.reason, "获取失败")
        self.assertEqual(len(self.requests), 1)
        for _, options in self.requests:
            self.assertIsNot(options.get("ssl", True), False)
            self.assertIsNot(options.get("verify_ssl", True), False)
        for session in self.sessions:
            self.assertIsNot(session.options.get("ssl", True), False)
            self.assertTrue(session.closed)
        formatted = "".join(traceback.format_exception(caught.exception))
        self.assertNotIn("sensitive-image-token", formatted)

    async def test_http_session_setup_oserror_becomes_a_safe_fetch_error(self):
        self.session_error = OSError("sensitive-session-setup-details")
        chunks = []
        with self.assertRaises(resources.ImageFetchError) as caught:
            await resources.fetch_image(HTTP_SOURCE, chunks.append, 1024)
        self.assertEqual(caught.exception.reason, "获取失败")
        self.assertEqual(str(caught.exception), "获取失败")
        self.assertEqual(chunks, [])
        self.assertEqual(self.requests, [])

    async def test_sink_failures_propagate_unchanged_and_leave_the_original_file_intact(self):
        sources = (str(self.source_file), "base64://" + base64.b64encode(PNG).decode(), HTTP_SOURCE)
        for source in sources:
            with self.subTest(kind=source[:15]):
                failure = OSError("cache quota rejected this chunk")

                def reject(chunk):
                    raise failure

                with self.assertRaises(OSError) as caught:
                    await resources.fetch_image(source, reject, 1024)
                self.assertIs(caught.exception, failure)
        self.assertEqual(self.source_file.read_bytes(), PNG)
        self.assertTrue(self.response.released)
        self.assertTrue(self.sessions[0].closed)

    async def test_cancelled_reads_do_not_keep_writing_in_the_background(self):
        payload = PNG + bytes(range(256)) * 2048
        self.source_file.write_bytes(payload)
        encoded = base64.b64encode(payload).decode()
        sources = (str(self.source_file), "base64://" + encoded, encoded, HTTP_SOURCE)
        for source in sources:
            with self.subTest(kind=source[:15]):
                self.response = FakeResponse([b"first", b"second", b"third"])
                chunks = []

                def cancel_after_first_write(chunk):
                    chunks.append(chunk)
                    asyncio.current_task().cancel()

                task = asyncio.create_task(resources.fetch_image(source, cancel_after_first_write, len(payload)))
                with self.assertRaises(asyncio.CancelledError):
                    await task
                for _ in range(3):
                    await asyncio.sleep(0)
                self.assertEqual(len(chunks), 1)
                if source == HTTP_SOURCE:
                    self.assertTrue(self.response.released)
                    self.assertTrue(self.sessions[-1].closed)
        self.assertEqual(self.source_file.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
