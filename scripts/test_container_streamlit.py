"""Offline container-adapter tests, including a real TCP half-close roundtrip."""
from pathlib import Path
import socket
import socketserver
import sys
import threading
import unittest
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.container_streamlit import ProxyRelay, container_environment
from psycopg.conninfo import conninfo_to_dict


class EnvironmentTests(unittest.TestCase):
    def test_loopback_rewrite_preserves_credentials_and_options(self):
        original = {
            "APEXLOGIC_CONTAINER_MODE": "1",
            "APEXLOGIC_POSTGRES_DSN": "host=127.0.0.1 port=5433 dbname=example user=person password='space and quote' sslmode=prefer",
            "APEXLOGIC_QUEUE_REDIS_URL": "redis://user:p%40ss@127.0.0.1:6380/2?socket_timeout=3",
            "HTTP_PROXY": "http://127.0.0.1:7890",
            "HTTPS_PROXY": "http://127.0.0.1:7890",
        }
        result = container_environment(original)
        info = conninfo_to_dict(result["APEXLOGIC_POSTGRES_DSN"])
        self.assertEqual((info["host"], info["port"]), ("postgres", "5432"))
        self.assertEqual(info["password"], "space and quote")
        self.assertEqual(info["sslmode"], "prefer")
        queue = urlsplit(result["APEXLOGIC_QUEUE_REDIS_URL"])
        self.assertEqual((queue.hostname, queue.port, queue.password, queue.path, queue.query),
                         ("queue", 6379, "p%40ss", "/2", "socket_timeout=3"))
        self.assertEqual(result["APEXLOGIC_REDIS_HOST"], "redis")
        self.assertEqual(result["HTTP_PROXY"], original["HTTP_PROXY"])
        self.assertEqual(result["HTTPS_PROXY"], original["HTTPS_PROXY"])
        self.assertIn("127.0.0.1", original["APEXLOGIC_POSTGRES_DSN"])

    def test_remote_services_and_non_container_environment_are_unchanged(self):
        original = {"APEXLOGIC_CONTAINER_MODE": "1", "APEXLOGIC_POSTGRES_DSN":
                    "host=db.example port=15432 dbname=example", "APEXLOGIC_REDIS_HOST": "cache.example",
                    "APEXLOGIC_QUEUE_REDIS_URL": "rediss://queue.example:16379/4"}
        self.assertEqual(container_environment(original), original)
        self.assertEqual(container_environment({"HTTP_PROXY": "http://127.0.0.1:7890"}),
                         {"HTTP_PROXY": "http://127.0.0.1:7890"})

    def test_ipv6_loopback_queue_uri(self):
        result = container_environment({"APEXLOGIC_CONTAINER_MODE": "1",
                                        "APEXLOGIC_QUEUE_REDIS_URL": "redis://[::1]:6380/0"})
        self.assertEqual(result["APEXLOGIC_QUEUE_REDIS_URL"], "redis://queue:6379/0")


class EchoHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while data := self.request.recv(65536):
            self.request.sendall(data)


class RelayTests(unittest.TestCase):
    def test_large_bidirectional_payload_and_half_close(self):
        upstream = socketserver.ThreadingTCPServer(("127.0.0.1", 0), EchoHandler)
        relay = ProxyRelay(("127.0.0.1", 0), upstream.server_address)
        servers = (upstream, relay)
        for server in servers:
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            payload = bytes(range(256)) * 1024
            with socket.create_connection(relay.server_address, timeout=5) as client:
                client.sendall(payload)
                client.shutdown(socket.SHUT_WR)
                received = bytearray()
                while data := client.recv(65536):
                    received.extend(data)
            self.assertEqual(received, payload)
        finally:
            for server in reversed(servers):
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
