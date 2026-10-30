import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from map_projection.service import Notifier


@unittest.skipUnless(hasattr(socket,'AF_UNIX'),'Unix datagram socket required')
class ServiceTests(unittest.TestCase):
    def test_readiness_and_rate_limited_watchdog(self):
        with tempfile.TemporaryDirectory() as folder:
            address=str(Path(folder)/'notify')
            with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as server:
                server.bind(address);server.settimeout(.02)
                with patch.dict(os.environ,{'NOTIFY_SOCKET':address,'WATCHDOG_USEC':'3000000'}):
                    notifier=Notifier();notifier.ready()
                    self.assertIn(b'READY=1',server.recv(512))
                    notifier.tick(0);self.assertEqual(server.recv(512),b'WATCHDOG=1')
                    notifier.tick(.2)
                    with self.assertRaises(socket.timeout):server.recv(512)
                    notifier.tick(1);self.assertEqual(server.recv(512),b'WATCHDOG=1')

    def test_missing_manager_is_nonfatal(self):
        with patch.dict(os.environ,{'NOTIFY_SOCKET':'/nonexistent/map-projection-test-notify'}):
            notifier=Notifier();notifier.ready();notifier.tick(0)


if __name__=='__main__':unittest.main()
