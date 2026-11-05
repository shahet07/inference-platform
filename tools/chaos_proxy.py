#!/usr/bin/env python3
"""A TCP proxy that injects faults, for placing between the router and a node.

Modes, which can change at any time:
  pass       Forward normally.
  delay      Hold each chunk of data for `delay_ms` first.
  blackhole  Accept connections, and swallow everything without answering. Looks alive, like a hung
             process or a silent network partition, and is the hardest failure to detect.
  refuse     Close new connections, and existing ones, immediately.
  reset      Abort connections with a TCP reset once data arrives.
"""
import socket
import struct
import threading
import time


class ChaosProxy:
    def __init__(self, listen_port, target_port):
        self.target_port = target_port
        self.mode = "pass"
        self.delay_ms = 0
        self._stopping = False
        self._connections = set()
        self._lock = threading.Lock()
        self._server = socket.socket()
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", listen_port))
        self._server.listen(256)
        self._server.settimeout(0.1)
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def set_mode(self, mode, delay_ms=0):
        self.delay_ms = delay_ms
        self.mode = mode
        if mode in ("refuse", "reset", "blackhole"):
            # Existing connections are affected too, as they would be in a real failure. A blackhole
            # keeps them open and silent.
            with self._lock:
                connections = list(self._connections)
            if mode != "blackhole":
                for connection in connections:
                    self._abort(connection)

    def stop(self):
        self._stopping = True
        self._server.close()
        with self._lock:
            connections = list(self._connections)
        for connection in connections:
            self._abort(connection)

    def _abort(self, connection):
        try:
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            connection.close()
        except OSError:
            pass

    def _accept_loop(self):
        while not self._stopping:
            try:
                client, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            if self.mode == "refuse":
                self._abort(client)
                continue
            threading.Thread(target=self._serve, args=(client,), daemon=True).start()

    def _serve(self, client):
        try:
            upstream = socket.create_connection(("127.0.0.1", self.target_port), timeout=2)
            upstream.settimeout(None)
        except OSError:
            self._abort(client)
            return
        for sock in (client, upstream):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self._lock:
            self._connections.update((client, upstream))
        threading.Thread(target=self._pump, args=(upstream, client), daemon=True).start()
        self._pump(client, upstream)

    def _pump(self, source, destination):
        try:
            while True:
                data = source.recv(65536)
                if not data:
                    break
                if self.mode == "reset":
                    self._abort(source)
                    self._abort(destination)
                    return
                while self.mode == "blackhole" and not self._stopping:
                    time.sleep(0.01)
                if self.mode == "delay" and self.delay_ms:
                    time.sleep(self.delay_ms / 1000)
                destination.sendall(data)
        except OSError:
            pass
        finally:
            with self._lock:
                self._connections.discard(source)
                self._connections.discard(destination)
            for sock in (source, destination):
                try:
                    sock.close()
                except OSError:
                    pass
