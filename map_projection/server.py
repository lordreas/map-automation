"""Small LAN control server; no frontend build or cloud service required."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit, parse_qs

STATIC=Path(__file__).resolve().parent / "web"


def start(engine,host,port):
    from .preview_stream import PreviewStream
    previews=PreviewStream(engine)
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        disable_nagle_algorithm=True
        def log_message(self,*args):
            pass

        def send(self,status,body,kind="application/json"):
            self.send_response(status)
            self.send_header("Content-Type",kind)
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass

        def json(self,status,value):
            self.send(status,json.dumps(value,allow_nan=False).encode())

        def do_GET(self):
            path=urlsplit(self.path).path
            if path=="/api/live":
                from .websocket import serve
                return serve(self,engine)
            if path=="/preview.mjpg":return previews.serve(self)
            if path=="/api/state":return self.json(200,engine.snapshot())
            if path=="/api/events":
                self.send_response(200)
                self.send_header('Content-Type','text/event-stream')
                self.send_header('Cache-Control','no-cache')
                self.send_header('X-Accel-Buffering','no')
                self.end_headers()
                sequence=-1
                try:
                    while not engine.stopped.is_set():
                        with engine.events:
                            engine.events.wait_for(lambda:sequence!=engine.event_sequence or engine.stopped.is_set(),timeout=10)
                            event=engine.time_event();sequence=event['sequence']
                        self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode());self.wfile.flush()
                except OSError:pass
                return
            if path in ("/preview.png","/frame.png"):
                query=parse_qs(urlsplit(self.path).query)
                try:
                    size=(max(64,min(1920,int(query.get("width",[960])[0]))),
                          max(64,min(1080,int(query.get("height",[540])[0]))))
                except ValueError:return self.json(400,{"error":"Invalid preview dimensions."})
                image=engine.preview(full=path=="/frame.png",size=size)
                return self.send(200,image,"image/png") if image else self.json(503,{"error":"First frame is still rendering."})
            files={"/":("index.html","text/html; charset=utf-8"),"/app.js":("app.js","text/javascript; charset=utf-8"),
                   "/style.css":("style.css","text/css; charset=utf-8")}
            if path in files:
                name,kind=files[path]
                return self.send(200,(STATIC/name).read_bytes(),kind)
            self.json(404,{"error":"Not found"})

        def do_POST(self):
            # JSON + same-origin checks prevent drive-by browser control on a
            # trusted LAN. This is not an Internet authentication mechanism.
            origin=self.headers.get("Origin")
            if origin and urlsplit(origin).netloc!=self.headers.get("Host"):
                return self.json(403,{"error":"Cross-origin control is disabled."})
            if self.headers.get_content_type()!="application/json":
                return self.json(415,{"error":"Use application/json."})
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0<length<=65536:raise ValueError("Request must contain 1–65536 bytes.")
                def invalid_constant(value):raise ValueError("Nonfinite JSON number: "+value)
                body=json.loads(self.rfile.read(length),parse_constant=invalid_constant)
                if not isinstance(body,dict):raise ValueError("Expected a JSON object.")
                path=urlsplit(self.path).path
                if path=="/api/config":engine.update(body,persist=parse_qs(urlsplit(self.path).query).get('transient')!=['1'])
                elif path=="/api/nudge":engine.nudge(body.get("corner"),body.get("dx"),body.get("dy"),body.get("step",1))
                elif path=="/api/query":engine.request_query(body)
                elif path=="/api/bake":engine.bake_day()
                elif path=="/api/track/replay":engine.replay_track()
                elif path=="/api/rivers/replay":engine.replay_rivers()
                elif path=="/api/now":engine.now()
                elif path=="/api/play":engine.play()
                elif path=="/api/stop":engine.stop_animation()
                elif path=="/api/cancel":
                    with engine.lock:engine.bake=[]
                else:return self.json(404,{"error":"Not found"})
                self.json(200,engine.snapshot())
            except (ValueError,TypeError,KeyError) as exc:
                self.json(400,{"error":str(exc)})
            except Exception as exc:
                self.json(500,{"error":str(exc)})

    server=ThreadingHTTPServer((host,port),Handler)
    server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,name="control-server",daemon=True)
    thread.start()
    return server
