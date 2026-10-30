"""One fixed-cadence, latest-frame JPEG producer shared by all viewers."""
import threading
import time
import socket


class PreviewStream:
    def __init__(self,engine,fps=15):
        self.engine=engine;self.period=1/fps;self.condition=threading.Condition()
        self.clients=0;self.sequence=0;self.image=None
        self.thread=threading.Thread(target=self.run,name='preview-stream',daemon=True);self.thread.start()

    def run(self):
        next_tick=0.;last=None
        while not self.engine.stopped.is_set():
            with self.condition:
                self.condition.wait_for(lambda:self.clients or self.engine.stopped.is_set(),timeout=.5)
                if not self.clients:continue
            if self.engine.stopped.wait(max(0,next_tick-time.monotonic())):break
            with self.engine.lock:
                overlay=self.engine.overlay() if hasattr(self.engine,'overlay') else None
                version=(self.engine.frame_id,getattr(self.engine,'presentation_id',0),(overlay['generation'],round(overlay['progress'],3)) if overlay else None)
            if version!=last or self.image is None:
                image=self.engine.preview(size=(960,540),encoding='JPEG')
                if image:
                    with self.condition:
                        self.image=image;last=version
            # Complete multipart frames at a steady cadence even while paused;
            # reuse the encoded JPEG without another GPU capture or encoding.
            with self.condition:
                if self.image:self.sequence+=1;self.condition.notify_all()
            next_tick=max(next_tick+self.period,time.monotonic())

    def serve(self,handler):
        handler.send_response(200);handler.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
        handler.send_header('Cache-Control','no-store');handler.send_header('X-Accel-Buffering','no');handler.send_header('Connection','close');handler.end_headers();handler.close_connection=True
        handler.connection.settimeout(2)
        handler.connection.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,65536)
        with self.condition:self.clients+=1;self.condition.notify_all()
        sequence=-1
        try:
            handler.wfile.write(b'--frame\r\n')
            while not self.engine.stopped.is_set():
                with self.condition:
                    self.condition.wait_for(lambda:self.sequence!=sequence or self.engine.stopped.is_set(),timeout=1)
                    if self.sequence==sequence or not self.image:continue
                    sequence=self.sequence;image=self.image
                handler.wfile.write(b'Content-Type: image/jpeg\r\nContent-Length: '+str(len(image)).encode()+b'\r\n\r\n'+image+b'\r\n--frame\r\n');handler.wfile.flush()
        except OSError:pass
        finally:
            with self.condition:self.clients-=1
