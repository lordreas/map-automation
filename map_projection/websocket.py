"""Small RFC6455 control channel; images use an independent MJPEG stream."""
import base64
import hashlib
import json
import os
import struct
import threading
from urllib.parse import urlsplit


def exact(stream,n):
    result=bytearray()
    while len(result)<n:
        part=stream.read(n-len(result))
        if not part:raise EOFError
        result.extend(part)
    return bytes(result)


def read_frame(stream):
    first,second=exact(stream,2);opcode=first&15
    if first&0x70 or not second&128:raise ValueError('Invalid WebSocket frame')
    length=second&127
    if length==126:length=struct.unpack('!H',exact(stream,2))[0]
    elif length==127:length=struct.unpack('!Q',exact(stream,8))[0]
    if length>65536 or (opcode>=8 and (length>125 or not first&128)):raise ValueError('WebSocket message too large')
    mask=exact(stream,4);body=exact(stream,length)
    return bool(first&128),opcode,bytes(value^mask[i&3] for i,value in enumerate(body))


def write_frame(stream,body,opcode=1):
    if isinstance(body,str):body=body.encode()
    length=len(body)
    header=bytes([128|opcode,length]) if length<126 else bytes([128|opcode,126])+struct.pack('!H',length) if length<65536 else bytes([128|opcode,127])+struct.pack('!Q',length)
    stream.write(header+body);stream.flush()


def merge_patch(old,new):
    result=dict(old)
    for key,value in new.items():
        result[key]=merge_patch(result.get(key,{}) if isinstance(result.get(key),dict) else {},value) if isinstance(value,dict) else value
    return result


class ControlInbox:
    """Read sockets promptly; one latest pending patch, never a stale FIFO."""
    def __init__(self,engine,send):
        self.engine,self.send=engine,send;self.condition=threading.Condition();self.pending=None;self.stopped=False
        self.thread=threading.Thread(target=self.run,name='latest-controls',daemon=True);self.thread.start()

    def submit(self,message):
        with self.condition:
            previous=self.pending
            self.pending={'patch':merge_patch(previous['patch'] if previous else {},message['patch']),
                          'persist':bool((previous and previous['persist']) or message.get('persist') is True),'seq':message.get('seq')}
            self.condition.notify_all()

    def run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda:self.pending is not None or self.stopped)
                if self.stopped:return
                message=self.pending;self.pending=None
            try:
                self.engine.update(message['patch'],persist=message['persist'])
                answer={'type':'ack','seq':message['seq'],'revision':self.engine.revision}
            except (ValueError,TypeError,KeyError) as exc:answer={'type':'error','seq':message['seq'],'error':str(exc)}
            try:self.send(json.dumps(answer))
            except (OSError,ValueError):return

    def close(self):
        with self.condition:self.stopped=True;self.pending=None;self.condition.notify_all()


def serve(handler,engine):
    origin=handler.headers.get('Origin')
    if origin and urlsplit(origin).netloc!=handler.headers.get('Host'):
        return handler.json(403,{'error':'Cross-origin control is disabled.'})
    try:
        key=handler.headers.get('Sec-WebSocket-Key','')
        if len(base64.b64decode(key,validate=True))!=16 or handler.headers.get('Sec-WebSocket-Version')!='13':raise ValueError
    except Exception:return handler.json(400,{'error':'Invalid WebSocket handshake'})
    handler.send_response(101);handler.send_header('Upgrade','websocket');handler.send_header('Connection','Upgrade')
    accept=base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
    handler.send_header('Sec-WebSocket-Accept',accept);handler.end_headers();handler.close_connection=True
    fragments=bytearray();fragmented=False;write_lock=threading.Lock()
    def send(body,opcode=1):
        with write_lock:write_frame(handler.wfile,body,opcode)
    inbox=ControlInbox(engine,send)
    try:
        while not engine.stopped.is_set():
            final,opcode,body=read_frame(handler.rfile)
            if opcode==8:send(body,8);break
            if opcode==9:send(body,10);continue
            if opcode==10:continue
            if opcode==1:
                if fragmented:raise ValueError('Interleaved messages')
                fragments=bytearray(body);fragmented=not final
            elif opcode==0 and fragmented:fragments.extend(body);fragmented=not final
            else:raise ValueError('Expected JSON text')
            if len(fragments)>65536:raise ValueError('Message too large')
            if not final:continue
            sequence=None
            try:
                message=json.loads(fragments.decode(),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite number')))
                if not isinstance(message,dict):raise ValueError('Expected a JSON object')
                sequence=message.get('seq')
                if message.get('type')=='ping':send(json.dumps({'type':'pong'}));continue
                if message.get('type')!='config' or not isinstance(message.get('patch'),dict):raise ValueError('Expected configuration patch')
                inbox.submit(message)
            except (ValueError,TypeError,KeyError,UnicodeDecodeError) as exc:
                send(json.dumps({'type':'error','seq':sequence,'error':str(exc)}))
    except (OSError,EOFError):pass
    except ValueError:
        try:send(struct.pack('!H',1002),8)
        except OSError:pass

    finally:inbox.close()
