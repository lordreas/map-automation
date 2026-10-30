import base64,copy,io,json,os,socket,struct,threading,time,unittest
from urllib.request import urlopen
from PIL import Image
from map_projection.server import start
from map_projection.websocket import read_frame


def masked(message,opcode=1,final=True):
    body=message if isinstance(message,bytes) else json.dumps(message).encode();mask=b'abcd'
    n=len(body);head=bytes([(128 if final else 0)|opcode,128|n]) if n<126 else bytes([(128 if final else 0)|opcode,254])+struct.pack('!H',n)
    return head+mask+bytes(c^mask[i%4] for i,c in enumerate(body))


def response(stream):
    a,b=stream.read(2);n=b&127
    if n==126:n=struct.unpack('!H',stream.read(2))[0]
    return a&15,stream.read(n)


class FakeEngine:
    def __init__(self):
        self.stopped=threading.Event();self.lock=threading.RLock();self.frame_id=1;self.presentation_id=0;self.revision=0;self.patches=[];self.previews=0
    def update(self,patch,persist=True):self.patches.append((patch,persist));self.revision+=1
    def preview(self,**kwargs):
        self.previews+=1;buf=io.BytesIO();Image.new('RGB',(16,16),(self.frame_id,20,30)).save(buf,'JPEG');return buf.getvalue()
    def snapshot(self):raise AssertionError('WebSocket controls must not request a snapshot')


class LiveTransportTests(unittest.TestCase):
    def setUp(self):self.engine=FakeEngine();self.server=start(self.engine,'127.0.0.1',0);self.port=self.server.server_port
    def tearDown(self):self.engine.stopped.set();self.server.shutdown();self.server.server_close()

    def connect(self):
        s=socket.create_connection(('127.0.0.1',self.port),timeout=3)
        key=base64.b64encode(os.urandom(16)).decode()
        s.sendall((f'GET /api/live HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\nOrigin: http://127.0.0.1:{self.port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
        stream=s.makefile('rb');self.assertIn(b'101',stream.readline())
        while stream.readline()!=b'\r\n':pass
        return s,stream

    def test_multiple_commands_no_preview_or_snapshot_roundtrip(self):
        s,stream=self.connect()
        try:
            s.sendall(b''.join(masked({'type':'config','seq':i,'persist':i==19,'patch':{'sun':{'time':str(i)}}}) for i in range(20)))
            answers=[]
            while not answers or answers[-1].get('seq')!=19:answers.append(json.loads(response(stream)[1]))
            self.assertEqual(answers[-1]['seq'],19);self.assertEqual(self.engine.patches[-1],({'sun':{'time':'19'}},True))
            self.assertLessEqual(len(self.engine.patches),20);self.assertEqual(self.engine.previews,0)
            s.sendall(masked(b'hi',opcode=9));opcode,body=response(stream);self.assertEqual((opcode,body),(10,b'hi'))
        finally:stream.close();s.close()

    def test_fragmented_json_and_unmasked_frames_rejected(self):
        s,stream=self.connect()
        try:
            body=json.dumps({'type':'config','patch':{}}).encode();s.sendall(masked(body[:10],final=False)+masked(body[10:],opcode=0))
            self.assertEqual(json.loads(response(stream)[1])['type'],'ack')
            s.sendall(b'\x81\x02{}');self.assertEqual(response(stream)[0],8)
        finally:stream.close();s.close()

    def test_shared_fixed_cadence_stream_not_one_image_per_command(self):
        with urlopen(f'http://127.0.0.1:{self.port}/preview.mjpg',timeout=3) as image:
            self.assertEqual(image.readline(),b'--frame\r\n');headers={}
            while True:
                line=image.readline()
                if line==b'\r\n':break
                k,v=line.decode().split(':',1);headers[k]=v.strip()
            data=image.read(int(headers['Content-Length']));self.assertEqual(Image.open(io.BytesIO(data)).size,(16,16))
            for n in range(50):self.engine.update({'x':n})
            time.sleep(.15);self.assertEqual(self.engine.previews,1)
            self.engine.frame_id+=1
            self.assertEqual(image.readline(),b'\r\n');self.assertEqual(image.readline(),b'--frame\r\n')
            self.assertEqual(image.readline(),b'Content-Type: image/jpeg\r\n')
            end=time.monotonic()+2
            while self.engine.previews<2 and time.monotonic()<end:time.sleep(.02)
            self.assertEqual(self.engine.previews,2)
