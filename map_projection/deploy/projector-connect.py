#!/usr/bin/python3
"""SSH transport: stable IPv6 link-local first, mDNS fallback; host key checked by SSH."""
import json,os,selectors,socket,subprocess,sys
port=int(sys.argv[1]);remote=None
links=json.loads(subprocess.check_output(['ip','-j','link','show']))
for link in links:
 name=link['ifname']
 if link.get('operstate')!='UP' or not os.path.isdir('/sys/class/net/'+name+'/wireless'):continue
 candidate=socket.socket(socket.AF_INET6,socket.SOCK_STREAM);candidate.settimeout(2)
 try:candidate.connect(('fe80::9230:d6ff:fe03:56a4',port,0,socket.if_nametoindex(name)));remote=candidate;break
 except OSError:candidate.close()
if remote is None:
 try:remote=socket.create_connection(('projector.local',port),timeout=5)
 except OSError as exc:print('Projector unavailable on this local network: '+str(exc),file=sys.stderr);sys.exit(1)
remote.settimeout(None)
selector=selectors.DefaultSelector();selector.register(remote,selectors.EVENT_READ,'remote');selector.register(sys.stdin.buffer,selectors.EVENT_READ,'stdin')
while True:
 for event,_ in selector.select():
  if event.data=='remote':
   data=remote.recv(65536)
   if not data:sys.exit(0)
   sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
  else:
   data=os.read(sys.stdin.fileno(),65536)
   if data:remote.sendall(data)
   else:selector.unregister(sys.stdin.buffer);remote.shutdown(socket.SHUT_WR)
