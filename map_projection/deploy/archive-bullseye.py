"""Keep Debian 11 apt sources usable without disabling signature verification."""
from pathlib import Path
import re
for path in [Path('/etc/apt/sources.list'),*Path('/etc/apt/sources.list.d').glob('*.list')]:
 if not path.exists():continue
 old=path.read_text();new=old
 for host in ('deb.debian.org','security.debian.org','ftp.debian.org'):
  new=new.replace('http://'+host+'/debian','https://archive.debian.org/debian').replace('https://'+host+'/debian','https://archive.debian.org/debian')
 lines=[]
 for line in new.splitlines():
  if re.match(r'^\s*deb\s+',line) and 'archive.debian.org' in line:
   if '[' in line:
    if 'check-valid-until=' not in line:line=line.replace('[','[check-valid-until=no ',1)
   else:line=re.sub(r'^(\s*deb\s+)',r'\1[check-valid-until=no] ',line)
  lines.append(line)
 new='\n'.join(lines)+'\n'
 if new!=old:
  backup=path.with_suffix(path.suffix+'.before-projector')
  if not backup.exists():backup.write_text(old)
  path.write_text(new)
