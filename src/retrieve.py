"""Cache the official archive and release only the required anonymous measurements."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
URL='https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip'

def main():
    data=ROOT/'data';data.mkdir(exist_ok=True)
    archive=data/'source.zip'
    if not archive.exists():
        request=urllib.request.Request(URL,headers={'User-Agent':'PublicResearchPortfolio/1.0 (https://github.com/S3R3Bii)'})
        with urllib.request.urlopen(request,timeout=180) as response:
            archive.write_bytes(response.read())
    manifest=[]
    with zipfile.ZipFile(archive) as outer:
        inner_name=next((n for n in outer.namelist() if n.endswith('UCI HAR Dataset.zip')),None)
        if inner_name:
            import io
            selected=zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
        else:selected=outer
        for name in selected.namelist():
            parts=name.split('/')
            if name.endswith('/') or name.startswith('__MACOSX'):continue
            relative='/'.join(parts[1:])
            if not (relative in ['README.txt','activity_labels.txt'] or
                    any(relative==f'{part}/{kind}_{part}.txt' for part in ['train','test'] for kind in ['subject','y']) or
                    ('/Inertial Signals/' in relative and relative.endswith('.txt'))):continue
            path=data/relative;path.parent.mkdir(parents=True,exist_ok=True)
            blob=selected.read(name);path.write_bytes(blob)
            manifest.append({'file':path.relative_to(ROOT).as_posix(),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
    record={'url':URL,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'license':'CC BY 4.0','doi':'10.24432/C54S4K','source_landing':'https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones','files':manifest}
    (data/'manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(manifest),'archive_bytes':archive.stat().st_size,'archive_sha256':record['archive_sha256']}))

if __name__=='__main__':main()
