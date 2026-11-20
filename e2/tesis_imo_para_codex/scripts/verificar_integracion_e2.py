"""Control del informe compilado y paquete local de revisión; no publica."""
import hashlib
import json
import re
import zipfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote

P=Path(__file__).resolve().parents[1]; B=P/'libro'; H=B/'_build/html'
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('src','href') and value:self.links.append(value)

def main():
    issues=[];checked=0
    pages=['index','problema','eda','literatura','modelos_e2','metodos_e2','resultados_e2',
           'cierre_e2','checklist_e2','reproducibilidad_e2','implementacion','resultados']
    for name in pages:
        file=H/(name+'.html'); parser=Links();parser.feed(file.read_text(encoding='utf-8'))
        for link in parser.links:
            u=urlsplit(link)
            if u.scheme or u.netloc or not u.path:continue
            checked+=1; target=(file.parent/unquote(u.path)).resolve()
            if not target.exists():issues.append({'page':name,'missing':link})
    text=(B/'index.md').read_text(encoding='utf-8')
    abstract=text.split('## Resumen\n')[1].split('## Estado')[0]
    words=len(abstract.split())
    if words>250:issues.append({'abstract_words':words})
    checklist=(P/'docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md').read_text(encoding='utf-8')
    mainrows=re.findall(r'^\| (\d{2}) \|.*?\| (C|P|F|D|NA) \|',checklist,re.M)
    brows=re.findall(r'^\| (B\d{2}) .*?\| (C|P|F|D|NA) \|',checklist,re.M)
    if len(mainrows)!=76 or len(brows)!=21:issues.append({'checklist_counts':[len(mainrows),len(brows)]})
    diagrams=list((B/'figuras/e2').glob('diagrama_*.png'))
    if len(diagrams)!=10:issues.append({'diagrams':len(diagrams)})
    report={'issues':issues,'local_links_checked':checked,'abstract_words':words,'model_diagrams':len(diagrams),
      'checklist_items':len(mainrows),'checklist_states':dict(Counter(state for _,state in mainrows)),
      'appendix_B_items':len(brows),'scope':'Artifact checks, not certification of missing scientific requirements.'}
    target=P/'resultados/integracion_e2/VERIFICACION.json'
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    assert not issues, issues
    package=P.parent/'Entregable2_integrado_revision_local.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(H.rglob('*')):
            if file.is_file():z.write(file,'libro_html/'+file.relative_to(H).as_posix())
        z.write(P/'docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md','CHECKLIST_ENTREGA_E2.md')
        z.write(target,'VERIFICACION.json')
        z.writestr('LEEME.txt','REVISION LOCAL, NO ENTREGA FINAL COMPLETA.\nExtraer y abrir libro_html/index.html.\nConsultar checklist para requisitos pendientes. No incluye pesos ni datasets completos.\nContiene figuras académicas locales; revisar licencias antes de redistribuir/publicar.\n')
    with zipfile.ZipFile(package) as z:assert z.testzip() is None
    print(json.dumps({**report,'zip':str(package),'zip_bytes':package.stat().st_size,
                     'zip_sha256':hashlib.sha256(package.read_bytes()).hexdigest()},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
