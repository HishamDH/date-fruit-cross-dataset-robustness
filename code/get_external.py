"""List and download the external date-fruit datasets used for the second-dataset and cross-dataset tests.

Usage:
    python get_external.py list        # print the file lists and sizes reported by the hosting sites
    python get_external.py soda        # Hugging Face: gfbati/soda
    python get_external.py mendeley    # Mendeley Data: w85f8zsdhw, version 2
    python get_external.py uci         # UCI: Ajwa or Medjool (dataset 879)
Everything is stored under ../_data/external/. No account or key is used.
"""
import json
import os
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(os.path.dirname(HERE), '_data', 'external')
UA = {'User-Agent': 'dataset-downloader/1.0'}
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def get(url, binary=False):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read() if binary else r.read().decode('utf-8', 'replace')


def fetch(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return os.path.getsize(path)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(path + '.part', 'wb') as fh:
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            fh.write(buf)
    os.replace(path + '.part', path)
    return os.path.getsize(path)


def soda_tree():
    out, url = [], 'https://huggingface.co/api/datasets/gfbati/soda/tree/main?recursive=true&limit=1000'
    while url:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            out += json.loads(r.read())
            link = r.headers.get('Link') or ''
        url = link[link.index('<') + 1:link.index('>')] if 'rel="next"' in link else None
    return out


def cmd_list():
    try:
        info = json.loads(get('https://huggingface.co/api/datasets/gfbati/soda'))
        print('SODA card:', {k: info.get(k) for k in ('id', 'lastModified', 'gated', 'private')}, '| licence tags:',
              [t for t in info.get('tags', []) if t.startswith('license')])
        tree = soda_tree()
        files = [t for t in tree if t['type'] == 'file']
        print('SODA files:', len(files), 'total MB %.1f' % (sum(f.get('size', 0) for f in files) / 2 ** 20))
        tops = {}
        for f in files:
            top = '/'.join(f['path'].split('/')[:2])
            tops[top] = tops.get(top, 0) + 1
        for k, v in sorted(tops.items())[:60]:
            print('   %5d  %s' % (v, k))
    except Exception as e:  # noqa: BLE001
        print('SODA error', repr(e)[:200])
    try:
        d = json.loads(get('https://data.mendeley.com/public-api/datasets/w85f8zsdhw?version=2'))
        print('MENDELEY:', d.get('name'), '| licence:', (d.get('data_licence') or {}).get('short_name'), '| published:', d.get('publish_date'))
        fl = json.loads(get('https://data.mendeley.com/public-api/datasets/w85f8zsdhw/files?folder_id=root&version=2'))
        print('   root files:', [(f.get('filename'), f.get('size')) for f in fl][:20])
        fo = json.loads(get('https://data.mendeley.com/public-api/datasets/w85f8zsdhw/folders/2'))
        print('   folders:', [(f.get('name'), f.get('id')) for f in fo][:40])
    except Exception as e:  # noqa: BLE001
        print('MENDELEY error', repr(e)[:200])
    try:
        req = urllib.request.Request('https://archive.ics.uci.edu/static/public/879/ajwa+or+medjool.zip', headers=UA, method='HEAD')
        with urllib.request.urlopen(req, timeout=60) as r:
            print('UCI zip:', r.status, r.headers.get('Content-Length'), r.headers.get('Content-Type'))
    except Exception as e:  # noqa: BLE001
        print('UCI error', repr(e)[:200])


def cmd_soda():
    os.environ.setdefault('HF_HOME', os.path.join(EXT, '_hf_home'))       # keep the Hugging Face cache on this drive
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    from huggingface_hub import snapshot_download
    dst = os.path.join(EXT, 'soda')
    snapshot_download(repo_id='gfbati/soda', repo_type='dataset', local_dir=dst, max_workers=4)
    n, size = 0, 0
    for d, _, files in os.walk(dst):
        if '.cache' in d:
            continue
        for f in files:
            n += 1
            size += os.path.getsize(os.path.join(d, f))
    print('SODA DONE', n, 'files', '%.0f MB' % (size / 2 ** 20), flush=True)


def cmd_mendeley_probe():
    for url in ('https://data.mendeley.com/public-api/datasets/w85f8zsdhw?version=2',
                'https://data.mendeley.com/public-api/datasets/w85f8zsdhw/files?folder_id=root&version=2',
                'https://data.mendeley.com/public-api/datasets/w85f8zsdhw/folders/2'):
        try:
            print(url, '->', get(url)[:700].replace('\n', ' '))
        except Exception as e:  # noqa: BLE001
            print(url, 'ERROR', repr(e)[:200])


def cmd_mendeley():
    def walk(folder_id, rel):
        fl = json.loads(get('https://data.mendeley.com/public-api/datasets/w85f8zsdhw/files?folder_id=%s&version=2' % folder_id))
        n = 0
        for f in fl:
            url = (f.get('content_details') or {}).get('download_url')
            if url:
                fetch(url, os.path.join(EXT, 'mendeley_w85f8zsdhw', rel, f['filename']))
                n += 1
        return n
    folders = json.loads(get('https://data.mendeley.com/public-api/datasets/w85f8zsdhw/folders/2'))
    by_id = {f['id']: f for f in folders}

    def path_of(f):
        parts = [f['name']]
        while f.get('parent_id') in by_id:
            f = by_id[f['parent_id']]
            parts.append(f['name'])
        return os.path.join(*reversed(parts))
    total = walk('root', '')
    for f in folders:
        rel = path_of(f)
        if 'augment' in rel.lower():
            continue                      # the augmented copies are not needed
        k = walk(f['id'], rel)
        total += k
        print('mendeley', rel, k, flush=True)
    print('MENDELEY DONE', total, 'files', flush=True)


KAGGLE = {
    'soda_kaggle': 'gfbati/soda-saudi-arabian-organic-date-fruits-dataset',      # full SODA (the Hugging Face copy is partial)
    'medina27': 'firasaalsaidi/medina-27-dates-hackathon-dataset-v1',
}


def cmd_kaggle():
    """Public Kaggle datasets download without an account through the public API endpoint."""
    for name in sys.argv[2:] or list(KAGGLE):
        p = os.path.join(EXT, name, name + '.zip')
        try:
            n = fetch('https://www.kaggle.com/api/v1/datasets/download/' + KAGGLE[name], p)
            import zipfile
            print('KAGGLE DONE', name, n, 'bytes', 'zip ok:', zipfile.is_zipfile(p), flush=True)
        except Exception as e:  # noqa: BLE001
            print('KAGGLE ERROR', name, repr(e)[:200], flush=True)
    print('KAGGLE ALL DONE', flush=True)


def cmd_uci():
    p = os.path.join(EXT, 'uci_ajwa_or_medjool', 'ajwa_or_medjool.zip')
    print('UCI DONE', fetch('https://archive.ics.uci.edu/static/public/879/ajwa+or+medjool.zip', p), 'bytes', flush=True)


def cmd_zenodo():
    """Alsirhani et al., "Date Fruit Dataset", Zenodo record 4639543 (CC BY 4.0): single files, no class folders."""
    rec = json.loads(get('https://zenodo.org/api/records/4639543'))
    files = rec.get('files', [])
    print('ZENODO', rec.get('metadata', {}).get('title'), '| licence:', (rec.get('metadata', {}).get('license') or {}).get('id'), '| files', len(files),
          '| MB %.0f' % (sum(f.get('size', 0) for f in files) / 2 ** 20), flush=True)
    done = 0
    for i, f in enumerate(files):
        url = f.get('links', {}).get('self') or f.get('links', {}).get('download')
        name = os.path.basename(f.get('key') or f.get('filename'))
        for attempt in range(3):
            try:
                fetch(url, os.path.join(EXT, 'alsirhani', name))
                done += 1
                break
            except Exception as e:  # noqa: BLE001
                print('retry', name, repr(e)[:120], flush=True)
        if (i + 1) % 50 == 0:
            print('zenodo', i + 1, 'of', len(files), flush=True)
    print('ZENODO DONE', done, 'of', len(files), 'files', flush=True)


if __name__ == '__main__':
    {'list': cmd_list, 'soda': cmd_soda, 'mendeley': cmd_mendeley, 'uci': cmd_uci, 'mendeley-probe': cmd_mendeley_probe,
     'kaggle': cmd_kaggle, 'zenodo': cmd_zenodo}[sys.argv[1]]()
