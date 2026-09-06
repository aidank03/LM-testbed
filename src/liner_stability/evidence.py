"""Local, explicit-file lexical retrieval with stable, versioned evidence IDs."""
from collections import Counter
import hashlib
import math
from pathlib import Path
import re

STOP = set("a an and are as at be by can could do does for from has have how i in is it of on or our should that the their this to us was we what when which will with would you".split())


def tokens(text):
    return [x for x in re.findall(r"[a-z0-9]+", text.lower()) if x not in STOP and len(x)>1]


def index_files(paths):
    chunks, seen, files = [], set(), []
    for name in paths:
        path = Path(name).resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.suffix.lower() not in (".md", ".txt") or not path.is_file():
            raise ValueError("Evidence indexing accepts explicitly named .md or .txt files")
        raw = path.read_bytes()
        if len(raw) > 2_000_000:
            raise ValueError("Split evidence files larger than 2 MB before indexing")
        text = raw.decode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()
        files.append({"path": str(path), "sha256": sha})
        lines = text.splitlines()
        begin, current = 1, []
        for number, line in enumerate(lines + [""], 1):
            if current and (not line.strip() or sum(map(len,current))+len(line)>1800):
                value = "\n".join(current)
                identifier = hashlib.sha256((sha+":"+str(begin)).encode()).hexdigest()[:20]
                # A heading alone is not usable evidence. Keep the source line
                # offsets on substantive passages rather than retrieving titles.
                if any(not re.match(r"^\s*#{1,6}\s", v) for v in current):
                    chunks.append({"evidence_id": "ev_"+identifier, "source_path": str(path), "source_sha256": sha,
                                   "line_start": begin, "line_end": number-1, "text": value})
                current = []
            if line.strip():
                if not current:
                    begin = number
                current.append(line)
    if not chunks:
        raise ValueError("No evidence text was found")
    return {"schema_version": "1", "method": "lexical_tfidf", "files": files, "chunks": chunks,
            "limitations": "Retrieval is not source verification or claim entailment. No papers are fetched automatically."}


def search(index, query, limit=5):
    if not isinstance(query, str) or not query.strip() or not 1 <= limit <= 20:
        raise ValueError("A nonempty query and limit between 1 and 20 are required")
    if index.get("schema_version") != "1" or not isinstance(index.get("chunks"), list):
        raise ValueError("Invalid evidence index")
    chunks = index["chunks"]
    q = Counter(tokens(query))
    counts = [Counter(tokens(c["text"])) for c in chunks]
    df = Counter(t for count in counts for t in count)
    idf = {t: 1+math.log((1+len(chunks))/(1+f)) for t, f in df.items()}
    qvec = {t: v*idf.get(t,0) for t,v in q.items()}
    qnorm = math.sqrt(sum(v*v for v in qvec.values()))
    result = []
    if not qnorm:
        return []
    for c, count in zip(chunks, counts):
        vec = {t: (1+math.log(v))*idf[t] for t,v in count.items()}
        dot = sum(v*vec.get(t,0) for t,v in qvec.items())
        norm = math.sqrt(sum(v*v for v in vec.values()))
        if dot and norm:
            result.append({**c, "retrieval_score": dot/(norm*qnorm)})
    return sorted(result, key=lambda c:(-c["retrieval_score"],c["evidence_id"]))[:limit]


def verify_references(answer, passages):
    allowed = {p["evidence_id"] for p in passages}
    refs = answer.get("evidence_ids")
    if not isinstance(refs,list) or not all(isinstance(x,str) for x in refs):
        raise ValueError("evidence_ids must be a list of strings")
    if set(refs)-allowed:
        raise ValueError("Model cited evidence IDs that were not supplied")
    return {"reference_ids_valid": True, "claim_entailment_checked": False}
