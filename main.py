import os
import json
import numpy as np
from Bio import SeqIO
import faiss

# ===== Константы =====
AA = "ACDEFGHIKLMNPQRSTVWY"  # 20 стандартных аминокислот
aa_to_idx = {aa: i for i, aa in enumerate(AA)}
K = 2                       # k=2 вместо 3
VOCAB_SIZE = len(AA) ** K   # 400 признаков

DB_VECTORS = "protein_vectors_k2.npy"
DB_META = "protein_meta.json"
INDEX_FILE = "protein_index_k2.faiss"


# ===== Векторизация последовательности =====
def seq_to_vector(seq, k=K):
    vec = np.zeros(VOCAB_SIZE, dtype=np.float32)  # сначала float32
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if all(c in aa_to_idx for c in kmer):
            idx = 0
            for c in kmer:
                idx = idx * len(AA) + aa_to_idx[c]
            vec[idx] += 1
    # Нормировка (для cosine similarity)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.astype(np.float16)  # храним как float16


# ===== Построение базы =====
def build_db(fasta_file):
    vectors, meta = [], []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        vectors.append(seq_to_vector(seq))
        meta.append({"id": record.id, "seq": seq})
    X = np.vstack(vectors).astype(np.float16)
    return X, meta


def build_index(X):
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X.astype(np.float32))
    return index


# ===== Запрос =====
def query(index, seq, meta, top_k=100):
    qvec = seq_to_vector(seq).reshape(1, -1).astype(np.float32)
    D, I = index.search(qvec, top_k)
    results = [(meta[i]["id"], meta[i]["seq"], float(D[0][j])) for j, i in enumerate(I[0])]
    return results


# ===== MAIN =====
if __name__ == "__main__":
    fasta_file = "uniprot_sprot.fasta"

    if not (os.path.exists(DB_VECTORS) and os.path.exists(DB_META) and os.path.exists(INDEX_FILE)):
        print("\rПервый запуск: строим базу и индекс...", end="", flush=True)

        X, meta = build_db(fasta_file)

        # Сохраняем данные
        np.save(DB_VECTORS, X)
        with open(DB_META, "w") as f:
            json.dump(meta, f)

        index = build_index(X)
        faiss.write_index(index, INDEX_FILE)

    else:
        print("\rЗагружаем готовую базу и индекс...", end="", flush=True)
        X = np.load(DB_VECTORS)
        with open(DB_META, "r") as f:
            meta = json.load(f)
        index = faiss.read_index(INDEX_FILE)

    # Пример входного белка
    input_seq = input("\nВведите белок: ")

    print("\rИщем топ-100 ближайших...\r", end="", flush=True)
    results = query(index, input_seq, meta, top_k=100)
    for rank, (pid, pseq, score) in enumerate(results, 1):
        print(f"{rank:3d}. {pid}  (similarity={score:.4f})")
        print(f"     {pseq}")
