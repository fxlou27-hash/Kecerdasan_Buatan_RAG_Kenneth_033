from pathlib import Path
import random
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# ---------- Load & split ----------
def baca_pdf(path):
    reader = PdfReader(path)
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def potong(teks, ukuran=800, tumpang=100):
    hasil, mulai = [], 0
    while mulai < len(teks):
        hasil.append(teks[mulai:mulai + ukuran])
        mulai += ukuran - tumpang
    return [h for h in hasil if h.strip()]

# ---------- Embedding & DB ----------
emb = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./chroma_db")

def bangun_koleksi(nama, ukuran, tumpang):
    koleksi = client.get_or_create_collection(
        name=nama, embedding_function=emb,
        metadata={"hnsw:space": "cosine"})
    for pdf in Path("dokumen").glob("*.pdf"):
        chunks = potong(baca_pdf(pdf), ukuran, tumpang)
        koleksi.upsert(
            ids=[f"{pdf.stem}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"sumber": pdf.name, "chunk": i} for i in range(len(chunks))],
        )
        print(f"[{nama}] {pdf.name} → {len(chunks)} chunk")
    return koleksi

# ---------- Query ----------
def cari(koleksi, pertanyaan, k=3):
    hasil = koleksi.query(query_texts=[pertanyaan], n_results=k)
    print(f"\nPertanyaan: {pertanyaan}")
    for doc, meta, jarak in zip(hasil["documents"][0], hasil["metadatas"][0], hasil["distances"][0]):
        print(f"[{meta['sumber']} · chunk {meta['chunk']}] jarak={jarak:.3f}")
        print(doc[:200], "...\n")

# ---------- Evaluasi ----------
def hit_at_k(koleksi, uji, k=3, tampil_gagal=False):
    benar = 0
    for pertanyaan, kunci in uji:
        docs = koleksi.query(query_texts=[pertanyaan], n_results=k)["documents"][0]
        gabung = [" ".join(d.split()).lower() for d in docs]
        if any(kunci.lower() in d for d in gabung):
            benar += 1
        elif tampil_gagal:
            print(f"   GAGAL (k={k}): {pertanyaan}")
    return benar / len(uji)


# LAB 1
def lab1():
    pdf = sorted(Path("dokumen").glob("*.pdf"))[0]
    print("File:", pdf.name)
    teks = baca_pdf(pdf)
    print("Total karakter:", len(teks))
    print("\n=== 500 karakter pertama ===")
    print(teks[:500])

    for ukuran in (500, 1000, 1500):
        tumpang = int(ukuran * 0.15)
        chunks = potong(teks, ukuran, tumpang)
        print(f"\n##### ukuran {ukuran}, overlap {tumpang} → {len(chunks)} chunk #####")
        for c in random.sample(chunks, 3):
            print("\n-----")
            print(c)


# LAB 2
def lab2():
    koleksi = bangun_koleksi("uji_1000", ukuran=1000, tumpang=150)

    # 5 pertanyaan biasa
    cari(koleksi, "Berapa sks mata kuliah Grafika Komputer?")
    cari(koleksi, "Apa saja CPMK mata kuliah Pemrograman Web Lanjut?")
    cari(koleksi, "Bagaimana ketentuan lama dan SKS kegiatan magang?")
    cari(koleksi, "Apa itu program MBKM di Informatika?")
    cari(koleksi, "Apa prasyarat mata kuliah Kewirausahaan dan Desain Inovasi?")

    # 2 pertanyaan parafrase (kata berbeda dari teks buku)
    cari(koleksi, "Mata kuliah yang mengajarkan membuat gambar tiga dimensi di komputer")
    cari(koleksi, "Belajar di luar prodi lewat kampus merdeka")


# LAB 3
def lab3():
    UJI = [
        # fakta singkat / kode
        ("Apa kode mata kuliah Grafika Komputer?", "INF254"),
        ("Apa kode mata kuliah Kewirausahaan dan Desain Inovasi?", "INF416"),
        ("Apa mata kuliah prasyarat Pemrograman Web Lanjut?", "(P) Pemrograman Web"),
        ("Apa alamat kampus Kalijudan?", "Kalijudan 37"),
        # nama / istilah
        ("Algoritma kurva apa yang dipelajari di Grafika Komputer?", "NURBS"),
        ("Siapa penulis buku acuan wajib Grafika Komputer?", "Edward Angel"),
        ("Teknologi apa untuk visualisasi interaktif multi-platform?", "WebGL"),
        ("Siapa penulis buku acuan wajib Digital Product Management?", "Brennan"),
        ("Siapa penulis buku Computer Organization and Design?", "Patterson"),
        ("Siapa penulis Kuliah Agama Katolik di Perguruan Tinggi Umum?", "Ismartono"),
        # tabel / isi bagian
        ("Berapa lama dan SKS kegiatan magang?", "Tabel 3.5"),
        ("Apa CPMK Kewirausahaan tentang presentasi bisnis?", "kanvas model bisnis"),
        # parafrase (kata beda dari buku)
        ("Mata kuliah yang membahas cahaya dan bayangan pada gambar 3D", "Ray Tracing"),
        ("Bagaimana mahasiswa bisa belajar di luar prodi sendiri?", "studi lain"),
        ("Program kampus merdeka di Informatika itu apa?", "Merdeka"),
    ]
    k500 = bangun_koleksi("uji_500", ukuran=500, tumpang=75)
    k1500 = bangun_koleksi("uji_1500", ukuran=1500, tumpang=225)
    for nama, kol in (("500", k500), ("1500", k1500)):
        print(f"\n=== chunk {nama} ===")
        for k in (1, 3, 5):
            print(f"Hit@{k} = {hit_at_k(kol, UJI, k, tampil_gagal=(k == 5)):.2f}")


# PILIH LAB YANG DIJALANKAN: ubah angkanya (1, 2, atau 3)
LAB = 3

if __name__ == "__main__":
    if LAB == 1:
        lab1()
    elif LAB == 2:
        lab2()
    elif LAB == 3:
        lab3()