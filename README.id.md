# IndoNLU 
![Ditunggu Pull Requestsnya](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat) [![Lisensi Github](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/indobenchmark/indonlu/blob/master/LICENSE) [![Perjanjian Kontributor](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

*Read this README in [English](README.md).*

<b>IndoNLU</b> adalah sebuah koleksi sumber untuk riset dalam topik Natural Language Understanding (NLU) untuk Bahasa Indonesia dengan 12 aplikasi. Kami menyediakan kode untuk mereproduksi hasil dan model besar yang sudah dilatih sebelumnya (<b>IndoBERT</b> and <b>IndoBERT-lite</b>) yang dilatih dengan kumpulan tulisan berisi sekitar 4 miliar kata (<b>Indo4B</b>) dan lebih dari 20 GB dalam ukuran data teks. Proyek ini awalnya dimulai dari kerjasama antara universitas dan industri, seperti Institut Teknologi Bandung, Universitas Multimedia Nusantara, The Hong Kong University of Science and Technology, Universitas Indonesia, Gojek, dan Prosa.AI.

## Makalah Penelitian
IndoNLU telah diterima oleh AACL-IJCNLP 2020 dan Anda dapat menemukan detailnya di paper kami https://www.aclweb.org/anthology/2020.aacl-main.85.pdf.
Jika Anda menggunakan komponen apa pun di IndoNLU termasuk Indo4B, FastText-Indo4B, atau IndoBERT dalam pekerjaan Anda, harap kutip makalah berikut:

```
@inproceedings{wilie2020indonlu,
  title={IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
  author={Bryan Wilie and Karissa Vincentio and Genta Indra Winata and Samuel Cahyawijaya and X. Li and Zhi Yuan Lim and S. Soleman and R. Mahendra and Pascale Fung and Syafri Bahar and A. Purwarianti},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  year={2020}
}
```

## Bagaimana cara untuk berkontribusi ke IndoNLU?
Pastikan anda mengecek [pedoman kontribusi](https://github.com/indobenchmark/indonlu/blob/master/CONTRIBUTING.md) dan hubungi pengelola atau buka issue untuk mengumpulkan umpan balik sebelum memulai PR Anda.

## 12 Aplikasi
- Cek disini: [[Tautan]](https://github.com/indobenchmark/indonlu/tree/master/dataset)
- Kami menyediakan train, valid, dan test set. Label set pengujian disamarkan (tidak ada label sebenarnya) untuk menjaga integritas evaluasi. Silakan kirim prediksi Anda ke portal pengiriman di [CodaLab](https://competitions.codalab.org/competitions/26537)

### Contoh
- Panduan untuk memuat model IndoBERT dan menyempurnakan model pada tugas Sequence Classification dan Sequence Tagging.
- Cek disini: [tautan](https://github.com/indobenchmark/indonlu/tree/master/examples)

### Susunan Pengiriman
Dimohon untuk memeriksa [tautan ini] (https://github.com/indobenchmark/indonlu/tree/master/submission_examples). Untuk setiap tugas, ada format yang berbeda. Setiap file pengiriman selalu dimulai dengan kolom `index` (id sampel pengujian mengikuti urutan set pengujian yang disamarkan).

Untuk pengiriman, pertama-tama Anda perlu mengganti nama prediksi Anda menjadi `pred.txt`, lalu membuat file menjadi zip. Setelah itu, Anda perlu mengizinkan sistem untuk menghitung hasilnya. Anda dapat dengan mudah memeriksa kemajuan anda di tab `hasil` Anda.

## Indo4B Dataset
Kami menyediakan akses ke kumpulan data pra-pelatihan kami yang besar. Dalam versi ini, kami mengecualikan semua tweet Twitter karena pembatasan Kebijakan dan Perjanjian Pengembang Twitter.
- Indo4B Dataset (23 GB tidak dikompresi, 5.6 GB dikompresi) [[Link]](https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/dataset/preprocessed/dataset_wot_uncased_blanklines.tar.xz)

## Model IndoBERT dan IndoBERT-lite
Kami menyediakan 4 Model IndoBERT dan IndoBERT-lite yang sudah dilatih terlebih dahulu [[Link]](https://huggingface.co/indobenchmark)
- IndoBERT-base
  - Fase 1  [[Tautan]](https://huggingface.co/indobenchmark/indobert-base-p1)
  - Fase 2  [[Tautan]](https://huggingface.co/indobenchmark/indobert-base-p2)
- IndoBERT-large
  - Fase 1  [[Tautan]](https://huggingface.co/indobenchmark/indobert-large-p1)
  - Fase 2  [[Tautan]](https://huggingface.co/indobenchmark/indobert-large-p2)
- IndoBERT-lite-base
  - Fase 1  [[Tautan]](https://huggingface.co/indobenchmark/indobert-lite-base-p1)
  - Fase 2  [[Tautan]](https://huggingface.co/indobenchmark/indobert-lite-base-p2)
- IndoBERT-lite-large
  - Fase 1  [[Tautan]](https://huggingface.co/indobenchmark/indobert-lite-large-p1)
  - Fase 2  [[Tautan]](https://huggingface.co/indobenchmark/indobert-lite-large-p2)

## FastText (Indo4B)
Kami menyediakan file model FastText lengkap tanpa pengubahan pengkapitalan huruf (11,9 GB) dan file vektor yang bersesuaian (3,9 GB)
- Model FastText (11.9 GB) [[Tautan]](https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/models/fasttext/fasttext.4B.id.300.epoch5.uncased.bin) 
- File Vector (3.9 GB) [[Tautan]](https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/models/fasttext/fasttext.4B.id.300.epoch5.uncased.vec.zip)

Kami menyediakan model FastText yang lebih kecil dengan kosakata yang lebih kecil untuk masing-masing dari 12 aplikasi
- FastText-Indo4B [[Tautan]](https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/models/fasttext/fasttext-4B-id-uncased.zip)
- FastText-CC-ID [[Tautan]](https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/models/fasttext/fasttext-cc-id.zip)

## Papan Peringkat
- Portal Komunitas dan Public Papan Peringkat Publik [[Tautan]](https://www.indobenchmark.com/leaderboard.html)
- Portal Pengiriman https://competitions.codalab.org/competitions/26537
