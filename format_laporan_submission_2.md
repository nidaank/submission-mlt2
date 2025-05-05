# Laporan Proyek Machine Learning - Nida'an Khafiyya

## Project Overview

Seiring dengan meningkatnya jumlah buku digital yang tersedia di berbagai platform, pengguna semakin sulit menemukan bacaan yang sesuai dengan preferensi mereka. Hal ini menciptakan fenomena *information overload* yang menghambat akses pengguna terhadap materi yang relevan dan bermanfaat, terutama di lingkungan pendidikan tinggi. Sistem rekomendasi muncul sebagai solusi yang efektif untuk membantu pengguna menavigasi koleksi besar dengan menyajikan saran yang personal dan relevan. Namun, tantangan utama dalam sistem ini adalah bagaimana memahami preferensi pengguna dan karakteristik buku secara efisien, khususnya ketika data eksplisit seperti rating atau ulasan terbatas. Maka dari itu, pengembangan sistem rekomendasi yang adaptif menjadi penting untuk mengatasi masalah tersebut dan meningkatkan keterlibatan pengguna.

Penelitian sebelumnya oleh Ahmed dan Letta menunjukkan bahwa pendekatan berbasis *collaborative filtering* seperti matrix factorization (SVD) mampu menangani permasalahan klasik sistem rekomendasi, seperti *cold start* dan *data sparsity*, dengan hasil akurasi yang menjanjikan (RMSE 0.1623). Studi tersebut juga menyoroti kelemahan pendekatan konvensional seperti *association rule mining* dan *case-based reasoning* yang menghasilkan daftar item yang statis dan kurang responsif terhadap kebutuhan pengguna baru [[1]](https://onlinelibrary.wiley.com/doi/10.1155/2023/1514801). Oleh karena itu, penting untuk mengeksplorasi penggabungan pendekatan konten dan kolaboratif guna menghasilkan rekomendasi yang lebih kontekstual, adaptif, dan mampu memberikan pengalaman yang lebih personal kepada pengguna, khususnya dalam domain perpustakaan digital pendidikan.

Referensi: [Book Recommendation Using Collaborative Filtering Algorithm](https://onlinelibrary.wiley.com/doi/10.1155/2023/1514801)

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah:
- Bagaimana cara membantu pengguna menemukan buku yang sesuai dengan preferensi mereka, terutama ketika jumlah koleksi buku sangat besar dan membuat proses pencarian menjadi kurang efisien?
- Bagaimana cara membangun sistem rekomendasi yang lebih personal dan relevan, yang mampu mempertimbangkan pola interaksi serta preferensi individual pengguna secara lebih mendalam?
- Bagaimana cara mengatasi cold-start problem, yaitu kondisi di mana pengguna baru atau buku baru belum memiliki riwayat interaksi atau rating, sehingga tetap memungkinkan sistem memberikan rekomendasi yang akurat?

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Mengembangkan sistem rekomendasi berbasis content-based filtering yang dapat merekomendasikan lima buku serupa berdasarkan informasi penulis buku. Sistem ini akan bekerja dengan mengukur kesamaan antar buku untuk membantu pengguna mengeksplorasi bacaan yang sejenis.
- Membangun sistem rekomendasi berbasis collaborative filtering yang mampu merekomendasikan sepuluh buku baru kepada pengguna berdasarkan histori interaksi pengguna dengan buku-buku sebelumnya. Sistem ini akan memanfaatkan data rating untuk mempelajari pola preferensi pengguna.
- Mengatasi masalah cold-start pada pengguna atau buku baru dengan mengandalkan pendekatan berbasis konten, sehingga tetap memungkinkan sistem memberikan rekomendasi meskipun belum tersedia data eksplisit seperti rating.

### Solution statements

- Content-Based Filtering: Menggunakan pendekatan TF-IDF untuk merepresentasikan metadata buku seperti nama penulis dan nama penerbit dalam bentuk vektor numerik, menghitung kemiripan antar buku menggunakan cosine similarity untuk merekomendasikan lima buku yang mirip dengan buku yang disukai pengguna, dan sangat efektif dalam mengatasi kasus cold-start karena tidak bergantung pada data interaksi pengguna.
- Collaborative Filtering dengan RecommenderNet: Menerapkan RecommenderNet, model deep learning yang mempelajari representasi laten dari pengguna dan buku berdasarkan data interaksi berupa rating, untuk memprediksi skor relevansi buku yang belum dibaca dan merekomendasikan sepuluh buku dengan skor tertinggi, sehingga mampu memberikan rekomendasi yang personal dan akurat berdasarkan pola perilaku pengguna lain yang serupa.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
- E. Ahmed and A. Letta, “Book Recommendation Using Collaborative Filtering Algorithm,” *Applied Computational Intelligence and Soft Computing*, vol. 2023, Article ID 1514801, 2023. \[Online]. Available: [https://onlinelibrary.wiley.com/doi/10.1155/2023/1514801](https://onlinelibrary.wiley.com/doi/10.1155/2023/1514801)
