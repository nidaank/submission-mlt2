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
- Mengembangkan sistem rekomendasi berbasis content-based filtering yang dapat merekomendasikan lima buku serupa berdasarkan informasi penulis bukudan penerbit. Sistem ini akan bekerja dengan mengukur kesamaan antar buku untuk membantu pengguna mengeksplorasi bacaan yang sejenis.
- Membangun sistem rekomendasi berbasis collaborative filtering yang mampu merekomendasikan sepuluh buku baru kepada pengguna berdasarkan histori interaksi pengguna dengan buku-buku sebelumnya. Sistem ini akan memanfaatkan data rating untuk mempelajari pola preferensi pengguna.
- Mengatasi masalah cold-start pada pengguna atau buku baru dengan mengandalkan pendekatan berbasis konten, sehingga tetap memungkinkan sistem memberikan rekomendasi meskipun belum tersedia data eksplisit seperti rating.

### Solution statements

- Content-Based Filtering: Menggunakan pendekatan TF-IDF untuk merepresentasikan metadata buku seperti nama penulis dan nama penerbit dalam bentuk vektor numerik, menghitung kemiripan antar buku menggunakan cosine similarity untuk merekomendasikan lima buku yang mirip dengan buku yang disukai pengguna, dan sangat efektif dalam mengatasi kasus cold-start karena tidak bergantung pada data interaksi pengguna.
- Collaborative Filtering dengan RecommenderNet: Menerapkan RecommenderNet, model deep learning yang mempelajari representasi laten dari pengguna dan buku berdasarkan data interaksi berupa rating, untuk memprediksi skor relevansi buku yang belum dibaca dan merekomendasikan 10 buku dengan skor tertinggi, sehingga mampu memberikan rekomendasi yang personal dan akurat berdasarkan pola perilaku pengguna lain yang serupa.

## Data Understanding

Dataset yang digunakan dalam proyek ini bersumber dari **Kaggle**, dengan judul **Book Recommendation Dataset**. Dataset ini terdiri dari tiga tabel utama, yaitu `Books`, `Ratings`, dan `Users`, yang masing-masing disimpan dalam format CSV. Dataset ini dikumpulkan dari interaksi pengguna dengan buku di platform penjualan **Amazon Web Services**.
Adapun rincian data adalah sebagai berikut:

* **Books.csv** terdiri dari **271.360 baris data** dan **8 kolom fitur** yang berisi informasi detail mengenai buku.
* **Ratings.csv** terdiri dari **1.149.780 baris data** dan **3 kolom fitur** yang merepresentasikan penilaian pengguna terhadap buku.
* **Users.csv** terdiri dari **278.858 baris data** dan **3 kolom fitur** mengenai informasi pengguna.

Sumber Dataset: [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

### **Deskripsi Variabel**

**Books.csv**

* `ISBN` : Nomor identifikasi unik untuk setiap buku.
* `Book-Title` : Judul buku.
* `Book-Author` : Nama penulis buku.
* `Year-Of-Publication` : Tahun terbit buku.
* `Publisher` : Nama penerbit buku.
* `Image-URL-S` : URL gambar buku ukuran kecil.
* `Image-URL-M` : URL gambar buku ukuran sedang.
* `Image-URL-L` : URL gambar buku ukuran besar.

**Ratings.csv**

* `User-ID` : ID unik pengguna.
* `ISBN` : Nomor ISBN buku yang diberi rating.
* `Book-Rating` : Skor rating yang diberikan pengguna (rentang 0–10).

**Users.csv**

* `User-ID` : ID unik pengguna.
* `Location` : Lokasi tempat tinggal pengguna.
* `Age` : Usia pengguna.

### **Exploratory Data Analysis (Univariate)**
1. **Dataset: Books**

Dataset `books` berisi informasi mengenai buku-buku yang tersedia di dalam sistem Book-Crossing. Tujuan awal dari eksplorasi ini adalah untuk memahami karakteristik data buku secara umum.

Berikut adalah beberapa informasi statistik awal yang diperoleh:

* Jumlah total entri buku (ISBN): 271.360
* Jumlah judul buku unik (`Book-Title`): 242.135
* Jumlah penulis unik (`Book-Author`): 102.023
* Jumlah penerbit unik (`Publisher`): 16.808
* Jumlah nilai unik pada kolom tahun terbit (`Year-Of-Publication`): 202

**Perbedaan Jumlah ISBN dan Judul Buku**

Ditemukan perbedaan antara jumlah ISBN dan judul buku (Book-Title). Oleh karena itu, kode di bawah ini digunakan untuk melakukan pengecekan jumlah kemunculan setiap judul buku:
```
books['Book-Title'].value_counts()
```
Hasil dari kode tersebut menunjukkan bahwa terdapat beberapa buku dengan judul yang sama. 
![image](https://github.com/user-attachments/assets/9764d074-d0cd-4ca1-b439-9a22affd8e5d)

Sebagai contoh, pada judul "Selected Poems", pencarian dengan judul yang sama menghasilkan beberapa entri buku yang identik pada judul, tetapi berbeda pada ISBN, penulis, penerbit, atau tahun terbit.
| ISBN       | Book-Title     | Book-Author             | Year-Of-Publication | Publisher                             |
| ---------- | -------------- | ----------------------- | ------------------- | ------------------------------------- |
| 081120958X | Selected Poems | William Carlos Williams | 1985                | New Directions Publishing Corporation |
| 0811201465 | Selected Poems | K. Patchen              | 1957                | New Directions Publishing Corporation |
| 0679750800 | Selected Poems | Rita Dove               | 1993                | Vintage Books USA                     |

**Pembersihan Kolom Tahun Terbit**

![image](https://github.com/user-attachments/assets/fa91a3a1-17e5-4a3b-8f0e-4c820fb73f63)

Ketika melakukan eksplorasi terhadap kolom `Year-Of-Publication`, ditemukan bahwa beberapa entri berisi data yang bukan merupakan tahun, seperti nama penerbit (`DK Publishing Inc`, `Gallimard`) atau angka yang tidak logis (misalnya, `0`, `1376`, `2020`, `2050`).

Setelah dilakukan investigasi lebih lanjut, kasus-kasus seperti ini ternyata terjadi karena pergeseran data antar kolom saat proses input. Untuk menjaga integritas data, tiga entri dengan kesalahan input tersebut dihapus.

Kemudian untuk nilai-nilai tahun yang tidak realistis (di luar rentang wajar publikasi, seperti `0` atau di atas `2006`) diubah menjadi nilai kosong (`NaN`) dan diisi dengan rata-rata tahun yang valid menggunakan codingan di bawah ini.

```
books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
books.loc[(books['Year-Of-Publication'] > 2006) | (books['Year-Of-Publication'] == 0),'Year-Of-Publication'] = np.nan

# mengganti NaN dengan nilai rata-rata yearOfPublication
books['Year-Of-Publication'].fillna(round(books['Year-Of-Publication'].mean()), inplace=True)
```
Setelah tahapan tersebut, kolom tahun terbit telah berisi nilai yang realistis. Kemudian, tipe data pada kolom `Year-Of-Publication` diubah dari tipe data `object` menjadi `integer` agar bisa diproses secara numerik.
![image](https://github.com/user-attachments/assets/36e6c91d-e4c5-4f0f-83ab-3a5409e35a9d)

**Missing Values**

Berikut adalah tabel jumlah *missing values* (nilai kosong) dalam dataset `books`:

| Kolom               | Missing Values |
| ------------------- | -------------- |
| Book-Author         | 2              |
| Publisher           | 2              |
| ISBN                | 0              |
| Book-Title          | 0              |
| Year-Of-Publication | 0              |

Hanya ada dua kolom yang memiliki *missing values*, dan jumlahnya sangat kecil sehingga tidak signifikan terhadap keseluruhan dataset.

---

2. **Dataset: Ratings**

Dataset `ratings` mencatat penilaian yang diberikan oleh pengguna terhadap buku. Dataset ini menjadi sangat penting karena akan digunakan untuk analisis preferensi pengguna dan sistem rekomendasi.

Statistik dasar yang ditemukan:

* Jumlah pengguna yang memberikan rating: 105.283
* Jumlah total rating yang tercatat: 340.556
* Jumlah skala rating unik: 11

**Skala Rating**

Rating diberikan dalam skala **0 hingga 10**, dengan total 11 nilai unik. Berikut adalah interpretasi awal:

* **Rating 0** kemungkinan besar berarti tidak ada rating eksplisit yang diberikan oleh pengguna. Ini dikenal sebagai *implicit rating* atau bisa jadi kesalahan input.
* Rating 1–10 menunjukkan penilaian eksplisit, dengan nilai lebih tinggi menandakan tingkat kesukaan yang lebih besar.

Distribusi nilai rating:

| Skor Rating | Keterangan              |
| ----------- | ----------------------- |
| 0           | Tidak diberi / default  |
| 1–3         | Penilaian sangat rendah |
| 4–6         | Penilaian sedang        |
| 7–10        | Penilaian tinggi        |

**Statistik Deskriptif Rating**

Hasil dari `ratings.describe()` memberikan ringkasan sebagai berikut:

* **Count**: 1.149.780
* **Mean**: 2.866
* **Std (standar deviasi)**: 3.854
* **Min**: 0
* **25%**: 0
* **50% (median)**: 0
* **75%**: 8
* **Max**: 10

Nilai tengah (median) berada di 0, mengindikasikan bahwa sebagian besar rating tidak eksplisit (banyak 0). Namun, nilai kuartil atas (75%) adalah 8, yang memperlihatkan bahwa ketika rating diberikan, nilainya cenderung tinggi.

**Missing Values**

Berikut adalah tabel jumlah *missing values* (nilai kosong) dalam dataset `ratings`:

| Kolom               | Missing Values |
| ------------------- | -------------- |
| User-ID         | 0              |
| ISBN           | 0              |
| Book-Rating                | 0              |

Tidak ada missing value dalam dataset ini.

---

3. **Dataset: Users**

Dataset `users` mencatat informasi dasar tentang pengguna, seperti ID, lokasi, dan umur.

Informasi awal yang diperoleh:

* Jumlah user unik: 278.858
* Jumlah lokasi unik (`Location`): 57.339
* Jumlah nilai unik pada kolom `Age`: 166

**Umur Pengguna**

![image](https://github.com/user-attachments/assets/1ec0355e-bfef-4fa7-9b2f-e5557d08938c)

Kolom `Age` memiliki nilai-nilai yang tidak masuk akal seperti `0`, `231`, dan `244`. Maka dilakukan pembersihan dengan cara:

* Umur di bawah 5 tahun atau di atas 90 tahun dianggap tidak valid dan diubah menjadi `NaN`
* Nilai `NaN` kemudian diisi dengan rata-rata umur pengguna yang valid
* Tipe data diubah ke `integer` agar lebih konsisten

Berikut code yang digunakan
```
# Umur di bawah 5 dan di atas 90 tidak masuk akal, maka menggantinya dengan NaN
users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan

# Replacing NaNs with mean
users.Age = users.Age.fillna(users.Age.mean())

# Setting the data type as int
users.Age = users.Age.astype(np.int32)
```

Hasil deskriptif umur pengguna setelah dibersihkan:

| Statistik | Nilai |
| --------- | ----- |
| Rata-rata | 34.4  |
| Median    | 34    |
| Minimum   | 5     |
| Maksimum  | 90    |

Sebagian besar pengguna berada di rentang usia produktif (sekitar 29–35 tahun), menunjukkan demografi utama pengguna platform ini.

**Missing Values**

Berikut adalah tabel jumlah *missing values* (nilai kosong) dalam dataset `users`:

| Kolom               | Missing Values |
| ------------------- | -------------- |
| User-ID         | 0              |
| Location           | 0              |
| Age                | 0              |

Tidak ada missing value dalam dataset ini.

### Visualisasi data
1. **Books**

**Barplot 10 Author Teratas dengan Buku Terbitan Terbanyak**
![image](https://github.com/user-attachments/assets/5e920802-dc0c-40c5-bdf2-2e1640a0ba2f)
Insight:
Berdasarkan barplot yang menampilkan 10 penulis teratas dengan jumlah buku terbitan terbanyak, terlihat jelas bahwa Agatha Christie mendominasi dengan jumlah publikasi yang signifikan, jauh melampaui penulis lainnya. William Shakespeare menempati posisi kedua, diikuti oleh Stephen King di urutan ketiga. Secara keseluruhan, grafik ini memperlihatkan distribusi jumlah buku yang diterbitkan oleh para penulis terkemuka, dengan penurunan bertahap dari penulis dengan publikasi terbanyak hingga penulis di urutan kesepuluh, yaitu Charles Dickens.

**Barplot 10 Publisher Teratas dengan Buku Terbitan Terbanyak**
![image](https://github.com/user-attachments/assets/b1e14994-2f40-44e3-868d-8d151713c194)
Insight:
Barplot ini menyajikan informasi mengenai 10 penerbit teratas berdasarkan jumlah buku yang telah mereka terbitkan. Terlihat bahwa Harlequin menduduki posisi puncak dengan jumlah publikasi yang jauh lebih tinggi dibandingkan penerbit lainnya. Silhouette berada di urutan kedua, diikuti oleh Pocket dan Ballantine Books yang memiliki jumlah publikasi serupa. Secara keseluruhan, grafik ini menggambarkan adanya dominasi yang signifikan dari beberapa penerbit besar dalam industri penerbitan buku, dengan penurunan jumlah publikasi secara bertahap hingga penerbit di urutan kesepuluh, yaitu Warner Books.

2. **Ratings**

**Barplot Pembagian Pemeringkatan Buku**
![image](https://github.com/user-attachments/assets/ccaecce4-7fba-4c7d-89b6-a3150d58b37e)
Insight:
Barplot ini memperlihatkan distribusi peringkat buku, dengan jelas menunjukkan bahwa mayoritas besar peringkat terkonsentrasi pada nilai 0. Hal ini mengindikasikan bahwa terdapat sejumlah besar buku yang belum atau tidak mendapatkan peringkat. Setelah nilai 0, jumlah peringkat secara umum meningkat seiring dengan kenaikan nilai peringkat, mencapai puncaknya pada peringkat 8, kemudian sedikit menurun pada peringkat 9 dan 10. Distribusi ini menyiratkan adanya polarisasi dalam pemberian peringkat, di mana banyak buku tidak dinilai sama sekali, sementara di antara buku yang dinilai, terdapat kecenderungan pemberian peringkat yang lebih tinggi.

3. **Users**

**Barplot Distribusi Umur Pengguna**
![image](https://github.com/user-attachments/assets/fae6290b-d808-44ad-85a9-7ecb7cb44f76)
Insight:
Boxplot ini menyajikan distribusi usia pengguna. Terlihat bahwa sebagian besar data usia terkumpul di sekitar rentang usia 25 hingga 45 tahun, yang ditunjukkan oleh kotak utama. Garis hitam di dalam kotak merepresentasikan median usia. Ekor atau *whisker* pada kedua sisi kotak menunjukkan sebaran data usia di luar kuartil bawah dan atas. Titik di atas *whisker* atas mengindikasikan adanya nilai *outlier*, yaitu satu pengguna dengan usia yang jauh lebih tinggi dibandingkan dengan mayoritas pengguna lainnya.

## Data Preparation

Pada tahap ini, dilakukan berbagai langkah pembersihan dan transformasi data untuk memastikan data yang digunakan bersih, relevan, dan siap untuk proses analisis lebih lanjut maupun pembangunan model sistem rekomendasi. Tahapan yang dilakukan adalah sebagai berikut:

### Penghapusan Missing Value

Setelah dilakukan proses EDA, ditemukan bahwa masih terdapat beberapa nilai kosong (*missing values*) dalam gabungan dataset `booksrate`. Berikut ini adalah jumlah missing value dari masing-masing kolom:

| Kolom               | Missing Values |
| ------------------- | -------------- |
| User-ID             | 0              |
| ISBN                | 0              |
| Book-Rating         | 0              |
| Book-Title          | 0              |
| Book-Author         | 0              |
| Year-Of-Publication | 0              |
| Publisher           | 1              |

Terlihat bahwa hanya ada satu *missing value* pada kolom `Publisher`, dan sisanya sudah bersih. Untuk membersihkan data dari *missing value*, digunakan fungsi `dropna()`:

```python
booksrate_clean = booksrate.dropna()
```

Fungsi ini secara otomatis menghapus baris yang memiliki nilai kosong pada salah satu kolom. Karena hanya ada satu baris yang terpengaruh, maka tidak akan berdampak signifikan terhadap keseluruhan dataset. Dengan menghapus missing value maka dapat menghindari error pada saat melakukan analisis atau pemodelan dan memastikan integritas data tetap terjaga tanpa baris yang tidak lengkap.

### Penghapusan Duplikat Data

Setelah membersihkan *missing value*, langkah selanjutnya adalah memastikan tidak ada data duplikat dalam data buku. Duplikat yang dimaksud di sini adalah data dengan ISBN yang sama muncul lebih dari sekali.

```python
preparation = booksrate_clean.drop_duplicates('ISBN')
```

Kolom `ISBN` (International Standard Book Number) merupakan pengenal unik untuk setiap buku. Dengan menghapus data duplikat berdasarkan kolom ini, kita menjamin bahwa tidak ada entri buku yang redundan dalam data yang akan diproses lebih lanjut. Dengan langkah ini, maka dapat menghindari bias data karena satu buku yang sama muncul lebih dari sekali dan menjaga akurasi dalam sistem rekomendasi yang berbasis konten (content-based filtering), karena sistem akan mengenali setiap buku hanya satu kali.

### Mengonversi Kolom Menjadi List

Setelah data dibersihkan dari duplikat, kolom-kolom penting seperti `ISBN`, `Book-Title`, `Book-Author`, `Year-Of-Publication`, dan `Publisher` kemudian dikonversi ke dalam bentuk list.

```python
book_isbn = preparation['ISBN'].tolist()
book_title = preparation['Book-Title'].tolist()
book_author = preparation['Book-Author'].tolist()
book_year = preparation['Year-Of-Publication'].tolist()
book_publisher = preparation['Publisher'].tolist()
```

Semua list ini memiliki panjang yang sama yaitu 14.930, menandakan bahwa tidak ada data yang hilang dalam proses konversi.Dengan melakukan langkah ini, maka dapat mengubah data menjadi bentuk list memudahkan manipulasi data pada tahap pembuatan *content-based recommender* dan format list lebih fleksibel untuk digunakan dalam pembuatan *dictionary*, *TF-IDF vectorizer*, atau pembuatan fitur metadata gabungan.

### Data Preparation untuk Content-Based Filtering

1. **Pembuatan Data `books_new`**

Langkah awal adalah menyusun *dataframe* baru bernama `books_new`. 
```
# Membuat dictionary untuk data ‘book_isbn’, ‘book_title’, ‘book_author’, dan 'book_publisher'
books_new = pd.DataFrame({
    'isbn': book_isbn,
    'title': book_title,
    'author': book_author,
    'year': book_year,
    'publisher': book_publisher
})
books_new
```

Proses ini melibatkan konversi informasi buku dari format *series* (kemungkinan dari hasil pembacaan *dataset* sebelumnya) menjadi format *list*. *List* ini kemudian digunakan untuk membuat kolom-kolom dalam *dataframe* `books_new`, yang terdiri dari 'isbn', 'book_title', 'book_author', 'year_of_publication', dan 'publisher'. Tujuan dari pembentukan *dataframe* ini adalah untuk memudahkan manipulasi dan transformasi data ke dalam format yang sesuai untuk pemodelan *content-based filtering*.

2. **Penggabungan Fitur Teks**

Untuk merepresentasikan konten setiap buku, informasi dari kolom 'book_author' dan 'publisher' digabungkan menjadi satu kolom teks baru bernama 'combined'. 
```
# Gabungkan author dan publisher menjadi satu kolom string
data['combined'] = data['author'].fillna('') + ' ' + data['publisher'].fillna('')
```

Langkah ini dilakukan dengan mengisi nilai yang hilang (jika ada) pada kedua kolom tersebut dengan string kosong ('') dan kemudian menggabungkannya dengan spasi di antara keduanya. Kolom 'combined' ini akan menjadi dasar untuk perhitungan kemiripan konten antar buku.

**3. Ekstraksi Fitur dengan TF-IDF:**

Teknik *Term Frequency-Inverse Document Frequency* (TF-IDF) digunakan untuk mengekstrak fitur-fitur penting dari teks dalam kolom 'combined'.

* **Inisialisasi `TfidfVectorizer()`**
  ```
  # Inisialisasi TF-IDF
  tf = TfidfVectorizer()
  ```
  Sebuah objek `TfidfVectorizer` dibuat. *Vectorizer* ini akan mengubah teks menjadi matriks representasi numerik, di mana setiap kata dalam korpus akan menjadi sebuah fitur. Bobot TF-IDF diberikan kepada setiap kata dalam setiap dokumen (dalam hal ini, setiap entri di kolom 'combined'), yang mencerminkan seberapa penting kata tersebut dalam dokumen relatif terhadap seluruh korpus.
* **Fit dan Transform ke Kolom 'combined'**
  ```
  # Fit dan transform ke kolom gabungan
  tfidf_matrix = tf.fit_transform(data['combined'])
  ```
  Metode `fit_transform()` dari objek `TfidfVectorizer` dipanggil dengan kolom 'combined' sebagai input. Proses `fit` akan mempelajari kosakata dari seluruh teks dalam kolom 'combined', dan proses `transform` akan mengubah setiap entri teks menjadi vektor TF-IDF berdasarkan kosakata yang telah dipelajari. Hasilnya adalah matriks `tfidf_matrix`.
* **Melihat Fitur yang Dihasilkan**
  ```
  # Melihat fitur yang dihasilkan
  features = tf.get_feature_names_out()
  
  # print(tfidf_matrix.shape)
  tf.get_feature_names_out()
  ```
  `tf.get_feature_names_out()` digunakan untuk mendapatkan daftar semua fitur (kata unik) yang telah diekstrak oleh `TfidfVectorizer`.
* **Melihat Ukuran Matriks TF-IDF**
  ```
  # Melakukan fit lalu ditransformasikan ke bentuk matrix
  tfidf_matrix = tf.fit_transform(data['combined'])
  
  # Melihat ukuran matrix tfidf
  tfidf_matrix.shape
  ```
  `tfidf_matrix.shape` memberikan dimensi dari matriks TF-IDF. Jumlah baris sesuai dengan jumlah buku dalam *dataset*, dan jumlah kolom sesuai dengan jumlah fitur unik yang ditemukan dalam kolom 'combined'.
* **Mengubah ke Matriks Padat**
  ```
  # Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
  tfidf_matrix.todense()
  ```
  `tfidf_matrix.todense()` mengubah matriks *sparse* TF-IDF menjadi matriks padat. Meskipun representasi *sparse* lebih efisien untuk penyimpanan dan perhitungan dengan data teks berdimensi tinggi, representasi padat mungkin lebih mudah dipahami dalam beberapa kasus.
* **Membuat *Dataframe* Matriks TF-IDF**
  ```
  # Membuat dataframe untuk melihat tf-idf matrix
  # Kolom diisi dengan author, publisher
  # Baris diisi dengan nama buku
  
  pd.DataFrame(
      tfidf_matrix.todense(),
      columns=tf.get_feature_names_out(),
      index=data.title
  ).sample(22, axis=1).sample(10, axis=0)
  ```
  Sebuah *dataframe* dibuat dari matriks padat TF-IDF. Kolom-kolom *dataframe* ini diberi nama sesuai dengan fitur-fitur yang diekstrak (kata-kata dari 'author' dan 'publisher'), dan indeksnya adalah judul buku dari kolom 'title' (kemungkinan kolom 'book_title' telah diubah namanya menjadi 'title' pada tahap sebelumnya). Metode `.sample()` digunakan untuk menampilkan sebagian kecil dari *dataframe* ini, memudahkan visualisasi bobot TF-IDF untuk beberapa buku dan fitur secara acak.

4. **Perhitungan *Cosine Similarity*:**

* **Menghitung *Cosine Similarity***
  ```
  # Menghitung cosine similarity pada matrix tf-idf
  cosine_sim = cosine_similarity(tfidf_matrix)
  cosine_sim
  ```
  Fungsi `cosine_similarity()` dari *library* `sklearn.metrics.pairwise` digunakan untuk menghitung kemiripan *cosine* antara semua pasangan vektor TF-IDF dalam `tfidf_matrix`. *Cosine similarity* adalah ukuran kemiripan antara dua vektor dalam ruang multidimensi dan sering digunakan untuk mengukur kemiripan dokumen teks. Nilainya berkisar antara -1 (tidak mirip) hingga 1 (sangat mirip).
* **Membuat *Dataframe* *Cosine Similarity***
  ```
  # Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa nama buku
  cosine_sim_df = pd.DataFrame(cosine_sim, index=data['title'], columns=data['title'])
  print('Shape:', cosine_sim_df.shape)
  ```
  Hasil dari `cosine_similarity()` adalah matriks *numpy* yang kemudian diubah menjadi *dataframe* `cosine_sim_df`. Indeks dan kolom dari *dataframe* ini diatur menjadi judul buku, sehingga setiap sel *(i, j)* dalam *dataframe* berisi nilai *cosine similarity* antara buku *i* dan buku *j*.
* **Melihat Ukuran dan Contoh Matriks *Cosine Similarity***
  ```
  # Melihat similarity matrix pada setiap buku
  cosine_sim_df.sample(5, axis=1).sample(10, axis=0)
  ```
  `cosine_sim_df.shape` menunjukkan dimensi dari matriks *cosine similarity*, yang akan berbentuk persegi dengan ukuran jumlah buku dikali jumlah buku. Metode `.sample()` kembali digunakan untuk menampilkan sebagian kecil dari matriks *cosine similarity*, yang menunjukkan tingkat kemiripan antara beberapa pasangan buku secara acak.

**Tujuan Keseluruhan:**

Tujuan dari tahapan persiapan data ini adalah untuk mengubah representasi tekstual dari penulis dan penerbit buku menjadi representasi numerik (vektor TF-IDF) dan kemudian mengukur kemiripan antar buku berdasarkan representasi numerik ini menggunakan *cosine similarity*. Matriks *cosine similarity* yang dihasilkan (`cosine_sim_df`) akan menjadi dasar untuk sistem rekomendasi *content-based*, di mana buku-buku dengan nilai kemiripan *cosine* yang tinggi dianggap memiliki konten yang serupa dan dapat direkomendasikan satu sama lain.

### Data Preparation untuk Collaborative Filtering

Tentu, mari kita bahas tahapan persiapan data untuk *collaborative filtering* ini secara lebih rinci:

**1. Penggunaan *Dataframe* `reduced_ratings`:**

Langkah awal dalam persiapan data untuk *collaborative filtering* adalah menggunakan *dataframe* yang disebut `reduced_ratings`. Ini mengindikasikan bahwa mungkin telah dilakukan proses pengurangan atau pemilihan sebagian data rating dari *dataset* awal. Tujuannya bisa untuk mengurangi kompleksitas komputasi atau fokus pada interaksi pengguna dan item yang lebih relevan.

**2. Encoding User ID dan ISBN:**

Proses *encoding* adalah mengubah data kategorikal (dalam hal ini, ID pengguna dan ISBN buku yang kemungkinan besar berupa *string* atau angka unik) menjadi representasi numerik berupa indeks integer. Hal ini penting karena model *machine learning*, termasuk model *collaborative filtering*, umumnya bekerja lebih baik dengan input numerik.

* **Membuat *List* ID Unik:** Kode `df['User-ID'].unique().tolist()` dan `df['ISBN'].unique().tolist()` menghasilkan *list* yang berisi nilai-nilai unik dari kolom 'User-ID' dan 'ISBN'. Ini memastikan bahwa setiap pengguna dan setiap buku hanya direpresentasikan satu kali dalam *list*.

* **Membuat *Dictionary* Pemetaan (Encoding):** Dua *dictionary* dibuat untuk setiap ID pengguna dan ISBN buku:
    * `user_to_user_encoded` dan `book_to_book_encoded`: *Dictionary* ini memetakan setiap nilai unik (ID pengguna atau ISBN) ke sebuah indeks integer yang berurutan. Misalnya, pengguna dengan ID '22' mungkin dipetakan ke indeks 0, pengguna dengan ID '53' ke indeks 1, dan seterusnya.
    * `user_encoded_to_user` dan `book_encoded_to_book`: Ini adalah *dictionary* kebalikan dari yang sebelumnya, memetakan indeks integer kembali ke nilai asli (ID pengguna atau ISBN). Ini berguna untuk interpretasi hasil model.

* **Menerapkan Pemetaan ke *Dataframe*:** Metode `.map()` digunakan untuk menerapkan *dictionary* *encoding* ke kolom 'User-ID' dan 'ISBN' dalam *dataframe* `df`. Ini akan membuat dua kolom baru, 'user' dan 'book', yang berisi indeks integer yang sesuai untuk setiap interaksi (rating).

**3. Mendapatkan Jumlah Pengguna dan Buku:**

Setelah proses *encoding*, jumlah pengguna unik (`num_users`) dan jumlah buku unik (`num_book`) dihitung menggunakan panjang dari *dictionary* *encoding*. Informasi ini penting untuk menentukan dimensi dari *embedding layer* dalam model *neural network* untuk *collaborative filtering*.

**4. Konversi Tipe Data Rating:**

Kolom 'Book-Rating' diubah menjadi tipe data *float32*. Ini umum dilakukan untuk nilai *rating* karena model *machine learning* seringkali bekerja dengan bilangan *floating-point*.

**5. Identifikasi Rentang Rating:**

Nilai minimum (`min_rating`) dan maksimum (`max_rating`) dari kolom 'Book-Rating' diidentifikasi. Informasi ini akan digunakan untuk normalisasi nilai *rating* ke dalam rentang yang lebih kecil (biasanya 0 hingga 1).

**6. Pengacakan Dataset:**

*Dataframe* `df` diacak menggunakan `df.sample(frac=1, random_state=42)`. `frac=1` berarti semua baris akan dikembalikan, tetapi dalam urutan acak. `random_state=42` digunakan untuk memastikan bahwa pengacakan akan menghasilkan urutan yang sama setiap kali kode dijalankan, yang penting untuk *reproducibility*.

**7. Pemisahan Fitur dan Label:**

* **Fitur (`x`):** Kolom 'user' dan 'book' (yang berisi indeks *encoding*) dipilih sebagai fitur (`x`). Pasangan indeks pengguna dan buku ini akan menjadi input untuk model *collaborative filtering*. Metode `.values` digunakan untuk mengubah *dataframe* menjadi *array numpy*.
* **Label (`y`):** Kolom 'Book-Rating' digunakan sebagai label (`y`). Nilai *rating* dinormalisasi ke dalam rentang 0 hingga 1 menggunakan formula: `(x - min_rating) / (max_rating - min_rating)`. Normalisasi ini membantu model untuk belajar dengan lebih stabil dan efisien.

**8. Pembagian Data Latih dan Validasi:**

Dataset dibagi menjadi set pelatihan (80%) dan validasi (20%). `train_indices` dihitung untuk menentukan titik pemisahan. Kemudian, fitur (`x`) dan label (`y`) dibagi menjadi `x_train`, `x_val`, `y_train`, dan `y_val`. Set pelatihan akan digunakan untuk melatih model, sedangkan set validasi akan digunakan untuk mengevaluasi kinerja model selama pelatihan dan membantu dalam *tuning hyperparameter*.

Secara keseluruhan, tahapan ini mempersiapkan data *rating* pengguna dan buku ke dalam format numerik yang sesuai untuk melatih model *collaborative filtering*. Proses *encoding* mengubah identitas pengguna dan buku menjadi indeks, normalisasi skala *rating*, dan pembagian data memastikan bahwa model dapat dipelajari dan dievaluasi dengan baik.

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
