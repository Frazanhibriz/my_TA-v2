**Review naskah IEEE — 24 Oktober 2026**

Naskah utama: [paper.tex](paper.tex). Penilaian mencakup source LaTeX, PDF hasil build yang tersedia, bibliografi, kode training, notebook preprocessing/training/evaluasi, serta CSV hasil RTX 5070 Ti. Angka diperiksa terhadap artefak tersimpan; training, prediksi, dan bootstrap tidak dijalankan ulang. Dokumen ini merupakan catatan review; naskah asli tidak diubah.

**Penilaian utama**

Bagian Methods sudah memiliki dasar yang baik: konfigurasi input jelas, konteks parent diuji melalui ablation, pemilihan model menggunakan validasi, dan hasil utama memakai tiga seed. Bahasa umumnya mudah dipahami. Namun, versi sekarang belum menjadi paper lengkap dan beberapa pernyataan perlu diperbaiki agar sesuai dengan implementasi serta batas bukti eksperimen.

Abstract, keywords, Introduction, Results, Discussion, dan Conclusion masih kosong. Karena itu, kualitas argumentasi keseluruhan, kebaruan, serta kecocokan kesimpulan dengan hasil belum dapat dinilai dari naskah. Methods dapat dinilai secara substantif, tetapi kesiapan submit belum tercapai.

**Temuan yang perlu didahulukan**

1. **Bukti penguncian konfigurasi perlu diperjelas — paper.tex:131.** Naskah menyebut 29 run sudah dicatat dan di-commit sebelum test set dibuka. Manifest berisi 29 run memang tersedia dan mempunyai kolom `locked_at`. Notebook 07 juga menyebut commit `2ac243fc`. Namun, pada repository yang sedang diperiksa, manifest belum tracked dan commit tersebut tidak ditemukan. Ini berarti bukti commit belum dapat diverifikasi dari salinan lokal ini; bukan bukti bahwa commit tidak pernah ada di tempat lain. Jika commit berasal dari mesin/repository lain, lampirkan hash dan akses ke riwayat tersebut. Jika bukti tidak tersedia, hilangkan klaim tentang commit. Membuat commit sekarang tidak membuktikan urutan kejadian yang lalu.

2. **“The test set was not touched” terlalu luas — paper.tex:70.** `src/training.py:342–351` memuat test set dan menghasilkan prediksi pada setiap run; fungsi prediksi juga menghitung loss test, meskipun nilainya dibuang dan tidak dipakai memilih model. Notebook 07 baru mengolah metrik final setelah membaca manifest. Jelaskan pemisahan antara menghasilkan prediksi, memeriksa hasil test, dan menggunakan test untuk seleksi. Kode yang diperiksa memilih epoch dan learning rate berdasarkan validasi; menghasilkan prediksi test lebih awal sendiri tidak membuktikan adanya kebocoran dalam seleksi.

3. **Definisi waktu inferensi belum cukup spesifik — paper.tex:125.** Di `measure_latency`, padding dan pemindahan tensor ke GPU dilakukan sebelum timer. Timer mencakup forward pass dan overhead loop pemanggilannya, dengan sinkronisasi CUDA sebelum dan sesudah pengukuran. Tokenisasi, padding, transfer CPU–GPU, softmax, serta pengambilan hasil tidak tercakup. Sebut sebagai *forward-pass inference time with inputs resident on the GPU*. Angka batch 32 adalah waktu total dibagi jumlah contoh, sehingga harus diberi satuan *ms/example at batch size 32*; angka itu bukan waktu respons satu batch. Prosedur ini berguna untuk membandingkan implementasi, tetapi belum mengukur waktu respons aplikasi lengkap.

4. **Pernyataan semua skor memakai tiga seed memiliki pengecualian — paper.tex:90 dan 131.** Ada 18 run utama, 9 run partial freezing, dan 2 run head-only. Dua konfigurasi head-only masing-masing hanya memakai seed 42. Tulis bahwa mean ± sample SD dilaporkan untuk konfigurasi dengan tiga seed, sedangkan head-only dilaporkan sebagai nilai satu run. Jangan membawa `± nan` dari tabel CSV ke paper.

5. **Uji statistik memerlukan interpretasi yang lebih hati-hati — paper.tex:133.** McNemar menguji perbedaan pola benar/salah, sedangkan paired bootstrap di sini menghitung interval selisih macro-F1. Kedua prosedur menjawab pertanyaan yang berkaitan tetapi berbeda. Holm pada kode diterapkan pada 11 p-value McNemar per seed; interval bootstrap 95% tetap interval per perbandingan, bukan interval simultan yang dikoreksi. Syarat lolos pada ketiga seed merupakan kriteria konsistensi yang ketat, bukan bukti bahwa suatu selisih “real” secara mutlak. Ganti frasa “accepted as real” dengan “met the predefined significance criterion”. Rujukan pemilihan uji: [Dror et al., 2018](https://aclanthology.org/P18-1128/).

6. **Jangan menyimpulkan kesetaraan dari hasil tidak signifikan.** Results masih kosong, tetapi beberapa interpretasi di notebook berpotensi terbawa ke paper. Untuk RoBERTa dengan enam layer dibekukan, macro-F1 turun rata-rata 0,239 poin. Perbedaan signifikan menurut kriteria per seed pada seed 42 dan 43, tetapi tidak pada seed 44. Pernyataan yang didukung adalah “the decrease was small but statistically significant in two of three seeds.” Kesetaraan atau non-inferiority membutuhkan margin dan pengujian yang sesuai. Hal yang sama berlaku untuk early vs late fusion DistilBERT: tidak ada bukti perbedaan konsisten pada protokol ini, tetapi kesetaraan belum dibuktikan.

7. **Asal tambahan parameter late fusion RoBERTa kurang tepat — paper.tex:87.** CSV menunjukkan kenaikan 2.360.064 parameter RoBERTa dan 1.769.472 parameter DistilBERT. Pelebaran linear layer dari 768→768 menjadi 3.072→768 sendiri menambah 3×768×768 = 1.769.472 parameter pada masing-masing model. Selisih 590.592 parameter RoBERTa cocok dengan pooler 768×768 + 768. Kode late fusion memakai `AutoModel` dengan default pooler RoBERTa dan mengambil `last_hidden_state`, sedangkan sequence classifier standar menonaktifkan pooler. Jadi penjelasan bahwa seluruh tambahan 2,4 juta berasal dari head kurang akurat. Untuk melaporkan eksperimen yang sudah selesai, nyatakan kenaikan total model dan jelaskan komponennya; perubahan implementasi pooler harus disertai verifikasi baru sebelum memakai ulang angka efisiensi. Dasar API: [implementasi resmi RoBERTa](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py).

**Perbaikan metode dan reproduksibilitas**

- **Identitas dataset — paper.tex:40.** Tambahkan sitasi paper asli SARC, URL sumber Kaggle, nama file yang benar-benar digunakan, dan versi/checksum atau tanggal pengambilan jika tersedia. Klaim “the same release” lebih kuat daripada bukti bahwa penelitian menggunakan sumber Kaggle yang sama. Zhang et al. menyebut 1,3 juta komentar, sedangkan data studi ini berisi 1.010.826 baris sebelum filtering. Perbedaan itu perlu dijelaskan sebelum menyatakan versi identik. Rujukan: [SARC asli](https://aclanthology.org/L18-1102/) dan [Zhang et al.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0334120).
- **Sifat label — paper.tex:40.** “Nobody annotated these labels by hand” mengaburkan fakta bahwa penulis komentar memberi penanda sendiri. Gunakan “Labels were derived from author-provided sarcasm markers rather than independent annotation.” Bahas kemungkinan label noise; tidak adanya `/s` tidak menjamin komentar benar-benar nonsarkastik. Paper asli SARC membahas persoalan ini.
- **Definisi split — paper.tex:70.** Kelompok dibentuk dari `comment_key` yang dinormalisasi, bukan hanya teks identik. Stratifikasi pada kode menggunakan empat strata: kelompok singleton/multirow dan label/label mayoritas. Nyatakan “approximately 80/10/10” karena proporsi baris aktual 79,88/10,02/10,10. Hindari kesan bahwa setiap split merupakan stratified random split per baris.
- **Batas pencegahan overlap.** Audit memastikan tidak ada overlap komentar terstandardisasi maupun pasangan antarsplit. Parent masih dapat berulang, dan split ini tidak membuktikan independensi berdasarkan thread, author, waktu, atau topik. Sebut batasnya dalam Discussion. Karena beberapa baris test berbagi komentar, pertimbangkan cluster bootstrap berdasarkan `comment_key` sebagai analisis sensitivitas sebelum menarik klaim populasi yang luas.
- **Filter fragmen `/s` — paper.tex:44.** Angka 8.599 baris dan dominasi label sarkastik layak dilaporkan, tetapi pola yang terdeteksi sebaiknya disebut *suspected annotation artifacts*. Keyakinan bahwa semua fragmen disebabkan penghapusan marker memerlukan pemeriksaan contoh. Sertakan regex atau lokasi implementasi, batas delapan kata, dan hasil audit manual bila dilakukan. Jangan menyebut audit manual seolah sudah dilakukan.
- **Layer freezing — paper.tex:90.** Bedakan layer dilatih dari layer dibekukan. `L3` pada nama run berarti tiga layer teratas dilatih; untuk RoBERTa berarti sembilan layer dibekukan. Jelaskan bahwa full fine-tuning juga melatih embedding, sedangkan embedding dibekukan pada konfigurasi partial/head-only. Learning rate partial freezing diwariskan dari full fine-tuning, sehingga studi tidak mencari optimum learning rate untuk setiap jumlah layer. Bahkan ada run DistilBERT tiga layer pada 2e-5 dengan validasi lebih tinggi daripada 1e-5 yang dipakai di konfigurasi final; ini tidak membatalkan protokol tetap, tetapi membatasi klaim tentang konfigurasi freezing terbaik.
- **Detail training yang belum disebut.** Tambahkan cross-entropy sebagai loss, dropout yang dipakai, seed split 42, serta identifier/revisi checkpoint bila tercatat. PyTorch 2.11.0+cu128 dan Transformers 5.17.0 sesuai output notebook yang tersimpan; tidak ada dasar dari audit ini untuk menyebut versi tersebut salah.
- **Agregasi efisiensi — paper.tex:125.** Rata-rata semua S1+S3 menggabungkan 8 run per kombinasi comment/early dan 6 run per kombinasi late. Ini sesuai penjelasan naskah, tetapi berbeda dari tiga run konfigurasi final yang menghasilkan skor test. Pilihan paling mudah dibandingkan adalah memakai tiga run final untuk tabel utama dan menaruh agregat lebih luas sebagai pelengkap. Bila tetap memakai agregat sekarang, tampilkan jumlah run dan variasinya. GB pada kode berarti pembagian dengan 10^9, bukan GiB.
- **Subgroup analysis — paper.tex:135.** Jelaskan batas panjang 1–3, 4–10, dan >10 kata; kelompok dapat overlap. Kelompok tidak yakin berasal dari probabilitas RoBERTa comment-only, termasuk saat menilai DistilBERT. Pada kode, VADER menerima teks mentah; jelaskan pilihan itu. P-value subgroup belum dikoreksi seperti 11 perbandingan utama, sehingga sebaiknya disebut analisis eksploratif atau diberi prosedur koreksi tersendiri. Kelompok yang dibentuk dari ketidakpastian baseline cenderung memberi ruang perbaikan lebih besar; kenaikan skor di situ belum membuktikan mekanisme linguistik tertentu.

**Kualitas bahasa**

Bahasa sudah jelas, tetapi sejumlah kalimat terdengar seperti penjelasan notebook. Pertahankan kalimat sederhana dan ubah ungkapan percakapan menjadi pernyataan yang terukur. Contoh penggantian berikut tidak membutuhkan istilah yang lebih rumit:

| Teks sekarang | Usulan |
|---|---|
| “It holds 1,010,826 pairs” | “The dataset contains 1,010,826 parent–comment pairs.” |
| “Nobody annotated these labels by hand.” | “Labels were derived from author-provided sarcasm markers rather than independent annotation.” |
| “The last filter needs a short explanation.” | Langsung jelaskan aturan filter dan alasannya. |
| “The real price is paid elsewhere” | “Late fusion additionally requires two encoder forward passes per example.” |
| “The learning rate was not copied from earlier work.” | “Learning rates were selected using validation macro-F1 over a four-value grid.” |
| “just to see whether the choice … would hold up” | “to assess the stability of learning-rate selection across seeds” |
| “A difference was accepted as real” | “A comparison met the significance criterion when …” |

Kalimat “the absolute difference and the product give … a direct signal of how far the reply drifts” juga terlalu interpretatif. Fitur tersebut menyediakan interaksi antarrepresentasi; eksperimen belum menunjukkan bahwa fitur itu secara khusus mengukur penyimpangan ironis. Usulan: “The absolute difference and element-wise product provide additional interaction features between the two representations.”

Paragraf tentang RoBERTa dengan F1 0% dan 0,97% sesuai tabel pada PDF Jiménez Castro et al. yang tersedia. Namun, hasil itu tidak dengan sendirinya membuktikan penyebabnya adalah learning rate. Gunakan sebagai motivasi umum pentingnya konfigurasi dan evaluasi, bukan diagnosis kausal kegagalan model.

**Bahan Results yang sudah tersedia**

Tabel berikut disalin dari ringkasan test RTX 5070 Ti. Waktu dan memori memakai agregasi S1+S3 sebagaimana naskah sekarang, sehingga jumlah run pengukuran efisiensi berbeda dari jumlah seed skor test.

| Model | Input | Test macro-F1 (%) | Training (menit/epoch) | Waktu inferensi BS1 (ms/contoh) | Peak allocated GPU memory (GB) |
|---|---|---:|---:|---:|---:|
| DistilBERT | Comment-only | 76,87 ± 0,08 | 8,55 | 4,22 | 2,63 |
| DistilBERT | Early fusion | 77,55 ± 0,09 | 9,88 | 4,21 | 2,64 |
| DistilBERT | Late fusion | 77,59 ± 0,02 | 14,96 | 7,28 | 4,29 |
| RoBERTa | Comment-only | 78,52 ± 0,06 | 16,51 | 8,40 | 5,09 |
| RoBERTa | Early fusion | 79,62 ± 0,06 | 18,88 | 8,39 | 5,11 |
| RoBERTa | Late fusion | 79,30 ± 0,02 | 28,87 | 15,14 | 8,38 |

Narasi yang didukung hasil:

- RoBERTa early fusion memiliki macro-F1 tertinggi di antara konfigurasi utama yang diuji. Dibanding DistilBERT early fusion, kenaikannya sekitar 2,07 poin dengan waktu training sekitar 1,91 kali dan waktu inferensi BS1 sekitar 1,99 kali.
- Parent meningkatkan macro-F1 dibanding comment-only pada kedua model: sekitar 1,10 poin untuk RoBERTa early fusion dan 0,69 poin untuk DistilBERT early fusion.
- Late fusion RoBERTa lebih mahal dan skornya lebih rendah daripada early fusion. Pada DistilBERT, selisih skor sangat kecil dan tidak memenuhi kriteria signifikansi konsisten.
- Membekukan enam layer RoBERTa menghasilkan macro-F1 79,38 ± 0,05. Dibanding tiga run final full fine-tuning, waktu training turun sekitar 44,8% dan peak memory sekitar 47,1%. Penurunan macro-F1 sekitar 0,24 poin harus tetap dilaporkan. Freezing menghemat training; semua layer tetap diperlukan saat inferensi.
- Biaya konteks bergantung pada batch size. Pada RoBERTa, early fusion hampir sama waktunya dengan comment-only di BS1, tetapi waktu per contoh pada BS32 naik dari sekitar 0,298 ke 0,609 ms. Karena itu, hindari pernyataan umum bahwa konteks diperoleh hampir tanpa biaya.

Gunakan “percentage points” untuk selisih skor persen, bukan “percent improvement” kecuali benar-benar menghitung perubahan relatif. Jangan menyebut 2,07 poin macro-F1 sebagai kenaikan accuracy.

**Susunan penulisan lanjutan**

Introduction perlu menjawab: masalah praktis apa yang memerlukan kompromi kualitas dan biaya; apa yang belum dijawab studi sebelumnya; dan mengapa gabungan jenis input, ukuran encoder, serta freezing membantu menjawabnya. Tiga pertanyaan riset yang sesuai dengan eksperimen adalah: (1) bagaimana trade-off kedua model, (2) berapa manfaat dan biaya konteks parent pada dua strategi fusion, dan (3) seberapa besar penghematan training melalui partial freezing beserta penurunan skornya.

Results dapat disusun mengikuti tiga pertanyaan tersebut, lalu menambahkan analisis subgroup sebagai eksplorasi. Discussion menjelaskan implikasi pemilihan model, biaya training versus inferensi, variasi menurut batch size, serta keterbatasan. Abstract ditulis setelah angka dan pesan utama Results sudah tetap.

Klaim kontribusi yang masuk akal adalah evaluasi empiris terkontrol pada dataset besar dengan grouped split, perbandingan manfaat/biaya konteks, dan karakterisasi penghematan partial freezing. Kebaruan itu perlu diposisikan terhadap literatur; jumlah run dan penggunaan model populer saja belum membuktikan kontribusi baru. RoBERTa dan DistilBERT juga berbeda dalam pretraining dan tokenizer, sehingga selisih skor tidak dapat diatribusikan hanya pada jumlah layer atau distillation.

Judul saat ini relevan, tetapi kata “Accuracy” kurang mewakili metrik utama macro-F1. Alternatif: **“Predictive Performance and Computational Cost of DistilBERT and RoBERTa for Context-Aware Sarcasm Detection.”**

**Sitasi dan format akhir**

Tambahkan sumber primer untuk [SARC](https://aclanthology.org/L18-1102/), [RoBERTa](https://arxiv.org/abs/1907.11692), [DistilBERT](https://arxiv.org/abs/1910.01108), dan [VADER](https://ojs.aaai.org/index.php/ICWSM/article/view/14550), serta rujukan pengujian statistik bila digunakan. Referensi penelitian terapan tetap berguna untuk positioning, tetapi tidak menggantikan sumber metode/dataset asli.

`\nocite{*}` saat ini menampilkan empat entri yang belum disitasi dalam teks. Setelah Introduction dan Discussion selesai, pertahankan hanya rujukan yang benar-benar mendukung argumen. Lindungi kapitalisasi akronim pada judul BibTeX dengan `{BERT}`, `{LLMs}`, dan `{SARC}`; hasil `.bbl` sekarang menampilkan beberapa akronim sebagai huruf kecil. Periksa pula penulisan aksen nama pada hasil PDF.

PDF build yang tersedia berjumlah tiga halaman. Log tidak menunjukkan citation undefined atau overfull box, tetapi ini belum menilai tata letak naskah lengkap. Isi identitas supervisor/email, hubungkan tiga gambar yang sudah disiapkan dengan Results, dan periksa kembali keterbacaan gambar pada lebar kolom setelah isi selesai. Batas halaman dan ketentuan submission harus mengikuti konferensi tujuan yang belum disebutkan.
