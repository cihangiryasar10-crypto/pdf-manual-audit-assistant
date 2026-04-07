# PDF Manual Audit Assistant

Bu proje, Streamlit ile hazırlanmış basit bir denetim destek uygulamasıdır.

## Özellikler

- Bilgisayardan bir veya birden fazla PDF manuel yükleme
- Mikrofon kaydını alma
- Whisper ile İngilizce konuşmayı metne çevirme
- Denetçi sorusunu manuel içeriklerinde arama
- Sadece en ilgili prosedür bölümünü ve kaynak sayfasını gösterme

## Kurulum

```bash
pip install -r requirements.txt
```

Whisper için ayrıca sisteminizde `ffmpeg` kurulu olmalıdır.

Windows örneği:

```bash
choco install ffmpeg
```

## Çalıştırma

```bash
streamlit run app.py
```

## Streamlit Community Cloud'a Yükleme

Bu proje online çalıştırma için hazırlandı.

Gerekli dosyalar:

- `app.py`: uygulama giriş dosyası
- `requirements.txt`: Python bağımlılıkları
- `packages.txt`: Linux tarafında `ffmpeg` kurulumu
- `.streamlit/config.toml`: Streamlit ayarları

İzlenecek adımlar:

1. Bu klasörü GitHub'da yeni bir repoya yükleyin.
2. [Streamlit Community Cloud](https://share.streamlit.io/) hesabı açın veya giriş yapın.
3. `New app` seçin.
4. GitHub repo'nuzu seçin.
5. Main file path olarak `app.py` yazın.
6. Deploy edin.

Notlar:

- İlk açılışta bağımlılık kurulumu birkaç dakika sürebilir.
- `openai-whisper` ve `torch` nedeniyle ücretsiz bulut ortamında ilk transkripsiyon yavaş olabilir.
- Çok büyük PDF'ler veya çok yoğun kullanım durumunda kaynak limitlerine takılabilirsiniz.

Resmi kaynaklar:

- [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud)
- [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Manage your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

## Notlar

- Uygulama İngilizce konuşma transkripsiyonu için ayarlanmıştır.
- PDF metni çıkarılamazsa, dosya taranmış görüntü PDF olabilir; bu durumda OCR gerekir.
- Arama ilk sürümde TF-IDF tabanlıdır; istenirse sonraki adımda vektör veritabanı ve embedding ile daha da güçlendirilebilir.
