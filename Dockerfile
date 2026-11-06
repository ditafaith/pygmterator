FROM condaforge/miniforge3:latest

# Instalasi GMT C-libraries, PyGMT, dan dependensi sistem
RUN conda install -y -c conda-forge \
    python=3.11 \
    gmt=6.5.0 \
    pygmt=0.17.0 \
    flask \
    flask-cors \
    pandas \
    numpy \
    gunicorn \
    && conda clean -afy

WORKDIR /app

# Salin seluruh isi folder proyek ke dalam container
COPY . /app

EXPOSE 8000

# Jalankan backend menggunakan gunicorn pada port 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "180", "app:app"]
