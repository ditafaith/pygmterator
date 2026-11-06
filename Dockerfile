cat << 'EOF' > Dockerfile
FROM condaforge/miniforge3:latest

# Instalasi GMT, PyGMT, dan dependensi sistem
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

# Salin file proyek
COPY . /app

EXPOSE 8000

# Bind ke PORT dari Railway jika ada, fallback ke 8000
CMD exec gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --timeout 180 app:app
EOF