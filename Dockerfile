FROM condaforge/miniforge3:latest

RUN mamba install -y -c conda-forge \
    pygmt \
    flask \
    flask-cors \
    pandas \
    numpy \
    gunicorn \
    && mamba clean -afy

WORKDIR /app
COPY . /app
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "180", "app:app"]