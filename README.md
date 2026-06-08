# Workflow-CI

Repository untuk **Kriteria 3 - CI Workflow** pada proyek MSML menggunakan MLflow Project dan Docker Hub.

## Deskripsi

CI pipeline yang menjalankan training model secara otomatis menggunakan MLflow Project, membangun Docker image, dan push ke Docker Hub.

## Struktur

```
├── MLproject/
│   ├── MLproject          # MLflow Project definition
│   ├── conda.yaml         # MLflow-managed Conda environment
│   └── modelling.py       # Training script
├── .github/workflows/ci.yml  # GitHub Actions CI workflow
└── README.md
```

## Docker Hub

Image: `https://hub.docker.com/r/fiyanz/red-chili-pepper-pests-classifier`

## Setup

Tambahkan GitHub Secrets:
- `KAGGLE_JSON` - Kaggle API credentials (isi dengan isi file kaggle.json)
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password/token

## Run Locally

```bash
cd MLproject
mlflow run . -P dataset_dir=/path/to/dataset -P epochs=50 -P batch_size=32
```

## Author

- **Nama:** Bagus Alfiyan Yusuf
- **Dicoding Username:** fiyanz
