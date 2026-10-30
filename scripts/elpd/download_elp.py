
from pathlib import Path
from zipfile import ZipFile

from kaggle.api.kaggle_api_extended import KaggleApi


DATASET = "abdelhamidzakaria/european-license-plates-dataset"

# Ruta del almacenamiento de Legio2, fuera del overlay.
DATASET_DIR = Path(
    "/home/miguel.sanchez/datasets/ocr_matriculas/raw/ELPD"
)

ZIP_PATH = DATASET_DIR / "european-license-plates-dataset.zip"


def main():
    print("=== DESCARGA ELPD ===")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset:", DATASET)
    print("Destino:", DATASET_DIR)

    if ZIP_PATH.is_file():
        print("El archivo ya existe. No se vuelve a descargar.")
    else:
        print("Descargando ELPD desde Kaggle...")

        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(
            dataset=DATASET,
            path=str(DATASET_DIR),
            unzip=False,
            quiet=False,
        )

    # Comprobar que el ZIP existe y contiene datos correctos.
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el ZIP: {ZIP_PATH}"
        )

    with ZipFile(ZIP_PATH) as zip_file:
        archivo_incorrecto = zip_file.testzip()
        if archivo_incorrecto is not None:
            raise RuntimeError(
                f"Archivo ZIP dañado: {archivo_incorrecto}"
            )

        num_archivos = len(zip_file.namelist())

    print("\n=== VERIFICACIÓN ===")
    print("Ruta:", ZIP_PATH)
    print("Tamaño:", round(ZIP_PATH.stat().st_size / 1024**2, 2), "MB")
    print("Archivos dentro del ZIP:", num_archivos)
    print("Integridad ZIP: CORRECTA")

    print("\n=== DESCARGA FINALIZADA ===")


if __name__ == "__main__":
    main()
