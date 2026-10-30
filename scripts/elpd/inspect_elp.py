
from pathlib import Path
from collections import Counter
from zipfile import ZipFile


ZIP_PATH = Path(
    "/home/miguel.sanchez/datasets/ocr_matriculas/"
    "raw/ELPD/european-license-plates-dataset.zip"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    print("=== INSPECCION ELPD ===")

    with ZipFile(ZIP_PATH) as zf:
        archivos = [
            info.filename
            for info in zf.infolist()
            if not info.is_dir()
        ]

        imagenes = [
            nombre for nombre in archivos
            if Path(nombre).suffix.lower() in IMAGE_EXTENSIONS
        ]

        extensiones = Counter(
            Path(nombre).suffix.lower()
            for nombre in archivos
        )

        print("Total de archivos:", len(archivos))
        print("Total de imagenes:", len(imagenes))

        print("\n=== EXTENSIONES ===")
        for ext, cantidad in sorted(extensiones.items()):
            print(ext or "(sin extension)", cantidad)

        print("\n=== PRIMEROS 15 ARCHIVOS ===")
        for nombre in archivos[:15]:
            print(nombre)

        print("\n=== POSIBLES ANOTACIONES ===")
        anotaciones = [
            nombre for nombre in archivos
            if Path(nombre).suffix.lower()
            in {".csv", ".json", ".txt", ".xml"}
        ]

        if anotaciones:
            for nombre in anotaciones[:20]:
                print(nombre)
        else:
            print("No se encontraron archivos de anotaciones.")

    print("\n=== INSPECCION FINALIZADA ===")


if __name__ == "__main__":
    main()
