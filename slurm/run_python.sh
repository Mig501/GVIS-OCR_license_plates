#!/bin/bash -l

# ==========================================
# GVIS - EJECUCION DE SCRIPTS PYTHON
# ==========================================

set -eo pipefail

# 1. Inicializar el sistema de modulos Lmod
if ! type module >/dev/null 2>&1; then
    for init in \
        /opt/ohpc/admin/lmod/lmod/init/bash \
        /usr/share/lmod/lmod/init/bash \
        /etc/profile.d/lmod.sh
    do
        if [ -f "$init" ]; then
            source "$init"
            break
        fi
    done
fi

if ! type module >/dev/null 2>&1; then
    echo "ERROR: No se pudo cargar Lmod"
    exit 1
fi

# 2. Cargar nuestro entorno Apptainer
module use "$HOME/privatmodules"
module load OCR_Matriculas

# Habilitar los alias del modulo en scripts Bash
shopt -s expand_aliases

# 3. Acceder al proyecto
cd "$HOME/ocr_matriculas"

# 4. Seleccionar el script Python
SCRIPT="${1:-scripts/prueba_cluster.py}"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: No existe el script $SCRIPT"
    exit 1
fi

# 5. Conservar el identificador SLURM en Apptainer
export APPTAINERENV_SLURM_JOB_ID="${SLURM_JOB_ID:-No_asignado}"

# 6. Mostrar informacion de la ejecucion
echo "======================================"
echo "       GVIS - OCR MATRICULAS"
echo "======================================"
echo "Nodo: $(hostname)"
echo "Trabajo SLURM: ${SLURM_JOB_ID:-No_asignado}"
echo "Script: $SCRIPT"
echo "Fecha: $(date)"
echo "======================================"

# 7. Ejecutar Python mediante Apptainer
python "$SCRIPT"

echo "======================================"
echo "Ejecucion finalizada correctamente"
echo "======================================"
