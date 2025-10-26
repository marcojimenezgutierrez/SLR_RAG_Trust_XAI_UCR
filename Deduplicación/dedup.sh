# Listar los archivos .bib y otros en el espacio compartido
ls -lh /home/oai/share

# (Intento) Verificar bibtexparser - *este entorno no tiene salida a internet, fallará instalar*
python - <<'PY'
import sys
try:
    import bibtexparser
    print('bibtexparser found', bibtexparser.__version__)
except Exception as e:
    print('bibtexparser not found')
PY

# (Intento) Instalar bibtexparser - fallará por red restringida (documentado aquí por trazabilidad)
pip install bibtexparser==1.4.0 --quiet

# Inspeccionar rápidamente el encabezado de un .bib
head -n 20 "/home/oai/share/SLR - JoCiCi - 2025 - ACM Library.bib"

# Comprobar pybtex (no disponible en el entorno)
python - <<'PY'
try:
    from pybtex.database import parse_file
    print('pybtex available')
except ImportError:
    print('pybtex not available')
PY

# Ejecutar el script de deduplicación (versión inline); abajo te dejo el script completo para archivo .py
python dedup.py