# racetraQ on a QuBins base image (core Qiskit + Aer + IBM Runtime, amd64 + arm64).
FROM ghcr.io/qubins/images:latest-small

COPY --chown=1000:100 . /opt/racetraq
# Floors for packages Trivy flags from a superseded base layer (the image
# already has newer ones; see .trivyignore): keeps a real regression out.
RUN pip install --no-cache-dir --upgrade "msgpack>=1.2.1" "setuptools>=78.1.1" "urllib3>=2.8.0" \
 && pip install --no-cache-dir /opt/racetraq

EXPOSE 8000
CMD ["python", "-m", "racetraq", "--host", "0.0.0.0", "--port", "8000"]
