# racetraQ on a QuBins base image (core Qiskit + Aer + IBM Runtime, amd64 + arm64).
FROM ghcr.io/qubins/images:latest-small

COPY --chown=1000:100 . /opt/racetraq
RUN pip install --no-cache-dir /opt/racetraq

EXPOSE 8000
CMD ["python", "-m", "racetraq", "--host", "0.0.0.0", "--port", "8000"]
