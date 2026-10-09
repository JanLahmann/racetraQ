# racetraQ on a QuBins base image (core Qiskit + Aer + IBM Runtime, amd64 + arm64).
FROM ghcr.io/qubins/images:latest-small

COPY --chown=1000:100 . /opt/racetraq
# The base image's copies of these trip the publish workflow's Trivy gate
# (HIGH: GHSA-6v7p-g79w-8964, CVE-2025-47273, CVE-2026-97687/97689).
RUN pip install --no-cache-dir --upgrade "msgpack>=1.2.1" "setuptools>=78.1.1" "urllib3>=2.8.0" \
 && pip install --no-cache-dir /opt/racetraq

EXPOSE 8000
CMD ["python", "-m", "racetraq", "--host", "0.0.0.0", "--port", "8000"]
