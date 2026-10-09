import { useEffect } from 'react';
import { outbound } from '../analytics';
import type { Manifest } from '../data';

const REPO = 'https://github.com/JanLahmann/racetraQ';

export function About({ manifest, onClose }: { manifest: Manifest; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  const quantum = manifest.drivers.filter((d) => d.agent === 'quantum').length;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-label="About" onClick={(e) => e.stopPropagation()}>
        <button type="button" className="close" onClick={onClose} aria-label="Close">
          ✕
        </button>
        <h2>A race car with a quantum circuit for a brain</h2>
        <p>
          Every tenth of a second the car measures its surroundings: three lidar rays (the purple lines) and its
          speed. Those numbers are written into qubits as rotation angles, a short quantum circuit entangles the
          qubits, and measuring each qubit gives one score per action — steer right, go straight, steer left,
          brake. The car takes the action with the highest score.
        </p>
        <p>
          The circuit's {manifest.drivers.find((d) => d.id === 'quantum_oval')?.n_params ?? 56} numbers were learned
          by reinforcement learning (double deep Q-learning): thousands of practice laps, rewarded for progress
          along the track and penalised for leaving it. Training happened offline in{' '}
          <a href={REPO} target="_blank" rel="noopener" {...outbound(REPO)}>
            racetraQ
          </a>{' '}
          (Python and Qiskit). This page only drives: it ships {quantum} trained quantum drivers and runs every
          circuit in your browser on{' '}
          <a href="https://qamposer.org" target="_blank" rel="noopener" {...outbound('https://qamposer.org')}>
            QAMPoser
          </a>
          's state-vector simulator.
        </p>
        <h3>What this is — and isn't</h3>
        <ul>
          <li>
            <strong>Exact and noiseless.</strong> The simulator computes the qubits' state exactly. A real quantum
            computer estimates the same numbers from repeated measurements and adds noise; on IBM's{' '}
            <em>ibm_marrakesh</em> a full lap took the same action as this simulation in about 90% of decisions.
          </li>
          <li>
            <strong>No quantum advantage.</strong> The classical rival — a small neural network trained the same
            way — learns faster and more reliably in our multi-seed studies, and drives faster laps on the hairpin
            tracks. The point is to see a quantum circuit make decisions, not to beat classical computers.
          </li>
          <li>
            <strong>Small on purpose.</strong> Four to ten qubits: small enough to simulate in a browser, the same
            size that runs on today's quantum hardware.
          </li>
        </ul>
        <h3>Modes</h3>
        <ul>
          <li>
            <strong>Watch</strong> — follow one decision at a time: sensors, the live circuit, the measured qubits
            and the scores. Pause and step, or open the exact circuit of a decision in IBM Quantum Composer.
          </li>
          <li>
            <strong>Race</strong> — drive yourself (arrow keys, WASD or the on-screen pedals) against the quantum
            car.
          </li>
          <li>
            <strong>Evolution</strong> — four snapshots of one training run racing each other: how the driver learned.
          </li>
        </ul>
        <p className="hint">
          More: the{' '}
          <a href={`${REPO}/blob/main/docs/EXPLAINER.md`} target="_blank" rel="noopener" {...outbound(REPO)}>
            one-page explainer
          </a>
          , the{' '}
          <a href={`${REPO}/blob/main/docs/SCIENCE.md`} target="_blank" rel="noopener" {...outbound(REPO)}>
            science notes
          </a>{' '}
          and the{' '}
          <a href={`${REPO}/blob/main/docs/REPORT.md`} target="_blank" rel="noopener" {...outbound(REPO)}>
            technical report
          </a>
          .
        </p>
        <p className="hint">
          Learn how it works: eight{' '}
          <a href={`${REPO}#notebooks`} target="_blank" rel="noopener" {...outbound(REPO)}>
            notebooks
          </a>{' '}
          build the whole thing from scratch and open in your browser on Binder, nothing to install. The{' '}
          <a href={`${REPO}/blob/main/docs/LEARN.md`} target="_blank" rel="noopener" {...outbound(REPO)}>
            learning path
          </a>{' '}
          suggests where to start, from a 5-minute tour to a research track.
        </p>
      </div>
    </div>
  );
}
