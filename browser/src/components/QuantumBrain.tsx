/** What the focused car's driver sees, computes and decides — per decision. */
import { circuitToQasm } from '@qamposer/react';
import LZString from 'lz-string';
import { track } from '../analytics';
import { actionLabels } from '../sim/actions';
import { QuantumDriver } from '../sim/agents';
import type { WorldCar } from '../sim/world';
import { CircuitView } from './CircuitView';
import { layoutCircuit } from './CircuitView';

const COMPOSER_BASE = 'https://quantum.cloud.ibm.com/composer';
const ARROWS: Record<string, string> = {
  Right: '↱',
  Straight: '↑',
  Left: '↰',
  Brake: '■',
  'Brake right': '■↱',
  'Brake left': '■↰',
  'Half right': '↗',
  'Half left': '↖',
};

/** IBM Quantum Composer URL with the circuit pre-loaded (the `?initial=`
 *  format Entangible verified: LZ-compressed {title, description, qasm}). */
export function composerUrl(qasm: string, title: string): string {
  const payload = JSON.stringify({ title, description: '', qasm });
  const url = `${COMPOSER_BASE}?initial=${encodeURIComponent(LZString.compressToEncodedURIComponent(payload))}`;
  return url.length > 7500 ? COMPOSER_BASE : url;
}

function Bar({ value, min = 0, max = 1, color }: { value: number; min?: number; max?: number; color: string }) {
  const zero = ((0 - min) / (max - min)) * 100;
  const pos = ((Math.min(Math.max(value, min), max) - min) / (max - min)) * 100;
  const left = Math.min(zero, pos);
  return (
    <span className="bar">
      {min < 0 && <span className="bar-zero" style={{ left: `${zero}%` }} />}
      <span className="bar-fill" style={{ left: `${left}%`, width: `${Math.abs(pos - zero)}%`, background: color }} />
    </span>
  );
}

export function QuantumBrain({
  car,
  wideCircuit,
  onToggleWide,
}: {
  car: WorldCar | undefined;
  /** the circuit is shown in the wide view under the track instead */
  wideCircuit: boolean;
  onToggleWide: () => void;
}) {
  if (!car || !car.driver || !car.observer) {
    return <p className="hint">Pick a driver to see what it computes.</p>;
  }
  const { driver, observer, decision, obs } = car;
  const labels = actionLabels(driver.nActions);
  const quantum = driver instanceof QuantumDriver ? driver : null;
  const qMin = decision ? Math.min(0, ...decision.q) : 0;
  const qMax = decision ? Math.max(0, ...decision.q) : 1;
  const qSpan = Math.max(qMax - qMin, 1e-9);

  const openInComposer = () => {
    if (!quantum || !decision?.gates) return;
    const qasm = circuitToQasm(layoutCircuit(quantum.nQubits, decision.gates));
    track('composer open', { qubits: quantum.nQubits, track: car.observer?.track.name });
    window.open(composerUrl(qasm, `traQmania decision (${quantum.nQubits} qubits)`), '_blank', 'noopener');
  };

  return (
    <div className="brain">
      <section>
        <h3>
          <span className="step">1</span> Sensors
          <span className="sub">
            {observer.nFeatures} numbers in [0, 1]{quantum ? ' — one per qubit' : ''}
          </span>
        </h3>
        <ul className="features">
          {observer.featureNames.map((name, i) => (
            <li key={name}>
              <span className="label">{name}</span>
              <Bar value={obs?.[i] ?? 0} color={name.startsWith('ray') ? '#7a5cff' : '#22d3ee'} />
              <span className="num">{(obs?.[i] ?? 0).toFixed(2)}</span>
            </li>
          ))}
        </ul>
      </section>

      {quantum ? (
        <>
          <section>
            <h3>
              <span className="step">2</span> Circuit
              <span className="sub">
                {quantum.nQubits} qubits · {quantum.nLayers} blocks · {quantum.nParams} trained numbers
              </span>
            </h3>
            <p className="hint">
              Each block writes the sensor values into the qubits as rotation angles (RY = λ·s, new every
              decision), applies trained RY/RZ rotations, and entangles neighbours with a CZ ring.
            </p>
            <button type="button" className="chip expand" onClick={onToggleWide}>
              {wideCircuit ? '⤡ Back into this panel' : '⤢ Show the whole circuit under the track'}
            </button>
            {wideCircuit ? null : decision?.gates ? (
              <CircuitView nQubits={quantum.nQubits} gates={decision.gates} zoom={quantum.nQubits > 6 ? 0.42 : 0.55} />
            ) : (
              <p className="hint">The circuit appears with the first decision.</p>
            )}
          </section>
          <section>
            <h3>
              <span className="step">3</span> Measurement
              <span className="sub">⟨Z⟩ of every qubit, exact (state vector in your browser)</span>
            </h3>
            <ul className="qubits">
              {Array.from({ length: quantum.nQubits }, (_, i) => {
                const z = decision?.expectations?.[i] ?? 1;
                const readout = i < driver.nActions;
                return (
                  <li key={i} className={readout ? 'readout' : 'spectator'}>
                    <span className="label">q{i}</span>
                    <Bar value={z} min={-1} max={1} color={readout ? '#b4a3ff' : '#4b5163'} />
                    <span className="num">{z >= 0 ? '+' : ''}{z.toFixed(2)}</span>
                    <span className="role">{readout ? `→ ${labels[i]}` : 'no readout'}</span>
                  </li>
                );
              })}
            </ul>
          </section>
        </>
      ) : (
        <section>
          <h3>
            <span className="step">2</span> Classical network
            <span className="sub">{driver.nParams.toLocaleString('en')} trained numbers</span>
          </h3>
          <p className="hint">A small neural network (one tanh hidden layer) maps the sensors to scores.</p>
        </section>
      )}

      <section>
        <h3>
          <span className="step">{quantum ? 4 : 3}</span> Decision
          <span className="sub">{quantum ? 'Q = w·⟨Z⟩ + b — highest score wins' : 'highest score wins'}</span>
        </h3>
        <ul className="actions">
          {labels.map((label, a) => {
            const q = decision?.q[a] ?? 0;
            const chosen = decision?.action === a;
            return (
              <li key={label} className={chosen ? 'chosen' : ''}>
                <span className="arrow">{ARROWS[label] ?? '·'}</span>
                <span className="label">{label}</span>
                <span className="bar">
                  <span
                    className="bar-fill"
                    style={{
                      left: `${((Math.min(q, 0) - qMin) / qSpan) * 100}%`,
                      width: `${(Math.abs(q) / qSpan) * 100}%`,
                      background: chosen ? '#ff9f1c' : '#5b6276',
                    }}
                  />
                </span>
                <span className="num">{q.toFixed(2)}</span>
              </li>
            );
          })}
        </ul>
      </section>

      {quantum && decision?.gates && (
        <section className="composer">
          <button type="button" className="link-button" onClick={openInComposer}>
            Open this decision in IBM Quantum Composer ↗
          </button>
          <p className="hint">
            The exact circuit with this decision's angles, ready to run on a real IBM quantum computer
            {quantum.nQubits > 4 ? ' (sign in to simulate more than 4 qubits there)' : ''}.
          </p>
        </section>
      )}
    </div>
  );
}
