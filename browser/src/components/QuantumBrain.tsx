/** What the focused car's driver sees, computes and decides — per decision. */
import { circuitToQasm } from '@qamposer/react';
import LZString from 'lz-string';
import { track } from '../analytics';
import { actionLabels } from '../sim/actions';
import { QuantumDriver } from '../sim/agents';
import type { WorldCar } from '../sim/world';
import { CircuitView } from './CircuitView';
import { layoutCircuit } from './CircuitView';
import { fmtInt, fmtNum, hasKey, t, useLang } from '../i18n';

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

/** An action label ("Brake right") in the active language. */
function actionLabel(label: string): string {
  const key = `action.${label.toLowerCase().replace(/ /g, '_')}`;
  return hasKey(key) ? t(key) : label;
}

/** An observer feature name ("ray +30°", "curvature ahead 12m") in the active language. */
export function featureLabel(name: string): string {
  const ray = /^ray (.+)$/.exec(name);
  if (ray) return t('feature.ray', { deg: ray[1] });
  const [, base, metres] = /^(.*?)(?: (\d+(?:\.\d+)?)m)?$/.exec(name)!;
  const key = `feature.${base.replace(/ /g, '_')}`;
  const label = hasKey(key) ? t(key) : base;
  return metres ? t('feature.lookahead', { name: label, m: fmtNum(Number(metres), 0) }) : label;
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
  useLang();
  if (!car || !car.driver || !car.observer) {
    return <p className="hint">{t('brain.pick')}</p>;
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
    window.open(composerUrl(qasm, t('brain.composer_title', { n: quantum.nQubits })), '_blank', 'noopener');
  };

  return (
    <div className="brain">
      <section>
        <h3>
          <span className="step">1</span> {t('brain.sensors')}
          <span className="sub">
            {t(quantum ? 'brain.sensors_sub_q' : 'brain.sensors_sub', { n: observer.nFeatures })}
          </span>
        </h3>
        <ul className="features">
          {observer.featureNames.map((name, i) => (
            <li key={name}>
              <span className="label">{featureLabel(name)}</span>
              <Bar value={obs?.[i] ?? 0} color={name.startsWith('ray') ? '#7a5cff' : '#22d3ee'} />
              <span className="num">{fmtNum(obs?.[i] ?? 0)}</span>
            </li>
          ))}
        </ul>
      </section>

      {quantum ? (
        <>
          <section>
            <h3>
              <span className="step">2</span> {t('brain.circuit')}
              <span className="sub">
                {t('brain.circuit_sub', {
                  n: quantum.nQubits,
                  layers: quantum.nLayers,
                  params: fmtInt(quantum.nParams),
                })}
              </span>
            </h3>
            <p className="hint">{t('brain.circuit_hint')}</p>
            <button type="button" className="chip expand" onClick={onToggleWide}>
              {t(wideCircuit ? 'brain.wide_back' : 'brain.wide_show')}
            </button>
            {wideCircuit ? null : decision?.gates ? (
              <CircuitView nQubits={quantum.nQubits} gates={decision.gates} zoom={quantum.nQubits > 6 ? 0.42 : 0.55} />
            ) : (
              <p className="hint">{t('brain.circuit_wait')}</p>
            )}
          </section>
          <section>
            <h3>
              <span className="step">3</span> {t('brain.measure')}
              <span className="sub">{t('brain.measure_sub')}</span>
            </h3>
            <ul className="qubits">
              {Array.from({ length: quantum.nQubits }, (_, i) => {
                // before the first decision there is no measurement yet
                const z = decision?.expectations?.[i];
                const readout = i < driver.nActions;
                return (
                  <li key={i} className={readout ? 'readout' : 'spectator'}>
                    <span className="label">q{i}</span>
                    <Bar value={z ?? 0} min={-1} max={1} color={readout ? '#b4a3ff' : '#4b5163'} />
                    <span className="num">{z === undefined ? '—' : fmtNum(z, 2, { sign: true })}</span>
                    <span className="role">{readout ? `→ ${actionLabel(labels[i])}` : t('brain.no_readout')}</span>
                  </li>
                );
              })}
            </ul>
          </section>
        </>
      ) : (
        <section>
          <h3>
            <span className="step">2</span> {t('brain.classical')}
            <span className="sub">{t('brain.classical_sub', { params: fmtInt(driver.nParams) })}</span>
          </h3>
          <p className="hint">{t('brain.classical_hint')}</p>
        </section>
      )}

      <section>
        <h3>
          <span className="step">{quantum ? 4 : 3}</span> {t('brain.decision')}
          <span className="sub">{t(quantum ? 'brain.decision_sub_q' : 'brain.decision_sub')}</span>
        </h3>
        <ul className="actions">
          {labels.map((label, a) => {
            const q = decision?.q[a] ?? 0;
            const chosen = decision?.action === a;
            return (
              <li key={label} className={chosen ? 'chosen' : ''}>
                <span className="arrow">{ARROWS[label] ?? '·'}</span>
                <span className="label">{actionLabel(label)}</span>
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
                <span className="num">{fmtNum(q)}</span>
              </li>
            );
          })}
        </ul>
      </section>

      {quantum && decision?.gates && (
        <section className="composer">
          <button type="button" className="link-button" onClick={openInComposer}>
            {t('brain.composer')}
          </button>
          <p className="hint">{t(quantum.nQubits > 4 ? 'brain.composer_hint_big' : 'brain.composer_hint')}</p>
        </section>
      )}
    </div>
  );
}
