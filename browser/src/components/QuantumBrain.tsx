/** What the focused car's driver sees, computes and decides — per decision. */
import { useEffect, useMemo, useRef, useState } from 'react';
import { circuitToQasm } from '@qamposer/react';
import LZString from 'lz-string';
import { track } from '../analytics';
import { actionLabels } from '../sim/actions';
import { argmax, QuantumDriver, type Decision } from '../sim/agents';
import { coverSensors, isLive, judgeGuess, nudgedDriver, trainedAngle, type AngleRef } from '../sim/whatif';
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
const FIRST_ANGLE: AngleRef = { block: 0, qubit: 0, gate: 'RY' };

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

/** The learner's guess against the driver's choice, and how close it was. */
function Verdict({ q, guess, labels }: { q: Float64Array; guess: number; labels: string[] }) {
  const { right, gap, margin } = judgeGuess(q, guess);
  const best = argmax(q);
  const action = actionLabel(labels[best]);
  return (
    <div className={right ? 'verdict right' : 'verdict wrong'} role="status">
      <strong>
        {right ? '✓ ' : '✗ '}
        {right
          ? t('brain.guess_right', { action })
          : t('brain.guess_wrong', { guess: actionLabel(labels[guess]), action })}
      </strong>{' '}
      {right
        ? labels.length > 1 && t('brain.guess_margin', { margin: fmtNum(margin) })
        : t('brain.guess_gap', { gap: fmtNum(gap), mine: fmtNum(q[guess]), best: fmtNum(q[best]) })}
    </div>
  );
}

/** Pick one trained RY/RZ rotation and shift it by up to ±π. */
function NudgePanel({
  driver,
  open,
  onOpen,
  angle,
  onAngle,
  offset,
  onOffset,
}: {
  driver: QuantumDriver;
  open: boolean;
  onOpen: (open: boolean) => void;
  angle: AngleRef;
  onAngle: (angle: AngleRef) => void;
  offset: number;
  onOffset: (offset: number) => void;
}) {
  const trained = trainedAngle(driver, angle);
  return (
    <div className="nudge">
      <button type="button" className={open ? 'chip active' : 'chip'} aria-expanded={open} onClick={() => onOpen(!open)}>
        {t('brain.nudge')}
      </button>
      {open && (
        <>
          <div className="nudge-pick">
            <label>
              {t('brain.nudge_block')}
              <select value={angle.block} onChange={(e) => onAngle({ ...angle, block: Number(e.target.value) })}>
                {Array.from({ length: driver.nLayers }, (_, l) => (
                  <option key={l} value={l}>
                    {l + 1}
                  </option>
                ))}
              </select>
            </label>
            <label>
              {t('brain.nudge_qubit')}
              <select value={angle.qubit} onChange={(e) => onAngle({ ...angle, qubit: Number(e.target.value) })}>
                {Array.from({ length: driver.nQubits }, (_, i) => (
                  <option key={i} value={i}>
                    q{i}
                  </option>
                ))}
              </select>
            </label>
            <label>
              {t('brain.nudge_gate')}
              <select value={angle.gate} onChange={(e) => onAngle({ ...angle, gate: e.target.value as 'RY' | 'RZ' })}>
                <option value="RY">RY</option>
                <option value="RZ">RZ</option>
              </select>
            </label>
          </div>
          <div className="nudge-slider">
            <input
              type="range"
              min={-Math.PI}
              max={Math.PI}
              step={0.01}
              value={offset}
              aria-label={t('brain.nudge_offset')}
              onChange={(e) => onOffset(Number(e.target.value))}
            />
            <span className="num">{fmtNum(offset, 2, { sign: true })}</span>
            <button type="button" className="mini" onClick={() => onOffset(0)} disabled={offset === 0}>
              {t('brain.nudge_reset')}
            </button>
          </div>
          <p className="hint">
            {t('brain.nudge_value', { from: fmtNum(trained), to: fmtNum(trained + offset) })}
          </p>
          {!isLive(driver, angle) && <p className="hint dead">{t('brain.nudge_dead')}</p>}
        </>
      )}
    </div>
  );
}

/** "With ray +30° covered: the choice changes, Straight → Left." */
function whatIfLine({
  covered,
  angle,
  offset,
  from,
  to,
}: {
  covered: string[];
  angle: AngleRef | null;
  offset: number;
  from: string;
  to: string;
}): string {
  const changes: string[] = [];
  if (covered.length) changes.push(t('brain.whatif_covered', { names: covered.join(', ') }));
  if (angle) {
    changes.push(
      t('brain.whatif_angle', {
        gate: angle.gate,
        qubit: angle.qubit,
        block: angle.block + 1,
        offset: fmtNum(offset, 2, { sign: true }),
      }),
    );
  }
  const joined = changes.join(' + ');
  return from === to
    ? t('brain.whatif_same', { changes: joined, action: from })
    : t('brain.whatif_changed', { changes: joined, from, to });
}

export function QuantumBrain({
  car,
  wideCircuit,
  onToggleWide,
  paused,
  guessMode,
  onGuessMode,
}: {
  car: WorldCar | undefined;
  /** the circuit is shown in the wide view under the track instead */
  wideCircuit: boolean;
  onToggleWide: () => void;
  /** the race is paused: the what-if tools and the guess are offered */
  paused: boolean;
  /** "Guess before you look": hide the scores until the learner picks a move */
  guessMode: boolean;
  onGuessMode: (on: boolean) => void;
}) {
  useLang();
  const driver = car?.driver;
  const decision = car?.decision;
  const obs = car?.obs;
  const quantum = driver instanceof QuantumDriver ? driver : null;

  // What-if state: inference on copies only (src/sim/whatif.ts), never the car.
  const [covered, setCovered] = useState<number[]>([]);
  const [nudgeOpen, setNudgeOpen] = useState(false);
  const [angle, setAngle] = useState<AngleRef>(FIRST_ANGLE);
  const [offset, setOffset] = useState(0);
  const [guess, setGuess] = useState<{ decision: Decision; action: number } | null>(null);
  const sent = useRef<{ sensor?: Decision; angle?: Decision }>({});

  // a running race shows the plain panel; another driver starts from scratch
  useEffect(() => {
    if (!paused) {
      setCovered([]);
      setOffset(0);
    }
  }, [paused]);
  useEffect(() => {
    setCovered([]);
    setAngle(FIRST_ANGLE);
    setOffset(0);
  }, [driver]);

  const active = paused && !!driver && !!decision && !!obs;
  const coverOn = active && covered.length > 0 && covered.every((i) => i < (driver?.nFeatures ?? 0));
  const angleValid = !!quantum && angle.block < quantum.nLayers && angle.qubit < quantum.nQubits;
  const angleOn = active && nudgeOpen && angleValid && offset !== 0;
  const whatIf = useMemo(() => {
    if (!driver || !obs || (!coverOn && !angleOn)) return null;
    const d = angleOn && quantum ? nudgedDriver(quantum, angle, offset) : driver;
    return coverSensors(d, obs, coverOn ? covered : []);
  }, [driver, quantum, obs, coverOn, angleOn, covered, angle, offset]);

  // analytics: at most once per decision and kind (not per slider move)
  useEffect(() => {
    if (!decision) return;
    if (coverOn && sent.current.sensor !== decision) {
      sent.current.sensor = decision;
      track('what-if', { kind: 'sensor' });
    }
    if (angleOn && sent.current.angle !== decision) {
      sent.current.angle = decision;
      track('what-if', { kind: 'angle' });
    }
  }, [decision, coverOn, angleOn]);

  if (!car || !driver || !car.observer) {
    return <p className="hint">{t('brain.pick')}</p>;
  }
  const { observer } = car;
  const labels = actionLabels(driver.nActions);
  // the guess belongs to one decision: Step (a new decision) hides the scores again
  const guessPending = active && guessMode && guess?.decision !== decision;
  const myGuess = active && guessMode && guess && guess.decision === decision ? guess.action : null;
  const qMin = decision ? Math.min(0, ...decision.q) : 0;
  const qMax = decision ? Math.max(0, ...decision.q) : 1;
  const qSpan = Math.max(qMax - qMin, 1e-9);

  const pickGuess = (a: number) => {
    if (!decision) return;
    setGuess({ decision, action: a });
    track('predict', { result: judgeGuess(decision.q, a).right ? 'right' : 'wrong', qubits: quantum?.nQubits });
  };
  const toggleCover = (i: number) =>
    setCovered((c) => (c.includes(i) ? c.filter((k) => k !== i) : [...c, i].sort((a, b) => a - b)));

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
        <ul className={active ? 'features coverable' : 'features'}>
          {observer.featureNames.map((name, i) => {
            const isCovered = active && covered.includes(i);
            const label = featureLabel(name);
            return (
              <li key={name} className={isCovered ? 'covered' : undefined}>
                <span className="label">{label}</span>
                <Bar value={obs?.[i] ?? 0} color={name.startsWith('ray') ? '#7a5cff' : '#22d3ee'} />
                <span className="num">{fmtNum(obs?.[i] ?? 0)}</span>
                {active && (
                  <button
                    type="button"
                    className={isCovered ? 'mini active' : 'mini'}
                    aria-pressed={isCovered}
                    aria-label={t('brain.cover_aria', { name: label })}
                    title={t('brain.cover_aria', { name: label })}
                    onClick={() => toggleCover(i)}
                  >
                    {t(isCovered ? 'brain.uncover' : 'brain.cover')}
                  </button>
                )}
              </li>
            );
          })}
        </ul>
        {active && <p className="hint">{t('brain.cover_hint')}</p>}
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
            {active && (
              <NudgePanel
                driver={quantum}
                open={nudgeOpen}
                onOpen={setNudgeOpen}
                angle={angleValid ? angle : FIRST_ANGLE}
                onAngle={(a) => {
                  setAngle(a);
                  setOffset(0);
                }}
                offset={offset}
                onOffset={setOffset}
              />
            )}
          </section>
          <section>
            <h3>
              <span className="step">3</span> {t('brain.measure')}
              <span className="sub">{t('brain.measure_sub')}</span>
            </h3>
            <ul className="qubits">
              {Array.from({ length: quantum.nQubits }, (_, i) => {
                // before the first decision there is no measurement yet; during a guess it stays hidden
                const z = guessPending ? undefined : decision?.expectations?.[i];
                const readout = i < driver.nActions;
                return (
                  <li key={i} className={readout ? 'readout' : 'spectator'}>
                    <span className="label">q{i}</span>
                    <Bar value={z ?? 0} min={-1} max={1} color={readout ? '#b4a3ff' : '#4b5163'} />
                    <span className="num">{z !== undefined ? fmtNum(z, 2, { sign: true }) : guessPending ? '?' : '—'}</span>
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
        <label className="toggle guess-toggle" title={t('brain.guess_tip')}>
          <input type="checkbox" checked={guessMode} onChange={(e) => onGuessMode(e.target.checked)} />
          {t('brain.guess_toggle')}
        </label>
        {guessMode && !active && <p className="hint">{t('brain.guess_paused')}</p>}
        {guessPending ? (
          <>
            <p className="ask">{t('brain.guess_ask')}</p>
            <div className="guess-grid">
              {labels.map((label, a) => (
                <button key={label} type="button" className="chip guess" onClick={() => pickGuess(a)}>
                  <span className="arrow">{ARROWS[label] ?? '·'}</span> {actionLabel(label)}
                </button>
              ))}
            </div>
            {whatIf && <p className="hint">{t('brain.whatif_wait')}</p>}
          </>
        ) : (
          <>
            {myGuess !== null && decision && <Verdict q={decision.q} guess={myGuess} labels={labels} />}
            <ul className={whatIf ? 'actions with-whatif' : 'actions'}>
              {whatIf && (
                <li className="head" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                  <span className="num">{t('brain.whatif_now')}</span>
                  <span className="num">{t('brain.whatif_col')}</span>
                </li>
              )}
              {labels.map((label, a) => {
                const q = decision?.q[a] ?? 0;
                const chosen = decision?.action === a;
                const classes = [chosen && 'chosen', myGuess === a && 'guessed'].filter(Boolean).join(' ');
                return (
                  <li key={label} className={classes || undefined} title={myGuess === a ? t('brain.guess_mine') : undefined}>
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
                    {whatIf && (
                      <span className={whatIf.action === a ? 'num whatif win' : 'num whatif'}>{fmtNum(whatIf.q[a])}</span>
                    )}
                  </li>
                );
              })}
            </ul>
            {whatIf && decision && (
              <p className={whatIf.action === decision.action ? 'whatif-line' : 'whatif-line changed'}>
                {whatIfLine({
                  covered: coverOn ? covered.map((i) => featureLabel(observer.featureNames[i])) : [],
                  angle: angleOn ? angle : null,
                  offset,
                  from: actionLabel(labels[decision.action]),
                  to: actionLabel(labels[whatIf.action]),
                })}
              </p>
            )}
            {myGuess !== null && <p className="hint">{t('brain.guess_next')}</p>}
          </>
        )}
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
