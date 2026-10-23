/**
 * "Train a tiny circuit — a toy": one or two qubits learn "steer toward the
 * longer ray" live, on a frozen batch of real readings (src/sim/toy.ts). The
 * page otherwise only shows the result of learning; this shows learning.
 *
 * Training runs in requestAnimationFrame chunks at STEPS_PER_SECOND, stops
 * when the panel closes (unmount) and pauses when the tab is hidden.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { track as trackEvent, outbound } from '../analytics';
import { loadTrack, type Manifest } from '../data';
import { Track } from '../sim/track';
import {
  DEFAULT_LEARNING_RATE,
  LEARNING_RATES,
  makeBatch,
  MAX_STEPS,
  N_TEST,
  N_TRAIN,
  paramRef,
  STEPS_PER_SECOND,
  TOY_TRACK,
  ToyTrainer,
  type ToyBatch,
  type ToyPoint,
  type ToyQubits,
  type ToySample,
} from '../sim/toy';
import { fmtNum, rich, t, useLang, type Key } from '../i18n';

const REPO = 'https://github.com/JanLahmann/racetraQ';
const GRID = 20;
/** validated pair (dataviz validator, dark surface): steer Left / steer Right */
const LEFT = '#d95926';
const RIGHT = '#3987e5';
/** training / held-out curves */
const TRAIN = '#9085e9';
const TEST = '#199e70';
const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const LEFT_RGB = rgb(LEFT);
const RIGHT_RGB = rgb(RIGHT);
/** --bg-panel, under the decision map */
const SURFACE_RGB = rgb('#161a22');
const LR_LABELS: Key[] = ['toy.lr_slow', 'toy.lr_medium', 'toy.lr_fast'];

function Out({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noopener" {...outbound(REPO)}>
      {children}
    </a>
  );
}

/** Stuck: 60 steps in, the loss flat over the last 20, the held-out accuracy still below 80 %. */
export function isStuck(history: ToyPoint[]): boolean {
  const last = history[history.length - 1];
  if (last.step < 60 || last.testAccuracy >= 0.8) return false;
  return Math.abs(history[history.length - 21].trainLoss - last.trainLoss) < 0.01;
}

// ---------------------------------------------------------------- scatter

/** Background: P(Left) per cell, orange (Left) through grey to blue (Right). */
function cellFill(p: number): string {
  const strength = Math.min(Math.abs(p - 0.5) * 2, 1) * 0.5;
  const pole = p >= 0.5 ? LEFT_RGB : RIGHT_RGB;
  // opaque: mixed with the panel colour, so neighbouring cells leave no seams
  const mix = pole.map((c, i) => Math.round(SURFACE_RGB[i] + (c - SURFACE_RGB[i]) * strength));
  return `rgb(${mix.join(',')})`;
}

function Scatter({ batch, grid }: { batch: ToyBatch; grid: Float64Array }) {
  const size = 240;
  const pad = { l: 34, b: 30, t: 6, r: 6 };
  const w = size + pad.l + pad.r;
  const h = size + pad.t + pad.b;
  const x = (v: number) => pad.l + v * size;
  const y = (v: number) => pad.t + (1 - v) * size;
  const cell = size / GRID;
  const label = (s: ToySample) => t(s.label === 0 ? 'action.left' : 'action.right');
  const point = (s: ToySample, test: boolean) => (
    <circle
      key={`${test ? 't' : 'b'}${s.left},${s.right}`}
      className="toy-point"
      cx={x(s.left)}
      cy={y(s.right)}
      r={test ? 4.5 : 4}
      fill={test ? 'none' : s.label === 0 ? LEFT : RIGHT}
      stroke={test ? (s.label === 0 ? LEFT : RIGHT) : 'var(--bg-panel)'}
      strokeWidth={test ? 2 : 1.5}
    >
      <title>
        {t(test ? 'toy.point_test' : 'toy.point', { left: fmtNum(s.left), right: fmtNum(s.right), label: label(s) })}
      </title>
    </circle>
  );
  return (
    <svg className="toy-scatter" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={t('toy.scatter_aria')}>
      <g>
        {Array.from(grid, (p, k) => {
          const r = Math.floor(k / GRID);
          const c = k % GRID;
          return <rect key={k} x={x(c / GRID)} y={y((r + 1) / GRID)} width={cell + 0.3} height={cell + 0.3} fill={cellFill(p)} />;
        })}
      </g>
      <rect x={pad.l} y={pad.t} width={size} height={size} fill="none" stroke="var(--border)" />
      {/* the rule itself: left = right */}
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--muted)" strokeDasharray="3 4" strokeWidth={1} />
      {[0, 0.5, 1].map((v) => (
        <g key={v} className="toy-tick">
          <text x={x(v)} y={pad.t + size + 13} textAnchor="middle">
            {fmtNum(v, v === 0.5 ? 1 : 0)}
          </text>
          <text x={pad.l - 5} y={y(v) + 4} textAnchor="end">
            {fmtNum(v, v === 0.5 ? 1 : 0)}
          </text>
        </g>
      ))}
      <text className="toy-axis" x={pad.l + size / 2} y={h - 3} textAnchor="middle">
        {t('toy.axis_left')}
      </text>
      <text className="toy-axis" transform={`translate(10 ${pad.t + size / 2}) rotate(-90)`} textAnchor="middle">
        {t('toy.axis_right')}
      </text>
      {batch.train.map((s) => point(s, false))}
      {batch.test.map((s) => point(s, true))}
    </svg>
  );
}

// ----------------------------------------------------------------- curves

function Curve({
  title,
  history,
  pick,
  max,
  percent,
}: {
  title: string;
  history: ToyPoint[];
  pick: (p: ToyPoint) => [number, number];
  max: number;
  percent?: boolean;
}) {
  const w = 300;
  const h = 96;
  const pad = { l: 34, r: 8, t: 8, b: 16 };
  const span = Math.max(30, history[history.length - 1].step);
  const x = (s: number) => pad.l + (s / span) * (w - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - Math.min(v, max) / max) * (h - pad.t - pad.b);
  const line = (i: 0 | 1) => history.map((p) => `${x(p.step).toFixed(1)},${y(pick(p)[i]).toFixed(1)}`).join(' ');
  const last = pick(history[history.length - 1]);
  const lastStep = history[history.length - 1].step;
  const fmt = (v: number) => (percent ? `${fmtNum(v * 100, 0)} %` : fmtNum(v));
  return (
    <figure className="toy-curve">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${title}: ${fmt(last[0])} / ${fmt(last[1])}`}>
        {[0, max / 2, max].map((v) => (
          <g key={v} className="toy-tick">
            <line x1={pad.l} x2={w - pad.r} y1={y(v)} y2={y(v)} stroke="var(--border)" />
            <text x={pad.l - 4} y={y(v) + 4} textAnchor="end">
              {percent ? fmtNum(v * 100, 0) : fmtNum(v, Number.isInteger(v) ? 0 : 1)}
            </text>
          </g>
        ))}
        <text className="toy-tick" x={w - pad.r} y={h - 3} textAnchor="end">
          {t('toy.steps', { n: span })}
        </text>
        <polyline points={line(0)} fill="none" stroke={TRAIN} strokeWidth={2} strokeLinejoin="round" />
        <polyline points={line(1)} fill="none" stroke={TEST} strokeWidth={2} strokeDasharray="5 3" strokeLinejoin="round" />
        {/* the latest point of each curve (also the only one before the first step) */}
        <circle cx={x(lastStep)} cy={y(last[0])} r={3.5} fill={TRAIN} stroke="var(--bg-panel)" strokeWidth={1.5} />
        <circle cx={x(lastStep)} cy={y(last[1])} r={3.5} fill={TEST} stroke="var(--bg-panel)" strokeWidth={1.5} />
      </svg>
    </figure>
  );
}

// ---------------------------------------------------------------- circuit

/** The circuit with live trained angles: encodings, rotations, CZ, readout. */
function ToyCircuit({ trainer }: { trainer: ToyTrainer }) {
  const { model, params } = trainer;
  type Col = { kind: 'enc'; ray: 'L' | 'R'; qubit: number }[] | { kind: 'rot'; k: number }[] | 'cz';
  const cols: Col[] = [];
  const rot = (ks: number[]) => ks.map((k) => ({ kind: 'rot' as const, k }));
  if (model.qubits === 1) {
    cols.push([{ kind: 'enc', ray: 'L', qubit: 0 }], rot([0]), rot([1]), [{ kind: 'enc', ray: 'R', qubit: 0 }], rot([2]), rot([3]));
  } else {
    for (let b = 0; b < model.blocks; b++) {
      cols.push(
        [
          { kind: 'enc', ray: 'L', qubit: 0 },
          { kind: 'enc', ray: 'R', qubit: 1 },
        ],
        rot([b * 4, b * 4 + 2]),
        rot([b * 4 + 1, b * 4 + 3]),
        'cz',
      );
    }
  }
  const colW = 50;
  const laneH = 46;
  const left = 30;
  const readW = 78;
  const w = left + cols.length * colW + readW;
  const h = model.qubits * laneH + 8;
  const lane = (q: number) => 4 + laneH / 2 + q * laneH;
  const dead = (k: number) => {
    const ref = paramRef(model, k);
    return ref.gate === 'RZ' && ref.block === (model.qubits === 1 ? 1 : model.blocks - 1);
  };
  const box = (cx: number, q: number, label: string, value: string, cls: string, key: string) => (
    <g key={key} className={cls}>
      <rect x={cx - 21} y={lane(q) - 16} width={42} height={32} rx={4} />
      <text x={cx} y={lane(q) - 3} textAnchor="middle" className="gate">
        {label}
      </text>
      <text x={cx} y={lane(q) + 10} textAnchor="middle" className="val">
        {value}
      </text>
    </g>
  );
  const readouts = model.qubits === 1 ? [t('toy.readout_1')] : [t('toy.readout_2', { q: 0, action: t('action.left') }), t('toy.readout_2', { q: 1, action: t('action.right') })];
  return (
    <div className="toy-circuit-scroll">
      <svg className="toy-circuit" viewBox={`0 0 ${w} ${h}`} style={{ minWidth: Math.min(w, 520) }} role="img" aria-label={t('toy.circuit_aria', { n: model.qubits })}>
        {Array.from({ length: model.qubits }, (_, q) => (
          <g key={q}>
            <line x1={left - 6} x2={w - readW + 6} y1={lane(q)} y2={lane(q)} className="wire" />
            <text x={2} y={lane(q) + 4} className="qlabel">
              q{q}
            </text>
            <text x={w - readW + 10} y={lane(q) + 4} className="readout">
              {readouts[q]}
            </text>
          </g>
        ))}
        {cols.map((col, c) => {
          const cx = left + c * colW + colW / 2;
          if (col === 'cz') {
            return (
              <g key={c} className="cz">
                <line x1={cx} x2={cx} y1={lane(0)} y2={lane(1)} />
                <circle cx={cx} cy={lane(0)} r={4} />
                <circle cx={cx} cy={lane(1)} r={4} />
              </g>
            );
          }
          return col.map((g) =>
            g.kind === 'enc'
              ? box(cx, g.qubit, 'RY', t(g.ray === 'L' ? 'toy.enc_left' : 'toy.enc_right'), 'enc', `${c}-${g.qubit}`)
              : (() => {
                  const ref = paramRef(model, g.k);
                  return box(cx, ref.qubit, ref.gate, fmtNum(params[g.k], 2), dead(g.k) ? 'rot dead' : 'rot', `${c}-${g.k}`);
                })(),
          );
        })}
      </svg>
    </div>
  );
}

// ------------------------------------------------------------------ panel

export function ToyPanel({ manifest, onClose }: { manifest: Manifest; onClose: () => void }) {
  useLang();
  const [trackObj, setTrackObj] = useState<Track | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [seed, setSeed] = useState(1);
  const [qubits, setQubits] = useState<ToyQubits>(2);
  const [lr, setLr] = useState(DEFAULT_LEARNING_RATE);
  const [running, setRunning] = useState(false);
  const [, setTick] = useState(0);
  const sentFor = useRef<ToyTrainer | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  useEffect(() => {
    let stale = false;
    loadTrack(TOY_TRACK).then(
      (json) => !stale && setTrackObj(new Track(json, manifest.resample_spacing)),
      (e) => !stale && setError(String(e)),
    );
    return () => {
      stale = true;
    };
  }, [manifest]);

  const batch = useMemo(() => (trackObj ? makeBatch(trackObj, manifest.physics, seed) : null), [trackObj, manifest, seed]);
  // the learning rate is not a dependency: changing it mid-run keeps the run
  const trainer = useMemo(() => (batch ? new ToyTrainer(batch, qubits, { learningRate: lr }) : null), [batch, qubits]);
  useEffect(() => {
    if (trainer) trainer.learningRate = lr;
  }, [trainer, lr]);

  // a new seed or circuit is a new run: stop, start from scratch
  useEffect(() => setRunning(false), [trainer]);

  /** Count a training run once, when it first moves (Train or Step). */
  const started = useCallback(() => {
    if (trainer && sentFor.current !== trainer) {
      sentFor.current = trainer;
      trackEvent('toy train', { qubits });
    }
  }, [trainer, qubits]);

  const stepOnce = () => {
    if (!trainer || trainer.done) return;
    started();
    trainer.step();
    setTick((n) => n + 1);
  };

  // the training loop: requestAnimationFrame chunks, STEPS_PER_SECOND
  useEffect(() => {
    if (!running || !trainer) return;
    let frame = 0;
    let last = performance.now();
    let budget = 0;
    const loop = () => {
      // performance.now(), not the frame timestamp: the two clocks may differ
      const now = performance.now();
      budget += Math.min(Math.max(now - last, 0), 250) / 1000;
      last = now;
      let moved = false;
      while (budget * STEPS_PER_SECOND >= 1 && !trainer.done) {
        trainer.step();
        budget -= 1 / STEPS_PER_SECOND;
        moved = true;
      }
      if (moved) setTick((n) => n + 1);
      if (trainer.done) {
        setRunning(false);
        return;
      }
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [running, trainer]);

  // a hidden tab pauses the run
  useEffect(() => {
    const onVisibility = () => document.hidden && setRunning(false);
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, []);

  const toggle = () => {
    if (!trainer || trainer.done) return;
    if (!running) {
      started();
      trainer.step(); // answer the click at once
      setTick((n) => n + 1);
    }
    setRunning((r) => !r);
  };

  const steps = trainer?.steps;
  const grid = useMemo(() => (trainer ? trainer.decisionGrid(GRID) : null), [trainer, steps]);
  const point = trainer?.last;
  const stuck = !!trainer && !!point && isStuck(trainer.history);
  const pct = (v: number) => `${fmtNum(v * 100, 0)} %`;
  // whole numbers, so the middle tick reads 0.5, 1 or 1.5
  const lossMax = trainer ? Math.max(1, Math.ceil(Math.max(...trainer.history.map((p) => Math.max(p.trainLoss, p.testLoss))))) : 1;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal toy" role="dialog" aria-modal="true" aria-label={t('toy.aria')} onClick={(e) => e.stopPropagation()}>
        <button type="button" className="close" onClick={onClose} aria-label={t('toy.close')}>
          ✕
        </button>
        <h2>
          {t('toy.title')} <span className="toy-badge">{t('toy.badge')}</span>
        </h2>
        <p className="toy-lead">{t('toy.intro', { train: N_TRAIN })}</p>
        <p className="toy-honest">
          <strong>{t('toy.honest_title')}</strong> {t('toy.honest')}
        </p>

        {error && <p className="hint">{t('app.error', { error })}</p>}
        {!trainer || !batch || !grid || !point ? (
          !error && <p className="hint">{t('toy.loading')}</p>
        ) : (
          <>
            <div className="toy-controls">
              <button type="button" className={running ? 'chip active' : 'chip'} onClick={toggle} disabled={trainer.done}>
                {t(running ? 'toy.pause' : 'toy.train')}
              </button>
              <button type="button" className="chip" onClick={stepOnce} disabled={running || trainer.done}>
                {t('toy.step')}
              </button>
              <button type="button" className="chip" onClick={() => setSeed((s) => s + 1)}>
                {t('toy.reset')}
              </button>
              <span className="toy-group" role="group" aria-label={t('toy.lr')}>
                <span className="control-label">{t('toy.lr')}</span>
                {LEARNING_RATES.map((v, i) => (
                  <button
                    key={v}
                    type="button"
                    className={v === lr ? 'chip active' : 'chip'}
                    aria-pressed={v === lr}
                    title={`η = ${fmtNum(v, 2)}`}
                    onClick={() => setLr(v)}
                  >
                    {t(LR_LABELS[i])}
                  </button>
                ))}
              </span>
              <span className="toy-group" role="group" aria-label={t('toy.qubits')}>
                <span className="control-label">{t('toy.qubits')}</span>
                {([2, 1] as const).map((q) => (
                  <button
                    key={q}
                    type="button"
                    className={q === qubits ? 'chip active' : 'chip'}
                    aria-pressed={q === qubits}
                    onClick={() => setQubits(q)}
                  >
                    {t(q === 1 ? 'toy.q1' : 'toy.q2')}
                  </button>
                ))}
              </span>
            </div>
            <p className="toy-status" role="status">
              {t('toy.status', {
                step: point.step,
                max: MAX_STEPS,
                seed,
                train: pct(point.trainAccuracy),
                test: pct(point.testAccuracy),
                loss: fmtNum(point.trainLoss),
              })}
            </p>
            {stuck ? (
              <p className="hint toy-stuck">{t('toy.stuck')}</p>
            ) : (
              trainer.done && <p className="hint">{t('toy.done', { n: MAX_STEPS })}</p>
            )}

            <div className="toy-grid">
              <section>
                <h3>{t('toy.scatter')}</h3>
                <Scatter batch={batch} grid={grid} />
                <ul className="toy-legend">
                  <li>
                    <span className="swatch" style={{ background: LEFT }} /> {t('toy.legend_left')}
                  </li>
                  <li>
                    <span className="swatch" style={{ background: RIGHT }} /> {t('toy.legend_right')}
                  </li>
                  <li>
                    <span className="swatch ring" /> {t('toy.legend_test', { n: N_TEST })}
                  </li>
                </ul>
                <p className="hint">{t('toy.scatter_hint')}</p>
              </section>
              <section>
                <h3>{t('toy.curves')}</h3>
                <Curve
                  title={t('toy.loss')}
                  history={trainer.history}
                  pick={(p) => [p.trainLoss, p.testLoss]}
                  max={lossMax}
                />
                <Curve
                  title={t('toy.accuracy')}
                  history={trainer.history}
                  pick={(p) => [p.trainAccuracy, p.testAccuracy]}
                  max={1}
                  percent
                />
                <ul className="toy-legend">
                  <li>
                    <span className="line" style={{ borderColor: TRAIN }} /> {t('toy.legend_train', { n: N_TRAIN })}
                  </li>
                  <li>
                    <span className="line dashed" style={{ borderColor: TEST }} /> {t('toy.legend_heldout', { n: N_TEST })}
                  </li>
                </ul>
              </section>
            </div>

            <section>
              <h3>
                {t('toy.circuit')}
                <span className="sub">
                  {t(qubits === 1 ? 'toy.circuit_sub_1' : 'toy.circuit_sub_2', { params: trainer.model.nParams })}
                </span>
              </h3>
              <ToyCircuit trainer={trainer} />
              <p className="hint">{t(qubits === 1 ? 'toy.circuit_hint_1' : 'toy.circuit_hint_2')}</p>
              <p className="hint">{t('toy.dead')}</p>
            </section>
          </>
        )}

        <section>
          <h3>{t('toy.how_title')}</h3>
          <p>{t('toy.how', { test: N_TEST })}</p>
          <p>{t('toy.shift')}</p>
        </section>

        <p className="hint">
          {rich('toy.more', {
            notebooks: <Out href={`${REPO}#notebooks`}>{t('about.notebooks')}</Out>,
            path: <Out href={`${REPO}/blob/main/docs/LEARN.md`}>{t('about.path')}</Out>,
          })}
        </p>
      </div>
    </div>
  );
}
