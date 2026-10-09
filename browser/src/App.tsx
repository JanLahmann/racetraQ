import { useEffect, useMemo, useRef, useState } from 'react';
import { findDriver, loadManifest, quantumDrivers, type DriverInfo, type Manifest } from './data';
import { buildRace, type BuiltRace, type Mode, type RaceSetup, type Rival } from './race';
import { outbound, track } from './analytics';
import { About } from './components/About';
import { Lapboard, fmtLap } from './components/Lapboard';
import { QuantumBrain } from './components/QuantumBrain';
import { CircuitView } from './components/CircuitView';
import { QuantumDriver } from './sim/agents';
import { TouchControls, keysToControls } from './components/TouchControls';
import { TrackStage } from './components/TrackStage';
import { CAMERA_VIEWS, type CameraView } from '@demo/race.js';

const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: 'watch', label: 'Watch', hint: 'a quantum driver races; see every decision' },
  { id: 'race', label: 'Race', hint: 'you against the quantum driver' },
  { id: 'evolution', label: 'Evolution', hint: 'four snapshots from one training run: watch it learn' },
];
const SPEEDS = [0.25, 0.5, 1, 2, 4];
const KEYMAP: Record<string, 'left' | 'right' | 'gas' | 'brake'> = {
  ArrowLeft: 'left',
  KeyA: 'left',
  ArrowRight: 'right',
  KeyD: 'right',
  ArrowUp: 'gas',
  KeyW: 'gas',
  ArrowDown: 'brake',
  KeyS: 'brake',
  Space: 'brake',
};

function driverChoices(manifest: Manifest, track: string): DriverInfo[] {
  const own = quantumDrivers(manifest, track);
  const universal = findDriver(manifest, 'quantum_universal');
  return universal ? [...own, universal] : own;
}

function driverChipLabel(d: DriverInfo): string {
  const n = d.circuit?.n_qubits ?? 4;
  return d.track === 'multi' ? `${n} qubits · all-round` : `${n} qubits`;
}

function evalLine(d: DriverInfo | undefined): string | null {
  const e = d?.fresh_eval;
  if (!e?.episodes || e.lapped_fraction == null) return null;
  const laps = Math.round(e.lapped_fraction * e.episodes);
  return `Tested on ${e.episodes} fresh episodes: lapped in ${laps}/${e.episodes}, mean lap ${fmtLap(e.mean_lap)}.`;
}

const ABOUT_SEEN_KEY = 'racetraq-about-seen';

/** First visit in this browser: open "What is this?" once on its own. */
function firstVisit(): boolean {
  try {
    return !localStorage.getItem(ABOUT_SEEN_KEY);
  } catch {
    return false; // storage blocked: don't pop the explainer on every load
  }
}

const CAMERA_LABELS: Record<CameraView, string> = { top: 'Top', chase: 'Chase', cockpit: 'Cockpit' };
const CAMERA_KEY = 'racetraq-camera';

/** Without a saved choice: the chase camera on touch screens, where the
 *  top-down car is a few pixels long, unless the visitor asked for reduced
 *  motion (the chase view turns with the car). */
function defaultCamera(): CameraView {
  const query = (q: string) => typeof window.matchMedia === 'function' && window.matchMedia(q).matches;
  return query('(pointer: coarse)') && !query('(prefers-reduced-motion: reduce)') ? 'chase' : 'top';
}

/** The camera view this browser used last (the default when unknown). */
function loadCamera(): CameraView {
  try {
    const saved = localStorage.getItem(CAMERA_KEY) as CameraView | null;
    return saved && CAMERA_VIEWS.includes(saved) ? saved : defaultCamera();
  } catch {
    return defaultCamera();
  }
}

export function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [setup, setSetup] = useState<RaceSetup>({
    track: 'oval',
    mode: 'watch',
    driver: 'quantum_oval',
    rival: 'mlp',
    ghost: true,
  });
  const [race, setRace] = useState<BuiltRace | null>(null);
  const [focusId, setFocusId] = useState('quantum');
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showRays, setShowRays] = useState(true);
  const [camera, setCamera] = useState<CameraView>(loadCamera);
  const [aboutOpen, setAboutOpen] = useState(firstVisit);
  const [wideCircuit, setWideCircuit] = useState(false);
  const [restartKey, setRestartKey] = useState(0);
  const [, setTick] = useState(0);
  const keys = useRef({ left: false, right: false, gas: false, brake: false });

  const closeAbout = () => {
    setAboutOpen(false);
    try {
      localStorage.setItem(ABOUT_SEEN_KEY, '1');
    } catch {
      // storage blocked: the explainer opens again next visit
    }
  };
  const openAbout = () => {
    track('about open');
    setAboutOpen(true);
  };

  const cycleCamera = () => {
    const next = CAMERA_VIEWS[(CAMERA_VIEWS.indexOf(camera) + 1) % CAMERA_VIEWS.length];
    setCamera(next);
    try {
      localStorage.setItem(CAMERA_KEY, next);
    } catch {
      // storage blocked: the choice just isn't remembered
    }
  };

  useEffect(() => {
    loadManifest().then(setManifest, (e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!manifest) return;
    let stale = false;
    buildRace(manifest, setup).then(
      (built) => {
        if (stale) return;
        if (import.meta.env.DEV) (window as unknown as { __racetraq: BuiltRace }).__racetraq = built;
        setRace(built);
        setFocusId(built.focusId);
        setPaused(false);
      },
      (e) => !stale && setError(String(e)),
    );
    return () => {
      stale = true;
    };
  }, [manifest, setup, restartKey]);

  // a visitor's finished race laps
  useEffect(() => {
    if (!race || setup.mode !== 'race') return;
    const { world } = race;
    return world.on((e) => {
      if (e.kind !== 'lap' || e.carId !== 'human') return;
      track('race lap', {
        track: setup.track,
        lap_time: e.lapTime.toFixed(1),
        clean: e.clean ? 'yes' : 'no',
        opponent: setup.driver,
      });
    });
  }, [race, setup.mode, setup.track, setup.driver]);

  // panels refresh at ~10 Hz; the canvas runs at the display rate
  useEffect(() => {
    const id = window.setInterval(() => setTick((t) => t + 1), 100);
    return () => window.clearInterval(id);
  }, []);

  // keyboard: driving in race mode, shortcuts everywhere
  useEffect(() => {
    const apply = () => {
      if (race) race.world.humanControls = keysToControls(keys.current);
    };
    const down = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      const key = KEYMAP[e.code];
      if (setup.mode === 'race' && key) {
        keys.current[key] = true;
        apply();
        e.preventDefault();
        return;
      }
      if (e.code === 'KeyC' && setup.mode === 'race' && !e.repeat && !e.metaKey && !e.ctrlKey) {
        cycleCamera();
        return;
      }
      if (e.code === 'KeyP' || (e.code === 'Space' && setup.mode !== 'race')) {
        setPaused((p) => !p);
        e.preventDefault();
      } else if (e.code === 'KeyR') {
        setRestartKey((k) => k + 1);
      }
    };
    const up = (e: KeyboardEvent) => {
      const key = KEYMAP[e.code];
      if (key) {
        keys.current[key] = false;
        apply();
      }
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, [race, setup.mode, camera]);

  const choices = useMemo(() => (manifest ? driverChoices(manifest, setup.track) : []), [manifest, setup.track]);
  const update = (patch: Partial<RaceSetup>) => setSetup((s) => ({ ...s, ...patch }));
  const pickTrack = (id: string) => {
    if (id === setup.track) return;
    track('track change', { track: id, mode: setup.mode });
    update({ track: id, driver: `quantum_${id}` });
  };
  const pickMode = (mode: Mode) => {
    if (mode === setup.mode) return;
    track('mode change', { mode, track: setup.track });
    update({ mode });
  };
  const pickDriver = (d: DriverInfo) => {
    if (d.id === setup.driver) return;
    track('driver change', { driver: d.id, qubits: d.circuit?.n_qubits });
    update({ driver: d.id });
  };
  const pickRival = (rival: Rival) => {
    if (rival === setup.rival) return;
    track('rival change', { rival });
    update({ rival });
  };

  if (error) {
    return (
      <div className="fatal">
        <h1>racetraQ</h1>
        <p>Could not load the race data: {error}</p>
      </div>
    );
  }
  if (!manifest) return <div className="loading">Loading racetraQ…</div>;

  const world = race?.world;
  const focus = world?.cars.find((c) => c.id === focusId);
  const human = world?.cars.find((c) => c.kind === 'human');
  const countdown = world && world.substep < world.holdUntil
    ? Math.ceil((world.holdUntil - world.substep) * world.physics.dt)
    : null;
  const selected = findDriver(manifest, setup.driver);
  const stepDecision = () => {
    if (!world) return;
    for (let k = 0; k < world.physics.substeps_per_decision; k++) world.step();
    setTick((t) => t + 1);
  };

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo">
            racetra<span className="q">Q</span>
          </span>
          <span className="edition">browser edition</span>
        </div>
        <nav className="tracks" aria-label="Track">
          {manifest.tracks.map((t) => (
            <button
              key={t.id}
              type="button"
              className={t.id === setup.track ? 'active' : ''}
              onClick={() => pickTrack(t.id)}
            >
              {t.name}
            </button>
          ))}
        </nav>
        <nav className="modes" aria-label="Mode">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              title={m.hint}
              className={m.id === setup.mode ? 'active' : ''}
              onClick={() => pickMode(m.id)}
            >
              {m.label}
            </button>
          ))}
        </nav>
        <button type="button" className="about-button" onClick={openAbout}>
          What is this?
        </button>
      </header>
      <p className="pitch">
        A tiny quantum circuit learned to drive this car. Watch it lap, then race it yourself.{' '}
        <button type="button" className="link" onClick={openAbout}>
          How does it work?
        </button>
      </p>

      <main>
        <div className="left">
          {race && world ? (
            <TrackStage
              world={world}
              track={race.track}
              mode={setup.mode === 'watch' ? 'attract' : setup.mode}
              paused={paused}
              speed={setup.mode === 'race' ? 1 : speed}
              showRays={showRays}
              camera={setup.mode === 'race' ? camera : 'top'}
            >
              <div className="hud">
                {focus && (
                  <div className="clock" style={{ borderColor: focus.color }}>
                    <span className="who">{setup.mode === 'race' && human ? 'You' : focus.label}</span>
                    <span className="time">{fmtLap(world.lapClock(human ?? focus))}</span>
                  </div>
                )}
              </div>
              {countdown !== null && <div className="countdown">{countdown}</div>}
              {setup.mode === 'race' && human?.frozenUntil !== null && human && (
                <div className="banner danger">Off track — back to the start</div>
              )}
              {paused && <div className="banner">Paused — press Step to advance one decision</div>}
              {setup.mode === 'race' && <TouchControls onChange={(patch) => {
                Object.assign(keys.current, patch);
                world.humanControls = keysToControls(keys.current);
              }} />}
            </TrackStage>
          ) : (
            <div className="stage loading">Loading track…</div>
          )}

          {wideCircuit && focus?.driver instanceof QuantumDriver && focus.decision?.gates && (
            <div className="wide-circuit">
              <div className="wide-circuit-head">
                <span>
                  Circuit of the current decision · {focus.label} · {focus.driver.nQubits} qubits ×{' '}
                  {focus.driver.nLayers} blocks
                </span>
                <button type="button" className="chip" onClick={() => setWideCircuit(false)}>
                  ✕
                </button>
              </div>
              <CircuitView
                nQubits={focus.driver.nQubits}
                gates={focus.decision.gates}
                zoom={focus.driver.nQubits > 6 ? 0.45 : 0.62}
              />
            </div>
          )}

          <div className="controls">
            {setup.mode !== 'evolution' && (
              <div className="control-group">
                <span className="control-label">Quantum driver</span>
                <div className="chips">
                  {choices.map((d) => (
                    <button
                      key={d.id}
                      type="button"
                      className={d.id === setup.driver ? 'chip active' : 'chip'}
                      onClick={() => pickDriver(d)}
                    >
                      {driverChipLabel(d)}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {setup.mode === 'watch' && (
              <div className="control-group">
                <span className="control-label">Rival</span>
                <div className="chips">
                  {(
                    [
                      ['none', 'none'],
                      ['mlp', 'classical MLP'],
                      ['pro', 'classical pro'],
                    ] as [Rival, string][]
                  ).map(([id, label]) => (
                    <button
                      key={id}
                      type="button"
                      className={id === setup.rival ? 'chip active' : 'chip'}
                      onClick={() => pickRival(id)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {setup.mode !== 'race' && (
              <div className="control-group">
                <span className="control-label">Speed</span>
                <div className="chips">
                  {SPEEDS.map((s) => (
                    <button
                      key={s}
                      type="button"
                      className={s === speed ? 'chip active' : 'chip'}
                      onClick={() => setSpeed(s)}
                    >
                      {s}×
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="control-group buttons">
              {setup.mode !== 'race' && (
                <>
                  <button type="button" className="chip" onClick={() => setPaused((p) => !p)}>
                    {paused ? '▶ Resume' : '❚❚ Pause'}
                  </button>
                  <button type="button" className="chip" onClick={stepDecision} disabled={!paused}>
                    Step
                  </button>
                </>
              )}
              <button type="button" className="chip" onClick={() => setRestartKey((k) => k + 1)}>
                ↺ Restart
              </button>
              {setup.mode === 'race' && (
                <button
                  type="button"
                  className="chip"
                  title="Camera (C): top-down, chase (turns with your car) or cockpit (closer, more road ahead)"
                  onClick={(e) => {
                    cycleCamera();
                    e.currentTarget.blur();
                  }}
                >
                  📷 {CAMERA_LABELS[camera]}
                </button>
              )}
              <label className="toggle">
                <input type="checkbox" checked={showRays} onChange={(e) => setShowRays(e.target.checked)} />
                lidar
              </label>
              {setup.mode !== 'evolution' &&
                !(setup.mode === 'watch' && setup.driver === `quantum_${setup.track}`) &&
                manifest.ghosts.some((g) => g.track === setup.track) && (
                <label className="toggle">
                  <input type="checkbox" checked={setup.ghost} onChange={(e) => update({ ghost: e.target.checked })} />
                  ghost
                </label>
              )}
            </div>
            {setup.mode === 'race' && (
              <p className="hint">
                Drive with the arrow keys or WASD (Space brakes) — or the on-screen pedals. C switches the
                camera: chase and cockpit turn with your car. The quantum car decides 10 times a second.
              </p>
            )}
            {setup.mode === 'evolution' && (
              <p className="hint">
                Four snapshots of the same quantum driver, saved at different points of one training run. Click a
                car in the table to look inside its circuit.
              </p>
            )}
            {setup.mode !== 'evolution' && evalLine(selected) && <p className="hint">{evalLine(selected)}</p>}
          </div>

          {world && (
            <Lapboard world={world} focusId={focusId} onFocus={(id) => setFocusId(id)} />
          )}
        </div>

        <aside className="panel">
          <h2>
            Inside the driver
            {focus && (
              <span className="focus-name" style={{ color: focus.color }}>
                {' '}
                · {focus.label}
              </span>
            )}
          </h2>
          <QuantumBrain car={focus} wideCircuit={wideCircuit} onToggleWide={() => setWideCircuit((w) => !w)} />
        </aside>
      </main>

      <footer>
        Inference only: the drivers were trained offline with{' '}
        <a href="https://github.com/JanLahmann/racetraQ" target="_blank" rel="noopener" {...outbound('https://github.com/JanLahmann/racetraQ')}>
          racetraQ
        </a>{' '}
        (Python, Qiskit). Every circuit is simulated in this browser by{' '}
        <a href="https://qamposer.org" target="_blank" rel="noopener" {...outbound('https://qamposer.org')}>
          QAMPoser
        </a>
        's state-vector simulator — no server.{' '}
        <a href="https://fun-with-quantum.org" target="_blank" rel="noopener" {...outbound('https://fun-with-quantum.org')}>
          Fun with Quantum
        </a>
      </footer>

      {aboutOpen && <About manifest={manifest} onClose={closeAbout} />}
    </div>
  );
}
