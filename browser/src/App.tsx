import { useEffect, useMemo, useRef, useState } from 'react';
import { findDriver, loadManifest, quantumDrivers, type DriverInfo, type Manifest } from './data';
import { buildRace, labelCars, type BuiltRace, type Mode, type RaceSetup, type Rival } from './race';
import { outbound, track } from './analytics';
import { About } from './components/About';
import { Lapboard, fmtLap } from './components/Lapboard';
import { QuantumBrain } from './components/QuantumBrain';
import { CircuitView } from './components/CircuitView';
import { QuantumDriver } from './sim/agents';
import { TouchControls, keysToControls } from './components/TouchControls';
import { TrackStage } from './components/TrackStage';
import { LangSwitch } from './components/LangSwitch';
import { CAMERA_VIEWS, type CameraView } from '@demo/race.js';
import { hasKey, rich, t, useLang, type Key } from './i18n';

const MODES: { id: Mode; label: Key; hint: Key }[] = [
  { id: 'watch', label: 'mode.watch', hint: 'mode.watch_hint' },
  { id: 'race', label: 'mode.race', hint: 'mode.race_hint' },
  { id: 'evolution', label: 'mode.evolution', hint: 'mode.evolution_hint' },
];
const RIVALS: [Rival, Key][] = [
  ['none', 'rival.none'],
  ['mlp', 'rival.mlp'],
  ['pro', 'rival.pro'],
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
  return t(d.track === 'multi' ? 'chip.qubits_multi' : 'chip.qubits', { n });
}

function evalLine(d: DriverInfo | undefined): string | null {
  const e = d?.fresh_eval;
  if (!e?.episodes || e.lapped_fraction == null) return null;
  const laps = Math.round(e.lapped_fraction * e.episodes);
  return t('hint.eval', { episodes: e.episodes, laps, lap: fmtLap(e.mean_lap) });
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

const CAMERA_LABELS: Record<CameraView, Key> = { top: 'camera.top', chase: 'camera.chase', cockpit: 'camera.cockpit' };

/** A track name in the active language (the manifest's name when unknown). */
function trackName(id: string, name: string): string {
  const key = `track.${id}`;
  return hasKey(key) ? t(key) : name;
}
const CAMERA_KEY = 'racetraq-camera';

const query = (q: string) => typeof window.matchMedia === 'function' && window.matchMedia(q).matches;
const isTouch = () => query('(pointer: coarse)');

/** Without a saved choice: the chase camera on touch screens, where the
 *  top-down car is a few pixels long, unless the visitor asked for reduced
 *  motion (the chase view turns with the car). */
function defaultCamera(): CameraView {
  return isTouch() && !query('(prefers-reduced-motion: reduce)') ? 'chase' : 'top';
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
  const lang = useLang();
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

  // car names follow the language, also mid-race
  useEffect(() => {
    if (manifest && race) labelCars(manifest, race);
  }, [manifest, race, lang]);

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
    // the four snapshots take a moment to load: don't show the old cars and laps meanwhile
    if (mode === 'evolution') setRace(null);
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
        <p>{t('app.error', { error })}</p>
      </div>
    );
  }
  if (!manifest) return <div className="loading">{t('app.loading')}</div>;

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
          <span className="edition">{t('header.edition')}</span>
        </div>
        <nav className="tracks" aria-label={t('header.tracks')}>
          {manifest.tracks.map((tr) => (
            <button
              key={tr.id}
              type="button"
              className={tr.id === setup.track ? 'active' : ''}
              onClick={() => pickTrack(tr.id)}
            >
              {trackName(tr.id, tr.name)}
            </button>
          ))}
        </nav>
        <nav className="modes" aria-label={t('header.modes')}>
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              title={t(m.hint)}
              className={m.id === setup.mode ? 'active' : ''}
              onClick={() => pickMode(m.id)}
            >
              {t(m.label)}
            </button>
          ))}
        </nav>
        <button type="button" className="about-button" onClick={openAbout}>
          {t('header.about')}
        </button>
        <LangSwitch />
      </header>
      <p className="pitch">
        {t('pitch.text')}{' '}
        <button type="button" className="link" onClick={openAbout}>
          {t('pitch.how')}
        </button>
      </p>

      <main>
        <div className="left">
          {race && world ? (
            <TrackStage
              world={world}
              track={race.track}
              mode={setup.mode === 'watch' ? 'attract' : setup.mode}
              paused={paused || (setup.mode === 'race' && aboutOpen)}
              speed={setup.mode === 'race' ? 1 : speed}
              showRays={showRays}
              camera={setup.mode === 'race' ? camera : 'top'}
            >
              <div className="hud">
                {focus && (
                  <div className="clock" style={{ borderColor: focus.color }}>
                    <span className="who">{setup.mode === 'race' && human ? t('hud.you') : focus.label}</span>
                    <span className="time">{fmtLap(world.lapClock(human ?? focus))}</span>
                  </div>
                )}
              </div>
              {countdown !== null && <div className="countdown">{countdown}</div>}
              {setup.mode === 'race' && human?.frozenUntil !== null && human && (
                <div className="banner danger">{t('banner.off_track')}</div>
              )}
              {paused && <div className="banner">{t('banner.paused')}</div>}
              {setup.mode === 'race' && <TouchControls onChange={(patch) => {
                Object.assign(keys.current, patch);
                world.humanControls = keysToControls(keys.current);
              }} />}
            </TrackStage>
          ) : (
            <div className="stage loading">{t('stage.loading')}</div>
          )}

          {wideCircuit && focus?.driver instanceof QuantumDriver && focus.decision?.gates && (
            <div className="wide-circuit">
              <div className="wide-circuit-head">
                <span>
                  {t('wide.heading', { label: focus.label, n: focus.driver.nQubits, layers: focus.driver.nLayers })}
                </span>
                <button
                  type="button"
                  className="chip"
                  aria-label={t('wide.close')}
                  title={t('wide.close')}
                  onClick={() => setWideCircuit(false)}
                >
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
                <span className="control-label">{t('controls.driver')}</span>
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
                <span className="control-label">{t('controls.rival')}</span>
                <div className="chips">
                  {RIVALS.map(([id, label]) => (
                    <button
                      key={id}
                      type="button"
                      className={id === setup.rival ? 'chip active' : 'chip'}
                      onClick={() => pickRival(id)}
                    >
                      {t(label)}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {setup.mode !== 'race' && (
              <div className="control-group">
                <span className="control-label">{t('controls.speed')}</span>
                <div className="chips">
                  {SPEEDS.map((s) => (
                    <button
                      key={s}
                      type="button"
                      className={s === speed ? 'chip active' : 'chip'}
                      onClick={() => setSpeed(s)}
                    >
                      {s.toLocaleString(lang)}×
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="control-group buttons">
              {setup.mode !== 'race' && (
                <>
                  <button type="button" className="chip" onClick={() => setPaused((p) => !p)}>
                    {t(paused ? 'controls.resume' : 'controls.pause')}
                  </button>
                  <button type="button" className="chip" onClick={stepDecision} disabled={!paused}>
                    {t('controls.step')}
                  </button>
                </>
              )}
              <button type="button" className="chip" onClick={() => setRestartKey((k) => k + 1)}>
                {t('controls.restart')}
              </button>
              {setup.mode === 'race' && (
                <button
                  type="button"
                  className="chip"
                  title={t('camera.tip')}
                  onClick={(e) => {
                    cycleCamera();
                    e.currentTarget.blur();
                  }}
                >
                  📷 {t(CAMERA_LABELS[camera])}
                </button>
              )}
              <label className="toggle">
                <input type="checkbox" checked={showRays} onChange={(e) => setShowRays(e.target.checked)} />
                {t('controls.lidar')}
              </label>
              {setup.mode !== 'evolution' &&
                !(setup.mode === 'watch' && setup.driver === `quantum_${setup.track}`) &&
                manifest.ghosts.some((g) => g.track === setup.track) && (
                <label className="toggle">
                  <input type="checkbox" checked={setup.ghost} onChange={(e) => update({ ghost: e.target.checked })} />
                  {t('controls.ghost')}
                </label>
              )}
            </div>
            {setup.mode === 'race' && (
              <p className="hint">
                {t(isTouch() ? 'hint.race_touch' : 'hint.race_keys')} {t('hint.race_decides')}
              </p>
            )}
            {setup.mode === 'evolution' && (
              <p className="hint">{t('hint.evolution')}</p>
            )}
            {setup.mode !== 'evolution' && evalLine(selected) && <p className="hint">{evalLine(selected)}</p>}
          </div>

          {world && (
            <Lapboard world={world} focusId={focusId} onFocus={(id) => setFocusId(id)} />
          )}
        </div>

        <aside className="panel">
          <h2>
            {t('panel.heading')}
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
        {rich('footer.text', {
          racetraq: (
            <a href="https://github.com/JanLahmann/racetraQ" target="_blank" rel="noopener" {...outbound('https://github.com/JanLahmann/racetraQ')}>
              racetraQ
            </a>
          ),
          qamposer: (
            <a href="https://qamposer.org" target="_blank" rel="noopener" {...outbound('https://qamposer.org')}>
              QAMPoser
            </a>
          ),
        })}{' '}
        <a href="https://fun-with-quantum.org" target="_blank" rel="noopener" {...outbound('https://fun-with-quantum.org')}>
          Fun with Quantum
        </a>
      </footer>

      {aboutOpen && <About manifest={manifest} onClose={closeAbout} />}
    </div>
  );
}
