/**
 * English strings of the browser edition (the reference table: every key
 * lives here, de.ts must have the same keys and placeholders). `{name}`
 * marks a placeholder, filled by t(key, { name }) or, for links and other
 * markup, by rich(key, { name: <a>…</a> }).
 */
export const en = {
  'app.title': 'racetraQ · browser edition',
  'app.loading': 'Loading racetraQ…',
  'app.error': 'Could not load the race data: {error}',

  'lang.group': 'Language',

  'header.edition': 'browser edition',
  'header.tracks': 'Track',
  'header.modes': 'Mode',
  'header.about': 'What is this?',
  'pitch.text': 'A tiny quantum circuit learned to drive this car. Watch it lap, then race it yourself.',
  'pitch.how': 'How does it work?',

  'mode.watch': 'Watch',
  'mode.watch_hint': 'a quantum driver races; see every decision',
  'mode.race': 'Race',
  'mode.race_hint': 'you against the quantum driver',
  'mode.evolution': 'Evolution',
  'mode.evolution_hint': 'four snapshots from one training run: watch it learn',

  'track.oval': 'Oval',
  'track.chicane': 'Chicane',
  'track.gp': 'Grand Prix',
  'track.combo': 'Combo',

  'stage.loading': 'Loading track…',
  'stage.aria': 'Race track',
  'hud.you': 'You',
  'race.you': 'YOU',
  'banner.off_track': 'Off track — back to the start',
  'banner.paused': 'Paused — press Step to advance one decision',

  'wide.heading': 'Circuit of the current decision · {label} · {n} qubits × {layers} blocks',
  'wide.close': 'Close the wide circuit',

  'controls.driver': 'Quantum driver',
  'controls.rival': 'Rival',
  'controls.speed': 'Speed',
  'controls.resume': '▶ Resume',
  'controls.pause': '❚❚ Pause',
  'controls.step': 'Step',
  'controls.restart': '↺ Restart',
  'controls.lidar': 'lidar',
  'controls.ghost': 'ghost',
  'chip.qubits': '{n} qubits',
  'chip.qubits_multi': '{n} qubits · all-round',
  'rival.none': 'none',
  'rival.mlp': 'classical MLP',
  'rival.pro': 'classical pro',
  'camera.tip': 'Camera (C): top-down, chase (turns with your car) or cockpit (closer, more road ahead)',
  'camera.top': 'Top',
  'camera.chase': 'Chase',
  'camera.cockpit': 'Cockpit',

  'hint.race_touch':
    'Drive with the on-screen pedals. The 📷 button switches the camera: chase and cockpit turn with your car.',
  'hint.race_keys':
    'Drive with the arrow keys or WASD (Space brakes) — or the on-screen pedals. C switches the camera: chase and cockpit turn with your car.',
  'hint.race_decides': 'The quantum car decides 10 times a second.',
  'hint.evolution':
    'Four snapshots of the same quantum driver, saved at different points of one training run. Click a car in the table to look inside its circuit.',
  'hint.eval': 'Tested on {episodes} fresh episodes: lapped in {laps}/{episodes}, mean lap {lap}.',

  'touch.brake': 'BRAKE',
  'touch.gas': 'GAS',

  'lap.car': 'Car',
  'lap.lap': 'Lap',
  'lap.last': 'Last',
  'lap.best': 'Best',
  'lap.ghost': "Ghost · 4-qubit driver's lap from a standing start",

  'car.you': 'You',
  'car.quantum': 'Quantum · {n} qubits',
  'car.quantum_multi': 'Quantum · {n} qubits · all-round',
  'car.pro': 'Classical pro · {params} params',
  'car.mlp': 'Classical MLP · {params} params',
  'car.mlp_unmatched': 'Classical MLP · {params} params ({rays} rays, not matched)',
  'car.ep': 'ep {n}',
  'car.stage': 'stage {n}',
  'car.best_of': 'best (of {n} ep run)',
  'car.best': 'best',

  'panel.heading': 'Inside the driver',
  'brain.pick': 'Pick a driver to see what it computes.',
  'brain.sensors': 'Sensors',
  'brain.sensors_sub': '{n} numbers in [0, 1]',
  'brain.sensors_sub_q': '{n} numbers in [0, 1] — one per qubit',
  'brain.circuit': 'Circuit',
  'brain.circuit_sub': '{n} qubits · {layers} blocks · {params} trained numbers',
  'brain.circuit_hint':
    'Each block writes the sensor values into the qubits as rotation angles (RY = λ·s, new every decision), applies trained RY/RZ rotations, and entangles neighbours with a CZ ring.',
  'brain.wide_show': '⤢ Show the whole circuit under the track',
  'brain.wide_back': '⤡ Back into this panel',
  'brain.circuit_wait': 'The circuit appears with the first decision.',
  'brain.measure': 'Measurement',
  'brain.measure_sub': '⟨Z⟩ of every qubit, exact (state vector in your browser)',
  'brain.no_readout': 'no readout',
  'brain.classical': 'Classical network',
  'brain.classical_sub': '{params} trained numbers',
  'brain.classical_hint': 'A small neural network (one tanh hidden layer) maps the sensors to scores.',
  'brain.decision': 'Decision',
  'brain.decision_sub_q': 'Q = w·⟨Z⟩ + b — highest score wins',
  'brain.decision_sub': 'highest score wins',
  'brain.composer': 'Open this decision in IBM Quantum Composer ↗',
  'brain.composer_hint': "The exact circuit with this decision's angles, ready to run on a real IBM quantum computer.",
  'brain.composer_hint_big':
    "The exact circuit with this decision's angles, ready to run on a real IBM quantum computer (sign in to simulate more than 4 qubits there).",
  'brain.composer_title': 'racetraQ decision ({n} qubits)',
  'circuit.aria': '{n}-qubit circuit of the current decision',

  'action.right': 'Right',
  'action.straight': 'Straight',
  'action.left': 'Left',
  'action.brake': 'Brake',
  'action.brake_right': 'Brake right',
  'action.brake_left': 'Brake left',
  'action.half_right': 'Half right',
  'action.half_left': 'Half left',

  'feature.ray': 'ray {deg}',
  'feature.speed': 'speed',
  'feature.curvature_ahead': 'curvature ahead',
  'feature.lateral_offset': 'lateral offset',
  'feature.heading_error': 'heading error',
  'feature.corner_speed': 'corner speed',
  'feature.lookahead': '{name} {m}m',

  'about.aria': 'About',
  'about.close': 'Close',
  'about.title': 'A race car with a quantum circuit for a brain',
  'about.p1':
    'Every tenth of a second the car measures its surroundings: for the 4-qubit driver, three lidar rays (the purple lines) and its speed. Those numbers are written into qubits as rotation angles, a short quantum circuit entangles the qubits, and measuring each qubit gives one score per action — steer right, go straight, steer left, brake. The car takes the action with the highest score.',
  'about.p2':
    "The 4-qubit oval circuit's {params} numbers were learned by reinforcement learning (double deep Q-learning): thousands of practice laps, rewarded for progress along the track and penalised for leaving it. Training happened offline in {racetraq} (Python and Qiskit). This page only drives: it ships {count} trained quantum drivers and runs every circuit in your browser on {qamposer}'s state-vector simulator.",
  'about.isnt': "What this is — and isn't",
  'about.exact_title': 'Exact and noiseless.',
  'about.exact':
    "The simulator computes the qubits' state exactly. A real quantum computer estimates the same numbers from repeated measurements and adds noise; on IBM's {backend} a full lap took the same action as this simulation in about 90% of decisions.",
  'about.noadv_title': 'No quantum advantage.',
  'about.noadv':
    'The classical rival — a small neural network trained the same way — learns faster and more reliably in our multi-seed studies, and drives faster laps on the hairpin tracks. The point is to see a quantum circuit make decisions, not to beat classical computers.',
  'about.small_title': 'Small on purpose.',
  'about.small':
    "Four to ten qubits: small enough to simulate in a browser, the same size that runs on today's quantum hardware.",
  'about.modes': 'Modes',
  'about.watch':
    '{mode} — follow one decision at a time: sensors, the live circuit, the measured qubits and the scores. Pause and step, or open the exact circuit of a decision in IBM Quantum Composer.',
  'about.race': '{mode} — drive yourself (arrow keys, WASD or the on-screen pedals) against the quantum car.',
  'about.evolution': '{mode} — four snapshots of one training run racing each other: how the driver learned.',
  'about.more': 'More: the {explainer}, the {science} and the {report}.',
  'about.explainer': 'one-page explainer',
  'about.science': 'science notes',
  'about.report': 'technical report',
  'about.learn':
    'Learn how it works: eight {notebooks} build the whole thing from scratch and open in your browser on Binder, nothing to install. The {path} suggests where to start, from a 5-minute tour to a research track.',
  'about.notebooks': 'notebooks',
  'about.path': 'learning path',
  'about.cite': 'Cite:',

  'footer.text':
    "Inference only: the drivers were trained offline with {racetraq} (Python, Qiskit). Every circuit is simulated in this browser by {qamposer}'s state-vector simulator — no server.",
} as const;

export type Key = keyof typeof en;
