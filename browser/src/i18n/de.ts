/**
 * German strings, in the vocabulary of the desktop app
 * (racetraq/web/i18n/de.json): Quanten-KI, Quantenfahrer, Runde, Bestzeit,
 * Schaltkreis, Blöcke, Geisterauto, informal "du". Same keys and
 * placeholders as en.ts (tests/i18n.test.ts checks both).
 */
import type { Key } from './en';

export const de: Record<Key, string> = {
  'app.title': 'racetraQ: Ein Quantenschaltkreis lernt Rennwagen fahren',
  'app.loading': 'racetraQ wird geladen…',
  'app.error': 'Die Renndaten ließen sich nicht laden: {error}',

  'lang.group': 'Sprache',

  'header.edition': 'Browser-Edition',
  'header.tracks': 'Strecke',
  'header.modes': 'Modus',
  'header.about': 'Was ist das?',
  'pitch.text':
    'Ein winziger Quantenschaltkreis hat gelernt, dieses Auto zu fahren. Sieh ihm bei seinen Runden zu und fahr dann selbst gegen ihn.',
  'pitch.how': 'Wie funktioniert das?',

  'mode.watch': 'Zuschauen',
  'mode.watch_hint': 'ein Quantenfahrer fährt – sieh jede seiner Entscheidungen',
  'mode.race': 'Rennen',
  'mode.race_hint': 'du gegen den Quantenfahrer',
  'mode.evolution': 'Evolution',
  'mode.evolution_hint': 'vier Zwischenstände aus einem Trainingslauf: sieh zu, wie er lernt',

  'track.oval': 'Oval',
  'track.chicane': 'Schikane',
  'track.gp': 'Grand Prix',
  'track.combo': 'Kombi',

  'stage.loading': 'Strecke wird geladen…',
  'stage.aria': 'Rennstrecke',
  'hud.you': 'Du',
  'race.you': 'DU',
  'banner.off_track': 'Von der Strecke abgekommen – zurück zum Start',
  'banner.paused': 'Pause – „Schritt“ geht eine Entscheidung weiter',

  'wide.heading': 'Schaltkreis der aktuellen Entscheidung · {label} · {n} Qubits × {layers} Blöcke',
  'wide.close': 'Breite Ansicht schließen',

  'controls.driver': 'Quantenfahrer',
  'controls.rival': 'Gegner',
  'controls.speed': 'Tempo',
  'controls.resume': '▶ Weiter',
  'controls.pause': '❚❚ Pause',
  'controls.step': 'Schritt',
  'controls.restart': '↺ Neustart',
  'controls.lidar': 'Lidar',
  'controls.ghost': 'Geisterauto',
  'chip.qubits': '{n} Qubits',
  'chip.qubits_multi': '{n} Qubits · Allrounder',
  'rival.none': 'keiner',
  'rival.mlp': 'klassisches MLP',
  'rival.pro': 'klassischer Profi',
  'camera.tip': 'Kamera (C): von oben, Verfolger (dreht sich mit deinem Auto) oder Cockpit (näher dran, mehr Straße voraus)',
  'camera.top': 'Oben',
  'camera.chase': 'Verfolger',
  'camera.cockpit': 'Cockpit',

  'hint.race_touch':
    'Fahr mit den Pedalen auf dem Bildschirm. Der 📷-Knopf wechselt die Kamera: Verfolger und Cockpit drehen sich mit deinem Auto.',
  'hint.race_keys':
    'Fahr mit den Pfeiltasten oder WASD (Leertaste bremst) – oder mit den Pedalen auf dem Bildschirm. C wechselt die Kamera: Verfolger und Cockpit drehen sich mit deinem Auto.',
  'hint.race_decides': 'Das Quantenauto entscheidet 10-mal pro Sekunde.',
  'hint.evolution':
    'Vier Zwischenstände desselben Quantenfahrers, gespeichert an verschiedenen Punkten eines Trainingslaufs. Klick ein Auto in der Tabelle an, um in seinen Schaltkreis zu schauen.',
  'hint.eval': 'Getestet in {episodes} neuen Episoden: ganze Runde in {laps}/{episodes}, mittlere Runde {lap}.',

  'touch.brake': 'BREMSE',
  'touch.gas': 'GAS',

  'lap.car': 'Auto',
  'lap.lap': 'Runde',
  'lap.last': 'Letzte',
  'lap.best': 'Beste',
  'lap.ghost': 'Geisterauto · Runde des 4-Qubit-Fahrers aus dem Stand',

  'car.you': 'Du',
  'car.quantum': 'Quanten-KI · {n} Qubits',
  'car.quantum_multi': 'Quanten-KI · {n} Qubits · Allrounder',
  'car.pro': 'Klassischer Profi · {params} Parameter',
  'car.mlp': 'Klassisches MLP · {params} Parameter',
  'car.mlp_unmatched': 'Klassisches MLP · {params} Parameter ({rays} Strahlen, nicht angeglichen)',
  'car.ep': 'Ep. {n}',
  'car.stage': 'Stufe {n}',
  'car.best_of': 'beste (aus Lauf mit {n} Ep.)',
  'car.best': 'beste',

  'panel.heading': 'Im Kopf des Fahrers',
  'brain.pick': 'Wähl einen Fahrer, um zu sehen, was er berechnet.',
  'brain.sensors': 'Sensoren',
  'brain.sensors_sub': '{n} Zahlen in [0, 1]',
  'brain.sensors_sub_q': '{n} Zahlen in [0, 1] – eine pro Qubit',
  'brain.circuit': 'Schaltkreis',
  'brain.circuit_sub': '{n} Qubits · {layers} Blöcke · {params} trainierte Zahlen',
  'brain.circuit_hint':
    'Jeder Block schreibt die Sensorwerte als Drehwinkel in die Qubits (RY = λ·s, bei jeder Entscheidung neu), wendet trainierte RY/RZ-Rotationen an und verschränkt Nachbarn mit einem CZ-Ring.',
  'brain.wide_show': '⤢ Ganzen Schaltkreis unter der Strecke zeigen',
  'brain.wide_back': '⤡ Zurück in dieses Feld',
  'brain.circuit_wait': 'Der Schaltkreis erscheint mit der ersten Entscheidung.',
  'brain.measure': 'Messung',
  'brain.measure_sub': '⟨Z⟩ jedes Qubits, exakt (Zustandsvektor in deinem Browser)',
  'brain.no_readout': 'keine Auslese',
  'brain.classical': 'Klassisches Netz',
  'brain.classical_sub': '{params} trainierte Zahlen',
  'brain.classical_hint':
    'Ein kleines neuronales Netz (eine verdeckte tanh-Schicht) macht aus den Sensorwerten die Bewertungen.',
  'brain.decision': 'Entscheidung',
  'brain.decision_sub_q': 'Q = w·⟨Z⟩ + b – der höchste Wert gewinnt',
  'brain.decision_sub': 'der höchste Wert gewinnt',
  'brain.composer': 'Diese Entscheidung im IBM Quantum Composer öffnen ↗',
  'brain.composer_hint':
    'Der genaue Schaltkreis mit den Winkeln dieser Entscheidung, bereit für einen echten IBM-Quantencomputer.',
  'brain.composer_hint_big':
    'Der genaue Schaltkreis mit den Winkeln dieser Entscheidung, bereit für einen echten IBM-Quantencomputer (melde dich dort an, um mehr als 4 Qubits zu simulieren).',
  'brain.composer_title': 'racetraQ-Entscheidung ({n} Qubits)',
  'circuit.aria': 'Schaltkreis der aktuellen Entscheidung mit {n} Qubits',

  'action.right': 'Rechts',
  'action.straight': 'Geradeaus',
  'action.left': 'Links',
  'action.brake': 'Bremsen',
  'action.brake_right': 'Bremsen rechts',
  'action.brake_left': 'Bremsen links',
  'action.half_right': 'Halb rechts',
  'action.half_left': 'Halb links',

  'feature.ray': 'Strahl {deg}',
  'feature.speed': 'Geschwindigkeit',
  'feature.curvature_ahead': 'Krümmung voraus',
  'feature.lateral_offset': 'seitlicher Versatz',
  'feature.heading_error': 'Richtungsfehler',
  'feature.corner_speed': 'Kurventempo',
  'feature.lookahead': '{name} {m} m',

  'about.aria': 'Über racetraQ',
  'about.close': 'Schließen',
  'about.title': 'Ein Rennauto mit einem Quantenschaltkreis als Gehirn',
  'about.p1':
    'Jede Zehntelsekunde misst das Auto seine Umgebung: beim 4-Qubit-Fahrer drei Lidar-Strahlen (die lila Linien) und seine Geschwindigkeit. Diese Zahlen werden als Drehwinkel in die Qubits geschrieben, ein kurzer Quantenschaltkreis verschränkt die Qubits, und die Messung jedes Qubits ergibt einen Wert pro Aktion – rechts lenken, geradeaus, links lenken, bremsen. Das Auto wählt die Aktion mit dem höchsten Wert.',
  'about.p2':
    'Die {params} Zahlen des 4-Qubit-Schaltkreises für das Oval wurden durch bestärkendes Lernen (Double Deep Q-Learning) gelernt: Tausende Übungsrunden, belohnt für Fortschritt entlang der Strecke und bestraft fürs Verlassen der Strecke. Trainiert wurde offline in {racetraq} (Python und Qiskit). Diese Seite fährt nur: Sie bringt {count} trainierte Quantenfahrer mit und rechnet jeden Schaltkreis in deinem Browser auf dem Zustandsvektor-Simulator von {qamposer}.',
  'about.isnt': 'Was das ist – und was nicht',
  'about.exact_title': 'Exakt und rauschfrei.',
  'about.exact':
    'Der Simulator berechnet den Zustand der Qubits exakt. Ein echter Quantencomputer schätzt dieselben Zahlen aus wiederholten Messungen und fügt Rauschen hinzu; auf IBMs {backend} traf eine ganze Runde in etwa 90\u00a0% der Entscheidungen dieselbe Aktion wie diese Simulation.',
  'about.noadv_title': 'Kein Quantenvorteil.',
  'about.noadv':
    'Der klassische Gegner – ein kleines neuronales Netz, genauso trainiert – lernt in unseren Studien mit vielen Seeds schneller und zuverlässiger und fährt auf den Strecken mit Haarnadelkurven schnellere Runden. Es geht darum, einem Quantenschaltkreis beim Entscheiden zuzusehen, nicht darum, klassische Computer zu schlagen.',
  'about.small_title': 'Absichtlich klein.',
  'about.small':
    'Vier bis zehn Qubits: klein genug, um sie im Browser zu simulieren, und genau die Größe, die auf heutiger Quantenhardware läuft.',
  'about.modes': 'Modi',
  'about.watch':
    '{mode} – verfolge eine Entscheidung nach der anderen: Sensoren, den Live-Schaltkreis, die gemessenen Qubits und die Werte. Halte an und geh schrittweise weiter, oder öffne den genauen Schaltkreis einer Entscheidung im IBM Quantum Composer.',
  'about.race': '{mode} – fahr selbst (Pfeiltasten, WASD oder die Pedale auf dem Bildschirm) gegen das Quantenauto.',
  'about.evolution': '{mode} – vier Zwischenstände eines Trainingslaufs fahren gegeneinander: So hat der Fahrer gelernt.',
  'about.more': 'Mehr dazu (auf Englisch): die {explainer}, die {science} und der {report}.',
  'about.explainer': 'einseitige Erklärung',
  'about.science': 'wissenschaftlichen Notizen',
  'about.report': 'technische Bericht',
  'about.learn':
    'So funktioniert es: Acht {notebooks} (auf Englisch) bauen alles von Grund auf nach und öffnen sich in deinem Browser auf Binder, ohne Installation. Der {path} schlägt vor, wo du anfängst – von einer 5-Minuten-Tour bis zu einem Forschungspfad.',
  'about.notebooks': 'Notebooks',
  'about.path': 'Lernpfad',
  'about.cite': 'Zitieren:',

  'footer.text':
    'Nur Inferenz: Die Fahrer wurden offline mit {racetraq} trainiert (Python, Qiskit). Jeden Schaltkreis simuliert der Zustandsvektor-Simulator von {qamposer} hier in deinem Browser – ohne Server.',
};
