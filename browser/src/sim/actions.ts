/** The discrete action sets — traqmania/agents/base.py. */

/** (steer, throttle, brake); steer +1 = left turn on screen. */
export const FULL_ACTIONS: readonly (readonly [number, number, number])[] = [
  [-1, 1, 0],
  [0, 1, 0],
  [1, 1, 0],
  [0, 0, 1],
  [-1, 0, 1],
  [1, 0, 1],
  [-0.5, 1, 0],
  [0.5, 1, 0],
];

export const FULL_LABELS = [
  'Right',
  'Straight',
  'Left',
  'Brake',
  'Brake right',
  'Brake left',
  'Half right',
  'Half left',
] as const;

export const ACTION_SIZES = [4, 6, 8] as const;

export function actionSet(nActions: number) {
  if (!(ACTION_SIZES as readonly number[]).includes(nActions)) {
    throw new Error(`n_actions must be one of ${ACTION_SIZES.join(', ')}, got ${nActions}`);
  }
  return FULL_ACTIONS.slice(0, nActions);
}

export function actionLabels(nActions: number): string[] {
  return FULL_LABELS.slice(0, actionSet(nActions).length);
}
