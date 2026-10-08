/** On-screen pedals and steering for touch devices (race mode). */
import type { Controls } from '../sim/world';

type Key = 'left' | 'right' | 'gas' | 'brake';

export function TouchControls({ onChange }: { onChange: (patch: Partial<Record<Key, boolean>>) => void }) {
  const bind = (key: Key) => ({
    onPointerDown: (e: React.PointerEvent) => {
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      onChange({ [key]: true });
    },
    onPointerUp: () => onChange({ [key]: false }),
    onPointerCancel: () => onChange({ [key]: false }),
    onContextMenu: (e: React.MouseEvent) => e.preventDefault(),
  });
  return (
    <div className="touch-controls" aria-hidden="true">
      <div className="steer">
        <button type="button" {...bind('left')}>◀</button>
        <button type="button" {...bind('right')}>▶</button>
      </div>
      <div className="pedals">
        <button type="button" className="brake" {...bind('brake')}>BRAKE</button>
        <button type="button" className="gas" {...bind('gas')}>GAS</button>
      </div>
    </div>
  );
}

/** Held keys -> car controls (server keys_to_controls: left+right cancel,
 *  brake overrides throttle; steer +1 turns left). */
export function keysToControls(keys: Record<Key, boolean>): Controls {
  const steer = (keys.left ? 1 : 0) - (keys.right ? 1 : 0);
  const brake = keys.brake ? 1 : 0;
  return { steer, throttle: keys.gas && !brake ? 1 : 0, brake };
}
