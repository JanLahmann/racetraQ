import type { World } from '../sim/world';

export const fmtLap = (t: number | null | undefined) => (t == null ? '—' : `${t.toFixed(2)} s`);

export function Lapboard({
  world,
  focusId,
  onFocus,
}: {
  world: World;
  focusId: string;
  onFocus?: (id: string) => void;
}) {
  return (
    <table className="lapboard">
      <thead>
        <tr>
          <th>Car</th>
          <th>Lap</th>
          <th>Last</th>
          <th>Best</th>
        </tr>
      </thead>
      <tbody>
        {world.cars.map((car) => (
          <tr
            key={car.id}
            className={car.id === focusId ? 'focus' : ''}
            onClick={car.driver && onFocus ? () => onFocus(car.id) : undefined}
            style={{ cursor: car.driver && onFocus ? 'pointer' : undefined }}
          >
            <td>
              <span className="swatch" style={{ background: car.color }} />
              {car.label}
            </td>
            <td className="num">{fmtLap(world.lapClock(car))}</td>
            <td className="num">{fmtLap(car.lastLap)}</td>
            <td className="num best">{fmtLap(car.bestLap)}</td>
          </tr>
        ))}
        {world.ghost && (
          <tr className="ghost-row">
            <td>
              <span className="swatch ghost" />
              Ghost · 4-qubit driver's lap from a standing start
            </td>
            <td />
            <td />
            <td className="num">{fmtLap(world.ghost.lap_time)}</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}
