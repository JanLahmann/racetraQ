import type { World } from '../sim/world';
import { fmtSeconds, t, useLang } from '../i18n';

/** A lap time in the active language: "12.20 s" / "12,20 s". */
export const fmtLap = (s: number | null | undefined) => (s == null ? '—' : fmtSeconds(s));

export function Lapboard({
  world,
  focusId,
  onFocus,
}: {
  world: World;
  focusId: string;
  onFocus?: (id: string) => void;
}) {
  useLang();
  return (
    <table className="lapboard">
      <thead>
        <tr>
          <th>{t('lap.car')}</th>
          <th>{t('lap.lap')}</th>
          <th>{t('lap.last')}</th>
          <th>{t('lap.best')}</th>
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
              {t('lap.ghost')}
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
