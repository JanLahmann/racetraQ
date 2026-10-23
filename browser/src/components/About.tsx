import { useEffect } from 'react';
import { outbound } from '../analytics';
import type { Manifest } from '../data';
import { fmtInt, rich, t, useLang } from '../i18n';

const REPO = 'https://github.com/JanLahmann/racetraQ';
/** concept DOI: resolves to the latest release (CITATION.cff) */
const DOI_URL = 'https://doi.org/10.5281/zenodo.23270600';

/** An external link, counted by Umami as an outbound click on `host`. */
function Out({ href, host = href, children }: { href: string; host?: string; children: React.ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noopener" {...outbound(host)}>
      {children}
    </a>
  );
}

export function About({
  manifest,
  onClose,
  onOpenToy,
}: {
  manifest: Manifest;
  onClose: () => void;
  /** open "Train a tiny circuit" (closes this dialog) */
  onOpenToy: () => void;
}) {
  useLang();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  const quantum = manifest.drivers.filter((d) => d.agent === 'quantum').length;
  const ovalParams = manifest.drivers.find((d) => d.id === 'quantum_oval')?.n_params ?? 56;
  const racetraq = <Out href={REPO}>racetraQ</Out>;
  const qamposer = <Out href="https://qamposer.org">QAMPoser</Out>;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={t('about.aria')} onClick={(e) => e.stopPropagation()}>
        <button type="button" className="close" onClick={onClose} aria-label={t('about.close')}>
          ✕
        </button>
        <h2>{t('about.title')}</h2>
        <p>{t('about.p1')}</p>
        <p>{rich('about.p2', { params: fmtInt(ovalParams), count: quantum, racetraq, qamposer })}</p>
        <h3>{t('about.isnt')}</h3>
        <ul>
          <li>
            <strong>{t('about.exact_title')}</strong> {rich('about.exact', { backend: <em>ibm_marrakesh</em> })}
          </li>
          <li>
            <strong>{t('about.noadv_title')}</strong> {t('about.noadv')}
          </li>
          <li>
            <strong>{t('about.small_title')}</strong> {t('about.small')}
          </li>
        </ul>
        <h3>{t('about.modes')}</h3>
        <ul>
          <li>
            {rich('about.watch', { mode: <strong>{t('mode.watch')}</strong> })} {t('about.predict')}
          </li>
          <li>{rich('about.race', { mode: <strong>{t('mode.race')}</strong> })}</li>
          <li>{rich('about.evolution', { mode: <strong>{t('mode.evolution')}</strong> })}</li>
        </ul>
        <p className="hint">
          {rich('about.more', {
            explainer: (
              <Out href={`${REPO}/blob/main/docs/EXPLAINER.md`} host={REPO}>
                {t('about.explainer')}
              </Out>
            ),
            science: (
              <Out href={`${REPO}/blob/main/docs/SCIENCE.md`} host={REPO}>
                {t('about.science')}
              </Out>
            ),
            report: (
              <Out href={`${REPO}/blob/main/docs/REPORT.md`} host={REPO}>
                {t('about.report')}
              </Out>
            ),
          })}
        </p>
        <p className="hint">
          {rich('about.learn', {
            notebooks: (
              <Out href={`${REPO}#notebooks`} host={REPO}>
                {t('about.notebooks')}
              </Out>
            ),
            path: (
              <Out href={`${REPO}/blob/main/docs/LEARN.md`} host={REPO}>
                {t('about.path')}
              </Out>
            ),
          })}
        </p>
        <p className="hint">
          {rich('about.toy', {
            toy: (
              <button type="button" className="link" onClick={onOpenToy}>
                {t('about.toy_link')}
              </button>
            ),
          })}
        </p>
        <p className="hint">
          {t('about.cite')} <Out href={DOI_URL}>doi:10.5281/zenodo.23270600</Out>
        </p>
      </div>
    </div>
  );
}
