/** The EN/DE switch in the header: a labelled group of two toggle buttons. */
import { LANGS, setLang, t, useLang } from '../i18n';

const NAMES = { en: 'English', de: 'Deutsch' } as const;

export function LangSwitch() {
  const lang = useLang();
  return (
    <div className="lang-switch" role="group" aria-label={t('lang.group')}>
      {LANGS.map((l) => (
        <button
          key={l}
          type="button"
          lang={l}
          aria-pressed={l === lang}
          aria-label={NAMES[l]}
          title={NAMES[l]}
          className={l === lang ? 'active' : ''}
          onClick={() => setLang(l)}
        >
          {l.toUpperCase()}
        </button>
      ))}
    </div>
  );
}
