// Licence states of a model (mirrors app/licensing.py): 'commercial' | 'check' | 'noncommercial'.
// The registry gives every model license: {name, commercial, note?, verify?}.
export function licenseState(lic) {
  if (!lic || !lic.commercial) return 'noncommercial';
  return lic.verify ? 'check' : 'commercial';
}
export const LICENSE_LABEL = { commercial: 'Commercial OK', check: 'Check licence', noncommercial: 'Non-commercial' };
export const LICENSE_HELP = {
  commercial: 'Commercial use of the output is allowed by the model licence.',
  check: 'Commercial use looks allowed, but the registry entry says the terms must be verified before you rely on it.',
  noncommercial: 'Research / non-commercial licence: do not use the output in commercial work.',
};
/** True when `model` must be hidden / refused under the "Commercial-safe models only" setting. */
export function licenseBlocked(model, { commercial_only, strict } = {}) {
  if (!commercial_only || !model) return false;
  const s = licenseState(model.license);
  return s === 'noncommercial' || (!!strict && s === 'check');
}
