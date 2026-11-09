// Actions offered in the Library's multi-select bar (after Export), registered by other modules: registerSelectionAction({id, label, icon,
// types?, min?, max?, run(ids, assets)}). `types`: asset types the action accepts (every selected asset must be one of them); min / max: how many
// assets must be selected (default 1 / no limit). Used by the project style guide ("Style guide from selection"); an engine export can use it too.
const actions = [];

export function registerSelectionAction(a) {
  if (!a?.id || typeof a.run !== 'function') return;
  const i = actions.findIndex(x => x.id === a.id);
  if (i >= 0) actions[i] = a; else actions.push(a);
}

/** The registered actions that fit the selected asset metas (what the Library bar renders). */
export function selectionActions(assets) {
  const n = assets.length;
  return actions.filter(a => n >= (a.min ?? 1) && (a.max == null || n <= a.max) && (!a.types || assets.every(x => a.types.includes(x.type))));
}
