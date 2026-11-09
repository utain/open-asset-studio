// Beginner texts for the fixed controls of the app (the (i) balloons that are not generated from a registry param or a settings field).
// One place for all of them: a control asks for `uiInfo('key')` (an (i) button, or null for an unknown key). Texts are 1-3 sentences,
// plain words, and say what the control does and what a good first choice is. Registry params get their text from the API (`info`).
// Add facts as [label, value, mono?] rows when a number or a name helps.
import { infoBtn } from './components/infotip.js';

export const UI_INFO = {
  // ------------------------------------------------------------- run forms (TaskStudio, PromptBox, ModelPicker, ParamForm)
  model: { title: 'Model', text: 'The AI model that makes the result. Different models trade speed, quality and licence: the badge shows whether the output may be used commercially, and greyed-out models are not installed or not ready yet. If you are unsure, keep the one that is already selected.' },
  settings: { title: 'Settings', text: 'The options of the selected model. The defaults are a sensible starting point, so you can press Generate without touching anything. Open Advanced only when you want finer control.' },
  presets: { title: 'Presets', text: 'One click sets several options at once: Fast gives a quick draft, Quality takes longer for a better result. With + Preset you can save your own combination and bring it back later.' },
  prompt: { title: 'Prompt', text: 'Describe what you want in plain words, in Thai or English. More concrete details (subject, style, mood) usually give better results; Improve can rewrite a short idea into the format this model understands.' },
  negative: { title: 'Negative prompt', text: 'Things you do not want to see, for example “blurry, extra fingers, text”. Only some models use it. Leave it empty if you are not sure.' },
  improve: { title: 'Improve', text: 'Asks a language model to rewrite your idea into a richer prompt written for the selected model, and to translate Thai to English when needed. You choose whether to replace your text, insert the result or keep your original; nothing is changed without a click.' },
  improve_chips: { title: 'Style hints', text: 'Switch on the looks and details you want and the improved prompt will include them, for example a camera angle, a lighting mood or an art style. They only affect Improve, not the prompt you typed.' },
  improve_n: { title: 'Variations', text: 'How many alternative prompts Improve writes besides the main one. 0 gives only the main prompt; more variations take a little longer.' },
  improve_with: { title: 'Improve with', text: 'Which language model rewrites the prompt. “Local” runs on this machine and sends nothing out; a cloud or CLI backend may be better at writing but sends your text to that provider.' },

  // input slots by role / type (AssetSlot); a task-specific entry `<task>.<role>` wins
  'slot.image': { title: 'Input image', text: 'The picture this task starts from. Drop a file here, or click to choose one from your library. A clear subject on a plain or transparent background works best.' },
  'slot.ref': { title: 'Reference', text: 'An extra file the model should look at, such as a style picture, an outfit, or a few seconds of a voice to copy. It guides the result but is not changed.' },
  'slot.input': { title: 'Input files', text: 'The files this tool works on. Drop them here or pick them from your library; your originals are never modified, the results are saved as new items.' },
  'slot.mesh': { title: '3D model', text: 'The 3D model (a GLB file) this task works on. Pick one from your library or drop a file here; the original is kept and the result is saved as a new model.' },
  'slot.audio': { title: 'Input sound', text: 'The sound clip this task starts from. Clean recordings without background noise give the best results.' },
  'slot.voice': { title: 'Voice sample', text: 'A short recording of the voice to copy or convert. Five to fifteen seconds of clear speech is usually enough.' },
  'slot.video': { title: 'Input video', text: 'The video this task works on. Drop a file here or pick one from your library; the original stays untouched.' },
  'slot.mesh_export': { title: 'Meshes to export', text: 'The 3D models to clean up, reduce and convert. You can add several at once; each one becomes a new file and the originals stay as they are.' },
  'texture.mesh': { title: 'Mesh to texture', text: 'The shape (a 3D model without colours, or one you want to repaint) that gets a new texture. Usually the result of an Image → 3D job.' },
  'texture.image': { title: 'Reference image', text: 'Optional picture the texture is painted from. If you leave it empty, the image that the shape was made from is used.' },
  'retopo.mesh': { title: 'Detailed model', text: 'The detailed model to rebuild: usually an Image → 3D or Texture result. It must not be rigged yet: retopologise first, then rig the clean copy. The original is kept.' },
  'rig.mesh': { title: 'Mesh to rig', text: 'A character or creature to give a skeleton, so that it can be posed and animated. It works best in a neutral A- or T-pose with arms and legs apart.' },
  'reskin.mesh': { title: 'Rigged mesh', text: 'A model that already has a skeleton, usually one you just edited with Edit rig. Its skeleton is kept exactly as it is and only the skin weights (how much each bone pulls the surface) are recomputed.' },
  'engine_export.mesh': { title: 'Models for the game engine', text: 'The 3D models (GLB) to package for a game engine. A model with a skeleton stays a GLB; a static model also gets an FBX for Unreal or Unity. Several models are packaged one after the other, each into its own result. The originals are not changed.' },
  'motion.mesh': { title: 'Rigged character', text: 'The character to animate: a rigged two-legged, two-armed model from the Rig tab, standing upright in a T- or A-pose. Other skeletons are refused with a reason.' },

  // ------------------------------------------------------------- export dialog
  export_preset: { title: 'Export preset', text: 'A ready-made set of conversion options for a target such as a game, a podcast or social media. Choosing one fills in the fields below, and you can still adjust them.' },
  export_audio: { title: 'Convert audio', text: 'When on, audio files are converted with the options below. When off, the original audio files are exported unchanged.' },
  export_audio_format: { title: 'Audio format', text: 'WAV and FLAC keep full quality (big files); MP3 and OGG are much smaller with a small loss. Use WAV for editing, MP3 or OGG for the web and most games.' },
  export_loudness: { title: 'Loudness', text: 'Makes every file about equally loud, measured in LUFS (closer to 0 is louder). −14 is typical for online video and music, −16 for podcasts and −23 for broadcast. “No change” keeps the original volume.' },
  export_sample_rate: { title: 'Sample rate', text: 'How many audio samples are stored per second. 44.1 or 48 kHz is right for almost everything; “Keep” leaves each file as it is.' },
  export_bit_depth: { title: 'Bit depth', text: 'The precision of each audio sample. 16-bit is CD quality and fine for finished sounds; 24-bit gives more room for further editing.' },
  export_images: { title: 'Convert images', text: 'When on, images are converted and resized with the options below. When off, the original image files are exported unchanged.' },
  export_image_format: { title: 'Image format', text: 'PNG is lossless and keeps transparency; WebP is smaller and also keeps transparency; JPEG is smallest but has no transparency (transparent areas turn white).' },
  export_aspect: { title: 'Aspect ratio', text: 'The shape of the picture, width to height, such as 1:1 (square) or 16:9 (widescreen). “Keep” leaves the shape as it is.' },
  export_size: { title: 'Size', text: 'The output size in pixels. “Long side 2048” scales the picture so its longer edge is 2048 px and keeps the proportions; a fixed size such as 1920 x 1080 sets both sides.' },
  export_fit: { title: 'When the aspect differs', text: 'What to do when the picture does not have the chosen shape. Crop cuts off the edges to fill the frame; Pad keeps everything and adds empty space.' },

  // ------------------------------------------------------------- Providers (cloud models and your own endpoints)
  cloud_allow: { title: 'Allow cloud models', text: 'Cloud models run on a provider’s computers instead of yours. They are paid per use, and your prompt (and any image or file you pick) is sent to that provider. Everything stays local while this is off. Turn it on only when you want a model that is not available locally.' },
  cloud_budget: { title: 'Monthly budget (USD)', text: 'The most the studio will let cloud jobs cost in one calendar month. A job whose estimated price would push the total over this amount is refused. Set it to what you are comfortable spending; 0 means no limit, so avoid that unless you track costs elsewhere.' },
  cloud_confirm: { title: 'Confirm above (USD)', text: 'Cloud jobs estimated at this price or more ask for your confirmation before they run, and jobs with an unknown price always ask. Raise it to be asked less often; set it to 0 to be asked about every paid job.' },
  api_key: { title: 'API key', text: 'A secret code from the provider that lets this studio use your account. It is stored on this computer only (data/secrets.json) and is never shown again or sent anywhere except to that provider. Treat it like a password.' },
  provider_test: { title: 'Test', text: 'Makes one small, free request to the provider to check that the key and connection work. It does not start a paid job.' },
  endpoint_name: { title: 'Name (id)', text: 'A short lowercase name used inside the studio to refer to this endpoint, for example “lab”. Pick something simple; it cannot be changed after saving.' },
  endpoint_label: { title: 'Label', text: 'The friendly name shown in menus and on cards. It is only for you.' },
  endpoint_base: { title: 'Base URL', text: 'The address of a server that speaks the OpenAI API, such as llama-server, vLLM, LM Studio or LiteLLM, usually ending in /v1. Local addresses (127.0.0.1, 192.168.x.x) work without the cloud switch and cost nothing.' },
  endpoint_key: { title: 'API key (optional)', text: 'Only needed if the server asks for one. Leave it empty for most local servers. When you edit an endpoint, leaving it empty keeps the key you saved before.' },
  endpoint_models: { title: 'Models', text: 'The model names this endpoint offers, separated by commas. Use Fetch models to read them from the server instead of typing them.' },
  endpoint_serves: { title: 'Serves', text: 'What the endpoint is used for: Text (LLM) makes it available to the prompt improver and the Assistant; Images adds its image models to Image › Generate.' },
  endpoint_price: { title: 'Price per image ($)', text: 'What one image costs on this endpoint, used for the cost estimate and the monthly budget. 0 means free, which is right for your own server.' },
  price_override: { title: 'Price override', text: 'The studio estimates cloud costs from a built-in price list. If the provider’s real price is different, edit it here; “Reset to list price” goes back to the built-in value.' },

  // ------------------------------------------------------------- Models page
  commercial_only: { title: 'Commercial-safe models only', text: 'Hides every model whose licence does not allow commercial use, in all pickers, and makes the API refuse them. Turn it on if you plan to sell or publish what you make. You can also switch it per project in Settings.' },
  models_recheck: { title: 'Re-check', text: 'Looks again at which models are installed and ready, for example after you downloaded weights or edited a model file. It does not download or change anything.' },
  models_default: { title: 'Default model (★)', text: 'The star marks the model that is pre-selected for a task. Click a star to make that model the default; click it again to go back to the built-in choice.' },
  models_memory: { title: 'Memory', text: 'How much free memory the model needs while it runs. If there is not enough, the studio unloads its own idle models first; it never touches other programs.' },
  models_license: { title: 'Licence', text: 'Whether the model’s output may be used commercially. Green means yes, amber means check the terms yourself, red means non-commercial or research use only.' },
  models_deep_check: { title: 'Deep check', text: 'Starts the backend and asks it to load each of its models. It is slow and uses memory, but it finds problems that the quick check cannot see.' },
  models_status: { title: 'Status filter', text: 'Show only models in a certain state, such as ready to use, missing files, or cloud-only.' },

  // ------------------------------------------------------------- 3D viewer and rig tools
  viewer_mode: { title: 'View mode', text: 'Textured shows the model as it will look. Shape is the plain white mesh, Edges and Wireframe show how the surface is built from triangles. This only changes the preview. Clean low-poly (retopo) models show their quads here instead of triangles.' },
  retopo_compare: { title: 'Before / after', text: 'The original (left or top) and the clean low-poly copy (right or bottom) side by side; dragging one view turns both. Use Clay to judge the baked detail without colours, and Edges to see the new quads.' },
  retopo_clay: { title: 'Clay', text: 'Shows both models in plain grey so you can see the shape and the baked normal detail without the colours distracting.' },
  retopo_normal_toggle: { title: 'Normal map', text: 'Turns the baked normal map off and on. The difference is the detail the bake carries over from the original.' },
  bones: { title: 'Bones', text: 'Draws the skeleton of a rigged model on top of it. The lines are bones and the dots are joints; they move with the animation. Turn it on to check the rig or to start editing it.' },
  edit_rig: { title: 'Edit rig', text: 'Opens the skeleton editor: move, rename, add or delete bones, then apply the changes as a new copy of the model. Your original is never changed.' },
  rige_select: { title: 'Selecting a bone', text: 'Click a dot in the viewer, or pick a bone from the list. The selected bone can then be moved, renamed, deleted or given a child.' },
  rige_step: { title: 'Move step', text: 'How far one press of Up, Down, Left, Right, Front or Back moves the selected bone, in centimetres. Dragging a dot in the viewer moves it freely instead.' },
  rige_follow: { title: 'Children follow', text: 'When on, the bones below the one you move (a hand when you move an elbow) are moved along. When off, only the selected bone moves and the ones below it stay where they are.' },
  rige_apply: { title: 'Apply', text: 'Writes your list of changes into a new copy of the model. New bones have no skin weight yet, so run Re-skin afterwards to make the surface follow them.' },
  rige_reskin: { title: 'Re-skin', text: 'Recomputes the skin weights (how strongly each bone pulls the surface) for the current skeleton. Needed after you add bones. It runs on the GPU and takes about half a minute; the skeleton itself is not changed.' },
  rige_undo: { title: 'Undo and Redo', text: 'Steps back or forward through your edits in the list. Nothing is saved until you press Apply.' },

  // ------------------------------------------------------------- game engine export (3D studio, Export > Game engine)
  engine_target: { title: 'What you get', text: 'A summary of the package for the engine you picked: which files are written (FBX or GLB), how many LOD levels, what kind of collision, and how the textures are named and packed. It updates as you change the settings below and is only a preview; the numbers come after the job runs.' },
  engine_lod: { title: 'LOD level', text: 'LODs are lighter copies of the model that a game shows when it is far away. Pick a level to see it in the viewer; the numbers in the report say how many triangles each one has. All levels share the same texture, so only the shape changes.' },
  engine_hulls: { title: 'Collision shapes', text: 'Shows the collision pieces the game physics will use, as coloured see-through shapes over the model. They should hug the model without sticking far out of it. Characters get a single capsule, props get several convex pieces.' },
  engine_report: { title: 'Export report', text: 'What the export did: the triangles of every LOD and how far each one drifts from the full model, the collision pieces, the textures that were repacked for the engine, and anything that needed fixing. Warnings are not errors; they tell you what to look at.' },
  engine_download: { title: 'Download package', text: 'A zip with everything to drop into a game project: the model file or files, the textures, a manifest and a short README. Unity packages also contain an optional editor script that sets up collision. The package is a copy; your original model is not changed.' },
  engine_deliver: { title: 'Deliver to project', text: 'Copies the package straight into a game project folder on this computer, into a sub folder named after the model. Folders are set up by hand in settings.json, so this button only appears when one exists. Files that you changed after the last delivery are never overwritten, and nothing is ever deleted.' },
  engine_overwrite: { title: 'Replace earlier delivery', text: 'When this model was delivered before and the files in the game project are still exactly as the studio wrote them, ticking this replaces them with the new ones. Files you edited in the meantime are always kept.' },
};

/** The (i) button for a UI_INFO key; null for an unknown key. `o`: {title} to rename the balloon, {facts: [[label, value, mono?], ...]} to add rows. */
export function uiInfo(key, o = {}) {
  const s = UI_INFO[key];
  return s ? infoBtn({ ...s, ...(o.title ? { title: o.title } : {}), ...(o.facts ? { facts: o.facts } : {}) }) : null;
}

/** The (i) button for an input slot: task-specific `<task>.<role>`, else `slot.<role>`, else `slot.<first type>`; null when none matches. */
export function slotInfo(task, { role, types } = {}) {
  const s = UI_INFO[`${task}.${role}`] || UI_INFO[`slot.${role}`] || UI_INFO[`slot.${(types || [])[0]}`] || null;
  return s ? infoBtn(s) : null;
}
