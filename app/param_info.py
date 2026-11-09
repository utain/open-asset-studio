"""Plain-English explanations for model parameters (the (i) balloon next to each field of a run form).

Written for people who have never used a generative model: what the parameter does in everyday words, what happens when you
raise or lower it (quality vs time vs memory, variety vs faithfulness) and a practical tip. The registry `help` text of a
parameter is a short technical hint and stays as it is; the UI shows default and range separately, so no per-model numbers here.

    PARAM_INFO[key]                  generic text, valid for any model that has this parameter
    PARAM_INFO_BY_TASK[(task, key)]  used instead when the meaning differs per task (size, mode, text, format, speed ...)
    param_info(key, task=None)       task-specific first, then generic, else ""

Keep every entry 2-4 short sentences, at most ~600 characters (tests/test_param_info.py checks this and that every parameter
key of the merged registry is covered).
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------------- shared building blocks
_EMOTIONS = {
    "afraid": "afraid", "angry": "angry", "anger": "angry", "calm": "calm", "disgust": "disgusted", "disgusted": "disgusted",
    "fear": "afraid", "happiness": "happy", "happy": "happy", "melancholic": "melancholic", "neutral": "neutral",
    "other": "'other' (a catch-all for feelings that fit none of the named ones)", "sad": "sad", "sadness": "sad",
    "surprise": "surprised", "surprised": "surprised",
}


def _emotion_text(word: str) -> str:
    return (f"One slider of an emotion mix: how much of a {word} tone goes into the voice. Raise the ones you want and leave the "
            "rest low. A little goes a long way; pushing several high at once, or one very high, tends to sound forced or unstable.")


PARAM_INFO: dict[str, str] = {
    # ------------------------------------------------------------------------------------------- common to many models
    "seed": "The starting number for the random dice the model rolls. The same seed with the same settings gives the same result, "
            "so keep it to reproduce or fine-tune a result you like. Leave it on random (-1) to explore, and change only the "
            "seed to get a new variation of the same idea.",
    "batch": "How many results to make in one go. Each one is a separate run with the next seed, so you get different variations to "
             "pick from. More takes cost proportionally more time; two to four is plenty for choosing the best one.",
    "steps": "How many refinement passes the model makes. More steps usually mean cleaner, more detailed results but take "
             "proportionally longer; after a point extra steps stop helping. Use a low value for quick drafts and the model's "
             "usual value (or a bit more) for the final render.",
    "guidance": "Guidance (also called CFG): how strictly the model follows your prompt instead of improvising. Low values are "
                "loose and creative, high values stick closely to the text but can look harsh, over-saturated or stiff. "
                "Stay close to the model's default and change it in small steps.",
    "cfg_scale": "Guidance (CFG): how strictly the model follows your text. Higher is more literal but can sound or look "
                 "over-processed; lower is looser and more natural but may ignore parts of the prompt. Change it in small steps "
                 "from the default.",
    "cfg_coef": "Guidance strength (CFG): how hard the model is pushed towards your text or settings. Higher follows them more "
                "literally but can sound harsh or artificial; lower is looser and more natural. Stay near the default and "
                "change it a little at a time.",
    "temperature": "How adventurous the model is when it picks the next piece. Low values are safe, steady and repeatable; high "
                   "values are more varied and lively but make mistakes or odd sounds more likely. Try small changes of 0.1 "
                   "and keep it moderate for anything that must be accurate.",
    "top_k": "Limits the model to its k most likely choices at every step. A small number keeps results safe and steady; a large "
             "number (or 0 for off) allows more variety and surprises. Usually leave it alone and adjust temperature instead.",
    "top_p": "Lets the model choose only among the most likely options that together make up this share of the probability. "
             "Lower is safer and more repeatable, 1.0 allows everything. A change of a few hundredths is already noticeable, "
             "so leave it unless results sound too random or too flat.",
    "min_p": "Throws away choices that are much less likely than the best one. Raising it makes the output steadier and "
             "less random; lowering it allows more unusual results. Small values are normal; change it only if the output "
             "sounds erratic.",
    "repetition_penalty": "Discourages the model from repeating itself, which helps against stuttering, looping or a stuck "
                          "syllable. Too high makes speech or music strange and avoids natural repeats. Raise it slowly only if "
                          "you hear repeats.",
    "language": "The language of the text or audio. Setting it explicitly is almost always better than letting the model guess, "
                "especially for short lines and for languages other than English. Pick the language that matches your text.",
    "speed": "How fast the result is played or spoken. Above 1 is faster, below 1 slower; extreme values can make the voice "
             "sound rushed or distorted. Small changes are usually enough.",
    "seconds": "How long the result should be, in seconds. Longer clips take more time and memory, and some models get less "
               "consistent the longer they run. Ask for what you need; you can trim afterwards.",
    "aspect": "The shape of the picture or video as width to height: 1:1 is square, 16:9 is wide like a film or YouTube, 9:16 is "
              "tall like a phone story. Pick it to match where the result will be used, because cropping later loses detail.",
    "size": "How large the picture is, in pixels along one side. Bigger gives more detail but takes much longer and needs more "
            "memory. Start small for drafts and go big only for the final render, or upscale afterwards.",
    "resolution": "How many pixels the result has. Higher means sharper detail but longer waits and more memory. Draft at a low "
                  "setting and render the final at a high one.",
    "fps": "Frames per second: how many pictures make up one second of video or animation. Higher looks smoother but means more "
           "frames to compute. Use the value the model was trained at unless you have a reason to change it.",
    "text": "The words to speak (or the text the model works on). Punctuation shapes the pauses and the rhythm: commas make short "
            "breaks, full stops longer ones. Write it as you would like to hear it, and spell out numbers and abbreviations.",
    "instruct": "A short description of how the voice or the delivery should sound, in everyday words, for example the age, the "
                "mood or the speaking style. The more specific it is, the better the model can follow it. If the result is "
                "off, change one detail at a time.",
    "instruction": "A plain-language instruction to the model, for example how the voice should sound or what role the model "
                   "should play. Specific descriptions work better than vague ones. Change one thing at a time to see what it "
                   "does.",
    "voice": "Which voice to use. Pick one from the list, or type the name or id the model expects. Hear a short sample first, "
             "since the same text sounds quite different with each voice.",
    "speaker": "Chooses a built-in speaker. The same speaker keeps the voice consistent from line to line. Try a few on a short "
               "sentence to find one you like.",
    "ref_text": "The exact words spoken in your reference recording. Giving the model the real transcript helps it match the voice "
                "more closely; a wrong or missing transcript lowers the likeness. If you leave it empty, the studio or model "
                "may guess it automatically.",
    "mode": "Chooses between the working modes of this model. Each mode keeps different qualities of the source. Try the default "
            "first and switch only if the result is not what you wanted.",
    "format": "The file format of the result. WAV and FLAC are lossless, which means the best quality but larger files. Choose "
              "what the next tool or your project needs.",
    "quality": "A trade-off between price or speed and detail. Higher looks better but takes longer and, for cloud models, costs "
               "more. Use low for tests and high only for the final version.",
    "remove_bg": "Automatically cuts out the main object and removes the background before the model starts. Models that expect "
                 "a clean object on plain background give better results this way. It is skipped when the image already has "
                 "transparency.",
    "offload": "Moves parts of the model out of the graphics memory while they are idle. That saves memory on small graphics "
               "cards but costs time. On this unified-memory machine it helps little, so leave it on the fastest option unless "
               "you run out of memory.",
    "vae_tiling": "Decodes the finished video or image in tiles instead of all at once. This needs much less peak memory with "
                  "almost no change in the result, but is slightly slower. Keep it on for big sizes; turn it off only if you "
                  "know you have memory to spare.",

    # ------------------------------------------------------------------------------------------- image generation / editing
    "true_cfg": "A second guidance setting that also uses your negative prompt (what you do not want). At 1 it is off and the "
                "run is fast; above 1 the model runs twice per step, which doubles the time but follows the prompt more "
                "strictly. Try small steps like 3 to 5 if details are ignored.",
    "cfg_trunc_ratio": "Applies the guidance only during the first part of the process and switches it off for the rest. A lower "
                       "value is faster and gives softer, more natural details; 1 applies it all the way. Leave it unless the "
                       "result looks over-sharpened or the speed matters.",
    "kv_cache": "Remembers part of the model's earlier work so it does not have to be recomputed on every step. This makes "
                "generation faster at the cost of a little extra memory and normally does not change the picture. Leave it on.",
    "output_resolution": "The size at which your reference images are fed to the model when editing or combining images. "
                         "Higher reads more detail from the references (so faces and logos survive better) but is slower and "
                         "uses more memory.",
    "transparent": "Makes the model draw the subject on a see-through background (RGBA, with an alpha channel), ready to place "
                   "on other pictures. The transparency is created while drawing, not cut out afterwards. Describe a single "
                   "clear object for the cleanest edges.",
    "threed_ready": "Adds wording to your prompt so the picture shows a single object, fully visible, on a plain background "
                    "with even light. That is the ideal input for turning a picture into a 3D model. Turn it on for 3D "
                    "sources, off for ordinary artwork.",
    "rendering_speed": "Trades speed for quality in the cloud model. Turbo is fastest and cheapest, Quality is slowest and gives "
                       "the best detail and text rendering. Use Turbo for ideas and Quality for the final picture.",

    # ------------------------------------------------------------------------------------------- image tools
    "alpha_matting": "A more careful way to cut out an object that keeps soft edges like hair, fur and smoke. It makes the "
                     "cut-out nicer but is noticeably slower. Turn it on for portraits and animals, leave it off for hard-edged "
                     "objects.",
    "post_process": "Cleans up the cut-out mask afterwards: it removes tiny speckles and rounds off ragged edges slightly. Helps "
                    "when the result has stray dots, but can soften fine details. Turn it on only if you see speckles.",
    "alpha_clean": "Makes nearly invisible pixels fully transparent. Many background removers leave a faint haze around the "
                   "object, which confuses later steps such as 3D recentering. Raise it if you still see a ghost halo; keep it "
                   "small so real soft edges survive.",
    "alpha_threshold": "How opaque a pixel must be to count as part of the object when the tool finds the object's outline for "
                       "cropping. Raise it to ignore faint haze; lower it to keep very soft edges inside the frame.",
    "margin": "The share of the picture left empty around the object after it is centered. More margin gives the object more "
              "breathing room; less margin makes it fill the frame. A little margin is good for 3D models.",
    "background": "What goes behind the object in the result: transparent keeps the see-through background, white fills it with "
                  "solid white. Use white for models that dislike transparency.",
    "out_size": "The side length of the finished square picture, in pixels. Choose 'keep' to leave the size unchanged, or a "
                "fixed size for models that need a certain input size.",
    "remove_bg_first": "If the picture has no transparency yet, the background is removed first so only the object is centered. "
                       "Without it, the whole picture counts as the object. Turn it off when the image is already cut out.",
    "scale": "How many times larger the picture becomes: 2 doubles width and height, 4 quadruples them. Bigger results need more "
             "time and memory. Pick 2 for a gentle boost and 4 for small images.",
    "tile": "Processes the picture in small squares one at a time, so very large images fit in memory. Smaller tiles need less "
            "memory but are slower and may show faint seams. Use 0 (no tiling) unless you run out of memory.",

    # ------------------------------------------------------------------------------------------- 3D: shape generation
    "foreground_ratio": "How much of the frame the object fills after the background is removed and the image is recentered. "
                        "A higher value makes the object larger in the frame, which gives more detail but risks cutting off "
                        "edges. Leave it unless parts of your object are being cut off or look tiny.",
    "octree": "The detail level of the 3D shape: it sets how fine the grid is that the surface is extracted from. Higher gives "
              "more detail and more triangles, but needs more time and memory. Use a lower value for quick tests and a "
              "higher one for the final model.",
    "mc_resolution": "How finely the 3D surface is sampled when the shape is turned into a mesh. Higher values give smoother, "
                     "more detailed meshes with many more triangles, and are slower. Raise it for detailed models and lower it "
                     "for quick drafts.",
    "mc_level": "Where the surface sits inside the model's density field. Positive values shrink the shape slightly, negative "
                "values fatten it. Leave it at the middle; change it by tiny amounts only to fix a too-thin or too-fat result.",
    "threshold": "Decides where the solid object ends and empty space begins in the model's density field. Lower makes the "
                 "shape fatter and blobbier, higher makes it thinner and sharper. Move it slowly if the surface looks too thin "
                 "or too bloated.",
    "box_v": "Scales the box in which the model builds the shape. Slightly above 1 leaves a safety margin so the object is not "
             "clipped at its edges; a larger value wastes resolution on empty space. Leave it unless an edge looks cut off.",
    "num_chunks": "How many points of the 3D space are calculated at once while building the mesh. Lower values need less "
                  "memory but take longer; higher values are faster if memory allows. It does not change the result.",
    "border_ratio": "The empty margin the object keeps around it when the input picture is prepared. A larger value shrinks the "
                    "object in the frame, which protects the edges; a smaller one gives more detail. Leave it on the default "
                    "unless the model's edges are clipped.",
    "sigma_shift": "Changes how the denoising steps are spread over the process. Higher values spend more effort on the coarse "
                   "shape, lower values on the fine detail. An advanced setting; leave it at the default unless you are "
                   "experimenting.",
    "flashvdm": "An experimental, faster way to turn the shape into a mesh. It can save a lot of time but might reduce quality "
                "or fail on some objects. Try it for quick drafts; turn it off if the result looks wrong.",
    "shape_cleanup": "Removes broken triangles and tiny floating bits right after the mesh is built. This gives a cleaner model "
                     "that is easier to texture and edit. Keep it on unless you need the raw mesh.",
    "floater_ratio": "Deletes loose pieces floating around the object. Any piece with fewer triangles than this share of the "
                     "whole mesh is removed. Raise it to clean up more aggressively, lower it if small real parts (like "
                     "buttons) disappear.",
    "ss_steps": "Number of passes for the first stage of this 3D model, which decides the overall structure. More steps are "
                "more reliable but slower. The default is a good balance; raise it a little if the overall form is wrong.",
    "ss_cfg": "Guidance for the first stage of the 3D model (the overall structure). Higher sticks closer to the input "
              "picture's outline, too high makes the form stiff. Change in small steps.",
    "slat_steps": "Number of passes for the second stage, which adds the surface detail and color. More steps give finer detail "
                  "but cost time. Raise it a little if the surface looks muddy.",
    "slat_cfg": "Guidance for the detail stage of the 3D model. Higher follows the input picture's details more closely; too "
                "high looks harsh. Adjust in small steps from the default.",
    "decimation_target": "How many triangles the finished mesh should have at most. Fewer triangles give a lighter model that "
                         "is faster to use in games, but with less fine detail. Pick it to match your target platform.",
    "target_polycount": "How many triangles you want in the final model; 0 lets the service decide. Lower numbers give lighter "
                        "models for games and the web, higher ones keep more detail.",
    "face_limit": "A cap on the number of triangles in the result; 0 means automatic. A lower limit gives a lighter model for "
                  "games, a higher limit keeps more detail.",
    "topology": "The kind of surface mesh to build: triangles are the standard and work everywhere, quads (four-sided faces) are "
                "preferred for hand-editing and smooth deformation. Choose quad if an artist will sculpt or animate it.",
    "texture": "Whether and how well the model is painted. 'No' gives just the grey shape and is cheapest; Standard and HD add "
               "color, with HD sharper but pricier and slower.",
    "texture_quality": "How detailed the color texture should be. Detailed gives sharper surface painting but takes longer and "
                       "may cost more. Use Standard for drafts.",
    "texture_resolution": "The size of the texture picture wrapped around the 3D model, in pixels. Larger textures look sharper "
                          "close up but make the file bigger and use more memory. 1024 is enough for most small props.",
    "texture_size": "The size of the texture picture wrapped around the 3D model, in pixels. Larger textures look sharper close "
                    "up but make the file bigger and use more memory. Choose by how close the camera will get.",
    "textured": "Also paints the 3D shape with color and surface detail in the same job. Turn it off if you only need the grey "
                "shape or want to texture it separately later. Texturing adds time and, for cloud models, cost.",
    "pbr": "Adds physically based rendering maps (metal, roughness, normal) so the surface reacts to light realistically in "
           "game engines. It needs more time and makes the file larger. Turn it on for game or render-ready assets.",
    "enable_pbr": "Adds physically based rendering maps (metal, roughness, normal) so the surface reacts to light realistically "
                  "in game engines. It makes the job slower and the file larger. Turn it on for game or render-ready assets.",
    "generate_type": "Chooses what kind of model is produced: Normal is a full textured model, LowPoly a simpler, lighter mesh, "
                     "Geometry only the grey shape without colors. Pick Geometry if you plan to texture it yourself.",

    # ------------------------------------------------------------------------------------------- 3D: texturing
    "tex_faces": "How many triangles the shape may have when it is painted. The mesh is simplified to this size first, which "
                 "keeps texturing fast and memory use low. Raise it for detailed models, lower it if texturing runs out of "
                 "memory.",
    "tex_steps": "Passes of the painting model for each camera view. More steps give cleaner color and detail but take longer; "
                 "past a point you will not see a difference. Use fewer for a quick look.",
    "tex_guidance": "How strictly the painter follows the reference picture. Higher sticks closer to it but can look flat or "
                    "harsh; lower is softer and freer. Change by small steps.",
    "tex_views": "How many camera angles are painted around the object. More views cover more surface and reduce unpainted "
                 "patches but cost more time and memory. Raise it for complex objects with many hidden parts.",
    "tex_resolution": "The size of each camera view the painting model creates. Larger views are sharper but use much more "
                      "memory and time. The default is a good balance.",
    "tex_render_size": "The size at which the painted views are projected back onto the 3D model. Larger gives crisper results "
                       "but takes more time and memory. 'auto' picks a value from the view size.",
    "tex_size": "The final texture picture size in pixels, wrapped around the model. Larger looks sharper close up but makes the "
                "file bigger and takes more memory. 2048 or 4096 suit most game assets.",
    "tex_seed": "The random start for the painting step. Keep it the same to get the same colors again, change it to get a "
                "different paint job on the same shape. It is separate from the shape seed.",
    "tex_reference": "Which picture the painter copies the look from. 'cleaned' and 'original' use the picture the shape was "
                     "made from; 'current' uses whatever is in the image slot now. Use 'current' to paint a model in a "
                     "different style.",
    "tex_full_res": "Keeps the texture at its full size instead of shrinking it when saving. You get sharper textures but a "
                    "larger file. Turn it on only if you need the extra detail.",
    "super_res": "Sharpens each painted view with an upscaling model before it is put on the 3D shape. Gives crisper textures "
                 "but costs time and memory. Turn it off for quick tests.",
    "sr_tile": "Splits the sharpening step into small squares so it fits in memory. Smaller tiles need less memory but are "
               "slower and can show faint seams. 0 means no tiling; use a tile size only if you run out of memory.",
    "bake_angle": "The steepest viewing angle at which a camera view may still paint a surface, in degrees. Lower values avoid "
                  "smeared color on surfaces seen at a glancing angle but leave more unpainted gaps. Raise it to fill more "
                  "area, lower it for cleaner paint.",
    "bake_exp": "How strongly views that look straight at a surface are preferred when several views overlap. Higher values give "
                "sharper but harder-edged blends, lower values blend more softly. An advanced setting; leave it at the default.",
    "uv_padding": "Extra pixels painted around each island of the unwrapped texture so seams do not show colored cracks. 0 uses "
                  "the default unwrap. Raise it a little if you see dark lines along the seams.",
    "vertex_inpaint": "Fills places that no camera view reached by copying color from neighbouring spots. This removes bare "
                      "patches and holes in the texture. Keep it on unless it smears color where you do not want it.",
    "metallic_scale": "How metallic the surface looks, from 0 (not metal) to 1 (full effect of the metal map). Lower it for "
                      "fabrics, wood and plastic; leave it up for metal objects.",
    "roughness_scale": "How rough or shiny the surface looks. Lower values make it glossier; 1 uses the full roughness map. "
                       "Change it to make materials look more polished or more matte.",

    # ------------------------------------------------------------------------------------------- 3D: multi-view, rigging, motion
    "decode_t": "How many video frames are decoded at the same time. Lower values need less memory but take longer. Reduce it "
                "if the job runs out of memory; it does not change the look.",
    "elevation": "How high the virtual camera sits above the object, in degrees. 0 looks from the side, larger values look "
                 "down on it from above. About 10 to 20 shows an object naturally.",
    "save_views": "Also saves every generated angle as its own picture. Handy for checking the views or using them elsewhere, "
                  "but it adds files to your project.",
    "view_stride": "Saves only every Nth frame of the turntable as a picture. A bigger number gives fewer images; 1 saves every "
                   "frame.",
    "num_beams": "How many alternative skeletons are explored at once when the skeleton is predicted. More gives a steadier, "
                 "more sensible skeleton but takes longer. Raise it if the bones look odd.",
    "skin_grid": "The detail of the grid used to decide how the skin follows each bone. A finer grid gives smoother bending "
                 "around joints but is slower. Use the default unless the mesh deforms badly.",
    "facing": "Which direction the character is looking at the start. 'auto' works it out from the feet. Pick a direction "
              "yourself if the character walks backwards or sideways.",
    "root_motion": "Whether the character travels through space or stays on the spot. 'root' lets it walk forward; 'in place' "
                   "keeps it where it is, which is what you want for looping walk cycles in a game.",
    "foot_lock": "Pins a foot in place while it touches the ground, to stop it from sliding. Keep it on for walking and "
                 "standing; turn it off for floaty moves like jumping or swimming.",
    "arm_offset": "Spreads the arms away from the body by this many degrees. Use a bit more for wide characters whose arms cut "
                  "into their body while moving.",
    "keep_clips": "Keeps the test animations from the rigging step next to the new animation. Turn it on if you want to compare "
                  "them; leave it off for a tidy file.",

    # ------------------------------------------------------------------------------------------- audio: generic generation
    "cfg_style": "How strongly the music follows the style you describe. Higher sticks to the described genre and mood but can "
                 "sound repetitive; lower is freer. Change it in small steps.",
    "loop": "Makes the end of the sound blend into the start so it can repeat without a click or a gap. Turn it on for "
            "ambient beds and game loops, leave it off for one-shot effects.",
    "extend_seconds": "How much new audio to add when you continue an existing clip. Longer means more to compute and a bigger "
                      "chance to drift from the original style. It only applies when a clip to extend is set.",
    "prompt_influence": "How literally the model follows your text. Higher gives exactly what you describe but less variety; "
                        "lower is more creative and surprising. Try 0.3 to 0.6 for most sounds.",
    "struct_prompt": "An optional structured description saying what happens when, for example a dog bark at the start and rain "
                     "throughout. Empty means your normal prompt is used for the whole clip. Use it for effects with several "
                     "events in a set order.",
    "frequency": "The pitch of the test tone in hertz (cycles per second). Low values sound deep, high values piercing. It is "
                 "only for checking that the studio works.",
    "waveform": "The shape of the test tone: sine is smooth, square is buzzy, saw is bright and harsh, noise is hiss. It is only "
                "for checking that the studio works.",
    "step_ms": "A pause between steps of the self-test model, in milliseconds. It simulates a slow job so you can test progress "
               "and cancel buttons.",
    "mask_video": "Hides the picture content from the model so your text decides WHAT it sounds like and the video decides only "
                  "WHEN things happen. Use it when the model picks the wrong sounds for the scene.",
    "mux_video": "Also saves a copy of the video with the new sound attached. Convenient for checking the sound against the "
                 "picture; it adds a video file to your project.",
    "cot": "A step-by-step description of the sounds in the video, which guides the model like a mini script. If left empty, the "
           "caption is used. Be concrete, for example 'footsteps on gravel, then a door closes'.",
    "audio": "Whether the video should contain sound. The model always creates it; if you turn this off it is simply left out "
             "of the file. Turn it off for silent clips to save size.",

    # ------------------------------------------------------------------------------------------- audio: music and song
    "lyrics": "The words to sing. Put section labels like [verse], [chorus] and [bridge] on their own lines to give the song a "
              "structure. Leave it empty for instrumental music. Keep lines short and singable.",
    "bpm": "Beats per minute, the tempo of the music. 60 is slow and calm, 120 is a steady pop beat, 160 and above is fast. "
           "0 lets the planner choose a tempo that fits your description.",
    "keyscale": "The musical key, for example D minor. Minor keys sound darker and sadder, major keys brighter. Leave empty to "
                "let the planner choose.",
    "lm_temperature": "How adventurous the planning model is when it designs the song structure. Lower is safer and more "
                      "conventional; higher is more surprising but can wander. Small changes are enough.",
    "thinking": "Lets a language model plan the song (structure, tempo, key) before the music is made. It usually gives better "
                "organised songs and fills in what you left empty, at the cost of some extra time.",

    # ------------------------------------------------------------------------------------------- audio: speech (TTS / cloning)
    "gender": "Whether the generated voice should sound female or male. Combine it with pitch and speed to shape the voice.",
    "pitch": "How high or low the voice sounds, from very low to very high. Moderate sounds natural; go to the extremes only "
             "for characters.",
    "speed_level": "How fast the voice speaks, from very slow to very fast. Moderate sounds natural; very fast can be hard to "
                   "understand.",
    "speaking_rate": "How fast the voice speaks, in sounds (phonemes) per second. Around 15 sounds like normal speech; lower is "
                     "slow and careful, much higher is rushed.",
    "pitch_std": "How much the pitch moves up and down while speaking. Low is flat and monotone, normal speech is in the middle, "
                 "high values sound expressive and theatrical and can turn wild.",
    "fmax": "The highest frequency the generated sound may contain, in hertz. Higher values keep more brightness and air; lower "
            "values sound duller. Use the value the model recommends for cloning.",
    "accent": "The accent of the base voice before it is changed to match the reference voice. Pick the one closest to the "
              "accent you want in the result.",
    "alpha": "How much of the reference voice's timbre (the colour of the voice) is copied. Low values stay close to the "
             "reference; high values let the model invent more. Try small steps if the voice does not feel right.",
    "beta": "How much of the reference voice's speaking rhythm and intonation is copied. Lower stays closer to the reference's "
            "prosody; higher lets the model choose its own. Adjust slowly.",
    "embedding_scale": "How strong and expressive the speaking style is. Higher is more emotional and dramatic but can become "
                       "unstable or exaggerated; lower is calm and safe.",
    "exaggeration": "How emotional and dramatic the voice is. About the middle is neutral; higher is more expressive and "
                    "faster, but very high values become unstable. Raise it for acting, lower it for narration.",
    "cfg_weight": "Balances how closely the voice follows the reference against how free the pacing is. Lower values give a "
                  "slower, more deliberate delivery and keep less of the reference accent; higher values copy the reference "
                  "more. Try lower values for fast reference speakers.",
    "guidance_scale": "How strictly the model follows your text. Higher is more literal but can sound tense or distorted; "
                      "lower is more relaxed. Stay near the default.",
    "cfg_strength": "How strongly the model follows the text and reference voice. Higher copies the voice and words more "
                    "faithfully but can sound harsher; lower is smoother and sloppier.",
    "cfg_rate": "How strongly the model is pushed towards the reference voice. Higher sounds more like the reference but may "
                "add artifacts; lower is smoother but less alike.",
    "duration_factor": "A multiplier for the length of the speech. Above 1 makes the speech longer and slower, below 1 shorter "
                       "and faster. Use it to fit the voice to a time slot.",
    "target_seconds": "The exact length you want the speech to have; 0 lets the model decide from the text. Fixing it helps "
                      "when the line must fit a time slot, but very short slots make the speech rushed.",
    "max_seconds": "The longest audio the model may produce. It is only a safety limit; generation stops at the end of the "
                   "text anyway. Raise it for long scripts and lower it to stop runaway output early.",
    "max_tokens": "A cap on how long the generated audio can get; 0 means until the text is finished. Use it as a safety limit "
                  "so a stuck run does not go on forever.",
    "max_segment_tokens": "Long lines are cut into pieces of this size and spoken one after another. Smaller pieces are safer "
                          "for long text but can add small pauses between pieces; larger pieces flow better.",
    "pause_ms": "Silence inserted between the pieces of a long line, in milliseconds. Raise it for a calmer, more deliberate "
                "delivery, lower it for a brisker one.",
    "pause_s": "A pause length in seconds. In speech it is the silence between parts; in subtitles, a gap in speech longer "
               "than this starts a new subtitle.",
    "noise_scale": "How much randomness goes into the voice. Higher is livelier and more varied between takes but less stable; "
                   "0 gives the same result every time. Small values sound the most natural.",
    "noise_scale_w": "How much randomness goes into the length of each sound. Higher varies the rhythm more; lower is more "
                     "regular. Leave the default unless the rhythm sounds robotic or sloppy.",
    "noise_w": "How much the rhythm and word lengths vary. Higher sounds more natural and lively but can be uneven; lower is "
               "regular and robotic. Leave it near the default.",
    "sdp_ratio": "Balances regular and varied rhythm in the speech. Higher adds more natural variation in how long sounds last "
                 "and in the pauses; lower is steadier. Change in small steps.",
    "tau": "How much of the reference voice's timbre is applied on top of the base voice. Lower stays close to the base "
           "voice's style, higher moves towards the reference voice's colour. Raise it if it does not sound like the "
           "reference.",
    "normalize": "Cleans up the script before speaking: it removes decorative punctuation and turns laughs like 'haha' into a "
                 "laugh sound. Turn it off only if the cleaning changes text you want read exactly as written.",
    "s2a_steps": "Number of refinement passes for the final sound. More steps give cleaner audio but take longer; the default "
                 "is usually enough.",
    "stability": "How steady the voice is between takes. Higher is consistent and calm; lower is more expressive and varied, "
                 "but can wander. Use lower values for acting and higher for narration.",
    "tts_model": "Which voice model to use. Newer ones are usually more expressive; older or 'flash' variants are faster and "
                 "cheaper. Leave it on the default unless you need a particular style.",
    "text_temp": "How adventurous the first stage (the one that decides the words and rhythm) is. Lower is steady and "
                 "predictable, higher is more varied and can go off script.",
    "waveform_temp": "How adventurous the last stage (the one that creates the actual sound) is. Lower is clean and steady, "
                     "higher adds more variation and noise.",
    "temperature_text": "How adventurous the model is when it chooses the words it will say. Lower is more predictable; higher "
                        "gives more varied answers but may wander off topic.",
    "ref_text2": "The exact words spoken in the reference recording of speaker 2. Having the correct transcript makes the voice "
                 "match better. If it is empty, the saved transcript of the voice is used.",
    "ref_text3": "The exact words spoken in the reference recording of speaker 3. Having the correct transcript makes the voice "
                 "match better.",
    "ref_text4": "The exact words spoken in the reference recording of speaker 4. Having the correct transcript makes the voice "
                 "match better.",
    "ref_language": "The language of your reference recording. Set it to the language that is actually spoken in the reference "
                    "so the voice is analysed correctly.",
    "source_text": "What the source recording says. Used in style mode to align the words; leave empty to let the studio transcribe "
                   "it automatically.",
    "use_random": "Picks a random emotion for variety when mixing emotions. It makes takes more lively but the voice can "
                  "drift away from the reference. Leave it off for a consistent voice.",
    "emo_text": "Describe the emotion in a few words, for example 'furious, shouting'. If left empty, the model infers the "
                "feeling from the line itself.",
    "emo_alpha": "How strongly the emotion is applied to the voice. Lower values sound more natural; high values can become "
                 "exaggerated. Around the middle is a good start.",
    "emotion_mode": "Where the emotion comes from: the reference voice itself, a separate emotion clip, or a text description. "
                    "Pick the one that matches the material you have.",

    # ------------------------------------------------------------------------------------------- audio: voice conversion and singing
    "auto_f0_adjust": "Automatically shifts the melody of your source into the range of the reference voice. Keep it on for "
                      "speech between different people. Turn it off when you set the pitch yourself or sing.",
    "f0_method": "How the pitch of your source recording is measured. The robust option copes with singing and noisy "
                 "recordings; the fast one is quicker but can be jittery. Use the robust one if unsure.",
    "pitch_shift": "Moves the pitch up or down in semitones (12 is one octave). Use +12 for a male source into a female voice, "
                   "-12 for the reverse, and 0 when the voices are similar.",
    "index_rate": "How much the voice's own training features replace the source's. Higher sounds more like the target voice "
                  "but can add accent or artifacts; 0 turns it off. Try moderate values.",
    "protect": "Protects breaths and consonants of the source from being altered, which avoids garbled sounds. Lower values "
               "protect more; the top value turns it off. Lower it if consonants sound smeared.",
    "rms_mix_rate": "Whether the loudness of the result follows the source (low) or the converted voice (high). Lower values "
                    "keep the natural loudness changes of the source.",
    "speaker_id": "Chooses the speaker inside a voice file that contains several. Leave it at 0 unless the voice file says "
                  "otherwise.",
    "length_adjust": "Changes the speed of the converted speech. Above 1 is slower, below 1 is faster. Keep it at 1 unless you "
                     "need to fit a time slot.",
    "singing": "Switch on when your source is singing: a model that follows the melody is used. Off uses the speech model, "
               "which is better for talking.",
    "key": "Moves the whole melody up or down in semitones (12 is one octave). Use it to fit the song to the singer's comfortable "
           "range.",
    "score": "The notes to sing, one per line: syllable, note and length in seconds, for example 'ni C4 0.5'. SP is a pause and "
             "AP a breath. A slur uses two notes on one syllable.",

    # ------------------------------------------------------------------------------------------- audio: separation, subtitles, ASR
    "stems": "Which parts to split the music into. Two parts (vocals and the rest) are quickest; all parts also give drums, "
             "bass and others as separate files.",
    "shifts": "Runs the separation several times with tiny time shifts and averages them. Each extra shift gives a slightly "
              "cleaner result but takes proportionally longer. 1 is fine for most uses.",
    "overlap": "How much neighbouring chunks of the song overlap while it is processed. More overlap hides seams between chunks "
               "but takes longer. The default is a good compromise.",
    "overlap_factor": "How many times neighbouring chunks overlap. Higher gives smoother joins between chunks but is slower.",
    "main_format": "The subtitle file format that is shown as the main result. SRT works almost everywhere, VTT is for web "
                   "players. The other format is saved with it.",
    "max_chars": "The longest allowed subtitle line in characters. Shorter lines are easier to read on small screens but "
                 "need more lines; about 40 is the usual limit.",
    "max_cue_s": "The longest time one subtitle may stay on screen, in seconds. Shorter limits make more, quicker subtitles "
                 "that are easier to read; longer limits give fewer, slower ones. Around 5 to 7 seconds is typical.",
    "max_lines": "How many lines a subtitle may have on screen at once. Two is standard; more lines hide more of the picture.",
    "split_sentences": "Starts a new subtitle at the end of every sentence. This makes subtitles easier to follow. Turn it off "
                       "if you prefer subtitles split only by length.",
    "translate": "Translates the speech into English while transcribing, whatever the spoken language. Handy but not perfect: "
                 "always check names and numbers.",
    "hotwords": "A comma-separated list of names or terms the model should spell correctly, such as product names or unusual "
                "words. It helps with rare words but does not change the rest.",
    "prompt_language": "The language of the instruction given to the model, not of your recording. English works best for most "
                       "users; choose Chinese only if you prefer it.",
    "max_new_tokens": "The longest text the model may write. Longer recordings need a higher value so the transcript or answer "
                      "is not cut off, but a higher value allows more time for runaway text.",
    "question": "What you want to know about the recording, in plain words, for example 'Describe the voice and emotion' or "
                "'What instruments are playing?'. Specific questions get more useful answers.",

    # ------------------------------------------------------------------------------------------- audio: dubbing pipeline
    "stage": "Runs the whole dubbing, or only the first part (transcript and translation) so you can fix the lines before "
             "voices are made. Run the draft first for important videos.",
    "source_language": "The language spoken in the video. Auto detects it, but naming it explicitly is more reliable for short "
                       "clips or noisy audio.",
    "target_language": "The language to dub into, as a short code like en, th or ja. What is available depends on the voice "
                       "model you picked.",
    "translate_backend": "Which engine translates the lines. Empty uses the local prompt model on this computer; a cloud or "
                         "command-line engine may translate better but sends the text away.",
    "separate_model": "Which model splits the speech from the background music. Leave empty for the default; pick a stronger one "
                      "when music is dense or loud.",
    "bg_gain_db": "Changes the volume of the original background in decibels. Negative is quieter, 0 unchanged, positive "
                  "louder. Lower it if the background fights the new voice.",
    "duck_db": "Lowers the background by this many decibels while someone is speaking, so the voice stays clear. Higher "
               "values make the voice clearer but the music pump more noticeably.",
    "lufs": "The target loudness of the final mix. -16 is a common level for podcasts and web video; lower is quieter. Use "
            "the same value for all your episodes so they match.",
    "max_speedup": "How much a dubbed line may be sped up to fit its time slot. Higher keeps timing tight but can sound rushed; "
                   "1 never speeds up. Around 1.2 to 1.3 is hardly noticeable.",
    "keep_original_track": "Keeps the original audio as a second audio track inside the MP4, so viewers can switch between "
                           "the dub and the original. Slightly larger file.",
    "segments": "The lines to use for the job. For dubbing, the edited list from a draft run; leave empty to transcribe and "
                "translate automatically. For songs, how many lyric sections to sing.",

    # ------------------------------------------------------------------------------------------- video tools
    "match_image": "Uses the shape of your start picture for the video. Off crops the picture to the aspect ratio you chose "
                   "instead. Keep it on to avoid cropping.",
    "factor": "How many times more frames the video gets. x2 doubles the frame rate for smoother motion, x4 quadruples it. "
              "The higher you go the longer it takes, and fast motion may show small errors.",
    "keep_audio": "Keeps the original sound in the smoothed video. Turn it off if you only need the picture.",

    # ------------------------------------------------------------------------------------------- mesh export
    "export_type": "The file format of the 3D model. GLB holds everything in one file and suits web and game engines; OBJ, PLY "
                   "and STL are older formats that other 3D programs and printers read. Choose GLB unless a tool needs another.",
    "include_texture": "Keeps the colors and texture on the exported model. When off, the mesh is processed and saved as a "
                       "plain grey shape. Keep it on for textured models.",
    "post_degenerate": "Removes broken triangles that have no area. This makes the model cleaner and safer for games and "
                       "printing. Keep it on.",
    "post_fix_normals": "Makes all surfaces face outward consistently. It fixes models that look inside-out or have patchy "
                        "lighting. Turn it on if shading looks wrong.",
    "post_floaters": "Deletes tiny loose pieces floating near the model. Keeps the file clean; turn it off if small real parts "
                     "disappear.",
    "post_simplify": "Reduces the number of triangles to the target. The model becomes lighter and faster but loses fine detail. "
                     "Turn it on for games and web.",
    "post_smooth": "Smooths bumpy surfaces by averaging neighbouring points, without shrinking the model. More iterations are "
                   "smoother but wash out small details. Use a few for noisy AI meshes.",
    "post_target_faces": "How many triangles the model should have after simplifying. Lower is lighter but less detailed. Pick "
                         "by where the model will be used, for example a few thousand for a game prop.",

    # ------------------------------------------------------------------------------------------- remaining
    "max_faces": "A limit on the number of triangles. Larger meshes keep more detail but take longer and more memory; a smaller "
                 "limit is faster and lighter.",
    # ------------------------------------------------------------------------------------------- retopo + bake
    "target_faces": "How many polygons the clean model should have. Fewer is lighter and faster in a game but loses shape detail; "
                    "the baked normal map brings back the fine surface detail, not the silhouette. Mobile games: a few "
                    "thousand; PC characters: about ten thousand; close-up hero models: thirty thousand.",
    "bake_normal": "Copies the fine surface detail of the original (wrinkles, seams, small bumps) into a normal map, so the light "
                   "model still looks detailed when lit. Keep it on unless you only need the shape.",
    "bake_color": "Copies the colours (and metallic / roughness) of the original onto the new model's texture. Turn it off for a "
                  "plain grey model, for example when you will retexture it.",
    "bake_ao": "Ambient occlusion: darkens creases and corners the light cannot reach, which adds depth in game engines. It is "
               "the slowest bake step; turn it off for quick tests.",
    "quad_mode": "Mostly quads keeps a few triangles where the flow needs them and follows the shape best. Pure quads has no "
                 "triangles at all, which some modelling tools prefer, but it fits the shape less closely for the same face count.",
    "crease_angle": "Makes the new edges follow sharp corners of the shape, which helps hard-surface props such as boxes, weapons "
                    "or furniture. 0 turns it off, which is right for organic characters; try 30-45 degrees for hard edges.",
    "smooth_iter": "How much the flow of the new edges is evened out. A little (the default) gives cleaner loops; more can round "
                   "off small details. Rarely needs changing.",
    "use_shape_mesh": "Many generated models keep a dense shape-only copy next to the lighter textured one. With this on, the "
                      "detail is baked from that dense copy and the colour from the textured one, which gives the sharpest result.",
    "fill_holes": "Small gaps in the new mesh up to this many edges are closed automatically. Bigger openings (a sleeve, a mouth) "
                  "are kept on purpose. Set 0 to keep every hole.",
    "min_part": "Removes tiny loose pieces (dust, stray fragments) that are smaller than this share of the model. Raise it if "
                "specks remain; lower it if small separate parts such as earrings disappear.",
    "ray_distance": "How far the bake looks for the original surface from the new one, as a share of the model size. Automatic "
                    "works for almost every model; raise it only if the maps show stretched smears, lower it if details from a "
                    "nearby part (an arm on the body) bleed through.",
    "ao_rays": "How many directions are tested for each pixel of the AO map. More is smoother but slower; 16 is usually clean, "
               "32 for hero assets.",
    "ao_distance": "How far away an object still casts ambient shadow, as a share of the model size. Small values darken only "
                   "tight creases; large values darken whole areas between limbs.",
    # ------------------------------------------------------------------------------------------- engine export
    "target": "Which game engine the package is made for. Unreal and Unity get an FBX file plus textures named the way that engine likes; Godot and the web get a GLB. Characters with a skeleton are always a GLB. Pick the engine you will import into.",
    "name": "The name the model gets in your game project (letters, digits and underscores). Empty uses the asset's name. Unreal adds its usual SM_ (static) or SK_ (skeleton) prefix for you.",
    "lods": "LOD means level of detail: lighter copies of the model that the engine shows when it is far from the camera, so a scene with many objects runs faster. 3 adds three lighter levels below the full model; 0 makes none. Godot makes its own levels and ignores this.",
    "lod_preset": "How much each lighter level keeps of the full model's triangles. Halves keeps 50%, 25% and 12.5%. Aggressive drops faster (40%, 15%, 6%) for mobile or crowds; gentle (70%, 50%, 35%) keeps more shape for hero objects. Custom uses your own list below.",
    "lod_ratios": "Your own list of levels for the Custom steps, separated by commas. Numbers up to 1 are fractions of the full model (0.5,0.25,0.1); bigger numbers are exact triangle counts (6000,2500,800).",
    "lod_max_error": "A safety limit: a level stops getting lighter when the shape would change by more than this share of the model size. Lower keeps the silhouette truer but may give fewer or heavier levels. The default suits most models.",
    "lod_min_tris": "No level is made lighter than this many triangles, because tiny levels look broken. Raise it for characters, lower it for simple props.",
    "collision": "The invisible shapes a game uses to decide what a model touches, much cheaper than its real surface. Auto picks several convex pieces that follow the shape for objects and one capsule for a character. Box and capsule are the simplest; None skips collision.",
    "collision_quality": "How closely the collision pieces follow the shape. Fast makes a few rough pieces, detailed makes more and tighter ones but takes longer to compute and costs more physics time in the game. Balanced is right for most props.",
    "max_hulls": "The most separate convex pieces the collision may use. More pieces fit the shape better but make physics slower in the game; hollow shapes like chairs or arches need more than blocks do.",
    "hull_max_verts": "The most corner points each collision piece may have. Lower is faster for the engine's physics; 64 is within Unity's limit for convex colliders, so pieces import without being simplified again.",
    "size_mode": "Keep leaves the model at the size it has now. Height scales it so it is a real-world height in metres, which is the easiest way to get a character or prop the right size in the engine.",
    "height_m": "The real-world height in metres the model is scaled to when Size is set to Height. 1.8 is an adult person, 0.9 a chair, 2.1 a door.",
    "pivot": "The point the model turns and scales around in the engine. Bottom centre puts it on the floor under the middle, which is right for things that stand on the ground. Keep leaves it as it is.",
    "tex_max": "Shrinks the texture pictures to at most this size (keeping power-of-two sizes) to save memory in the game. Keep leaves them as they are. 1024 suits mobile, 2048 PC, 4096 only for close-up hero items.",
    "unity_pipeline": "Unity can render with the Universal (URP) or High Definition (HDRP) pipeline and each wants its roughness, metal and occlusion maps packed differently. Both writes the files for each so you can use either; pick one to keep the package smaller.",
    "yaw": "Turns the model around the vertical axis before it is saved, in steps of 90 degrees. Use it if the model faces the wrong way in your engine after import.",
    "include_collision_preview": "Adds a separate GLB that shows only the collision pieces, so you can check them in any viewer, and lets the 3D studio draw them over the model. It is not needed in the game.",
}

# ------------------------------------------------------------------------------------------------- emotion sliders
for _k, _w in _EMOTIONS.items():
    PARAM_INFO[f"emo_{_k}"] = _emotion_text(_w)

# ----------------------------------------------------------------------------------------- same key, different meaning
PARAM_INFO_BY_TASK: dict[tuple[str, str], str] = {
    # size
    ("t2i", "size"): "The length of the picture's longest side in pixels; the other side follows the aspect ratio. Bigger gives "
                     "more detail but takes much longer and uses more memory. Draft small, then render big or upscale.",
    ("edit", "size"): "Sets the size of the edited picture: the model works at about this side length squared, keeping the shape "
                      "of your input image. Bigger keeps more detail but is slower. Choose bigger only for the final edit.",
    ("multi_ref", "size"): "Sets the size of the combined picture: the area of a square of this side length, with the shape of "
                           "your first image. Bigger keeps more detail but is slower and uses more memory.",
    ("rgba", "size"): "The length of the picture's longest side in pixels. Bigger gives more detail on the see-through object "
                      "but takes longer and uses more memory.",
    ("extract", "size"): "The size of the extracted subject picture in pixels. Bigger keeps more detail but is slower.",
    # resolution
    ("shape", "resolution"): "The voxel resolution: how fine the 3D grid is that the shape is built in. Higher gives more "
                             "surface detail and many more triangles, but takes more time and memory.",
    ("t2i", "resolution"): "How large the picture is, for example 1K, 2K or 4K. Higher means more detail but longer waits "
                           "and, for cloud models, higher cost. Draft at the lowest setting.",
    ("edit", "resolution"): "How large the edited picture is. Higher keeps more detail but takes longer and costs more in the "
                            "cloud. Use the lowest setting for tests.",
    ("multi_ref", "resolution"): "How large the combined picture is. Higher keeps more detail but takes longer and costs more "
                                 "in the cloud. Use the lowest setting for tests.",
    ("t2v", "resolution"): "The height of the video in pixels (480p, 720p ...). Higher is sharper but takes a lot more time "
                           "and memory; draft at a low resolution first.",
    ("i2v", "resolution"): "The height of the video in pixels (480p, 720p ...). Higher is sharper but takes a lot more time "
                           "and memory; draft at a low resolution first.",
    ("flf2v", "resolution"): "The height of the video in pixels (480p, 720p ...). Higher is sharper but takes a lot more time "
                             "and memory; draft at a low resolution first.",
    # mode
    ("transcribe", "mode"): "What the transcript contains. 'Speakers + timestamps' gives who spoke when, line by line; "
                            "'Speakers only' gives the labelled text without times and is a bit faster.",
    ("voice_convert", "mode"): "What is taken from the reference voice. Timbre only changes the colour of the voice and keeps "
                               "the source's rhythm; the other modes also copy the speaking style or accent. Start with "
                               "timbre only.",
    ("tts", "mode"): "Chooses between the working modes of this model, for example how it treats the reference voice or the "
                     "style. Try the default first and switch only if the result is not what you wanted.",
    ("voice_clone", "mode"): "Chooses between the working modes of this model, for example which qualities of the reference "
                             "(timbre, style, accent) are copied. Start with the default.",
    # text
    ("tts", "text"): "The words to speak. Punctuation shapes the pauses and rhythm; some models also accept tags such as "
                     "[laugh], [S1]/[S2] for speakers or [pause 1.5s]. Spell out numbers and abbreviations for best results.",
    ("voice_clone", "text"): "The words the cloned voice should say. Punctuation shapes the pauses; write numbers and "
                             "abbreviations as words. Any text works, it does not have to match the reference clip.",
    ("voice_design", "text"): "The words the designed voice should say. Use a sentence or two to judge the voice; punctuation "
                              "shapes the pauses.",
    ("s2s", "text"): "An optional typed question or message that you can give instead of, or in addition to, your spoken clip.",
    ("voice_convert", "text"): "The words spoken in the source recording, if the model needs them. Leave empty to let the "
                               "studio find them out automatically.",
    # format
    ("separate", "format"): "The file format of the separated tracks. WAV and FLAC are both lossless (no quality lost); FLAC "
                            "files are smaller. Both are high quality (44.1 kHz, 24 bit stereo).",
    # speed
    ("sing", "speed"): "The tempo of the singing. Above 1 is faster, below 1 slower; all note lengths are divided by it.",
    ("tts", "speed"): "How fast the voice speaks. Above 1 is faster, below 1 slower. Extreme values can sound rushed or "
                      "stretched; small changes sound the most natural.",
    ("voice_clone", "speed"): "How fast the cloned voice speaks. Above 1 is faster, below 1 slower. Extreme values can sound "
                              "rushed or stretched.",
    ("voice_design", "speed"): "How fast the designed voice speaks. Above 1 is faster, below 1 slower. Small changes sound the "
                               "most natural.",
    # seconds
    ("motion", "seconds"): "How long the animation lasts. Longer motions take more time, and the character may drift or lose "
                           "the thread of the move. Short moves of a few seconds are the most reliable.",
    ("s2s", "seconds"): "The longest spoken answer the model may give. It is an upper limit, so shorter answers are fine. "
                        "Raise it if answers are cut off.",
    ("tts", "seconds"): "The longest audio the model may produce; generation stops earlier when the text is finished. Raise it "
                        "for long scripts.",
    ("voice_clone", "seconds"): "The longest audio the model may produce; generation stops earlier when the text is finished. "
                                "Raise it for long scripts.",
    ("song", "seconds"): "The length of the song. This is an upper limit; the model may end it a little earlier. Longer songs "
                         "take more time and may repeat or drift.",
    ("music", "seconds"): "How long the music should be. Longer pieces take more time and may repeat or lose direction; make "
                          "short ones and loop or extend them if you need more.",
    ("t2v", "seconds"): "The length of the video clip. Longer clips need much more time and memory, and motion can become less "
                        "consistent. Short clips are the most reliable.",
    ("i2v", "seconds"): "The length of the video clip. Longer clips need much more time and memory, and motion can become less "
                        "consistent. Short clips are the most reliable.",
    ("flf2v", "seconds"): "The length of the video between your first and last picture. Longer clips need more time and "
                          "memory; short ones keep the transition tight.",
    # steps
    ("voice_convert", "steps"): "Number of refinement passes of the voice conversion. More steps give cleaner results but take "
                                "longer; the default is usually enough.",
    ("shape", "steps"): "Number of passes used to build the 3D shape. More steps give a cleaner, more detailed shape but take "
                        "longer. Use fewer for quick tests.",
    ("texture", "steps"): "Number of passes used to paint the 3D model. More steps give cleaner colors but take longer.",
    ("motion", "steps"): "Number of passes used to create the motion. More steps give smoother and more accurate movement but "
                         "take longer.",
    ("multiview", "steps"): "Number of passes used to generate the extra camera views. More steps give cleaner, more consistent "
                            "views but take longer.",
    ("t2v", "steps"): "Number of passes used for each video. More steps give cleaner motion and detail but take proportionally "
                      "longer; fewer steps make a rough preview.",
    ("i2v", "steps"): "Number of passes used for each video. More steps give cleaner motion and detail but take proportionally "
                      "longer; fewer steps make a rough preview.",
    ("sfx", "steps"): "Number of refinement passes for the sound. More steps give cleaner audio but take longer; 10 is a quick "
                      "preview and 30 to 50 suits final quality.",
    ("music", "steps"): "Number of refinement passes for the music. More steps give cleaner, richer audio but take longer; a "
                        "few steps make a quick preview.",
    # guidance
    ("shape", "guidance"): "How strictly the shape follows your input picture. Higher sticks to the picture's outline but can "
                           "be stiff or over-detailed; lower gives a softer, freer shape. Stay near the default.",
    ("texture", "guidance"): "How strictly the painting follows the reference picture. Higher is closer to it but can look "
                             "harsh; lower is softer.",
    ("multiview", "guidance"): "How strictly the new views follow the input picture. Higher sticks closer but can look "
                               "over-processed; lower is freer.",
    ("motion", "guidance"): "How strictly the motion follows your text description. Higher follows it more literally but can "
                            "look stiff; lower is looser and smoother.",
    ("t2i", "guidance"): "How strictly the picture follows your prompt. Low values are looser and more creative; high values "
                         "stick to the text but can look over-saturated or stiff.",
    ("edit", "guidance"): "How strictly the edit follows your instruction. Higher follows it more but may change more than "
                          "you wanted; lower keeps more of the original picture.",
    ("music", "guidance"): "How strictly the music follows your description. Higher is more literal but can sound harsh or "
                           "repetitive; lower is freer and more natural.",
    ("sfx", "guidance"): "How strictly the sound follows your description. Higher is more literal but can sound harsh or "
                         "artificial; lower is looser and more natural.",
    ("ambience", "guidance"): "How strictly the ambience follows your description. Higher is more literal but can sound "
                              "artificial; lower is looser and more natural.",
    ("tts", "guidance"): "How strictly the voice follows your text and settings. Higher is more literal but can sound tense; "
                         "lower is more relaxed.",
    ("voice_clone", "guidance"): "How strictly the cloned voice follows the reference and your text. Higher is closer to the "
                                 "reference but can sound harsh; lower is smoother.",
    ("voice_design", "guidance"): "How strictly the voice follows your description. Higher is more literal but can sound "
                                  "tense; lower is more natural.",
    # cfg_coef
    ("tts", "cfg_coef"): "How strongly the voice is held to the voice settings. Higher is more consistent with the chosen "
                         "voice; very high values can sound unnatural. The model was trained at a fixed value, so changes "
                         "are small.",
    ("music", "cfg_coef"): "How strictly the music follows your description. Higher is more literal but can sound harsh; "
                           "lower is freer. Stay close to the default.",
    ("sfx", "cfg_coef"): "How strictly the sound follows your description. Higher is more literal but can sound harsh; lower "
                         "is freer. Stay close to the default.",
    # temperature
    ("rig", "temperature"): "How adventurous the model is when it predicts the skeleton. Low values give a conventional, "
                            "steady skeleton; high values try more unusual bone layouts and may fail. Keep it low.",
    ("reskin", "temperature"): "How adventurous the model is when it predicts the skeleton. Low values give a conventional, "
                               "steady skeleton; high values try unusual layouts. Keep it low.",
    ("audio_ask", "temperature"): "How adventurous the answer is. 0 always picks the most likely words, which is best for "
                                  "transcribing and facts; higher gives more varied wording but more mistakes.",
    ("tts", "temperature"): "How lively the voice is. Low is steady and repeatable, high is more expressive and varied but "
                            "mispronunciations or odd noises become likelier. Make small changes.",
    ("voice_clone", "temperature"): "How lively the cloned voice is. Low is steady and close to the reference, high is more "
                                    "expressive but less stable. Make small changes.",
    ("voice_design", "temperature"): "How lively the designed voice is. Low is steady, high is more varied but less stable. "
                                     "Make small changes.",
    ("s2s", "temperature"): "How adventurous the spoken answer is. Low is predictable and safe, high is more varied and "
                            "lively but may wander off topic.",
    ("music", "temperature"): "How adventurous the music is. Low is predictable and safe, high is more varied and surprising "
                              "but can sound messy.",
    ("song", "temperature"): "How adventurous the singing and music are. Low is safe and conventional, high is more varied "
                             "but can wander.",
    ("sfx", "temperature"): "How adventurous the sound is. Low is steady and repeatable, high is more varied but more likely "
                            "to glitch.",
    # language
    ("qa_asr", "language"): "The language spoken in the recording. 'auto' lets the model detect it from the first seconds; "
                            "naming it is more reliable for short or noisy clips.",
    ("subtitles", "language"): "The language spoken in the recording. 'auto' lets the model detect it from the first seconds; "
                               "naming it is more reliable for short or noisy clips.",
    ("song", "language"): "The language of the lyrics. Set it to the language you actually sing in for the best pronunciation.",
    ("music", "language"): "The language of the lyrics, if the music has vocals. Set it to the language you actually sing in.",
    ("tts", "language"): "The language of your text. Setting it explicitly is usually better than auto, especially for short "
                         "lines and for languages other than English.",
    ("voice_clone", "language"): "The language of the text the cloned voice should speak. Setting it explicitly is usually "
                                 "better than auto, especially for short lines.",
    ("voice_design", "language"): "The language of the text the designed voice should speak. Set it explicitly for the best "
                                  "pronunciation.",
    ("voice_convert", "language"): "The language of the speech to convert. Set it to what is actually spoken.",
    # batch
    ("t2i", "batch"): "How many pictures to make in one go. Each uses the next seed so you get different variations. More "
                      "pictures take proportionally longer.",
    ("edit", "batch"): "How many edited versions to make. Each uses the next seed, so you can pick the best one. More versions "
                       "take proportionally longer.",
    ("multi_ref", "batch"): "How many combined pictures to make. Each uses the next seed, so you can pick the best one.",
    ("audio_ask", "batch"): "How many separate answers to produce. Each is a new try, so you can pick the best wording.",
    # remove_bg
    ("multiview", "remove_bg"): "Cuts the background out of your picture first so the model sees only the object. This gives "
                                "cleaner new views. It is skipped when the image already has transparency.",
    ("shape", "remove_bg"): "Cuts the background out of your picture first so the 3D model builds only the object. This "
                            "usually gives cleaner shapes. It is skipped when the image already has transparency.",
    # fps
    ("multiview", "fps"): "Frames per second of the turntable video made from the generated views. Higher plays the same views "
                          "faster; it does not add new detail.",
    ("motion", "fps"): "Frames per second of the animation. Higher is smoother and the file gets larger. Use the rate your "
                       "game engine or video expects.",
    ("t2v", "fps"): "Frames per second of the video. Higher plays the same frames faster or smoother depending on the model, "
                    "and means more frames to compute. Keep the rate the model was trained at unless you need another.",
    ("i2v", "fps"): "Frames per second of the video. Higher plays the same frames faster or smoother depending on the model, "
                    "and means more frames to compute. Keep the rate the model was trained at unless you need another.",
    # aspect (video)
    ("i2v", "aspect"): "The shape of the video. By default it follows your start picture; choose a shape to crop the picture "
                       "to it. 16:9 suits film and YouTube, 9:16 phone stories.",
    ("flf2v", "aspect"): "The shape of the video. By default it follows your first picture; choose a shape to crop the pictures "
                         "to it.",
    # instruction / instruct
    ("s2s", "instruction"): "An optional role or persona for the voice assistant, for example 'You are a gruff dwarven "
                            "blacksmith. Answer in two sentences.' The clearer the role, the more consistent the answers.",
    ("voice_design", "instruction"): "A description of the voice you want, in plain English: gender, age, pitch, pace, mood, "
                                     "accent, and recording quality. The more precise, the closer the voice. Change one detail "
                                     "at a time.",
    ("tts", "instruction"): "A description of how the line should be delivered, in plain English: mood, pace, pitch, accent "
                            "and recording quality. Specific wording works better than vague words.",
    ("voice_clone", "instruction"): "A description of how the cloned voice should deliver the line, for example mood and pace. "
                                    "It steers the delivery, not the voice's identity.",
    ("tts", "instruct"): "How the voice should sound or deliver the line, in short plain words, for example 'female, young, "
                         "high pitch' or 'say it with a hint of panic'. Be specific and change one detail at a time.",
    ("voice_clone", "instruct"): "How the cloned voice should deliver the line (mood, pace, intensity), in plain words. It "
                                 "steers the delivery; the voice itself comes from the reference.",
    ("voice_design", "instruct"): "Describe the voice you want in plain words, for example 'female, young adult, high pitch, "
                                  "British accent'. More precise descriptions give a closer voice.",
    # voice / speaker
    ("voice_convert", "voice"): "The voice the recording is turned into. Choose the voice file (without .pth) or its folder name "
                                "from the voices you have installed.",
    ("s2s", "voice"): "The voice the assistant answers in. Listen to a few short samples to find one you like.",
    ("tts", "speaker"): "Chooses a built-in speaker. The same speaker keeps the voice consistent from line to line, and the "
                        "description then only steers delivery. Try a few on a short sentence.",
    ("voice_clone", "speaker"): "Chooses a built-in speaker used as the base for the cloned voice. Try a few on a short "
                                "sentence to find one that fits.",
    ("voice_design", "speaker"): "Chooses a named speaker as the base of the voice. The same speaker keeps the voice consistent; "
                                 "your description then steers delivery and recording quality.",
    # ref_text
    ("voice_convert", "ref_text"): "The exact words spoken in the reference recording. The correct transcript makes the "
                                   "likeness better; if empty, the studio transcribes it automatically.",
    ("voice_design", "ref_text"): "The exact words of the sample voice, if you give one. A correct transcript helps the model "
                                  "match its tone.",
    # max_faces
    ("shape", "max_faces"): "A limit on the number of triangles in the 3D mesh; 0 keeps everything. Fewer triangles make a "
                            "lighter model that loads faster but loses fine detail.",
    ("rig", "max_faces"): "How many triangles of the mesh are used to predict the skeleton. A lower number is faster and uses "
                          "less memory; the full mesh is still used for the final result. Raise it only for complex models.",
    ("reskin", "max_faces"): "How many triangles of the mesh are used to predict the skeleton. A lower number is faster and "
                             "uses less memory. Raise it only for complex models.",
    # segments
    ("dub", "segments"): "The subtitle lines to dub, as edited from a draft run, with start, end, text and translation. Leave "
                         "empty to transcribe and translate the video automatically.",
    ("song", "segments"): "How many lyric sections the model sings in one run. More sections make a longer song but take "
                          "longer and may lose quality.",
    # pause_s
    ("subtitles", "pause_s"): "A silence in the speech longer than this starts a new subtitle. Lower values make more, "
                              "shorter subtitles that follow the speech closely.",
    ("tts", "pause_s"): "Silence inserted between the parts of a long text, in seconds. Raise it for a calmer delivery, "
                        "lower it for a brisker one.",
    # texture-related
    ("shape", "texture"): "Whether and how well the model is painted. 'No' gives just the grey shape and is cheapest; "
                          "Standard and HD add color, with HD sharper but pricier and slower.",
    ("shape", "textured"): "Also paints the 3D shape with color and surface detail in the same job. Turn it off if you only "
                           "want the grey shape or want to texture it separately later. Texturing adds time and, for cloud "
                           "models, cost.",
    ("texture", "textured"): "Whether the service paints the model with color in a second, refining pass. Turn it off to keep "
                             "the model as it is. Painting adds time and cost.",
    ("text3d", "textured"): "Also paints the 3D shape with color in the same job. Turn it off if you only need the grey "
                            "shape. Texturing adds time and cost.",
    # retopo: the texture size of the baked maps
    ("retopo", "tex_size"): "The size of the baked texture pictures (colour, normal, AO). Bigger keeps more fine detail but makes "
                            "the file and the GPU memory larger and the bake slower (4096 takes about three times longer than "
                            "2048). Match it to how close the camera gets: 1024 for mobile, 2048 for PC, 4096 for hero assets.",
    # alpha_clean
    ("rmbg", "alpha_clean"): "Makes nearly invisible pixels at the cut-out edge fully transparent, which removes the faint "
                             "haze that background removal can leave. Raise it if you still see a ghost halo.",
    # quality
    ("t2i", "quality"): "Trades price and speed for detail. Low is quick and cheap, high gives the finest detail and text "
                        "but takes longer and costs more. Use low for tests.",
    ("edit", "quality"): "Trades price and speed for detail. Low is quick and cheap, high keeps the most detail but takes "
                         "longer and costs more.",
}


def param_info(key: str, task: str | None = None) -> str:
    """Explanation for a parameter key: the task-specific text first, then the generic one, else an empty string."""
    if task:
        t = PARAM_INFO_BY_TASK.get((task, key))
        if t:
            return t
    return PARAM_INFO.get(key, "")
