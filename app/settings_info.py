"""Plain-English explanations for the Settings page (the `info` field of every setting, group and environment entry).

Written for people who have never configured a local AI studio: what the setting does in everyday terms, when to change it,
what happens if it is too high or too low, and a concrete recommendation. `settings_schema` attaches these texts; the page
shows them in a balloon behind the (i) icon. The short one-line `help` of each field is a different text and stays as it is.
Keep every entry 2-4 short sentences (tests/test_settings.py checks 40-600 characters and that it differs from `help`).
"""
from __future__ import annotations

GROUP_INFO: dict[str, str] = {
    "general": "The settings most people touch first. They say what kind of work you do (so the studio suggests the right "
               "sizes and formats) and whether you only want models you may use in a commercial product. "
               "Everything here can also be set per project.",
    "models": "Every kind of job (draw an image, make a sound, build a 3D shape ...) can be done by several models. "
              "Here you choose which one is used when you do not pick one yourself. "
              "Leave a row on 'Registry default' and the studio picks a good installed model for you.",
    "prompts": "Many studios have a sparkle button that rewrites your idea into a better prompt, and the Assistant and writers "
               "create text for you. These settings choose which text engine does that, in which language it explains itself, "
               "and the art direction it should always follow.",
    "cloud": "Cloud models run on someone else's computers and are paid per use, so they are off until you allow them. "
             "Set a monthly budget here and the studio will refuse jobs that would go over it. "
             "Your API keys are stored on the Providers page and never shown here.",
    "memory": "Image, video and 3D models are big and need a lot of memory to run. These settings control how much the studio "
              "keeps free for itself and when it unloads models it is not using. "
              "The defaults suit most machines; change them only if jobs fail with out-of-memory errors.",
    "workers": "Each family of models runs in its own helper program (a worker) that starts when first needed. "
               "These settings decide how long helpers stay alive when idle, how cancelling works and how many jobs may run at "
               "the same time. They are system-wide because the whole studio shares one machine.",
    "llm": "The prompt LLM is a small text model that runs on this computer and rewrites your prompts when you press the sparkle "
           "button. These settings control how it behaves, how long it stays loaded and which model files it uses. "
           "You can ignore this group unless the prompt improver is slow or fails to start.",
    "env": "These are the values the studio was started with: which network address and port it listens on, where your files "
           "live and which online services it may contact. They cannot be changed here. To change one, stop the studio and "
           "start it again with a different value.",
}

# ----------------------------------------------------------------------------------------------- fields (by exact key)
FIELD_INFO: dict[str, str] = {
    "profile": "Tells the studio what you are making, for example a game, a video or a podcast. It then offers the matching "
               "picture shapes (16:9 for video, square for a podcast cover), style chips and a sensible export format. "
               "It only changes suggestions and defaults, never what the models can do, so 'General' is a safe choice if unsure.",
    "license.commercial_only": "Many free AI models are for personal or research use only and may not be used in something you "
                               "sell or put on a monetised channel. Turn this on and those models are hidden in the pickers and "
                               "refused when you start a job. Turn it on for client or commercial work; leave it off for hobby use.",
    "license.strict": "A few models are marked 'check the licence' because their terms are unclear or depend on your company size. "
                      "With this on, and 'Commercial-safe models only' on too, those models are refused as well. "
                      "It does nothing while the commercial filter is off. Use it when a lawyer or client needs you to be extra careful.",
    "export.preset": "When you download or zip assets, the studio can convert them to a ready-to-use format, for example "
                     "a -14 LUFS MP3 for online video or a 2048 px WebP for the web. Your original files are never changed. "
                     "Leave it empty to follow the use-case profile; pick a preset only if you always want one particular format.",
    "ui.show_test_models": "The studio ships a few fake 'test' models (a plain tone instead of real music, for example) that "
                           "check the plumbing without a GPU. They are hidden so they do not clutter the pickers. "
                           "Turn this on only if you are developing or debugging the studio.",
    "textgen.improve": "Chooses the engine behind the sparkle (Improve) button. 'local' is the prompt LLM on this computer: free "
                       "and private. A 'cli:' or 'api:' choice sends your prompt to that company's service, which can be "
                       "smarter but uses your subscription or costs money. Empty means automatic, which is local.",
    "textgen.write": "Chooses the engine for writing helpers such as character barks, sound-effect lists and dialogue ideas. "
                     "'local' stays on this computer. A 'cli:' or 'api:' choice sends the text to that service. "
                     "Empty means automatic: the fast Claude Code model if it is installed here, otherwise local.",
    "textgen.assistant": "Chooses the engine behind the Assistant page, the chat that plans jobs for you. Planning is harder than "
                         "rewriting a prompt, so a stronger engine helps. Local works but is weaker. "
                         "Empty means automatic: Claude Code (Sonnet) if it is installed here, otherwise local.",
    "prompt.lang": "When the studio improves a prompt, the actual image or sound prompt is written in English because most models "
                   "understand that best. This setting is the language of the short explanation, caption and lyrics that come "
                   "with it. 'Auto' follows the language you typed, which is right for almost everybody.",
    "prompt.context.image": "A line of standing instructions added every time the sparkle button improves an image prompt, so all "
                            "your pictures share one look. Write it like a short brief, for example: 'low-poly mobile game, warm "
                            "palette, no text'. Keep it short; a very long brief can crowd out your actual idea. Empty means none.",
    "prompt.context.video": "A line of standing instructions added every time the sparkle button improves a video prompt, so your "
                            "clips keep a consistent style. Example: 'warm 35 mm documentary look, handheld, natural light'. "
                            "Keep it short. Empty means none.",
    "prompt.context.sound": "A line of standing instructions added every time the sparkle button improves a sound or music prompt. "
                            "Example: 'cozy farming game, soft acoustic instruments, no vocals'. "
                            "Keep it short. Empty means none.",
    "textgen.timeout_s": "How long the studio waits for a text engine to answer one request before giving up. Online and "
                         "command-line engines can be slow when busy. Too low and good answers are cut off with a timeout error; "
                         "too high and a stuck request makes you wait. Leave it empty (180 s online, 300 s command-line) unless "
                         "you see timeouts.",
    "textgen.assistant_timeout_s": "The Assistant may think, look things up and plan in several steps, so one turn can take a while. "
                                   "This is the longest the studio lets a whole turn run. Raise it if long plans end with a "
                                   "timeout; lowering it only makes failures arrive sooner. Empty means 420 s (7 minutes).",
    "textgen.api_models": "The list of online models you have added to the text engine pickers. It is managed with a dialog, "
                          "not typed here, so you cannot break it by mistake. Use the button to add or remove models.",
    "textgen.endpoints": "Your own text servers, for example one on your network that speaks the OpenAI format. "
                         "They are added and removed in the Assistant > Text backends dialog, so use the button instead of "
                         "editing this by hand.",
    "memory.margin_gb": "Before a job starts, the studio checks that the model fits in free memory with this much to spare. If "
                        "it does not, it unloads its own idle models first. Too low and jobs can crash with out-of-memory errors; "
                        "too high and the studio unloads models it did not need to. Leave it at 6 GB, and raise it if other "
                        "programs share this machine.",
    "memory.llm_yield_gb": "The prompt LLM stays loaded between uses so Improve is quick. But a very large job (a model needing "
                           "this much memory or more) stops it first to make room. Lower the number if big jobs keep failing "
                           "while the LLM is loaded; raise it to keep the LLM alive more often. Memory is in gigabytes (GB).",
    "memory.evict_wait_s": "After the studio unloads a model, the operating system hands the memory back a little later. This is "
                           "how long it waits for that before deciding to unload another model. Too low and it may unload more "
                           "than needed; too high wastes time when the memory is really held by another program. 12 s is fine.",
    "workers.model_idle_s": "A worker keeps a model loaded after a job so the next one starts instantly. If the model is not used "
                            "for this long, the worker unloads it to free memory. Shorter frees memory sooner but makes the next "
                            "job wait for a reload; longer is faster for repeated work. Applies to workers that start after the change.",
    "workers.exit_idle_s": "When a worker has no model loaded for this long, the whole helper program quits. It frees the small "
                           "amount of memory a worker still uses; starting it again takes a few seconds. Keep it longer than "
                           "'Unload an idle model after'. Applies to workers that start after the change.",
    "workers.cancel_grace_s": "When you press Cancel, the job is asked to stop politely. If it has not stopped after this many "
                              "seconds, the studio forces the worker to quit (the model then reloads on the next job). "
                              "Too low can kill a job that was just finishing; too high makes Cancel feel stuck. 20 s is a good balance.",
    "jobs.cpu_concurrency": "Only one job at a time can use the GPU, but small tools that only use the processor (converting, "
                            "trimming, exporting) can run beside it. This is how many of those run together. More is faster for big "
                            "batches but can slow everything on a weak CPU. Needs a studio restart.",
    "jobs.cloud_concurrency": "Cloud jobs mostly wait for an answer over the internet, so several can run at once, even while a GPU "
                              "job runs. Higher finishes batches sooner but can hit a provider's rate limit and spends money "
                              "faster. Your monthly budget still applies. Needs a studio restart.",
    "jobs.history_keep": "The Queue page remembers finished jobs, and when the studio restarts it reloads this many of the newest. "
                         "Older ones disappear from the list. Your generated files and projects are not affected. Lower it if "
                         "the studio is slow to start, raise it if you want a longer history. Needs a studio restart.",
    "llm.temperature": "Controls how adventurous the prompt rewrites are. 0 gives the safest, most repeatable wording; higher gives "
                       "more surprising ideas but more odd results (above about 1.2 the text can get messy). 0.5 to 0.9 works "
                       "well. If an answer fails the studio's checks it retries once with a slightly higher value.",
    "llm.kill_idle_s": "After this long without any Improve request the studio shuts the prompt LLM down completely, freeing all "
                       "its memory. The next use starts it again, which takes a while. Raise it if you improve prompts in "
                       "bursts hours apart, lower it if you need the memory for other work. 1800 s is 30 minutes.",
    "llm.start_timeout_s": "How long the studio waits for the prompt LLM to finish starting. Big models on a slow disk can take "
                           "minutes the first time. Too low and a model that would have started is reported as failed; too high "
                           "and you wait long for a broken setup. Raise it if the first Improve of the day times out.",
    "llm.sleep_idle_s": "After this long unused, the LLM drops its model from memory but the program stays running, so waking it "
                        "is faster than a full start. 0 means never sleep. This is gentler than 'Stop after idle'. "
                        "Takes effect the next time the prompt LLM starts.",
    "llm.primary.name": "A friendly label shown in the studio so you know which prompt LLM is running. It is only a name: "
                        "changing it does not change which model file is loaded. Takes effect the next time the prompt LLM starts.",
    "llm.primary.model": "The model file (a .gguf) used for prompt improvement when there is enough free memory. Write it as "
                         "lmstudio:publisher/repo/file.gguf, hf:org/repo/file.gguf or a full path. If the file is missing, the "
                         "studio shows 'file missing' and falls back to the smaller model. Takes effect on the next LLM start.",
    "llm.primary.mmproj": "An optional second file that lets the model look at pictures, so Improve can describe an image you "
                          "attach. It must belong to the same model. Leave it empty for a text-only model; the model then simply "
                          "ignores attached images. Takes effect on the next LLM start.",
    "llm.primary.memory_gb": "How much memory you expect this model to take once loaded. It is not a limit: the studio uses it to "
                             "plan which models to unload to make room. A good guess is the model file size plus a few GB. Too low "
                             "and the plan can be wrong; too high unloads things unnecessarily.",
    "llm.primary.min_free_gb": "The main prompt model is only started when at least this much memory is free; otherwise the studio "
                               "uses the smaller fallback model. Raise it so the big model does not squeeze out image or video "
                               "jobs; lower it if the studio uses the small model when you wanted the big one.",
    "llm.fallback.name": "A friendly label for the backup prompt LLM. It is only a name: changing it does not change which model "
                         "file is loaded. Takes effect the next time the prompt LLM starts.",
    "llm.fallback.model": "The backup model file (a .gguf), used when memory is too tight for the main one. Write it as "
                          "lmstudio:publisher/repo/file.gguf, hf:org/repo/file.gguf or a full path. A small file is best here, so "
                          "Improve keeps working in tight situations. Takes effect on the next LLM start.",
    "llm.fallback.mmproj": "An optional file that lets the backup model look at pictures. It must belong to the same model. "
                           "Leave it empty for a text-only model; attached images are then ignored. "
                           "Takes effect on the next LLM start.",
    "llm.fallback.memory_gb": "How much memory you expect the backup model to take once loaded. It is only used to plan what to "
                              "unload, not as a limit. A good guess is the model file size plus a little extra for working space.",
    "llm.fallback.min_free_gb": "The backup model is only started when at least this much memory is free. If even that is not "
                                "available, Improve reports that memory is short and asks you to unload something on the Queue & "
                                "System page. Keep it a little above the model's memory need.",
    "llm.server": "The program (llama.cpp's llama-server) that actually runs the prompt LLM. The studio starts it for you. It is "
                  "locked on this page because it is a program the studio runs on your computer, so only editing settings.json "
                  "can change it.",
    "llm.host": "The network address the prompt LLM listens on. 127.0.0.1 means only this computer can talk to it, which is the "
                "safe choice. It is locked here; change it only in settings.json and only if you know why.",
    "llm.port": "The network port the prompt LLM listens on (7880 by default). Change it in settings.json only if another "
                "program already uses that port. It is locked here because other parts of the studio depend on it.",
    "llm.args": "The extra options passed to the prompt LLM program, for example how much of it runs on the GPU (-ngl), how much "
                "text it can remember at once (-c, 16384 tokens) and how many requests it serves together (-np). Wrong values "
                "can stop it from starting, so they are locked here; edit settings.json if you need to.",
}

# ----------------------------------------------------------------------------------------------- templated fields
CLI_MODEL_INFO = ("The model name the studio asks {label} to use, for example one of the suggestions in the list. Cheaper, faster "
                  "models answer sooner; bigger ones write better. Leave it empty to use {label}'s own default. "
                  "Only used when a text engine is set to this tool.")
CLI_BINARY_INFO = ("Where the {label} program is installed on this computer. The studio finds it automatically; you only set it "
                   "if it lives somewhere unusual. It is locked here because the studio runs this file, so only editing "
                   "settings.json can change it.")
TASK_INFO = ("Chooses which model is used for '{label}' in the {studio} studio when you have not picked one yourself in that "
             "studio. Leave 'Registry default' and the studio uses the best installed model for the job, preferring local "
             "ones. Pick a model here if you always want the same one, and set it per project if projects differ.")

# ----------------------------------------------------------------------------------------------- environment entries
ENV_INFO: dict[str, str] = {
    "STUDIO_HOST": "The network address the studio listens on. 127.0.0.1 means only this computer can open it; 0.0.0.0 means "
                   "anyone on your network can. There is no login, so only use 0.0.0.0 on a network you trust.",
    "STUDIO_PORT": "The port number in the address you type in the browser, for example :7860. Start another copy of the "
                   "studio on a different port if two must run side by side.",
    "STUDIO_DATA": "The folder where your projects, assets, uploads and exports are stored. Back this folder up to keep your "
                   "work. Moving the studio elsewhere means pointing this at the old folder.",
    "STUDIO_LOGS": "The folder where the studio writes its logs. Look here first when a job fails: each helper program has "
                   "its own log file, and the main log is studio.log.",
    "STUDIO_SETTINGS": "The file that stores the changes you make on this page for the whole studio. Deleting it brings every "
                       "setting back to its default. Project overrides are stored with each project instead.",
    "STUDIO_MODELS": "The file that lists every model the studio knows about, with its licence and memory need. Extra lists in "
                     "the models.d folder are merged into it. Press Re-check on the Models page after editing.",
    "HF_HOME": "The base folder where Hugging Face (the main site for open AI models) keeps its downloads and settings on this "
               "computer. Model weights usually live inside it.",
    "HF_HUB_CACHE": "The folder where the downloaded model files themselves are kept. These files are large, often tens of "
                    "gigabytes, so this is the folder to look at if the disk is filling up.",
    "HF_HUB_OFFLINE": "When this is 1, the studio never goes online to fetch a model: every model must already be on disk. "
                      "It is the normal setting and keeps the studio private. A model that was not downloaded shows as "
                      "'missing weights'.",
    "HF_HUB_DISABLE_XET": "Turns off Hugging Face's newer download method (Xet) and uses the classic one. It only matters "
                          "when models are downloaded, and the start script sets it for you. You can ignore it.",
    "LMSTUDIO_MODELS": "The folder where LM Studio keeps its downloaded text models. The prompt LLM can load a model file from "
                       "here when its path starts with 'lmstudio:', so you do not need a second copy.",
    "LLAMA_SERVER": "The path of the llama-server program that runs the prompt LLM. If it is empty the studio searches your "
                    "usual folders. If Improve says the prompt LLM is not installed, check this path first.",
    "STUDIO_ONLINE": "When set, the model helper programs are allowed to go online, for example to download a missing model. "
                     "Unset (the normal case), they stay fully offline and only use files already on disk. "
                     "Set it only while you install a new model.",
    "COMFYUI_URL": "The address of a ComfyUI server (a separate image tool) that some image models use. The default is "
                   "http://127.0.0.1:8188 on this computer. Models that need it show as unavailable while it is not running.",
    "STUDIO_NO_COMFYUI": "When set, the studio never tries to contact ComfyUI, and the models that need it stay unavailable. "
                         "Use it if you do not run ComfyUI and want to skip that check.",
    "STUDIO_CLOUD_MOCK": "A switch for developers. When on, cloud model calls go to a fake local server instead of the real "
                         "service, so nothing is sent online and nothing is charged. Normally off.",
    "STUDIO_MCP_API": "The address the built-in tool server (MCP, used by Claude Code and other assistants) calls back to "
                      "reach the studio. It defaults to this studio's own port, so you rarely need to set it.",
}
