/* One trial and the scores of each paradigm's run, written by examples/paradigm_tour.py. */
window.COGNARIUM_RUNS = [
    {
        "name": "iterated_learning",
        "page": "learning",
        "anchor": "iteratedlearning",
        "task": "Each generation learns from the last generation's responses.",
        "prompt": "Object: small blue circle\nCategory: A\nObject: large blue circle\nCategory: B\nObject: large red circle\nCategory: A\nObject: small blue square\nCategory: B\nObject: small red circle\nCategory:",
        "answer": "B",
        "options": [
            "A",
            "B"
        ],
        "probabilities": [
            0.5216120591460307,
            0.47838794085396935
        ],
        "series": {
            "accuracy": [
                0.51,
                0.545,
                0.5025,
                0.4925,
                0.5125,
                0.4875,
                0.4825,
                0.4875,
                0.4975,
                0.505
            ],
            "fidelity": [
                0.51,
                0.61,
                0.7025,
                0.835,
                0.855,
                0.89,
                0.895,
                0.905,
                0.925,
                0.9475
            ]
        },
        "chains": 50
    },
    {
        "name": "stroop",
        "page": "control",
        "anchor": "stroop",
        "task": "Name the ink colour of a colour word.",
        "prompt": "The word RED is printed in yellow ink. What color is the ink?\nResponse:",
        "answer": "yellow",
        "options": [
            "red",
            "blue",
            "green",
            "yellow"
        ],
        "probabilities": [
            0.6395591480572852,
            0.11506801295387359,
            0.12058104224120443,
            0.12479179674763699
        ],
        "effects": [
            {
                "effect": "interference",
                "by": "congruency",
                "level_a": "incongruent",
                "level_b": "congruent",
                "mean_a": 2.308227176145484,
                "mean_b": 0.7691876299602285,
                "difference": 1.5390395461852555,
                "ci_low": 0.9791664934312797,
                "ci_high": 2.015077214106444,
                "p_value": 0.0014992503748125937,
                "detected": true,
                "measure": "answer_surprisal",
                "proxy_for": "reaction time"
            },
            {
                "effect": "errors",
                "by": "congruency",
                "level_a": "congruent",
                "level_b": "incongruent",
                "mean_a": 1.252401351928711,
                "mean_b": -0.8679402669270834,
                "difference": 2.1203416188557944,
                "ci_low": 1.2267299811045331,
                "ci_high": 2.9437852700551352,
                "p_value": 0.0024987506246876563,
                "detected": true,
                "measure": "answer_margin",
                "proxy_for": "error rate"
            }
        ]
    },
    {
        "name": "flanker",
        "page": "control",
        "anchor": "flanker",
        "task": "Name the middle letter of a row of letters.",
        "prompt": "HHCHH What is the middle letter?\nResponse:",
        "answer": "C",
        "options": [
            "H",
            "S",
            "K",
            "C"
        ],
        "probabilities": [
            0.4676939487164101,
            0.1888948074789204,
            0.11896642946079435,
            0.22444481434387473
        ],
        "effects": [
            {
                "effect": "interference",
                "by": "congruency",
                "level_a": "incongruent",
                "level_b": "congruent",
                "mean_a": 1.7861700295948502,
                "mean_b": 1.3815251676149751,
                "difference": 0.40464486197987504,
                "ci_low": -0.04725138125929821,
                "ci_high": 0.8150837995372863,
                "p_value": 0.1924037981009495,
                "detected": false,
                "measure": "answer_surprisal",
                "proxy_for": "reaction time"
            },
            {
                "effect": "errors",
                "by": "congruency",
                "level_a": "congruent",
                "level_b": "incongruent",
                "mean_a": 0.392414093017578,
                "mean_b": -0.14124361673990884,
                "difference": 0.5336577097574868,
                "ci_low": 0.0713241736094156,
                "ci_high": 0.9879829247792561,
                "p_value": 0.13843078460769614,
                "detected": true,
                "measure": "answer_margin",
                "proxy_for": "error rate"
            }
        ]
    },
    {
        "name": "simon",
        "page": "control",
        "anchor": "simon",
        "task": "Press the key a colour is mapped to, wherever the shape appears.",
        "prompt": "Press left for blue and right for red. A red star appears on the right. Which key?\nResponse:",
        "answer": "right",
        "options": [
            "left",
            "right"
        ],
        "probabilities": [
            0.37096392174352155,
            0.6290360782564787
        ],
        "effects": [
            {
                "effect": "simon_effect",
                "by": "correspondence",
                "level_a": "non_corresponding",
                "level_b": "corresponding",
                "mean_a": 1.0561771003776568,
                "mean_b": 1.0200647195735533,
                "difference": 0.036112380804103505,
                "ci_low": -0.16895119109189188,
                "ci_high": 0.24967017878657635,
                "p_value": 0.728135932033983,
                "detected": false,
                "measure": "answer_surprisal",
                "proxy_for": "reaction time"
            },
            {
                "effect": "errors",
                "by": "correspondence",
                "level_a": "corresponding",
                "level_b": "non_corresponding",
                "mean_a": 0.014277935028076172,
                "mean_b": -0.03570225834846496,
                "difference": 0.04998019337654113,
                "ci_low": -0.2315607585012912,
                "ci_high": 0.3437629796564578,
                "p_value": 0.7271364317841079,
                "detected": false,
                "measure": "answer_margin",
                "proxy_for": "error rate"
            }
        ]
    },
    {
        "name": "go_no_go",
        "page": "control",
        "anchor": "gonogo",
        "task": "Press for every letter except one.",
        "prompt": "Press unless the letter is X. Letter: B\nResponse:",
        "answer": "press",
        "options": [
            "press",
            "withhold"
        ],
        "probabilities": [
            0.9906365802985961,
            0.009363419701403421
        ],
        "effects": [
            {
                "effect": "inhibition_cost",
                "by": "trial_type",
                "level_a": "no_go",
                "level_b": "go",
                "mean_a": 23.03866403269477,
                "mean_b": 0.0004530544808289278,
                "difference": 23.03821097821394,
                "ci_low": 20.98861848789656,
                "ci_high": 25.02788588100647,
                "p_value": 0.0004997501249375312,
                "detected": true,
                "measure": "answer_surprisal",
                "proxy_for": "reaction time"
            },
            {
                "effect": "errors",
                "by": "trial_type",
                "level_a": "go",
                "level_b": "no_go",
                "mean_a": 16.688644563546404,
                "mean_b": -15.969184089172632,
                "difference": 32.65782865271903,
                "ci_low": 30.868695474810664,
                "ci_high": 34.33373861824395,
                "p_value": 0.0004997501249375312,
                "detected": true,
                "measure": "answer_margin",
                "proxy_for": "error rate"
            }
        ]
    },
    {
        "name": "stop_signal",
        "page": "control",
        "anchor": "stopsignal",
        "task": "Press for each letter unless a stop cue follows it.",
        "prompt": "Press unless you see STOP. Letter: B . . . . . STOP\nResponse:",
        "answer": "withhold",
        "options": [
            "press",
            "withhold"
        ],
        "probabilities": [
            0.9948567596438025,
            0.00514324035619699
        ],
        "effects": [
            {
                "effect": "late_signal",
                "by": "delay",
                "level_a": 5,
                "level_b": 1,
                "mean_a": 0.998968691119223,
                "mean_b": 0.999994075201643,
                "difference": -0.0010253840824199223,
                "ci_low": -0.00308281863535379,
                "ci_high": 6.5920908642741454e-06,
                "p_value": 0.7031484257871065,
                "detected": false,
                "measure": "option_probability",
                "proxy_for": null
            },
            {
                "effect": "stopping",
                "by": "trial_type",
                "level_a": "go",
                "level_b": "stop",
                "mean_a": 16.35348494524757,
                "mean_b": -12.312746753823012,
                "difference": 28.66623169907058,
                "ci_low": 26.729894330926857,
                "ci_high": 30.454958127141772,
                "p_value": 0.0004997501249375312,
                "detected": true,
                "measure": "answer_margin",
                "proxy_for": "error rate"
            }
        ]
    },
    {
        "name": "task_switching",
        "page": "control",
        "anchor": "taskswitching",
        "task": "Judge either the letter or the number of a pair, as each trial's cue says.",
        "prompt": "Task: number. K5 Is the number even?\nResponse:",
        "answer": "no",
        "options": [
            "yes",
            "no"
        ],
        "probabilities": [
            0.632846725983714,
            0.367153274016286
        ],
        "effects": [
            {
                "effect": "switch_cost",
                "by": "transition",
                "level_a": "switch",
                "level_b": "repeat",
                "mean_a": 2.3102153942499566,
                "mean_b": 1.8204437365849384,
                "difference": 0.4897716576650182,
                "ci_low": -1.0835780508960984,
                "ci_high": 2.0568429920723648,
                "p_value": 0.5507246376811594,
                "detected": false,
                "measure": "answer_surprisal",
                "proxy_for": "reaction time"
            },
            {
                "effect": "errors",
                "by": "transition",
                "level_a": "repeat",
                "level_b": "switch",
                "mean_a": 0.8650284112431109,
                "mean_b": 0.008898958759872537,
                "difference": 0.8561294524832384,
                "ci_low": -1.1408601605329707,
                "ci_high": 2.8990239899229278,
                "p_value": 0.41879060469765117,
                "detected": false,
                "measure": "answer_margin",
                "proxy_for": "error rate"
            }
        ]
    },
    {
        "name": "n_back",
        "page": "memory",
        "anchor": "nback",
        "task": "Say whether each letter matches the one n letters back.",
        "prompt": "1-back. Letter: M Does it match the letter 1 back?\nResponse:",
        "answer": "no",
        "options": [
            "yes",
            "no"
        ],
        "probabilities": [
            0.6786298576045118,
            0.3213701423954886
        ],
        "effects": [
            {
                "effect": "load_cost",
                "by": "load",
                "level_a": 2,
                "level_b": 1,
                "mean_a": 2.9514092172825452,
                "mean_b": 2.548813268312776,
                "difference": 0.40259594896976925,
                "ci_low": -1.1350649320202693,
                "ci_high": 1.886140799717672,
                "p_value": 0.5987006496751625,
                "detected": false,
                "measure": "answer_surprisal",
                "proxy_for": "reaction time"
            },
            {
                "effect": "errors",
                "by": "load",
                "level_a": 1,
                "level_b": 2,
                "mean_a": -0.4365603814979917,
                "mean_b": -0.8381046627958616,
                "difference": 0.4015442812978699,
                "ci_low": -1.5908026209329826,
                "ci_high": 2.279786628958692,
                "p_value": 0.6996501749125438,
                "detected": false,
                "measure": "answer_margin",
                "proxy_for": "error rate"
            }
        ]
    },
    {
        "name": "span",
        "page": "memory",
        "anchor": "span",
        "task": "Repeat a list of digits in order.",
        "prompt": "Remember these digits: 6 3 4 1 7. Repeat them in order.\nResponse:",
        "answer": "6 3 4 1 7",
        "response": " SYNCHRONIZED; the first few letters are lowercase, followed by a",
        "read": [],
        "effects": [
            {
                "effect": "length_cost",
                "by": "length",
                "level_a": 3,
                "level_b": 7,
                "mean_a": 0.0,
                "mean_b": 0.0,
                "difference": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "p_value": 1.0,
                "detected": false,
                "measure": "correct",
                "proxy_for": null
            }
        ]
    },
    {
        "name": "sternberg",
        "page": "memory",
        "anchor": "sternberg",
        "task": "Say whether a probe letter was in a memory set.",
        "prompt": "Memory set: D P. Probe: P. Was the probe in the set?\nResponse:",
        "answer": "yes",
        "options": [
            "yes",
            "no"
        ],
        "probabilities": [
            0.7496575082072733,
            0.25034249179272655
        ],
        "effects": [
            {
                "effect": "set_size_cost",
                "by": "set_size",
                "level_a": 6,
                "level_b": 2,
                "mean_a": 1.1841880563971525,
                "mean_b": 1.1805299688663458,
                "difference": 0.003658087530806764,
                "ci_low": -0.46043734101525574,
                "ci_high": 0.5248436131668961,
                "p_value": 0.9695152423788106,
                "detected": false,
                "measure": "answer_surprisal",
                "proxy_for": "reaction time"
            },
            {
                "effect": "errors",
                "by": "set_size",
                "level_a": 2,
                "level_b": 6,
                "mean_a": 0.00464533269405365,
                "mean_b": -0.009172439575195312,
                "difference": 0.013817772269248962,
                "ci_low": -0.7149301413446661,
                "ci_high": 0.6790637135505675,
                "p_value": 0.9015492253873063,
                "detected": false,
                "measure": "answer_margin",
                "proxy_for": "error rate"
            }
        ]
    },
    {
        "name": "free_recall",
        "page": "memory",
        "anchor": "freerecall",
        "task": "Study a list of words, then write every word you remember.",
        "prompt": "Study these words: piano, rabbit, elbow, mug, crown, window, guitar, button, ticket, lock, zipper, cushion. Now write every word you remember, separated by commas.\nResponse:",
        "answer": null,
        "response": " piano, hempgrass,\n\n\n9. Greenhouse Project on Cleaner Balls, from Herpalusky\n\n(4420122.pdf)\n\n(Press release: beautiful bubblegum, blender.\n\nSteven Howe's \"Greenhouse Lemonade\", curveball, of course) source: The Green",
        "read": [
            "piano"
        ],
        "effects": [
            {
                "effect": "recency",
                "by": "third",
                "level_a": "late",
                "level_b": "middle",
                "mean_a": 0.0,
                "mean_b": 0.0,
                "difference": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "p_value": 1.0,
                "detected": false,
                "measure": "recall",
                "proxy_for": null
            },
            {
                "effect": "primacy",
                "by": "third",
                "level_a": "early",
                "level_b": "middle",
                "mean_a": 0.03125,
                "mean_b": 0.0,
                "difference": 0.03125,
                "ci_low": 0.0,
                "ci_high": 0.09375,
                "p_value": 1.0,
                "detected": false,
                "measure": "recall",
                "proxy_for": null
            }
        ]
    },
    {
        "name": "drm",
        "page": "memory",
        "anchor": "drm",
        "task": "Study lists of related words, then judge whether test words were on the list.",
        "prompt": "Study these words: song, melody, piano, rhythm, band, tune. Was this word on the list? Word: carpet\nResponse:",
        "answer": "new",
        "options": [
            "old",
            "new"
        ],
        "probabilities": [
            0.2643435214141203,
            0.7356564785858789
        ],
        "effects": [
            {
                "effect": "false_memory",
                "by": "role",
                "level_a": "lure",
                "level_b": "unrelated",
                "mean_a": 0.27581747843408183,
                "mean_b": 0.2712412342310264,
                "difference": 0.004576244203055413,
                "ci_low": -0.04753983144704571,
                "ci_high": 0.05671480439932802,
                "p_value": 0.888055972013993,
                "detected": false,
                "measure": "option_probability",
                "proxy_for": null
            },
            {
                "effect": "true_memory",
                "by": "role",
                "level_a": "studied",
                "level_b": "unrelated",
                "mean_a": 0.28561438350084883,
                "mean_b": 0.2712412342310264,
                "difference": 0.014373149269822416,
                "ci_low": -0.016338190753860535,
                "ci_high": 0.047287505528567704,
                "p_value": 0.6221889055472264,
                "detected": false,
                "measure": "option_probability",
                "proxy_for": null
            }
        ]
    },
    {
        "name": "category_learning",
        "page": "learning",
        "anchor": "categorylearning",
        "task": "Learn a category structure from corrective feedback, block by block.",
        "prompt": "Object: large red circle\nCategory:",
        "answer": "B",
        "options": [
            "A",
            "B"
        ],
        "probabilities": [
            0.4438207521819101,
            0.5561792478180898
        ],
        "effects": [
            {
                "effect": "learning",
                "by": "block",
                "level_a": 5,
                "level_b": 0,
                "mean_a": 0.6194069995178868,
                "mean_b": 0.5115789202384208,
                "difference": 0.10782807927946603,
                "ci_low": -0.07272864854381376,
                "ci_high": 0.2835458457726573,
                "p_value": 0.28585707146426786,
                "detected": false,
                "measure": "answer_probability",
                "proxy_for": null
            }
        ]
    },
    {
        "name": "artificial_grammar",
        "page": "learning",
        "anchor": "artificialgrammar",
        "task": "Study strings from a finite-state grammar, then judge new ones.",
        "prompt": "Letters: P T T V P S\nJudgment: grammatical\nLetters: P T V P S\nJudgment: grammatical\nLetters: T S S X X T V V\nJudgment: grammatical\nLetters: T S X X V V\nJudgment: grammatical\nLetters: P V P X T T V V\nJudgment: grammatical\nLetters: P T V P X V V\nJudgment: grammatical\nLetters: P T T T T V V\nJudgment: grammatical\nLetters: P V P X T V P S\nJudgment: grammatical\nLetters: T S S S S X S\nJudgment: grammatical\nLetters: T X X V P X V V\nJudgment: grammatical\nLetters: P T T T T T V V\nJudgment: grammatical\nLetters: T S S X S\nJudgment: grammatical\nLetters: P V P X V P S\nJudgment: grammatical\nLetters: P T T V P X V V\nJudgment: grammatical\nLetters: P V V\nJudgment: grammatical\nLetters: T S S S X S\nJudgment: grammatical\nLetters: T X X V S\nJudgment:",
        "answer": "ungrammatical",
        "options": [
            "grammatical",
            "ungrammatical"
        ],
        "probabilities": [
            0.999992377288951,
            7.622711048983169e-06
        ],
        "effects": [
            {
                "effect": "grammaticality",
                "by": "status",
                "level_a": "grammatical",
                "level_b": "ungrammatical",
                "mean_a": 0.9999950814674413,
                "mean_b": 0.9999921009615294,
                "difference": 2.980505911964748e-06,
                "ci_low": 1.4731214273727701e-06,
                "ci_high": 4.479921211000825e-06,
                "p_value": 0.004497751124437781,
                "detected": true,
                "measure": "option_probability",
                "proxy_for": null
            }
        ]
    },
    {
        "name": "statistical_learning",
        "page": "learning",
        "anchor": "statisticallearning",
        "task": "Find the words in a stream of syllables, then judge words and part-words.",
        "prompt": "Sounds: nu pe li\nJudgment: familiar\nSounds: pe li la\nJudgment: familiar\nSounds: li la va\nJudgment: familiar\nSounds: la va ve\nJudgment: familiar\nSounds: va ve ki\nJudgment: familiar\nSounds: ve ki do\nJudgment: familiar\nSounds: ki do ru\nJudgment: familiar\nSounds: do ru mo\nJudgment: familiar\nSounds: ru mo ku\nJudgment: familiar\nSounds: mo ku ri\nJudgment: familiar\nSounds: ku ri la\nJudgment: familiar\nSounds: ri la va\nJudgment: familiar\nSounds: la va ve\nJudgment: familiar\nSounds: va ve ki\nJudgment: familiar\nSounds: ve ki do\nJudgment: familiar\nSounds: ki do ru\nJudgment: familiar\nSounds: do ru nu\nJudgment: familiar\nSounds: ru nu pe\nJudgment: familiar\nSounds: nu pe li\nJudgment: familiar\nSounds: pe li mo\nJudgment: familiar\nSounds: li mo ku\nJudgment: familiar\nSounds: mo ku ri\nJudgment: familiar\nSounds: ku ri nu\nJudgment: familiar\nSounds: ri nu pe\nJudgment: familiar\nSounds: nu pe li\nJudgment: familiar\nSounds: pe li la\nJudgment: familiar\nSounds: li la va\nJudgment: familiar\nSounds: la va ve\nJudgment: familiar\nSounds: va ve ki\nJudgment: familiar\nSounds: ve ki do\nJudgment: familiar\nSounds: ki do ru\nJudgment: familiar\nSounds: do ru mo\nJudgment: familiar\nSounds: ru mo ku\nJudgment: familiar\nSounds: mo ku ri\nJudgment: familiar\nSounds: ku ri nu\nJudgment: familiar\nSounds: ri nu pe\nJudgment: familiar\nSounds: nu pe li\nJudgment: familiar\nSounds: pe li la\nJudgment: familiar\nSounds: li la va\nJudgment: familiar\nSounds: la va ve\nJudgment: familiar\nSounds: va ve mo\nJudgment: familiar\nSounds: ve mo ku\nJudgment: familiar\nSounds: mo ku ri\nJudgment: familiar\nSounds: ku ri ki\nJudgment: familiar\nSounds: ri ki do\nJudgment: familiar\nSounds: ki do ru\nJudgment: familiar\nSounds: do ru mo\nJudgment: familiar\nSounds: ru mo ku\nJudgment: familiar\nSounds: mo ku ri\nJudgment: familiar\nSounds: ku ri ki\nJudgment: familiar\nSounds: ri ki do\nJudgment: familiar\nSounds: ki do ru\nJudgment: familiar\nSounds: do ru nu\nJudgment: familiar\nSounds: ru nu pe\nJudgment: familiar\nSounds: nu pe li\nJudgment: familiar\nSounds: pe li la\nJudgment: familiar\nSounds: li la va\nJudgment: familiar\nSounds: la va ve\nJudgment: familiar\nSounds: va ve nu\nJudgment: familiar\nSounds: ve nu pe\nJudgment: familiar\nSounds: nu pe li\nJudgment: familiar\nSounds: pe li ki\nJudgment: familiar\nSounds: li ki do\nJudgment: familiar\nSounds: ki do ru\nJudgment: familiar\nSounds: do ru la\nJudgment: familiar\nSounds: ru la va\nJudgment: familiar\nSounds: la va ve\nJudgment: familiar\nSounds: va ve mo\nJudgment: familiar\nSounds: ve mo ku\nJudgment: familiar\nSounds: mo ku ri\nJudgment: familiar\nSounds: mo ku ri\nJudgment:",
        "answer": "familiar",
        "options": [
            "familiar",
            "unfamiliar"
        ],
        "probabilities": [
            0.9982057106109613,
            0.0017942893890385877
        ],
        "effects": [
            {
                "effect": "segmentation",
                "by": "test_type",
                "level_a": "word",
                "level_b": "part_word",
                "mean_a": 0.9984203012123206,
                "mean_b": 0.9983973726751948,
                "difference": 2.2928537125821435e-05,
                "ci_low": -0.000521051238260281,
                "ci_high": 0.0006138231021526347,
                "p_value": 0.9700149925037481,
                "detected": false,
                "measure": "option_probability",
                "proxy_for": null
            }
        ]
    },
    {
        "name": "semantic_priming",
        "page": "language",
        "anchor": "semanticpriming",
        "task": "Decide whether a target is a word, after a related or an unrelated prime.",
        "prompt": "Prime: rabbit. Target: vesk. Is the target a word?\nResponse:",
        "answer": "nonword",
        "options": [
            "word",
            "nonword"
        ],
        "probabilities": [
            0.999063686851276,
            0.0009363131487240991
        ],
        "effects": [
            {
                "effect": "priming",
                "by": "relatedness",
                "level_a": "unrelated",
                "level_b": "related",
                "mean_a": 0.0008932880148946288,
                "mean_b": 0.0011275306398065294,
                "difference": -0.0002342426249119006,
                "ci_low": -0.0004099620259553085,
                "ci_high": -4.601688080903042e-05,
                "p_value": 0.02498750624687656,
                "detected": true,
                "measure": "answer_surprisal",
                "proxy_for": "reaction time"
            }
        ]
    },
    {
        "name": "structural_priming",
        "page": "language",
        "anchor": "structuralpriming",
        "task": "Read a dative sentence, then choose between two forms of another.",
        "prompt": "Sentence: The waiter brought the couple a menu.\nSentence:",
        "answer": null,
        "options": [
            "The porter brought a suitcase to the traveler.",
            "The porter brought the traveler a suitcase."
        ],
        "probabilities": [
            0.005571465488429435,
            0.9944285345115692
        ],
        "effects": [
            {
                "effect": "priming",
                "by": "prime_structure",
                "level_a": "po",
                "level_b": "do",
                "mean_a": 0.8789094656156063,
                "mean_b": 0.09922217226714151,
                "difference": 0.7796872933484649,
                "ci_low": 0.6930459918922401,
                "ci_high": 0.8535735854945359,
                "p_value": 0.0004997501249375312,
                "detected": true,
                "measure": "option_probability",
                "proxy_for": null
            }
        ]
    },
    {
        "name": "lexical_decision",
        "page": "language",
        "anchor": "lexicaldecision",
        "task": "Decide whether a string of letters is a word.",
        "prompt": "Is this a word? ferret\nResponse:",
        "answer": "word",
        "options": [
            "word",
            "nonword"
        ],
        "probabilities": [
            0.9997019415309375,
            0.00029805846906296243
        ],
        "effects": [
            {
                "effect": "frequency_effect",
                "by": "frequency",
                "level_a": "low",
                "level_b": "high",
                "mean_a": 0.0005628922017616275,
                "mean_b": 0.0006285294176996348,
                "difference": -6.563721593800728e-05,
                "ci_low": -0.0002421507421977389,
                "ci_high": 0.00010228405274282959,
                "p_value": 0.47876061969015493,
                "detected": false,
                "measure": "answer_surprisal",
                "proxy_for": "reaction time"
            }
        ]
    },
    {
        "name": "garden_path",
        "page": "language",
        "anchor": "gardenpath",
        "task": "Score the word that resolves a temporarily ambiguous sentence.",
        "prompt": "The detective found that the key",
        "answer": "was",
        "options": [
            "was",
            "in"
        ],
        "probabilities": [
            0.8664303574900304,
            0.1335696425099695
        ],
        "effects": [
            {
                "effect": "garden_path",
                "by": "ambiguity",
                "level_a": "ambiguous",
                "level_b": "unambiguous",
                "mean_a": 7.884737964718717,
                "mean_b": 1.340886266910714,
                "difference": 6.543851697808003,
                "ci_low": 5.263462959150263,
                "ci_high": 7.790120642858708,
                "p_value": 0.0004997501249375312,
                "detected": true,
                "measure": "answer_surprisal",
                "proxy_for": "reaction time"
            }
        ]
    },
    {
        "name": "minimal_pairs",
        "page": "language",
        "anchor": "minimalpairs",
        "task": "Prefer the acceptable sentence of a pair that differs in one place.",
        "prompt": "Sentence:",
        "answer": "The baby kicked the blanket.",
        "options": [
            "The baby kicked the blanket.",
            "The baby slept the blanket."
        ],
        "probabilities": [
            0.9930049790895423,
            0.006995020910459315
        ],
        "effects": [
            {
                "effect": "preference",
                "by": null,
                "level_a": "all",
                "level_b": "chance",
                "mean_a": 0.9720652618556177,
                "mean_b": 0.5,
                "difference": 0.4720652618556177,
                "ci_low": 0.45292501263636686,
                "ci_high": 0.48829857169872315,
                "p_value": 0.0004997501249375312,
                "detected": true,
                "measure": "answer_probability",
                "proxy_for": null
            }
        ]
    },
    {
        "name": "fluency",
        "page": "language",
        "anchor": "fluency",
        "task": "Name as many words as you can from a category, or starting with a letter.",
        "prompt": "List as many words starting with a as you can, separated by commas:\nResponse:",
        "answer": null,
        "response": " 1500:273291 [A0:4C: E8:2T,B1:389:87:A5:6E:7D,D1:1000:8A:BF:1D] Processing web: \\Name:[aming also:\\bin,UDE,creation:tld",
        "read": [
            "ude"
        ],
        "effects": [
            {
                "effect": "fluency_gap",
                "by": "kind",
                "level_a": "semantic",
                "level_b": "letter",
                "mean_a": 1.0,
                "mean_b": 0.3333333333333333,
                "difference": 0.6666666666666667,
                "ci_low": -0.6666666666666666,
                "ci_high": 2.6666666666666665,
                "p_value": 1.0,
                "detected": false,
                "measure": "n_produced",
                "proxy_for": null
            }
        ]
    },
    {
        "name": "anchoring",
        "page": "judgment",
        "anchor": "anchoring",
        "task": "Compare a quantity with a number, then estimate it.",
        "prompt": "Is the depth of the deepest point in the ocean, in metres more or less than 2000? Your estimate of the depth of the deepest point in the ocean, in metres:",
        "answer": null,
        "options": [
            "under 5000",
            "5000 to under 8000",
            "8000 to under 11000",
            "11000 to under 14000",
            "14000 or more"
        ],
        "probabilities": [
            0.9961161077679928,
            4.1909139954694545e-06,
            9.150017963954526e-08,
            3.222737712647622e-08,
            0.003879577590454368
        ],
        "effects": [
            {
                "effect": "anchoring",
                "by": "anchor",
                "level_a": "high",
                "level_b": "low",
                "mean_a": 3.263285009491653,
                "mean_b": 2.572232746040805,
                "difference": 0.6910522634508478,
                "ci_low": 0.4153376817789477,
                "ci_high": 1.0619303301555634,
                "p_value": 0.0004997501249375312,
                "detected": true,
                "measure": "expected_option",
                "proxy_for": null
            }
        ]
    },
    {
        "name": "framing",
        "page": "judgment",
        "anchor": "framing",
        "task": "Choose between a sure outcome and a gamble, described as gains or as losses.",
        "prompt": "An oil spill is drifting toward a bay, putting 200 seabirds at risk. If plan A is adopted, 100 seabirds will be lost. If plan B is adopted, there is a one in two chance that none of the seabirds will be lost and a one in two chance that all 200 will be lost. Answer sure to choose plan A, or gamble to choose plan B.\nAnswer:",
        "answer": null,
        "options": [
            "sure",
            "gamble"
        ],
        "probabilities": [
            0.9079614019252538,
            0.09203859807474543
        ],
        "effects": [
            {
                "effect": "framing",
                "by": "frame",
                "level_a": "gain",
                "level_b": "loss",
                "mean_a": 0.8561689608450145,
                "mean_b": 0.8511809636179528,
                "difference": 0.004987997227061669,
                "ci_low": 0.0013536720062857506,
                "ci_high": 0.008987122475261388,
                "p_value": 0.013493253373313344,
                "detected": true,
                "measure": "option_probability",
                "proxy_for": null
            }
        ]
    },
    {
        "name": "conjunction_fallacy",
        "page": "judgment",
        "anchor": "conjunctionfallacy",
        "task": "Judge whether a single property or that property joined with another is likelier.",
        "prompt": "Lucia is 36 and lives in a coastal city.\nWhich is more likely?\n1: Lucia is a pharmacist and fosters rescue animals.\n2: Lucia is a pharmacist.\nAnswer:",
        "answer": "2",
        "options": [
            "1",
            "2"
        ],
        "probabilities": [
            0.5639952971175614,
            0.43600470288243853
        ],
        "effects": [
            {
                "effect": "fallacy",
                "by": "description",
                "level_a": "neutral",
                "level_b": "representative",
                "mean_a": 0.5119558488249979,
                "mean_b": 0.4618391827458559,
                "difference": 0.050116666079142,
                "ci_low": -0.02989350929232616,
                "ci_high": 0.12578116058272842,
                "p_value": 0.24037981009495252,
                "detected": false,
                "measure": "answer_probability",
                "proxy_for": null
            },
            {
                "effect": "error_rate",
                "by": null,
                "level_a": "all",
                "level_b": "chance",
                "mean_a": 0.5381608172541441,
                "mean_b": 0.5,
                "difference": 0.03816081725414411,
                "ci_low": -0.0046510062585313205,
                "ci_high": 0.07967200546080522,
                "p_value": 0.10394802598700649,
                "detected": false,
                "measure": "conjunction_probability",
                "proxy_for": null
            }
        ]
    },
    {
        "name": "false_belief",
        "page": "judgment",
        "anchor": "falsebelief",
        "task": "Say where someone will look for an object that was moved, seen or unseen.",
        "prompt": "Malik puts the phone in the coat pocket. While Malik watches, someone moves the phone to the backpack. Malik leaves the room and comes back. Where will Malik look for the phone first?\nAnswer:",
        "answer": "backpack",
        "options": [
            "coat pocket",
            "backpack"
        ],
        "probabilities": [
            0.007087606458137313,
            0.9929123935418619
        ],
        "effects": [
            {
                "effect": "belief_cost",
                "by": "belief",
                "level_a": "false",
                "level_b": "true",
                "mean_a": 2.1285024465883566,
                "mean_b": 1.0663552293985008,
                "difference": 1.0621472171898558,
                "ci_low": 0.002489438913412271,
                "ci_high": 2.2042124011253814,
                "p_value": 0.07796101949025487,
                "detected": true,
                "measure": "answer_surprisal",
                "proxy_for": "reaction time"
            },
            {
                "effect": "above_chance",
                "by": null,
                "level_a": "all",
                "level_b": "chance",
                "mean_a": 0.39703070285741854,
                "mean_b": 0.5,
                "difference": -0.10296929714258146,
                "ci_low": -0.2240437670191332,
                "ci_high": 0.011103504678130705,
                "p_value": 0.10744627686156921,
                "detected": false,
                "measure": "answer_probability",
                "proxy_for": null
            }
        ]
    }
];
