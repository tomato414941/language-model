import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from .data import file_digest

SOURCE_URL = "https://huggingface.co/datasets/{repository}/resolve/{revision}/{path}"
# JGLUE and KUCI publish their files on GitHub, where the evaluation reads them too.
GITHUB_URL = "https://raw.githubusercontent.com/{repository}/{revision}/{path}"
# BLiMP keeps each of its 67 phenomena in a file of its own.
BLIMP_FILES = {
    "adjunct_island": "094b1a6c4d8e88d2a193d25b6a8eaf6dc746158c14eec537802dd2b67f57847b",
    "anaphor_gender_agreement": "3c3f1d8d9e6aa071151e6455bf121c097ca4f7d9a0a7349ebfc9ec0e193c3472",
    "anaphor_number_agreement": "5dc16d19b9b3694589a11a66c7d0d0ca5707d3762a45d8cf0306133151575784",
    "animate_subject_passive": "d4c957edec77c6fb5b8afc3cee3b5b54c759458e0a19bca54ba1c35570c1d84e",
    "animate_subject_trans": "ef2f412c3f47a8bddc0a365aba51c7ce0678bc2a0d52874df923e008ea32bf2d",
    "causative": "cfec48a03e32498177214dd7add31df1200c2e049c6723678cbd3b66b5da52c4",
    "complex_NP_island": "4a01cfd34626f006bcf0ae90ca3f340348b20d1f6c0530450f35b7218e1b8db7",
    "coordinate_structure_constraint_complex_left_branch": "b776239fe67070fb13d64671fdeb4a31a81dd0693204abf4447fbe2e582c8f8a",
    "coordinate_structure_constraint_object_extraction": "55d830a519e53f2400d3c436f751295a83a123f46084f247ce802ebec9585816",
    "determiner_noun_agreement_1": "eac75d95c9fa5a1e6e560918d8a2edf64e4c6e0fae8622a7389bbd5ec514727f",
    "determiner_noun_agreement_2": "17520554fff0a78a122cc5bf92708e64099b6907d1791c029a1c254e20089bca",
    "determiner_noun_agreement_irregular_1": "b73d7c301e81f55ce3631cf2c0a02de83a1c5fdf70fdaef4bff90ac5d4f60c7d",
    "determiner_noun_agreement_irregular_2": "73c404f947d00d35d8b21cce80d0c23b51268873fa834d162018fb45927d1c6c",
    "determiner_noun_agreement_with_adj_2": "732f6dc785f282ef4757e119c847feef552c324b7125bb5191eb1f1377ef060d",
    "determiner_noun_agreement_with_adjective_1": "dd61dd23aa32be9bfe5f8c440a1bfc1720bde1543428cbe919a1359f53fa8cfb",
    "determiner_noun_agreement_with_adj_irregular_1": "939567391e6ec007ef20a9f8306f41b2ee0e25c380842550ca46bcf0484ceca9",
    "determiner_noun_agreement_with_adj_irregular_2": "e71c6a9baf9cf00d9e4bee09ac01590f9f3f96265ea80821ce309696d840811a",
    "distractor_agreement_relational_noun": "2e0cef46676c822f3443d91aab2daaab23e177bc9b033d2eb50324df29052ae9",
    "distractor_agreement_relative_clause": "c23c03ee033d92d9c22061089a80cbca29da2d337764bfa9bd20dc9d14d42afc",
    "drop_argument": "4acf04febbe94ca4d8e0d12d2e02c808fb028c3501f4e66e62b1a1e2bfc8fdf3",
    "ellipsis_n_bar_1": "58edad00d53c2abe73e721a8afdd6220c06c5bf3aeb860da8670249b181a5fa1",
    "ellipsis_n_bar_2": "7506bd80abb5c2d54afddf64042bc2df6d587b271f37832c2dcdfa1b66c11d70",
    "existential_there_object_raising": "1c63d5b77b5d40e0a8d475342d9893f2809093653b08c04f69f8a17252a98809",
    "existential_there_quantifiers_1": "ac2e197d55304a262bfeddaa498163f475e6da53dcab819892c799f58a229d99",
    "existential_there_quantifiers_2": "df307122fda88d067682497963044cfff29e3de5d9a88ace338e61d91e016f55",
    "existential_there_subject_raising": "1e281bf8a80a598b73f173f575003f6c3a7899143dd27412e18c06e5b561e171",
    "expletive_it_object_raising": "45a0a957726a018283bb5eaefc7a902257b363234f9b8793a73f8c4acfd20347",
    "inchoative": "cb452cd6ee3795f19a6b6e8b4328172610610c8cd9ed943c58c1ebbcd7e769cd",
    "intransitive": "f04f66e0a30a5f0ca8bbe130ed8f680ccf09d42a1e7971a1520352d37b667627",
    "irregular_past_participle_adjectives": "77d65967bbf67977e0ccaa13f190a3ebcf2066222fa6bb343841257d6ad564c3",
    "irregular_past_participle_verbs": "f6acd480c04c37beac4a65252edf51b2472c391646ec55ff04794d29b0f1338e",
    "irregular_plural_subject_verb_agreement_1": "b4b07609019d870a0b95ee8245d0e3d97ea9999098d58eb03b49dbfe1a975e2c",
    "irregular_plural_subject_verb_agreement_2": "66a1bf6bba6daad40a9d90676e5f7971bf2f12543f96557968fa60ab40e25adc",
    "left_branch_island_echo_question": "41a535903bc7f138d556dc01cff7cb6d9d8cc8e5d40fca1cb17e5da18ce12ae4",
    "left_branch_island_simple_question": "8c718b6f37cb5674c5f8dfd2094d7cd54c43fe6f86154cf3a73eaf2dc74aa721",
    "matrix_question_npi_licensor_present": "a7ed0834104a8fe81b389072c856c10779d660295f924067362740d7901d3532",
    "npi_present_1": "cca3e48e8a08528c06eaec33e9a53aa2875048642f6b38f8c03355c008a55c4f",
    "npi_present_2": "44cdf2e5c7964d7618370927e5a86753f87de84b4b53555ed19f710eb61f8240",
    "only_npi_licensor_present": "8e26a57bc8d8625bfcc1cc018d6c46944abf202aa5f874a39a0f04c24df9e178",
    "only_npi_scope": "c5c14f42c274448aaaaa7fe0e94440f9fdbe5c01a9cf1673af824189678d90e7",
    "passive_1": "3bcb3d7397215b8fc4804d74c6c2accf37c649cdf3976a56d8964004ed4f0ee5",
    "passive_2": "ee29e69ca55ec89cf648a57595f13c2b7271995ffb3329e10233a6703dbdd1bf",
    "principle_A_case_1": "6615f7f1572b9bcc335be59e7645002ceb2c95d92dc2371551c91db48d9a9a55",
    "principle_A_case_2": "e3a13d2ab24f231adbc9b3dda2fcd448883855199419db178a449c42eee1d097",
    "principle_A_c_command": "6d18116ede2c686232348552e632f04be358cc62ce1a8c3d65e32263c625c887",
    "principle_A_domain_1": "5efcec2af824973d3eca500ee3591d444b5cc56569394c7bb1e0fb8a95efcd35",
    "principle_A_domain_2": "5de4982ab8ba2914d476335ba48956cc5c26e24c5ae0b64b8aed7ab06cf2903b",
    "principle_A_domain_3": "25227d39f6168b88f5b3ee0b35d0f6dd8105545016d76764d11c27f6031844d5",
    "principle_A_reconstruction": "bd57fef2d8e907730bfdf9be1048bfaa75571b8a5c3481219b08254df544feb1",
    "regular_plural_subject_verb_agreement_1": "602d82e663e7d8d73aad82eeb8b4f6d1f578dcf33121a54fd639e3f0d5196aeb",
    "regular_plural_subject_verb_agreement_2": "47b11c839024e89b06eae89dc874c1f5c447b384dfadec7ed0f812e97e4ba5a3",
    "sentential_negation_npi_licensor_present": "2f2e30c846249f580be81a8a1276db371351bae9bf7c569e59679169404608dc",
    "sentential_negation_npi_scope": "06afb479723e7bff4071ca2f1511c1a78b8849331d09c608aa963fee057f8ac2",
    "sentential_subject_island": "a096e1ea8ff2e05ae00f0a52663f46598acc535ef108a7d3b2bbd73607b082f1",
    "superlative_quantifiers_1": "eba2e172058aaa24858464ab6e79455564e2e220d18863395897895ce8fcec4e",
    "superlative_quantifiers_2": "76eadb284f7c21aefe83c8eff7cf43afb446d237efbfbcb74962311650e5ba5e",
    "tough_vs_raising_1": "defbb5c0cb9b3d4b4b5f1571e64df8ba37428619bb4943d4d228ad5caf83f4d7",
    "tough_vs_raising_2": "01d9c6a6b0df268db59ce28ef6904541df2c0755cb5915b2f1f5dbcf08c3d82a",
    "transitive": "1c92b23164784bdb7b721a87969a883ce76fcb13b7b2e2240230f67800544432",
    "wh_island": "63e455edde12fd76bca6e9287cb80920833b705ac656bc5ad105831b3327f219",
    "wh_questions_object_gap": "5107d0b8a4057d6f778a01253de18ee1de6c125b42ee6dc75111aeee43e20b41",
    "wh_questions_subject_gap": "a58be1f30174106b6934ca13a7d89716e380a8f2ccd051f6bc20e85ba19904a4",
    "wh_questions_subject_gap_long_distance": "ca30473b9569b256cc7707d02668db436d097c810466c8cb60b1b7315ddb10e9",
    "wh_vs_that_no_gap": "92fc7e67b32b10f6e94b6b4c094204c51bb1b622d48812d714e33f91b6603fe2",
    "wh_vs_that_no_gap_long_distance": "e1aa6fd1c7406b3bea5206e954b75f1dec7dd17c1019d526abe6005e715b38b3",
    "wh_vs_that_with_gap": "c905f68e44327d8a970a0341a4a023f4f70e402f382c0262f702cd5946b58c68",
    "wh_vs_that_with_gap_long_distance": "c2adf0cb1714de75ac960e59eeed74c0fada5babee2e0cd6c2721b90aa8ed829",
}
# Collect all splits to exclude benchmark questions from instruction training.
# Evaluation uses the harness's held-out splits in pretrain.evaluation.
TASKS = {
    "hellaswag": {
        "language": "en",
        "repository": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "license": "MIT",
        "files": {
            "data/validation-00000-of-00001.parquet": "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd",
        },
    },
    "arc_easy": {
        "language": "en",
        "repository": "allenai/ai2_arc",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "license": "CC-BY-SA-4.0",
        "files": {
            "ARC-Easy/train-00000-of-00001.parquet": "b315db8a4be597dc7daa50a4e70d48dd7c990c32085629e6ccd8c926beaa80b5",
            "ARC-Easy/validation-00000-of-00001.parquet": "ed890ff1e4cef7a7140d3a30dcea3ed2c9d467c6458f447ad9ef0176d8dcbb74",
            "ARC-Easy/test-00000-of-00001.parquet": "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177",
        },
    },
    "jcommonsenseqa": {
        "language": "ja",
        "repository": "sbintuitions/JCommonsenseQA",
        "revision": "b72d3115609bed00c386701fcb40234a1fd08c7a",
        "license": "CC-BY-SA-4.0",
        "files": {
            "data/train-00000-of-00001.parquet": "8e1851eb5d7f9579432f2af2ff474f654977b077ef2a83da197d81518c6024ba",
            "data/validation-00000-of-00001.parquet": "b72e6c05535a7ccde7b66f8a721c3b0707e67097a7f33bab626195354180994f",
        },
    },
    "sciq": {
        "language": "en",
        "repository": "allenai/sciq",
        "revision": "2c94ad3e1aafab77146f384e23536f97a4849815",
        "license": "CC-BY-NC-3.0",
        "files": {
            "data/train-00000-of-00001.parquet": "19644360954006d06e9ad3df07bddb34f8535c081b831d48f604603c713ac167",
            "data/validation-00000-of-00001.parquet": "455dd9f1d725cd3ecbce369799a2fbbdbbfecf51ab84a86d56ba3370dc847b8a",
            "data/test-00000-of-00001.parquet": "3a719356a29b127fc54ef3c7f51a034db4bd105d5717215e8c85d2aa58d60667",
        },
    },
    "piqa": {
        "language": "en",
        "repository": "baber/piqa",
        "revision": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
        # Neither this copy nor the original dataset card states a license.
        "license": "unknown",
        "files": {
            "piqa_train.parquet": "3114602b31fa724309a6c493cc86368c4eab7ab6e79e285056fd5ce3b60ddf02",
            "piqa_validation.parquet": "2b17ee1215b1ceb2ddabf4c90a05c5a8b833bef9d207960f57d111c43a2080f3",
            "piqa_test.parquet": "037e4175b967f4082e45a5ccf5c9d31ca3fd73030804aa7d4db0f7ca1fcd9b34",
        },
    },
    "lambada": {
        "language": "en",
        "repository": "EleutherAI/lambada_openai",
        "revision": "900124bf3b8235c6daf21033af9948b3f07346c4",
        "license": "MIT",
        "files": {
            "default/test/default.parquet": "99d8a9e54cda761c6340ddd4fa6feda7f846ffb0e716afa52968dea75495eea7",
        },
    },
    "blimp": {
        "language": "en",
        "repository": "nyu-mll/blimp",
        "revision": "877fba0801ffb7cbd8c39c1ff314a46f053f6036",
        "license": "CC-BY-4.0",
        "files": {
            f"{phenomenon}/train-00000-of-00001.parquet": digest
            for phenomenon, digest in BLIMP_FILES.items()
        },
    },
    "kuci": {
        "language": "ja",
        "host": "github",
        "repository": "ku-nlp/KUCI",
        "revision": "79afa1d63d1f262e6709e9be714d9ad5dae8d6f6",
        "license": "CC-BY-SA-4.0",
        "files": {
            "dataset/train.jsonl": "77cd991d35cdc5cba23d0e2208cdc66e3173577449dffdf1503bf49e0624b076",
            "dataset/dev.jsonl": "a5205847d0c5a6a7952445a772b858e909d0fd379ea7f5b1a4effa7b1e045aa3",
            "dataset/test.jsonl": "d4741c5a786018e7094ed83be12bd71b812650ce347394d05d3725f9e8482a19",
        },
    },
    "xwinograd_ja": {
        "language": "ja",
        "repository": "polm-stability/xwinograd-ja",
        "revision": "2b841a0b715e00b389c38976daf25374368dee30",
        "license": "CC-BY-4.0",
        "files": {
            "test/xwinograd-ja.json": "bc272a8bf08d52de84ee12bf1f575b440451e4ed450a8c7f6be8cf8e5cd9facd",
        },
    },
    "jnli": {
        "language": "ja",
        "host": "github",
        "repository": "yahoojapan/JGLUE",
        "revision": "c96c6af54200a404f31a3a4cd0b289786aa78809",
        "license": "CC-BY-SA-4.0",
        "files": {
            "datasets/jnli-v1.2/train-v1.2.json": "c506354d1310807f7300cfa8ad49482c6359a84fe85eea8d5e667d90d338d1c4",
            "datasets/jnli-v1.2/valid-v1.2.json": "ca0353efc7c2eebfb6de4e13f16295053c8b1ee65e7b0849190c90426fbc495f",
        },
    },
    "ifeval": {
        "language": "en",
        "repository": "google/IFEval",
        "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "license": "Apache-2.0",
        "files": {
            "ifeval_input_data.jsonl": "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
        },
    },
    "arc_challenge": {
        "language": "en",
        "repository": "allenai/ai2_arc",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "license": "CC-BY-SA-4.0",
        "files": {
            "ARC-Challenge/train-00000-of-00001.parquet": "e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb",
            "ARC-Challenge/validation-00000-of-00001.parquet": "395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05",
            "ARC-Challenge/test-00000-of-00001.parquet": "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9",
        },
    },
    "mmlu": {
        "language": "en",
        "repository": "cais/mmlu",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "license": "MIT",
        "files": {
            "all/dev-00000-of-00001.parquet": "2b19bde1ed8ca6b482fb283abc90e8e0d9d228947029c0b91795d64b28b3bc3f",
            "all/validation-00000-of-00001.parquet": "66cdf0b090ccb657d18d13cd81e31fdc55c3467da9642ffb178653268a97c8ef",
            "all/test-00000-of-00001.parquet": "74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7",
        },
    },
    "gsm8k": {
        "language": "en",
        "repository": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "license": "MIT",
        "files": {
            "main/train-00000-of-00001.parquet": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
            "main/test-00000-of-00001.parquet": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
        },
    },
    "humaneval": {
        "language": "en",
        "repository": "openai/openai_humaneval",
        "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "license": "MIT",
        "files": {
            "openai_humaneval/test-00000-of-00001.parquet": "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6",
        },
    },
    "mbpp": {
        "language": "en",
        "repository": "google-research-datasets/mbpp",
        "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        "license": "CC-BY-4.0",
        "files": {
            "full/prompt-00000-of-00001.parquet": "a053e4bb85ceb77430ae80592addb4ca4dc6ba087592f9e04537800ee88b7431",
            "full/train-00000-of-00001.parquet": "09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae",
            "full/validation-00000-of-00001.parquet": "3f0ec060987432d99fe8fb409d31e6c67445b208a01741c5583517c80a10fe80",
            "full/test-00000-of-00001.parquet": "566fd53060ffba5766dace1d1e2f4c38906781526de222b0dfbdbc325b696c77",
        },
    },
    "jmmlu": {
        "language": "ja",
        "repository": "nlp-waseda/JMMLU",
        "revision": "3637b25e444ccfdcde4d23a783cbe8e674faa01b",
        "license": "CC-BY-NC-ND-4.0",
        "files": {
            "JMMLU.zip": "3ba7d912943ede44fb7ec06aa1df067ac6bee157e65c5588736b9b983f0e684d",
        },
    },
    "mgsm_ja": {
        "language": "ja",
        "repository": "juletxara/mgsm",
        "revision": "b2f13d426afe3be8d69a7e739b36724db8b66bbc",
        "license": "CC-BY-SA-4.0",
        "files": {
            "ja/train-00000-of-00001.parquet": "eff3111bb371ee8748406995264230280ea314c5254c3811b793f439a821a3c8",
            "ja/test-00000-of-00001.parquet": "9f8868e9245f99454171fca529d9c5ab5c77f2b70c0297379a67eec4464e7aea",
        },
    },
    "jhumaneval": {
        "language": "ja",
        "repository": "kogi-jwu/jhumaneval",
        "revision": "35c54b74c11ec8f5699283c29e0d3ec9a94702ca",
        "license": "MIT",
        "files": {
            "jhumaneval/test-00000-of-00001.parquet": "eb3b085bf70d9c2a934a9da52eefa5e3fbaae7dbacea1f715232f777095be53d",
        },
    },
    "jsquad": {
        "language": "ja",
        "host": "github",
        "repository": "yahoojapan/JGLUE",
        "revision": "c96c6af54200a404f31a3a4cd0b289786aa78809",
        "license": "CC-BY-SA-4.0",
        # SQuAD's JSON nests the questions in the paragraphs of articles.
        "format": "squad",
        "files": {
            "datasets/jsquad-v1.2/train-v1.2.json": "24a0233a5d2cc80571e499de487fe2188981126312f159fc90136db82f3584b5",
            "datasets/jsquad-v1.2/valid-v1.2.json": "fb0a57a35281ff03bbe7f0e878cff5b577985fbe6fcb242e7e8ddb63d1c0ab11",
        },
    },
    "mt_bench": {
        "language": "en",
        "repository": "HuggingFaceH4/mt_bench_prompts",
        "revision": "e3a795c5e9a82ee40611c416b8a7786c73198991",
        "license": "Apache-2.0",
        # The questions of MT-Bench, which grades conversations in two turns.
        "files": {
            "raw/question.jsonl": "420ae1d63cbf0f499d8afc3b28d86714e77f13e44c133eedf942ed16c544dd34",
        },
    },
    "ja_mt_bench": {
        "language": "ja",
        "host": "github",
        "repository": "Stability-AI/FastChat",
        "revision": "e37eb28ee490480f6f4989274ed47c55f7dfce82",
        "license": "Apache-2.0",
        # Japanese MT-Bench, all eight categories.
        "files": {
            "fastchat/llm_judge/data/japanese_mt_bench/question_full.jsonl": "8f56c8bd5ff93d2120409bc8f77f0020e673c74548c9637f2aace12a323834e5",
        },
    },
    "elyza_tasks_100": {
        "language": "ja",
        "repository": "elyza/ELYZA-tasks-100",
        "revision": "240fc800c4695d86ce0f32a713e199cd2ba66ab6",
        "license": "CC-BY-SA-4.0",
        "files": {
            "test.csv": "51412fbfc955e06e59f6cae2cb564b43758cb5360d6dd65715e6558861dd524c",
        },
    },
    "math500": {
        "language": "en",
        "repository": "HuggingFaceH4/MATH-500",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "license": "MIT",
        "files": {
            "test.jsonl": "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
        },
    },
    "m_ifeval_ja": {
        "language": "ja",
        "repository": "tokyotech-llm/M-IFEval-Ja",
        "revision": "f42d9f0c9bd3e1be76b47c84143cd94d98eada8b",
        "license": "Apache-2.0",
        "files": {
            "test.jsonl": "d74fbbd10fe512ad0a487e1dc85823ecfb9addcdc08fb9182d68616d6056782e",
        },
    },
    "bfcl": {
        "language": "en",
        "repository": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "revision": "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
        "license": "Apache-2.0",
        # The Berkeley Function Calling Leaderboard's requests; its chat-only category is left out.
        "files": {
            "BFCL_v3_exec_multiple.json": "180ccd36814856574a2a488e026ace22ef2631fe17f404a8a4168ceffaab4cb6",
            "BFCL_v3_exec_parallel.json": "67826cdd638e1416a3b9f0243872ce91b95f438b467faa3f8d9351e80d39153d",
            "BFCL_v3_exec_parallel_multiple.json": "e3134c946b215d34a79dbe3eef0fbfe78f9e3a18e67e9a9adf891f01b2e7977d",
            "BFCL_v3_exec_simple.json": "120eb383a8ec801b47e56d7337b61158944ae35b935c86e8d846948fa27215f6",
            "BFCL_v3_irrelevance.json": "975f51c51f688649fd190078efd87081241e0a326f9114a2ea3c1ca2440d8690",
            "BFCL_v3_java.json": "261d327d50c1bb505614dd5f75a29167d5bd9f938e5cefefef989c325cb67848",
            "BFCL_v3_javascript.json": "b69b6a99548f72e151be7abed4beb0ed70873d3370d5b78feed75b66f33795d3",
            "BFCL_v3_live_irrelevance.json": "0d259e1c3ab6ba06c2a51911e2f49ecaa2eac3723a4a14f183a7687955e61338",
            "BFCL_v3_live_multiple.json": "af2520ac5bb0b05aa9b8abe14a79b87198e5a3137a707d4f5633c792330f0cee",
            "BFCL_v3_live_parallel.json": "6c26e9fdc3350cf596e6d1ea9c179cbff834761bccf562f4141ed29a839ca421",
            "BFCL_v3_live_parallel_multiple.json": "21d4b9319c1faac431e22757b367ea28917fe467364c3a4b17f16ec06d4f6e79",
            "BFCL_v3_live_relevance.json": "294ef3f21909e6f84c735faf688f5cbd72226ad012b7ea53c1e177c40cbbf053",
            "BFCL_v3_live_simple.json": "b398743e6c0dfd2a87cee97a3bc1b9b90b81edc228c3153a820a93b51186fc00",
            "BFCL_v3_multi_turn_base.json": "edaea7666e0089062c30f3ac511914d5c27414dbe9b8214488191980d873b145",
            "BFCL_v3_multi_turn_composite.json": "1fda0310e176317af7184132ff5fb5daa8ee7984b1cfbf69ce4bfb58c3151ae1",
            "BFCL_v3_multi_turn_long_context.json": "a7c2b44cfe14e243e66acbb0737970f1085460fed5793bfa0137fb4782ff8fc4",
            "BFCL_v3_multi_turn_miss_func.json": "e6cfbd225ac1182c82bfcd0a83410babd410a8cf2050942894017829e9a57e23",
            "BFCL_v3_multi_turn_miss_param.json": "21a850b8bb0b41aba63e0bbcd9b4af6c7997144180eadd4bcdf4bdfd98077c1a",
            "BFCL_v3_multiple.json": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
            "BFCL_v3_parallel.json": "fad375d68295776efb63e5aa331dd884f8635e7bb50e05908113c74f14281c3c",
            "BFCL_v3_parallel_multiple.json": "8863ea8433239f55c5f016154cf0830853c89f693c6ea270396a2fa121960579",
            "BFCL_v3_rest.json": "b064f9eefc257dba383227177f570fd9038588dd058e5a9e869007f81d5cc5f2",
            "BFCL_v3_simple.json": "fbc37b2ad252bf9af985582e0e07b456173fe627d957491472ea9cef5fb83158",
            "BFCL_v3_sql.json": "2cf021e2e4f81f1097aab31763423b785b3aaa73df8203a4f4ae3900822b0d53",
        },
    },
}


def clean_hellaswag(text):
    # Remove WikiHow markup such as [header] in the usual HellaSwag way.
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ").strip()


def hellaswag_item(row):
    context = f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}"
    return {
        "id": f"{row['source_id']}/{row['ind']}",
        "context": clean_hellaswag(context),
        "choices": [" " + clean_hellaswag(ending) for ending in row["endings"]],
        "label": int(row["label"]),
    }


def arc_item(row):
    return {
        "id": row["id"],
        "context": f"Question: {row['question']}\nAnswer:",
        "choices": [" " + text for text in row["choices"]["text"]],
        "label": row["choices"]["label"].index(row["answerKey"]),
    }


def jcommonsenseqa_item(row):
    return {
        "id": str(row["q_id"]),
        "context": f"質問：{row['question']}\n回答：",
        "choices": [row[f"choice{index}"] for index in range(5)],
        "label": row["label"],
    }


def sciq_item(row):
    # The question alone: the evaluation shows the supporting passage anyway.
    return {"context": f"Question: {row['question']}\nAnswer:"}


def piqa_item(row):
    return {"context": f"Question: {row['goal']}\nAnswer:"}


def lambada_item(row):
    return {"context": row["text"]}


def blimp_item(row):
    # Only the grammatical sentence of each pair is one that texts would contain.
    return {"context": row["sentence_good"]}


def kuci_item(row):
    # The clause with its right continuation, as the web text KUCI came from wrote it.
    answer = row[f"choice_{row['label']}"]
    return {"id": str(row["id"]), "context": (row["context"] + answer).replace(" ", "")}


def xwinograd_item(row):
    # The sentence with the right referent filled in, as a text would write it.
    return {"context": row[f"sentence{row['answer']}"]}


def jnli_item(row):
    # The premise, a published image caption, is the longer of the pair.
    return {"context": row["sentence1"]}


def ifeval_item(row):
    return {"id": str(row["key"]), "context": row["prompt"]}


def question_item(row):
    return {"context": f"Question: {row['question'].strip()}\nAnswer:"}


def program_item(row):
    # The function's signature and description, which its copies share.
    return {"id": row["task_id"], "context": row["prompt"]}


def mbpp_item(row):
    return {"id": str(row["task_id"]), "context": row["text"]}


def jmmlu_item(row):
    return {"context": f"質問：{row['question'].strip()}\n回答："}


def mgsm_item(row):
    # The examples label their questions, which texts would not.
    return {"context": row["question"].removeprefix("問題：")}


def jsquad_item(row):
    return {"id": row["id"], "context": f"質問：{row['question']}\n回答："}


def turn_items(field, key):
    """Convert a question asked over several turns into an item for each turn."""

    def items(row):
        return [
            {"id": f"{row[key]}:{turn}", "context": text}
            for turn, text in enumerate(row[field])
        ]

    return items


def elyza_item(row):
    return {"context": row["input"]}


def math_item(row):
    return {"id": row["unique_id"], "context": row["problem"]}


def tool_items(row):
    # Each of the user's requests; the functions offered are documentation.
    question = row["question"]
    # Requests come in turns of messages, or as one list of messages as SQL's do.
    turns = [question] if question and isinstance(question[0], dict) else question
    return [
        {"id": f"{row['id']}:{turn}:{index}", "context": message["content"]}
        for turn, messages in enumerate(turns)
        for index, message in enumerate(messages)
        if message["role"] == "user"
    ]


CONVERTERS = {
    "hellaswag": hellaswag_item,
    "arc_easy": arc_item,
    "jcommonsenseqa": jcommonsenseqa_item,
    "sciq": sciq_item,
    "piqa": piqa_item,
    "lambada": lambada_item,
    "blimp": blimp_item,
    "kuci": kuci_item,
    "xwinograd_ja": xwinograd_item,
    "jnli": jnli_item,
    "ifeval": ifeval_item,
    "arc_challenge": arc_item,
    "mmlu": question_item,
    "gsm8k": question_item,
    "humaneval": program_item,
    "mbpp": mbpp_item,
    "jmmlu": jmmlu_item,
    "mgsm_ja": mgsm_item,
    "jhumaneval": program_item,
    "jsquad": jsquad_item,
    "mt_bench": turn_items("prompt", "question_id"),
    "ja_mt_bench": turn_items("turns", "question_id"),
    "elyza_tasks_100": elyza_item,
    "math500": math_item,
    "m_ifeval_ja": ifeval_item,
    "bfcl": tool_items,
}


def source_url(task, path):
    template = GITHUB_URL if task.get("host") == "github" else SOURCE_URL
    return template.format(path=path, **task)


def download(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def read_rows(task, path, data):
    """The records of one of a task's source files."""
    if path.endswith(".zip"):
        # JMMLU keeps each subject in a CSV file; macOS added files of its own.
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return [
                row
                for name in sorted(archive.namelist())
                if name.endswith(".csv") and not name.startswith("__MACOSX/")
                for row in csv.DictReader(
                    io.TextIOWrapper(archive.open(name), encoding="utf-8-sig")
                )
            ]
    if task.get("format") == "squad":
        return [
            {**question, "title": article["title"], "context": paragraph["context"]}
            for article in json.loads(data)["data"]
            for paragraph in article["paragraphs"]
            for question in paragraph["qas"]
        ]
    if path.endswith(".csv"):
        return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    if path.endswith((".json", ".jsonl")):
        # JGLUE and XWinograd also write one question per line in .json.
        return [
            json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()
        ]
    from pyarrow import parquet

    return parquet.read_table(io.BytesIO(data)).to_pylist()


def fetch_benchmarks(output):
    try:
        import pyarrow.parquet  # noqa: F401  # read_rows reads parquet sources with it
    except ImportError as error:
        raise ValueError("Install corpus dependencies with --extra corpus") from error
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; select a new benchmark directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        manifest = {
            "version": 1,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "tasks": {},
        }
        for name, task in TASKS.items():
            items = []
            for path, expected in task["files"].items():
                data = download(source_url(task, path))
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"{name} source checksum does not match: {path}")
                for index, row in enumerate(read_rows(task, path, data)):
                    converted = CONVERTERS[name](row)
                    # A question asked in several messages gives an item for each.
                    for item in converted if isinstance(converted, list) else [converted]:
                        # Sources without identifiers number questions by file and row.
                        item.setdefault("id", f"{path}:{index}")
                        items.append(item)
            if len({item["id"] for item in items}) != len(items):
                raise ValueError(f"{name} items need unique identifiers")
            for item in items:
                if not isinstance(item["context"], str) or not item["context"].strip():
                    raise ValueError(f"{name} item {item['id']} is malformed")
                if "choices" in item and (
                    not 0 <= item["label"] < len(item["choices"])
                    or not all(choice.strip() for choice in item["choices"])
                ):
                    raise ValueError(f"{name} item {item['id']} is malformed")
            filename = f"{name}.jsonl"
            with (staging / filename).open("w", encoding="utf-8") as destination:
                for item in items:
                    destination.write(json.dumps(item, ensure_ascii=False) + "\n")
            manifest["tasks"][name] = {
                **task,
                "dataset_url": f"https://github.com/{task['repository']}"
                if task.get("host") == "github"
                else f"https://huggingface.co/datasets/{task['repository']}",
                "filename": filename,
                "items": len(items),
                "sha256": file_digest(staging / filename),
            }
        (staging / "sources.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_benchmarks(path, tasks=None, limit=None):
    path = Path(path)
    manifest = json.loads((path / "sources.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("unsupported benchmark format")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    benchmarks = {}
    for name in tasks or manifest["tasks"]:
        if name not in manifest["tasks"]:
            raise ValueError(f"unknown benchmark: {name}")
        metadata = manifest["tasks"][name]
        filename = path / metadata["filename"]
        if file_digest(filename) != metadata["sha256"]:
            raise ValueError(f"benchmark checksum does not match: {name}")
        with filename.open(encoding="utf-8") as source:
            items = [json.loads(line) for line in source]
        if limit is not None and limit < len(items):
            # A seeded sample keeps limited runs comparable across checkpoints.
            selected = random.Random(name).sample(range(len(items)), limit)
            items = [items[index] for index in sorted(selected)]
        benchmarks[name] = items
    return benchmarks
