import hashlib
import io
import json
import os
import random
import re
import shutil
import tempfile
import urllib.request
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
}


def source_url(task, path):
    template = GITHUB_URL if task.get("host") == "github" else SOURCE_URL
    return template.format(path=path, **task)


def download(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def fetch_benchmarks(output):
    try:
        from pyarrow import parquet
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
                rows = (
                    [
                        json.loads(line)
                        for line in data.decode("utf-8").splitlines()
                        if line.strip()
                    ]
                    # JGLUE and XWinograd also write one question per line in .json.
                    if path.endswith((".json", ".jsonl"))
                    else parquet.read_table(io.BytesIO(data)).to_pylist()
                )
                for index, row in enumerate(rows):
                    item = CONVERTERS[name](row)
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
