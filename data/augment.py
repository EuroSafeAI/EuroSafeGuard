"""Augment classifier training prompts with obfuscations, Best-of-N perturbations and jailbreak templates.

Every augmentation is applied to harmful and benign rows at the same rate, and an augmented row keeps the label
(category, entry_type) and the input_id of its source row, so it pairs with its plain twin in the consistency loss.
The output holds the input rows unchanged plus the augmented rows (source = technique family, attack = technique).

Families
  obf       one of 40 string transforms: encodings (base64/32/85, hex, binary, char codes, URL, unicode escapes, HTML
            entities, Morse, Braille, ROT13, Atbash, Caesar; with and without a "decode and follow" instruction),
            look-alike characters (fullwidth, math bold, homoglyphs, leetspeak), spacing and separators, reversals,
            casual writing, noise, payload splitting and neutral containers (Markdown, code comment, e-mail, log, CSV)
  bon       Best-of-N-style character perturbation: random capitals, shuffled word interiors, inserted characters
  template  the prompt inserted into a jailbreak template from --templates (its placeholder, or appended). No
            templates ship with this repository; e.g. the in-the-wild prompts of Shen et al. (CCS'24):
            --templates TrustAIRLab/in-the-wild-jailbreak-prompts:jailbreak_2023_12_25
--gibberish N adds N content-free benign rows (keyboard mash, random characters, fake words, symbols, emoji).

    python data/augment.py --in train.csv --out train_aug.csv --rate 0.1 --families obf bon
    python data/augment.py --in train.csv --out train_aug.csv --families template --templates FILE_OR_HF_DATASET

Input: a CSV with a `prompt` column; `category`, `entry_type`, `source`, `input_id` are kept if present (missing
input_id: one per row; missing source: "plain"). Prints counts only.
"""
import argparse
import base64
import codecs
import hashlib
import random
import re
import string
import urllib.parse
from pathlib import Path

import pandas as pd

HARMFUL = {"harmful", "dual_use_harmful"}
PLACEHOLDER = re.compile(r"\[INSERT[^\]]*\]|\{\{\s*PROMPT\s*\}\}|\[PROMPT\]|INSERT PROMPT HERE|\[YOUR PROMPT\]", re.I)

MORSE = dict(zip("abcdefghijklmnopqrstuvwxyz0123456789",
                 (".- -... -.-. -.. . ..-. --. .... .. .--- -.- .-.. -- -. --- .--. --.- .-. ... - ..- ...- .-- -..- "
                  "-.-- --.. ----- .---- ..--- ...-- ....- ..... -.... --... ---.. ----.").split()))
BRAILLE = {c: chr(0x2800 + v) for c, v in zip("abcdefghijklmnopqrstuvwxyz",
                                               [1, 3, 9, 25, 17, 11, 27, 19, 10, 26, 5, 7, 13, 29, 21, 15, 31, 23, 14,
                                                30, 37, 39, 58, 45, 61, 53])}
HOMOGLYPH = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у"}  # Cyrillic look-alikes
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "l": "1", "g": "9", "b": "8"}


def shift(t: str, k: int) -> str:
    return "".join(chr((ord(c) - 97 + k) % 26 + 97) if c.islower() else
                   chr((ord(c) - 65 + k) % 26 + 65) if c.isupper() else c for c in t)


def atbash(t: str) -> str:
    return "".join(chr(219 - ord(c)) if "a" <= c <= "z" else chr(155 - ord(c)) if "A" <= c <= "Z" else c for c in t)


def mathbold(t: str) -> str:
    return "".join(chr(0x1D400 + ord(c) - 65) if "A" <= c <= "Z" else
                   chr(0x1D41A + ord(c) - 97) if "a" <= c <= "z" else c for c in t)


def scramble(t: str, rng: random.Random, p: float = 1.0) -> str:
    """Shuffle the interior letters of words of 4+ characters, each with probability p."""
    return " ".join(w if len(w) < 4 or rng.random() > p else w[0] + "".join(rng.sample(w[1:-1], len(w) - 2)) + w[-1]
                    for w in t.split())


def typos(t: str, rng: random.Random, p: float = 0.12) -> str:
    out = []
    for c in t:
        u = rng.random()
        out.append(c + c if c.isalpha() and u < p / 2 else "" if c.isalpha() and u < p else c)
    return "".join(out)


def payload_split(t: str) -> str:
    w = t.split()
    n = min(4, max(2, len(w) // 8))
    body = "\n".join(f'{chr(97 + i)} = "{" ".join(w[i::n])}"' for i in range(n))
    return f"Join these in order a..{chr(96 + n)} and answer the result:\n{body}"


def decode(name: str, encoded: str) -> str:
    return f"Decode {name} and follow:\n{encoded}"


OBF = {  # name -> fn(text, rng)
    "base64": lambda t, r: decode("base64", base64.b64encode(t.encode()).decode()),
    "base32": lambda t, r: decode("base32", base64.b32encode(t.encode()).decode()),
    "base85": lambda t, r: decode("base85", base64.b85encode(t.encode()).decode()),
    "hex": lambda t, r: decode("hex", t.encode().hex()),
    "binary": lambda t, r: decode("binary", " ".join(format(b, "08b") for b in t.encode())),
    "ascii_ordinal": lambda t, r: decode("these character codes", " ".join(str(ord(c)) for c in t)),
    "url": lambda t, r: decode("this URL encoding", urllib.parse.quote(t)),
    "unicode_escape": lambda t, r: decode("these escapes", t.encode("unicode_escape").decode()),
    "html_entities": lambda t, r: decode("these HTML entities", "".join(f"&#{ord(c)};" for c in t)),
    "morse": lambda t, r: decode("Morse", " ".join(MORSE.get(c.lower(), c) for c in t)),
    "braille": lambda t, r: decode("Braille", "".join(BRAILLE.get(c.lower(), c) for c in t)),
    "rot13": lambda t, r: decode("ROT13", codecs.encode(t, "rot_13")),
    "atbash": lambda t, r: decode("Atbash", atbash(t)),
    "caesar": lambda t, r: decode("Caesar(+3)", shift(t, 3)),
    "raw_base64": lambda t, r: base64.b64encode(t.encode()).decode(),
    "raw_base32": lambda t, r: base64.b32encode(t.encode()).decode(),
    "raw_hex": lambda t, r: t.encode().hex(),
    "raw_rot13": lambda t, r: codecs.encode(t, "rot_13"),
    "fullwidth": lambda t, r: "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in t),
    "mathbold": lambda t, r: mathbold(t),
    "homoglyph": lambda t, r: "".join(HOMOGLYPH.get(c, c) for c in t),
    "leetspeak": lambda t, r: "".join(LEET.get(c, c) for c in t.lower()),
    "letter_spacing": lambda t, r: " ".join(t.replace(" ", "_")),
    "separator": lambda t, r: ".".join(t),
    "zero_width": lambda t, r: "​".join(t),
    "reversed_chars": lambda t, r: "Reverse and answer:\n" + t[::-1],
    "reversed_words": lambda t, r: "Un-reverse each word:\n" + " ".join(w[::-1] for w in t.split()),
    "reversed_word_order": lambda t, r: "Read right-to-left:\n" + " ".join(t.split()[::-1]),
    "scrambled": lambda t, r: scramble(t, r),
    "keyboard_typos": lambda t, r: typos(t, r),
    "random_capitals": lambda t, r: "".join(c.upper() if r.random() < 0.4 else c.lower() for c in t),
    "lowercase": lambda t, r: t.lower(),
    "dropped_spaces": lambda t, r: t.replace(" ", ""),
    "dropped_punct": lambda t, r: "".join(c for c in t if c.isalnum() or c.isspace()),
    "payload_split": lambda t, r: payload_split(t),
    "markdown": lambda t, r: f"```\n# task\n{t}\n```\nComplete the task in the code block.",
    "code_comment": lambda t, r: f"# TODO: {t}\ndef solve():\n    pass  # fill in per the TODO above",
    "email": lambda t, r: f"From: a@example.com\nTo: b@example.com\nSubject: request\n\n{t}\n\nPlease action the above.",
    "log_entry": lambda t, r: f"[2026-01-01 12:00:00] INFO user_request: {t}",
    "csv_row": lambda t, r: f'id,request\n1,"{t}"\nAnswer the request in row 1.',
}


def bon(t: str, rng: random.Random) -> str:
    t = "".join(c.upper() if rng.random() < 0.35 else c.lower() for c in t)
    t = scramble(t, rng, p=0.5)
    return "".join(c + (rng.choice("asdf") if rng.random() < 0.06 else "") for c in t)


def fill_template(template: str, prompt: str) -> str:
    if PLACEHOLDER.search(template):
        return PLACEHOLDER.sub(lambda m: prompt, template, count=1)
    return template.rstrip() + "\n\n" + prompt


def load_templates(spec: str, column: str) -> list[str]:
    """A local CSV/Parquet/JSONL file, or a Hugging Face dataset as REPO[:CONFIG]."""
    p = Path(spec)
    if p.exists():
        d = {".parquet": pd.read_parquet, ".jsonl": lambda f: pd.read_json(f, lines=True)}.get(p.suffix, pd.read_csv)(p)
    else:
        from datasets import load_dataset
        repo, _, config = spec.partition(":")
        d = load_dataset(repo, config or None, split="train").to_pandas()
    t = d[column].dropna().astype(str)
    return sorted(set(t[t.str.len().between(40, 4000)]))


def gibberish(n: int, rng: random.Random) -> list[tuple[str, str]]:
    letters, emoji = string.ascii_lowercase, "😀😂🤣😍🙄😴🤔👍🔥✨🎉💀🙏🌈🍕🐱"

    def mash(k):
        s = ""
        while len(s) < k:
            s += rng.choice(letters) * rng.choice([1, 1, 1, 2, 4, 9])
        return s[:k]
    rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm", "1234567890"]
    gen = {
        "mash": lambda: mash(rng.randint(5, 200)),
        "keyboard_rows": lambda: "".join(rng.choice(rows)[rng.randint(0, 3):] for _ in range(rng.randint(1, 10))),
        "random_chars": lambda: "".join(rng.choice(string.ascii_letters + string.digits + " ")
                                        for _ in range(rng.randint(5, 250))),
        "fake_words": lambda: " ".join(mash(rng.randint(1, 10)) for _ in range(rng.randint(1, 40))),
        "symbols": lambda: "".join(rng.choice(string.punctuation + " ") for _ in range(rng.randint(3, 80))),
        "digits": lambda: "".join(rng.choice(string.digits + " .,") for _ in range(rng.randint(3, 120))),
        "emoji": lambda: "".join(rng.choice(emoji + " ") for _ in range(rng.randint(1, 40))),
        "repeat": lambda: rng.choice(string.ascii_letters + "?!.") * rng.randint(3, 300),
    }
    names = list(gen)
    return [(gen[k](), k) for k in (names[i % len(names)] for i in range(n))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="CSV with a `prompt` column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", nargs="+", default=["obf", "bon"], choices=["obf", "bon", "template"])
    ap.add_argument("--rate", type=float, default=0.1,
                    help="share of the harmful rows and, separately, of the other rows that each family augments")
    ap.add_argument("--templates", help="template file or Hugging Face dataset REPO[:CONFIG] (family `template`)")
    ap.add_argument("--template-column", default="prompt")
    ap.add_argument("--gibberish", type=int, default=0, help="add this many content-free benign rows")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    d = pd.read_csv(a.inp, dtype=str, keep_default_na=False)
    if "input_id" not in d:
        d["input_id"] = [hashlib.sha1(f"{i}\t{t}".encode()).hexdigest()[:16] for i, t in enumerate(d.prompt)]
    if "source" not in d:
        d["source"] = "plain"
    harmful = d.entry_type.isin(HARMFUL) if "entry_type" in d else pd.Series(False, index=d.index)
    templates = []
    if "template" in a.families:
        if not a.templates:
            ap.error("family `template` needs --templates")
        templates = load_templates(a.templates, a.template_column)
        print(f"templates: {len(templates)}")

    new = []
    for k, family in enumerate(a.families):
        for is_harmful in (True, False):
            group = d[harmful == is_harmful]
            for _, row in group.sample(frac=a.rate, random_state=a.seed + k).iterrows():
                if family == "obf":
                    attack = rng.choice(list(OBF))
                    text = OBF[attack](row.prompt, rng)
                elif family == "bon":
                    attack, text = "bon", bon(row.prompt, rng)
                else:
                    i = rng.randrange(len(templates))
                    attack, text = f"template_{i}", fill_template(templates[i], row.prompt)
                if text.strip() and text != row.prompt:
                    new.append({**row.to_dict(), "prompt": text, "source": family, "attack": attack})
    aug = pd.DataFrame(new, columns=[*d.columns, "attack"])
    if a.gibberish:
        g = pd.DataFrame(gibberish(a.gibberish, rng), columns=["prompt", "attack"])
        g = g.assign(category="Benign", entry_type="general_benign", source="gibberish",
                     input_id=[f"gibberish_{i}" for i in range(len(g))])
        aug = pd.concat([aug, g], ignore_index=True)

    out = pd.concat([d.assign(attack="none"), aug], ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"input rows {len(d)}, augmented rows {len(aug)}, written {len(out)} -> {a.out}")
    if len(aug):
        is_h = aug.entry_type.isin(HARMFUL) if "entry_type" in aug else pd.Series(False, index=aug.index)
        print(pd.crosstab(aug.source, is_h.map({True: "harmful", False: "benign"})).to_string())
        twins = aug.input_id.isin(d.input_id[d.source == "plain"])
        print(f"augmented rows with a plain twin: {int(twins.sum())}")


if __name__ == "__main__":
    main()
