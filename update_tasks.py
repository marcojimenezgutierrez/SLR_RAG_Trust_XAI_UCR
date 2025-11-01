import json
from pathlib import Path

TASKS_PATH = Path('tasks.jsonl')
SCRIPT_PATH = Path('Deduplicación/enrich_abstracts.py')


def main() -> None:
    lines = [json.loads(line) for line in TASKS_PATH.read_text(encoding='utf-8').splitlines()]
    script_body = SCRIPT_PATH.read_text(encoding='utf-8')
    cmd = "@'\n" + script_body + "\n'@ | Set-Content -Encoding UTF8 'Deduplicaci\\u00f3n/enrich_abstracts.py'"
    for obj in lines:
        if obj.get('id') == '05_script':
            obj['cmd'] = cmd
    TASKS_PATH.write_text('\n'.join(json.dumps(obj, ensure_ascii=False) for obj in lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
