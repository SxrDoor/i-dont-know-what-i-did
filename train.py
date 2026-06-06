# train.py —— 全自动中文分词训练 & 前端模型导出（已过滤纯中文标点）
import os, re, json, math, datetime
from collections import defaultdict, Counter

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ========== 参数配置 ==========
RANDOM_AMPLITUDE = 0.0
OFFICIAL_PMI_THRESHOLD = 3.0
OFFICIAL_BASE = 200.0
BAD_SUFFIXES = {'们', '的', '了', '着', '过', '地', '得', '在', '和', '是'}
SEED_FREQ = 10000

RE_CHINESE = re.compile(r'[\u4e00-\u9fff]')
RE_CHINESE_BLOCK = re.compile(r'[\u4e00-\u9fff]+')

# ========== 全局统计 ==========
unigram_freq = defaultdict(int)
bigram_freq = defaultdict(int)
word_stats = {}
semi_count = defaultdict(int)
is_official = defaultdict(bool)

def load_seed_words(seed_dir='dict'):
    if not os.path.isdir(seed_dir):
        return
    for fname in os.listdir(seed_dir):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(seed_dir, fname)
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                w = line.strip()
                if len(w) < 2:
                    continue
                if w not in word_stats:
                    word_stats[w] = {'freq': 0, 'right_neighbors': defaultdict(int), 'left_neighbors': defaultdict(int)}
                word_stats[w]['freq'] = SEED_FREQ
                semi_count[w] = 100
                is_official[w] = True

def update_global_ngrams(text):
    n = len(text)
    for i in range(n):
        unigram_freq[text[i]] += 1
        if i + 1 < n:
            bigram_freq[text[i:i+2]] += 1

def add_word_stats(word, left, right):
    if word not in word_stats:
        word_stats[word] = {'freq': 0, 'right_neighbors': defaultdict(int), 'left_neighbors': defaultdict(int)}
    stats = word_stats[word]
    stats['freq'] += 1
    if right is not None:
        stats['right_neighbors'][right] += 1
    if left is not None:
        stats['left_neighbors'][left] += 1

def calc_pmi_min(word):
    L = len(word)
    if L < 2:
        return 0.0
    total_uni = sum(unigram_freq.values())
    if total_uni == 0:
        return 0.0
    pmi_vals = []
    for i in range(L - 1):
        a, b = word[i], word[i+1]
        pair = word[i:i+2]
        f_pair = bigram_freq.get(pair, 0) + 1
        f_a = unigram_freq.get(a, 0) + 1
        f_b = unigram_freq.get(b, 0) + 1
        pmi = math.log2((f_pair * total_uni) / (f_a * f_b))
        pmi_vals.append(pmi)
    return min(pmi_vals)

def update_all_stats(chinese_str, words):
    update_global_ngrams(chinese_str)
    n = len(chinese_str)
    for i in range(n):
        for k in range(2, min(5, n - i + 1)):
            word = chinese_str[i:i+k]
            left = chinese_str[i-1] if i > 0 else None
            right = chinese_str[i+k] if i+k < n else None
            add_word_stats(word, left, right)
    for w in words:
        if len(w) > 1:
            semi_count[w] += 1
            if (semi_count[w] >= 3 and not is_official[w] and
                calc_pmi_min(w) >= OFFICIAL_PMI_THRESHOLD):
                if len(w) == 2 and w[-1] in BAD_SUFFIXES:
                    continue
                is_official[w] = True

def entropy_from_counts(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for cnt in counter.values():
        p = cnt / total
        ent -= p * math.log2(p)
    return ent

def get_freedom(word, direction):
    stats = word_stats.get(word)
    if stats:
        neighbor_counts = stats['right_neighbors'] if direction == 'forward' else stats['left_neighbors']
        if neighbor_counts and len(neighbor_counts) > 0:
            return entropy_from_counts(neighbor_counts) + 1.0
    if is_official.get(word, False):
        return 2.5
    return 1.0

def word_score(word, direction):
    if len(word) == 1:
        HIGH_FREQ_SINGLE = {'的', '了', '在', '和', '是', '就', '也', '都', '很', '不',
                           '把', '被', '从', '对', '与', '但', '而', '所', '且', '或',
                           '已', '曾', '您'}
        return 0.35 if word in HIGH_FREQ_SINGLE else 0.2
    base_map = {1:0.7, 2:1.6, 3:1.9, 4:2.5}
    base = base_map.get(len(word), 2.0)
    bonus = 0.8 if semi_count.get(word, 0) > 0 else 0.0
    freedom = get_freedom(word, direction)
    cohesion = calc_pmi_min(word) + 1.0
    if cohesion < 0.1: cohesion = 0.1
    if cohesion > 10.0: cohesion = 10.0
    return (base + bonus) * freedom * cohesion

def official_total_score(word):
    free_fwd = get_freedom(word, 'forward')
    free_bwd = get_freedom(word, 'backward')
    ctx = min(free_fwd, free_bwd)
    return (OFFICIAL_BASE * len(word) - 2) * ctx

def segment_raw(chinese_str):
    n = len(chinese_str)
    if n == 0:
        return []
    max_seed_len = max((len(w) for w, off in is_official.items() if off), default=4)
    max_k = max(4, max_seed_len)
    scores = {}
    for i in range(n):
        for k in range(1, min(max_k, n - i) + 1):
            w = chinese_str[i:i+k]
            if not all('\u4e00' <= c <= '\u9fff' for c in w):
                continue
            if is_official.get(w, False):
                s = official_total_score(w)
            elif k <= 4:
                s = word_score(w, 'forward') + word_score(w, 'backward')
            else:
                continue
            scores[(i, k)] = s
    dp = [-float('inf')] * (n + 1)
    dp[0] = 0
    choice = [None] * (n + 1)
    for i in range(1, n + 1):
        for k in range(1, min(max_k, i) + 1):
            start = i - k
            if (start, k) in scores and dp[start] + scores[(start, k)] > dp[i]:
                dp[i] = dp[start] + scores[(start, k)]
                choice[i] = (start, k)
    words = []
    pos = n
    while pos > 0:
        start, k = choice[pos]
        words.append(chinese_str[start:start+k])
        pos = start
    words.reverse()
    return words

def segment_with_voting(chinese_str, times=5):
    candidates = [tuple(segment_raw(chinese_str)) for _ in range(times)]
    best = Counter(candidates).most_common(1)[0][0]
    return list(best)

def train():
    global RANDOM_AMPLITUDE
    try:
        amp = float(input("随机幅度（0~10，建议 3 以内）："))
        RANDOM_AMPLITUDE = max(0.0, min(10.0, amp)) * 0.01
    except:
        RANDOM_AMPLITUDE = 0.0

    load_seed_words('dict')
    train_folder = 'train'
    if not os.path.isdir(train_folder):
        print("train 文件夹不存在，请创建并放入 .txt 训练文本")
        return

    files = [f for f in os.listdir(train_folder) if f.endswith('.txt')]
    print(f"找到 {len(files)} 个训练文件，开始训练……")
    for fname in sorted(files):
        with open(os.path.join(train_folder, fname), 'r', encoding='utf-8') as f:
            text = f.read()
        sentences = re.split(r'(?<=[。！？…])', text)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            m = re.match(r'^(.*?)([。！？…]+)$', sent)
            body = sent if not m else m.group(1)
            for block in RE_CHINESE_BLOCK.findall(body):
                words = segment_with_voting(block)
                update_all_stats(block, words)

    model = {
        'unigram': dict(unigram_freq),
        'bigram': dict(bigram_freq),
        'words': {},
        'semi_count': dict(semi_count),
        'is_official': dict(is_official)
    }
    for w, st in word_stats.items():
        model['words'][w] = {'freq': st['freq'], 'right': dict(st['right_neighbors']), 'left': dict(st['left_neighbors'])}
    with open('model.json', 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    print(f"模型已保存到 model.json，包含 {len(word_stats)} 个多字词统计")

    official_words = [w for w, off in is_official.items() if off]
    today_str = datetime.date.today().strftime('%Y%m%d')
    dict_filename = f'dict.{today_str}.txt'
    with open(dict_filename, 'w', encoding='utf-8') as f:
        for w in sorted(official_words, key=lambda x: -word_stats[x]['freq'] if x in word_stats else 0):
            f.write(w + '\n')
    print(f"确认词库已保存到 {dict_filename}，共 {len(official_words)} 个词")

    choice = input("是否生成前端分词文件？(y/n): ").strip().lower()
    if choice == 'y':
        generate_frontend(model)

def generate_frontend(model):
    today = datetime.date.today().strftime('%Y%m%d')
    out_dir = f'output_{today}'
    os.makedirs(out_dir, exist_ok=True)

    words_export = {}
    for w, st in model['words'].items():
        if st['freq'] > 0 or is_official.get(w, False):
            words_export[w] = {'f': st['freq'], 'r': st['right'], 'l': st['left']}
    export_data = {
        'uni': model['unigram'],
        'bi': model['bigram'],
        'words': words_export,
        'official': {w: True for w, off in model['is_official'].items() if off},
        'semi': {w: c for w, c in model['semi_count'].items() if c > 0}
    }

    js_code = f'''// 自动生成的分词模型 - {today}
const WORDSEG_MODEL = {json.dumps(export_data, ensure_ascii=False)};

function splitChinese(text) {{
    if (!text) return [];
    const model = WORDSEG_MODEL;
    const uni = model.uni;
    const bi = model.bi;
    const words = model.words;
    const official = model.official || {{}};
    const semi = model.semi || {{}};
    const totalUni = Object.values(uni).reduce((a,b)=>a+b, 0);

    const OFFICIAL_BASE = 200.0;
    const baseMap = {{1:0.7, 2:1.6, 3:1.9, 4:2.5}};

    function entropy(counts) {{
        const total = Object.values(counts).reduce((a,b)=>a+b, 0);
        if (total === 0) return 0;
        let ent = 0;
        for (let c of Object.values(counts)) {{
            const p = c / total;
            ent -= p * Math.log2(p);
        }}
        return ent;
    }}

    function pmiMin(w) {{
        const L = w.length;
        if (L < 2) return 0;
        let minPmi = Infinity;
        for (let i = 0; i < L-1; i++) {{
            const pair = w.slice(i, i+2);
            const a = w[i], b = w[i+1];
            const fPair = (bi[pair] || 0) + 1;
            const fA = (uni[a] || 0) + 1;
            const fB = (uni[b] || 0) + 1;
            const pmi = Math.log2((fPair * totalUni) / (fA * fB));
            if (pmi < minPmi) minPmi = pmi;
        }}
        return minPmi;
    }}

    function freedom(w, direction) {{
        const wInfo = words[w];
        if (!wInfo) {{
            if (official[w]) return 2.5;
            return 1.0;
        }}
        const neighbors = direction === 'forward' ? wInfo.r : wInfo.l;
        if (!neighbors || Object.keys(neighbors).length === 0) {{
            if (official[w]) return 2.5;
            return 1.0;
        }}
        return entropy(neighbors) + 1.0;
    }}

    function scoreOrdinary(w, direction) {{
        if (w.length === 1) {{
            const highFreq = {{'的':1,'了':1,'在':1,'和':1,'是':1,'就':1,'也':1,'都':1,'很':1,'不':1,'把':1,'被':1,'从':1,'对':1,'与':1,'但':1,'而':1,'所':1,'且':1,'或':1,'已':1,'曾':1,'您':1}};
            return (w in highFreq) ? 0.35 : 0.2;
        }}
        let base = baseMap[w.length] || 2.0;
        let bonus = semi[w] > 0 ? 0.8 : 0.0;
        const free = freedom(w, direction);
        let coh = pmiMin(w) + 1.0;
        if (coh < 0.1) coh = 0.1;
        if (coh > 10) coh = 10;
        return (base + bonus) * free * coh;
    }}

    function officialScore(w) {{
        const freeFwd = freedom(w, 'forward');
        const freeBwd = freedom(w, 'backward');
        const ctx = Math.min(freeFwd, freeBwd);
        return (OFFICIAL_BASE * w.length - 2) * ctx;
    }}

    function segmentChunk(chunk) {{
        const n = chunk.length;
        if (n === 0) return [];
        const maxSeedLen = Object.keys(official).reduce((max, w) => Math.max(max, w.length), 4);
        const maxK = Math.max(4, maxSeedLen);
        const dp = new Array(n+1).fill(-Infinity);
        dp[0] = 0;
        const choice = new Array(n+1).fill(null);
        for (let i = 1; i <= n; i++) {{
            for (let k = 1; k <= Math.min(maxK, i); k++) {{
                const start = i - k;
                const w = chunk.slice(start, i);
                if (!/^[\\u4e00-\\u9fff]+$/.test(w)) continue;
                let s = 0;
                if (official[w]) {{
                    s = officialScore(w);
                }} else if (k <= 4) {{
                    s = scoreOrdinary(w, 'forward') + scoreOrdinary(w, 'backward');
                }} else continue;
                if (dp[start] + s > dp[i]) {{
                    dp[i] = dp[start] + s;
                    choice[i] = {{start, word: w}};
                }}
            }}
        }}
        const wordsOut = [];
        let pos = n;
        while (pos > 0) {{
            const {{start, word}} = choice[pos];
            wordsOut.unshift(word);
            pos = start;
        }}
        return wordsOut;
    }}

    function isChinese(c) {{
        return /[\\u4e00-\\u9fff]/.test(c);
    }}

    function isPunctuationOnly(str) {{
        const trimmed = str.replace(/\\s/g, '');
        if (trimmed.length === 0) return true;
        return /^[\\u2000-\\u206F\\u2E00-\\u2E7F\\u3000-\\u303F\\uFF00-\\uFFEF\\uFE30-\\uFE4F\\u0021-\\u002F\\u003A-\\u0040\\u005B-\\u0060\\u007B-\\u007E]+$/.test(trimmed);
    }}

    const result = [];
    let buf = '';
    let bufType = null;
    for (const ch of text) {{
        const chIsChinese = isChinese(ch);
        if (bufType === null) {{
            bufType = chIsChinese;
            buf = ch;
        }} else if (bufType === chIsChinese) {{
            buf += ch;
        }} else {{
            if (bufType) {{
                result.push(...segmentChunk(buf));
            }} else {{
                const spaced = buf.replace(/([^\\w\\s])/g, ' $1 ');
                const parts = spaced.split(/\\s+/).filter(s => s.length > 0);
                result.push(...parts.filter(p => !isPunctuationOnly(p)));
            }}
            buf = ch;
            bufType = chIsChinese;
        }}
    }}
    if (buf) {{
        if (bufType) {{
            result.push(...segmentChunk(buf));
        }} else {{
            const spaced = buf.replace(/([^\\w\\s])/g, ' $1 ');
            const parts = spaced.split(/\\s+/).filter(s => s.length > 0);
            result.push(...parts.filter(p => !isPunctuationOnly(p)));
        }}
    }}
    return result;
}}

if (typeof module !== 'undefined' && module.exports) {{
    module.exports = {{ splitChinese }};
}}
'''

    with open(os.path.join(out_dir, 'wordseg.js'), 'w', encoding='utf-8') as f:
        f.write(js_code)

    html_code = '''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>分词测试</title>
<script src="wordseg.js"></script>
</head>
<body>
<h2>前端分词测试</h2>
<textarea id="input" rows="4" cols="60" placeholder="输入文本..."></textarea><br>
<button onclick="run()">分词</button>
<h3>结果（JSON数组）：</h3>
<pre id="output"></pre>
<script>
function run() {
    const text = document.getElementById('input').value;
    const words = splitChinese(text);
    document.getElementById('output').textContent = JSON.stringify(words, null, 2);
    console.log('分词结果:', words);
}
window.onload = function() {
    document.getElementById('input').value = '1949年，中华人民共和国中央人民政府成立。';
    run();
};
</script>
</body>
</html>'''

    with open(os.path.join(out_dir, 'test.html'), 'w', encoding='utf-8') as f:
        f.write(html_code)

    print(f"前端文件已生成至 {out_dir}/ 文件夹")

if __name__ == '__main__':
    train()
    input("按回车键退出...")