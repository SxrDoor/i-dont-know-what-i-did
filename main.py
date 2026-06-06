# main.py —— 使用训练好的 model.json 对 data.txt 进行分词
import json
import math
import re
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ========== 配置参数（与训练保持一致） ==========
OFFICIAL_BASE = 200.0
BASE_MAP = {1: 0.7, 2: 1.6, 3: 1.9, 4: 2.5}
HIGH_FREQ_SINGLE = {'的', '了', '在', '和', '是', '就', '也', '都', '很', '不',
                    '把', '被', '从', '对', '与', '但', '而', '所', '且', '或',
                    '已', '曾', '您'}

# ========== 加载模型 ==========
def load_model(model_path='model.json'):
    with open(model_path, 'r', encoding='utf-8') as f:
        model = json.load(f)

    model['official'] = model.get('is_official', {})
    model['semi'] = model.get('semi_count', {})
    model['totalUni'] = sum(model['unigram'].values())
    return model

model = None  # 全局模型

def entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

def pmi_min(w):
    L = len(w)
    if L < 2:
        return 0.0
    uni = model['unigram']
    bi = model['bigram']
    total = model['totalUni']
    min_pmi = float('inf')
    for i in range(L - 1):
        pair = w[i:i+2]
        a, b = w[i], w[i+1]
        f_pair = bi.get(pair, 0) + 1
        f_a = uni.get(a, 0) + 1
        f_b = uni.get(b, 0) + 1
        pmi = math.log2((f_pair * total) / (f_a * f_b))
        if pmi < min_pmi:
            min_pmi = pmi
    return min_pmi

def freedom(w, direction):
    words_info = model['words'].get(w)
    official = model['official']
    if not words_info:
        return 2.5 if official.get(w) else 1.0
    neighbors = words_info['right'] if direction == 'forward' else words_info['left']
    if not neighbors or len(neighbors) == 0:
        return 2.5 if official.get(w) else 1.0
    return entropy(neighbors) + 1.0

def score_ordinary(w, direction):
    if len(w) == 1:
        return 0.35 if w in HIGH_FREQ_SINGLE else 0.2   # 可调整单字权重
    base = BASE_MAP.get(len(w), 2.0)
    bonus = 0.8 if model['semi'].get(w, 0) > 0 else 0.0
    free = freedom(w, direction)
    coh = pmi_min(w) + 1.0
    if coh < 0.1: coh = 0.1
    if coh > 10.0: coh = 10.0
    return (base + bonus) * free * coh

def official_score(w):
    free_fwd = freedom(w, 'forward')
    free_bwd = freedom(w, 'backward')
    ctx = min(free_fwd, free_bwd)
    return (OFFICIAL_BASE * len(w) - 2) * ctx

def segment_chunk(chunk):
    """对纯汉字字符串进行分词"""
    n = len(chunk)
    if n == 0:
        return []
    official = model['official']
    max_seed_len = max((len(w) for w in official if official[w]), default=4)
    max_k = max(4, max_seed_len)

    dp = [-float('inf')] * (n + 1)
    dp[0] = 0
    choice = [None] * (n + 1)

    for i in range(1, n + 1):
        for k in range(1, min(max_k, i) + 1):
            start = i - k
            w = chunk[start:i]
            if not all('\u4e00' <= c <= '\u9fff' for c in w):
                continue   # 跳过含非汉字的候选
            if official.get(w):
                s = official_score(w)
            elif k <= 4:
                s = score_ordinary(w, 'forward') + score_ordinary(w, 'backward')
            else:
                continue
            if dp[start] + s > dp[i]:
                dp[i] = dp[start] + s
                choice[i] = start

    # 回溯
    words = []
    pos = n
    while pos > 0:
        start = choice[pos]
        words.append(chunk[start:pos])
        pos = start
    words.reverse()
    return words

def split_text(text):
    result = []
    buf = ''
    buf_is_chinese = None

    for ch in text:
        ch_is_chinese = '\u4e00' <= ch <= '\u9fff'
        if buf_is_chinese is None:
            buf_is_chinese = ch_is_chinese
            buf = ch
        elif buf_is_chinese == ch_is_chinese:
            buf += ch
        else:
            if buf_is_chinese:
                result.extend(segment_chunk(buf))
            else:
                # 把非汉字块中的标点符号和相邻字符切开
                buf_spaced = re.sub(r'([^\w\s])', r' \1 ', buf)
                parts = buf_spaced.split()
                result.extend(parts)
            buf = ch
            buf_is_chinese = ch_is_chinese

    if buf:
        if buf_is_chinese:
            result.extend(segment_chunk(buf))
        else:
            buf_spaced = re.sub(r'([^\w\s])', r' \1 ', buf)
            parts = buf_spaced.split()
            result.extend(parts)

    return result

# ========== 主程序 ==========
def main():
    global model
    model = load_model()

    # 读取 data.txt，如果没有则从命令行参数读取
    input_file = 'data.txt'
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        words = split_text(line)
        print('/'.join(words))   # 分隔输出

if __name__ == '__main__':
    main()
    input("\n按回车键退出...")