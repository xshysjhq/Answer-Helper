# -*- coding: utf-8 -*-
"""
实时屏幕 OCR 答题助手 (Tkinter 版)

流程:
1. 启动后打开主配置窗口: 框选题目区域 / 选择题库(json) / 配置大模型 API-Key
2. 点击"开始识别"后主窗口隐藏, 变为置顶小悬浮窗
3. OCR 识别题目 -> 后端自动在所选题库中模糊匹配 -> 命中则直接显示答案
4. 题库未命中 -> 若配置了大模型, 自动交给大模型解答

热键: Ctrl+Alt+O 暂停/继续 | Ctrl+Alt+R 重选区域 | Ctrl+Alt+Q 退出
"""
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
import os
import sys
import re
import time
import json
import glob
import difflib
import urllib.request
import urllib.error
import tkinter as tk
from tkinter import ttk, messagebox
from mss import mss
import keyboard
import threading
import queue
from collections import Counter

# 打包成 windowed exe 后没有控制台, sys.stdout/stderr 为 None, print 会报错
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w', encoding='utf-8')

# --- Configuration ---
CONFIG_FILE = 'ocr_config.json'
OCR_INTERVAL_SECONDS = 0.5

# --- Hotkeys ---
TOGGLE_OCR_HOTKEY = 'ctrl+alt+o'
RESELECT_HOTKEY = 'ctrl+alt+r'
QUIT_HOTKEY = 'ctrl+alt+q'

# --- Matching / LLM ---
MATCH_THRESHOLD = 0.55   # 题库模糊匹配置信度阈值
TOP_N = 3                # 最多使用的匹配题目数
LLM_TIMEOUT = 60         # 大模型请求超时(秒)

# --- 小窗尺寸 ---
MINI_W, MINI_H = 420, 280

DEFAULT_SETTINGS = {
    "region": None,
    "bank_file": "",
    "bank_name": "all",   # all 或题库内的 bank_name 分类
    "bank_enabled": True, # 是否启用题库匹配 (关闭后直接问 AI, 更快)
    "llm": {
        "enabled": False,
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "",
        "model": "deepseek-chat",
    },
}

# --- Global variables ---
settings = dict(DEFAULT_SETTINGS)
capture_region_coords = None
ocr_instance = None
running = True
ocr_active = False
last_cleaned_text = ""
last_processed_image_hash = ""
last_query_fp = ""      # 上一次已处理的题目指纹, 防止重复请求大模型
query_seq = 0           # 识别题目序号, 防止异步结果乱序覆盖

banks = {}              # {文件名: {"count": n, "bank_names": {名称: 数量}}}
_bank_index_cache = None
bank_lock = threading.Lock()

ui_queue = queue.Queue()    # 后台线程 -> UI (识别/结果)
ui_events = queue.Queue()   # 热键 -> UI 主线程


# ==========================
# 配置管理
# ==========================
def load_config():
    global capture_region_coords
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            for key in ("bank_file", "bank_name", "bank_enabled"):
                if config.get(key) is not None:
                    settings[key] = config[key]
            if isinstance(config.get('region'), dict):
                settings['region'] = config['region']
                capture_region_coords = config['region']
            llm_cfg = config.get('llm')
            if isinstance(llm_cfg, dict):
                for k in DEFAULT_SETTINGS['llm']:
                    if llm_cfg.get(k) is not None:
                        settings['llm'][k] = llm_cfg[k]
            print(f"已加载配置: 区域={capture_region_coords}, 题库={settings['bank_file']}, "
                  f"AI={'开' if settings['llm']['enabled'] else '关'}")
        except Exception as e:
            print(f"加载配置失败: {e}")


def save_config():
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"保存配置失败: {e}")


# ==========================
# OCR 工具
# ==========================
def clean_ocr_text(text):
    if not text: return ""
    return re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', text)


def fingerprint(text):
    """题目指纹: 去掉所有符号空格, 只留汉字/字母/数字并转小写"""
    return re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', text or '').lower()


def preprocess_image(img_np):
    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    except Exception:
        return None


def images_are_similar(img1, img2):
    """像素差异比较, 变化小于1%认为画面静止"""
    if img1 is None or img2 is None: return False
    if img1.shape != img2.shape: return False
    try:
        diff = cv2.absdiff(img1, img2)
        non_zero_count = np.count_nonzero(diff)
        return (non_zero_count / img1.size) < 0.01
    except Exception:
        return False


# ==========================
# 题库管理
# ==========================
def discover_banks():
    """扫描目录下所有可作为题库的 json 文件"""
    global banks
    found = {}
    for f in glob.glob('*.json'):
        if f == CONFIG_FILE:
            continue
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
        except Exception:
            continue
        if isinstance(data, list) and data and all(isinstance(x, dict) for x in data[:100]):
            if any('text' in x for x in data[:100]):
                names = {}
                for q in data:
                    n = q.get('bank_name') or '未命名'
                    names[n] = names.get(n, 0) + 1
                found[f] = {"count": len(data), "bank_names": names}
    banks = found
    return found


def get_bank_index():
    """获取当前所选题库(文件+科目)的指纹索引, 带缓存"""
    global _bank_index_cache
    with bank_lock:
        key = (settings.get('bank_file'), settings.get('bank_name'))
        if not key[0]:
            return []
        if _bank_index_cache and _bank_index_cache[0] == key:
            return _bank_index_cache[1]
        try:
            with open(key[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"读取题库失败 {key[0]}: {e}")
            return []
        if key[1] and key[1] != 'all':
            data = [q for q in data if isinstance(q, dict) and (q.get('bank_name') or '') == key[1]]
        index = []
        for q in data:
            if not isinstance(q, dict):
                continue
            fp = fingerprint((q.get('text') or '') + ''.join(q.get('options') or []))
            if fp:
                # 预计算字符计数器, 用于覆盖率预筛(不受长度不对称惩罚)
                index.append((fp, Counter(fp), q))
        _bank_index_cache = (key, index)
        print(f"题库索引已加载: {key[0]} [{key[1]}] 共 {len(index)} 题")
        return index


def invalidate_bank_cache():
    global _bank_index_cache
    with bank_lock:
        _bank_index_cache = None


def partial_score(query, target, min_sub=6):
    """
    计算相似度, 兼容长度不对称的情况 (OCR 只截到题干时, 题库指纹含选项会更长):
    - 互为子串 -> 0.95
    - 否则将短串对齐到长串中相似度最高的窗口再比较
    """
    if not query or not target:
        return 0.0
    if min(len(query), len(target)) >= min_sub and (query in target or target in query):
        return 0.95
    short, long_ = (query, target) if len(query) <= len(target) else (target, query)
    if len(short) == len(long_):
        return difflib.SequenceMatcher(None, short, long_).ratio()
    # 用最长公共块定位对齐窗口, 避免滑窗全扫描
    matcher = difflib.SequenceMatcher(None, short, long_)
    blk = matcher.find_longest_match(0, len(short), 0, len(long_))
    if blk.size == 0:
        return 0.0
    start = max(0, blk.b - blk.a)
    window = long_[start:start + len(short)]
    if len(window) < len(short):
        window = long_[len(long_) - len(short):]
    return difflib.SequenceMatcher(None, short, window).ratio()


def fuzzy_search(query_fp, index, threshold=MATCH_THRESHOLD, top_n=TOP_N):
    """在题库索引中模糊搜索, 返回 [(score, question), ...] 降序"""
    scored = []
    if not query_fp:
        return scored
    q_counter = Counter(query_fp)
    q_len = len(query_fp)
    for fp, fp_counter, q in index:
        # 覆盖率预筛: 识别文本的字符有多少出现在该题目中 (廉价, 且不受长度差惩罚)
        overlap = sum(min(c, fp_counter.get(ch, 0)) for ch, c in q_counter.items())
        if overlap / q_len < threshold:
            continue
        score = partial_score(query_fp, fp)
        if score >= threshold:
            scored.append((score, q))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]


def serialize_question(q):
    return {
        "bank_name": q.get('bank_name'),
        "type": q.get('type'),
        "text": q.get('text'),
        "options": q.get('options') or [],
        "correct_answer": q.get('correct_answer'),
        "analysis": q.get('analysis'),
    }


# ==========================
# 大模型对接 (OpenAI 兼容接口)
# ==========================
LOG_FILE = 'log.txt'


def log_line(msg):
    """追加一行日志到 log.txt, 方便排查问题"""
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def ask_llm(question_text):
    cfg = settings['llm']
    url = cfg['base_url'].rstrip('/') + '/chat/completions'
    system_prompt = (
        "你是答题助手，题目来自OCR识别、可能有错字或乱序。理解题意后严格按以下格式输出，"
        "禁止重复题目、禁止寒暄废话：\n"
        "第一行： 答案：X  （选择题只写选项字母，多选用、分隔）\n"
        "第二行： 不超过50字的解析"
    )
    body = {
        "model": cfg['model'],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_text},
        ],
        "temperature": 0.8,
        "max_tokens": 300,   # 限制输出长度, 生成耗时与输出token成正比
    }
    # kimi-k3 等推理模型: 只允许 temperature=1, 且默认深度思考很慢, 答题场景用 low 提速
    if 'kimi-k3' in (cfg['model'] or ''):
        body['temperature'] = 1
        body['reasoning_effort'] = 'low'
        body['max_tokens'] = 1000   # 推理模型的思考过程也占输出额度, 放宽以免正文被截断
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode('utf-8'),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        },
        method="POST",
    )
    t0 = time.time()
    key_masked = (cfg['api_key'][:6] + '***' + cfg['api_key'][-4:]) if len(cfg['api_key'] or '') > 12 else '***'
    log_line(f"[LLM] 请求 {url} model={cfg['model']} key={key_masked} 问题: {question_text[:60]}")
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        # 解析服务端返回的具体原因 (余额不足/模型不存在/Key无效等)
        detail = ''
        try:
            err = json.loads(e.read().decode('utf-8'))
            detail = (err.get('error') or {}).get('message') or ''
        except Exception:
            pass
        log_line(f"[LLM] 失败 HTTP {e.code}: {detail or e.reason} (耗时 {time.time() - t0:.1f}s)")
        raise RuntimeError(f"HTTP {e.code}: {detail or e.reason}")
    except Exception as e:
        log_line(f"[LLM] 失败 {type(e).__name__}: {e} (耗时 {time.time() - t0:.1f}s)")
        raise
    answer = (data["choices"][0]["message"].get("content") or '').strip()
    log_line(f"[LLM] 成功 耗时{time.time() - t0:.1f}s 返回{len(answer)}字: {answer[:80]}")
    return answer


# ==========================
# 答题核心流程: 题库优先, 未命中走大模型
# ==========================
def process_question(raw_text, seq=None, dedup=True):
    """返回结果 dict (供 UI 显示)"""
    global last_query_fp
    q_fp = fingerprint(raw_text)
    if len(q_fp) < 2:
        return None
    if dedup and q_fp == last_query_fp:
        return None
    last_query_fp = q_fp

    result = {"query": raw_text, "seq": seq}

    # 1. 题库匹配 (主界面可关闭, 关闭后直接问 AI)
    if settings.get('bank_enabled', True):
        index = get_bank_index()
        if index:
            matches = fuzzy_search(q_fp, index)
            if matches:
                result.update({
                    "source": "bank",
                    "score": round(matches[0][0], 3),
                    "matches": [
                        {"score": round(s, 3), "question": serialize_question(q)}
                        for s, q in matches
                    ],
                })
                return result

    # 2. 题库未命中/已关闭 -> 大模型
    llm = settings['llm']
    if llm.get('enabled') and llm.get('api_key'):
        if seq is not None:
            hint = "题库未命中，正在请求大模型..." if settings.get('bank_enabled', True) else "正在请求大模型..."
            ui_queue.put(("stage", seq, {"query": raw_text, "message": hint}))
        try:
            answer = ask_llm(raw_text)
            result.update({"source": "llm", "answer": answer})
        except Exception as e:
            print(f"大模型请求失败: {e}")
            result.update({"source": "error", "error": str(e)})
        return result

    result.update({"source": "none", "message": "未命中题库，且未启用大模型" if settings.get('bank_enabled', True) else "题库匹配与 AI 均未启用"})
    return result


# ==========================
# 后台线程: 题目处理 / OCR 循环
# ==========================
def question_worker():
    while running:
        try:
            seq, text = question_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            result = process_question(text, seq=seq)
            if result:
                ui_queue.put(("result", seq, result))
        except Exception as e:
            print(f"题目处理错误: {e}")


question_queue = queue.Queue()


def perform_ocr_cycle(sct):
    global last_cleaned_text, last_processed_image_hash, query_seq

    if not capture_region_coords or not ocr_instance: return

    try:
        sct_img = sct.grab(capture_region_coords)
        img_np = np.array(sct_img)

        processed_image = preprocess_image(img_np)
        if processed_image is None: return

        if images_are_similar(processed_image, last_processed_image_hash):
            return  # 画面无变化，跳过 OCR

        last_processed_image_hash = processed_image.copy()

        result, _ = ocr_instance(processed_image)
        if not result: return

        # RapidOCR 返回 [[坐标, 文本, 置信度], ...]
        raw_text = " ".join((line[1] or '').strip() for line in result).strip()
        cleaned_text = clean_ocr_text(raw_text)

        if cleaned_text and len(cleaned_text) > 1 and cleaned_text != last_cleaned_text:
            timestamp = time.strftime('%H:%M:%S')
            print(f"[{timestamp}] 识别到新内容: {raw_text}")
            last_cleaned_text = cleaned_text
            query_seq += 1
            seq = query_seq
            ui_queue.put(("ocr", seq, {"text": raw_text}))
            question_queue.put((seq, raw_text))

    except Exception as e:
        print(f"OCR 循环错误: {e}")


def ocr_loop():
    with mss() as sct:
        while running:
            start_time = time.time()
            if ocr_active:
                perform_ocr_cycle(sct)
            elapsed = time.time() - start_time
            time.sleep(max(0.05, OCR_INTERVAL_SECONDS - elapsed))


# ==========================
# OCR 状态控制
# ==========================
def set_ocr_active(target):
    global ocr_active, last_cleaned_text, last_processed_image_hash, last_query_fp
    if target == ocr_active:
        return
    ocr_active = target
    if ocr_active:
        last_cleaned_text = "RESET"  # 强制刷新
        last_processed_image_hash = None
        last_query_fp = ""
        print(f">>> OCR 已开启 (间隔: {OCR_INTERVAL_SECONDS}s)")
    else:
        print("<<< OCR 已暂停")
    ui_events.put(("ocr_state", ocr_active))


# ==========================
# 区域选择 (全屏半透明覆盖框选)
# ==========================
def select_region_toplevel():
    region_holder = {}
    top = tk.Toplevel()
    top.attributes("-fullscreen", True)
    top.attributes("-alpha", 0.3)
    top.attributes("-topmost", True)

    canvas = tk.Canvas(top, cursor="cross", bg='gray')
    canvas.pack(fill=tk.BOTH, expand=tk.YES)
    canvas.create_text(top.winfo_screenwidth() // 2, 40,
                       text="拖动鼠标框选题目所在区域（Esc 取消）",
                       font=("Microsoft YaHei", 16, "bold"), fill="red")

    state = {"sx": None, "rect": None}

    def on_down(e):
        state["sx"], state["sy"] = e.x, e.y
        if state["rect"]: canvas.delete(state["rect"])

    def on_drag(e):
        if state["rect"]: canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(state["sx"], state["sy"], e.x, e.y, outline='red', width=2)

    def on_up(e):
        if state["sx"] is not None:
            x1, y1 = min(state["sx"], e.x), min(state["sy"], e.y)
            x2, y2 = max(state["sx"], e.x), max(state["sy"], e.y)
            if x2 - x1 > 10 and y2 - y1 > 10:
                region_holder["region"] = {'top': y1, 'left': x1, 'width': x2 - x1, 'height': y2 - y1}
        top.destroy()

    canvas.bind("<ButtonPress-1>", on_down)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_up)
    top.bind("<Escape>", lambda e: top.destroy())
    top.focus_force()
    top.wait_window(top)
    return region_holder.get("region")


# ==========================
# Tkinter GUI
# ==========================
class App:
    def __init__(self, root):
        self.root = root
        root.title("答题助手 · 配置")
        root.geometry("480x780")
        root.minsize(460, 560)

        self.display_seq = 0          # 当前小窗显示的结果序号(防止乱序覆盖)
        self.file_map = []            # combobox 显示名 -> 文件名
        self.name_map = []            # combobox 显示名 -> 科目名

        self._build_main()
        self._build_mini()
        self.root.after(120, self._poll)
        root.protocol("WM_DELETE_WINDOW", self.quit_app)

    # ---------- 主配置窗口 ----------
    def _build_main(self):
        pad = {"padx": 16, "pady": 4}

        # 可滚动容器: 屏幕较矮时也能滚到底部看到全部设置
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True)
        self.main_canvas = tk.Canvas(outer, highlightthickness=0, bg=self.root.cget("bg"))
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.main_canvas.pack(side="left", fill="both", expand=True)
        frm = ttk.Frame(self.main_canvas, padding=10)
        _win = self.main_canvas.create_window((0, 0), window=frm, anchor="nw")
        frm.bind("<Configure>", lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))
        self.main_canvas.bind("<Configure>", lambda e: self.main_canvas.itemconfigure(_win, width=e.width))
        self.main_canvas.bind_all("<MouseWheel>",
                                  lambda e: self.main_canvas.yview_scroll(int(-e.delta / 120), "units"))

        ttk.Label(frm, text="📝 答题助手", font=("Microsoft YaHei", 15, "bold")).pack(anchor="w")
        ttk.Label(frm, text="OCR 识别 → 题库匹配 → AI 兜底", foreground="#6b7280").pack(anchor="w", pady=(0, 10))

        # 1. 区域
        box1 = ttk.LabelFrame(frm, text=" 📷 题目识别区域 ", padding=10)
        box1.pack(fill="x", pady=5)
        self.region_var = tk.StringVar(value="未设置，请先框选")
        ttk.Label(box1, textvariable=self.region_var, foreground="#374151").pack(anchor="w")
        ttk.Button(box1, text="框选题目所在区域", command=self.on_select_region).pack(anchor="w", pady=(6, 0))

        # 2. 题库
        box2 = ttk.LabelFrame(frm, text=" 📚 题库选择（JSON） ", padding=10)
        box2.pack(fill="x", pady=5)
        self.bank_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(box2, text="启用题库匹配（取消则跳过题库、直接问 AI，更快）",
                        variable=self.bank_enabled).pack(anchor="w")
        ttk.Label(box2, text="题库文件", foreground="#6b7280").pack(anchor="w")
        self.bank_file_cb = ttk.Combobox(box2, state="readonly")
        self.bank_file_cb.pack(fill="x", **pad)
        self.bank_file_cb.bind("<<ComboboxSelected>>", self.on_bank_file_change)
        ttk.Label(box2, text="科目 / 分类", foreground="#6b7280").pack(anchor="w")
        self.bank_name_cb = ttk.Combobox(box2, state="readonly")
        self.bank_name_cb.pack(fill="x", **pad)
        self.bank_name_cb.bind("<<ComboboxSelected>>", lambda e: self.save_selection())
        ttk.Label(box2, text="将题库 json 放到本脚本同目录即可被自动识别",
                  foreground="#9ca3af", font=("Microsoft YaHei", 8)).pack(anchor="w", pady=(4, 0))

        # 3. AI 大模型
        box3 = ttk.LabelFrame(frm, text=" 🤖 AI 大模型（题库未命中时） ", padding=10)
        box3.pack(fill="x", pady=5)
        self.llm_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(box3, text="启用 AI 解答兜底", variable=self.llm_enabled).pack(anchor="w")
        ttk.Label(box3, text="API 地址（OpenAI 兼容）", foreground="#6b7280").pack(anchor="w")
        self.base_url_var = tk.StringVar()
        ttk.Entry(box3, textvariable=self.base_url_var).pack(fill="x", **pad)
        ttk.Label(box3, text="API Key", foreground="#6b7280").pack(anchor="w")
        self.api_key_var = tk.StringVar()
        ttk.Entry(box3, textvariable=self.api_key_var, show="*").pack(fill="x", **pad)
        ttk.Label(box3, text="模型名称", foreground="#6b7280").pack(anchor="w")
        self.model_var = tk.StringVar()
        ttk.Entry(box3, textvariable=self.model_var).pack(fill="x", **pad)

        # 测试连接
        test_row = ttk.Frame(box3)
        test_row.pack(fill="x", pady=(8, 0))
        self.test_btn = ttk.Button(test_row, text="🔌 测试连接", command=self.on_test_llm)
        self.test_btn.pack(side="left")
        self.test_status = tk.StringVar(value="填写后可先测试，结果记录在 log.txt")
        ttk.Label(test_row, textvariable=self.test_status, foreground="#6b7280").pack(side="left", padx=8)

        # 开始按钮
        self.start_btn = ttk.Button(frm, text="▶  开 始 识 别", command=self.on_start)
        self.start_btn.pack(fill="x", pady=(12, 4))
        ttk.Label(frm, text=f"热键：{TOGGLE_OCR_HOTKEY.upper()} 暂停/继续 · {RESELECT_HOTKEY.upper()} 重选区域 · {QUIT_HOTKEY.upper()} 退出\n"
                            f"开始后主窗口隐藏为置顶小窗，点小窗 ⚙ 可返回此界面",
                  foreground="#9ca3af", justify="left").pack(anchor="w", pady=2)

        self._load_settings_into_ui()

    def _load_settings_into_ui(self):
        # 区域
        r = settings.get('region')
        self.region_var.set(
            f"左上角({r['left']}, {r['top']})  尺寸 {r['width']} × {r['height']}" if r else "未设置，请先框选")

        # 题库
        self.bank_enabled.set(bool(settings.get('bank_enabled', True)))
        discover_banks()
        files = sorted(banks.keys())
        self.file_map = files
        disp_files = [f"{f}（{banks[f]['count']}题）" for f in files]
        self.bank_file_cb["values"] = disp_files
        if files:
            cur = settings['bank_file'] if settings['bank_file'] in files else files[0]
            settings['bank_file'] = cur
            self.bank_file_cb.set(f"{cur}（{banks[cur]['count']}题）")
            self._refresh_bank_names()
        else:
            self.bank_file_cb.set("（未发现题库 json）")
            self.bank_name_cb["values"] = ["全部"]
            self.bank_name_cb.set("全部")

        # AI
        llm = settings['llm']
        self.llm_enabled.set(bool(llm['enabled']))
        self.base_url_var.set(llm['base_url'])
        self.api_key_var.set(llm['api_key'])
        self.model_var.set(llm['model'])

    def _refresh_bank_names(self):
        f = settings['bank_file']
        bank = banks.get(f)
        names = sorted((bank or {}).get('bank_names', {}).keys())
        self.name_map = ['all'] + names
        disp = ["全部科目"] + [f"{n}（{bank['bank_names'][n]}题）" for n in names]
        self.bank_name_cb["values"] = disp
        cur = settings['bank_name']
        if cur in names:
            idx = names.index(cur) + 1
        else:
            idx = 0
            settings['bank_name'] = 'all'
        self.bank_name_cb.current(idx)

    # ---------- 小悬浮窗 ----------
    def _build_mini(self):
        self.mini = tk.Toplevel(self.root)
        self.mini.overrideredirect(True)
        self.mini.attributes("-topmost", True)
        self.mini.configure(bg="white")
        sw, sh = self.mini.winfo_screenwidth(), self.mini.winfo_screenheight()
        self.mini.geometry(f"{MINI_W}x{MINI_H}+{sw - MINI_W - 30}+{sh - MINI_H - 80}")

        # 顶栏 (可拖动)
        bar = tk.Frame(self.mini, bg="#1f2430", height=32)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        self.mini_status = tk.Label(bar, text="○ 已暂停", fg="#fbbf24", bg="#1f2430",
                                    font=("Microsoft YaHei", 9, "bold"))
        self.mini_status.pack(side="left", padx=10)
        self.mini_src = tk.Label(bar, text="等待识别...", fg="#e5e7eb", bg="#1f2430",
                                 font=("Microsoft YaHei", 9))
        self.mini_src.pack(side="left", padx=6)

        btn_style = {"fg": "#e5e7eb", "bg": "#1f2430", "bd": 0, "width": 3,
                     "font": ("Microsoft YaHei", 10), "activebackground": "#374151"}
        self.mini_toggle_btn = tk.Button(bar, text="⏸", command=lambda: self.ui_toggle_ocr(), **btn_style)
        self.mini_toggle_btn.pack(side="right", padx=2)
        tk.Button(bar, text="✕", command=self.quit_app, **btn_style).pack(side="right", padx=2)
        tk.Button(bar, text="⚙", command=self.show_main, **btn_style).pack(side="right", padx=2)

        # 拖动窗口
        drag = {"x": None, "y": None}
        bar.bind("<Button-1>", lambda e: drag.update(x=e.x, y=e.y))
        bar.bind("<B1-Motion>", lambda e: self.mini.geometry(
            f"+{self.mini.winfo_x() + e.x - drag['x']}+{self.mini.winfo_y() + e.y - drag['y']}") if drag["x"] is not None else None)

        # 答案区
        body = tk.Frame(self.mini, bg="white")
        body.pack(fill="both", expand=True, padx=1, pady=1)
        self.answer_text = tk.Text(body, wrap="char", bd=0, highlightthickness=0,
                                   font=("Microsoft YaHei", 10), bg="#fafafa", cursor="arrow")
        sb = ttk.Scrollbar(body, command=self.answer_text.yview)
        self.answer_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.answer_text.pack(fill="both", expand=True)
        # 滚轮滚动
        self.answer_text.bind("<Enter>", lambda e: self._bind_mousewheel())
        self.answer_text.bind("<Leave>", lambda e: self._unbind_mousewheel())

        for tag, cfg in {
            "head_bank": {"foreground": "#16a34a", "font": ("Microsoft YaHei", 10, "bold")},
            "head_llm": {"foreground": "#7c3aed", "font": ("Microsoft YaHei", 10, "bold")},
            "head_err": {"foreground": "#dc2626", "font": ("Microsoft YaHei", 10, "bold")},
            "head_none": {"foreground": "#6b7280", "font": ("Microsoft YaHei", 10, "bold")},
            "q": {"foreground": "#4b5563"},
            "ans": {"foreground": "#16a34a", "font": ("Microsoft YaHei", 13, "bold")},
            "analysis": {"foreground": "#6b7280", "font": ("Microsoft YaHei", 9)},
            "dim": {"foreground": "#9ca3af", "font": ("Microsoft YaHei", 9)},
        }.items():
            self.answer_text.tag_configure(tag, **cfg)

        self.answer_text.insert("1.0", "开启识别后，题目答案将显示在这里。\n\n"
                                       "⚙ 返回配置界面    ⏸ 暂停/继续    ✕ 退出\n"
                                       "顶栏可拖动窗口位置。", "dim")
        self.answer_text.config(state="disabled")
        self.mini.withdraw()

    def _bind_mousewheel(self):
        self.answer_text.bind_all("<MouseWheel>", lambda e: self.answer_text.yview_scroll(-e.delta // 40, "units"))

    def _unbind_mousewheel(self):
        self.answer_text.unbind_all("<MouseWheel>")

    # ---------- 结果渲染 ----------
    def _insert(self, text, tag=None):
        self.answer_text.insert("end", text, tag or ())

    def render_result(self, res):
        self.display_seq = max(self.display_seq, res.get("seq") or 0)
        self.answer_text.config(state="normal")
        self.answer_text.delete("1.0", "end")

        ts = time.strftime('%H:%M:%S')
        if res.get("source") == "bank" and res.get("matches"):
            m0 = res["matches"][0]
            q = m0["question"]
            self._insert(f"✔ 题库命中 {round(res['score'] * 100)}%", "head_bank")
            self._insert(f"   {ts}\n", "dim")
            self._insert(f"[{q.get('bank_name') or ''}] ", "dim")
            self._insert(f"{q.get('type') or ''}\n", "dim")
            self._insert(f"{q.get('text')}\n", "q")
            if q.get("options"):
                for i, opt in enumerate(q["options"]):
                    s = str(opt or "")
                    prefixed = bool(re.match(r'^\s*[A-Za-z][\.、．]', s))
                    self._insert(("  " + s if prefixed else f"  {chr(65 + i)}. {s}") + "\n", "q")
            self._insert("\n答案：", "q")
            self._insert(f"{q.get('correct_answer')}\n", "ans")
            if q.get("analysis") and str(q["analysis"]) not in ("无", "【无】", "无："):
                self._insert(f"\n解析：{q['analysis']}\n", "analysis")
            for m in res["matches"][1:]:
                mm = m["question"]
                self._insert(f"\n— 相似 {round(m['score'] * 100)}%：{str(mm.get('text'))[:30]}… 答案：{mm.get('correct_answer')}", "dim")
        elif res.get("source") == "llm":
            head = "🤖 AI 解答" + ("（题库未命中）" if settings.get('bank_enabled', True) else "")
            self._insert(head, "head_llm")
            self._insert(f"   {ts}\n", "dim")
            self._insert(f"{res.get('query')}\n\n", "q")
            self._insert(f"{res.get('answer')}\n", "q")
        elif res.get("source") == "error":
            self._insert("✖ AI 请求失败", "head_err")
            self._insert(f"   {ts}\n", "dim")
            self._insert(f"{res.get('error')}\n\n请检查 API 地址 / Key / 模型名或网络。", "q")
        else:
            self._insert("✖ 未找到答案", "head_none")
            self._insert(f"   {ts}\n", "dim")
            self._insert(f"{res.get('query')}\n\n{res.get('message') or '未找到答案'}", "q")

        self.answer_text.config(state="disabled")
        self.answer_text.yview_moveto(0)

        src_map = {"bank": "题库命中", "llm": "AI 解答", "error": "AI 失败", "none": "未命中"}
        self.mini_src.config(text=f"{src_map.get(res.get('source'), '')} · {ts}")

    def render_ocr(self, seq, text):
        if seq < self.display_seq:
            return
        self.answer_text.config(state="normal")
        self.answer_text.delete("1.0", "end")
        hint = "⏳ 识别中，正在匹配题库..." if settings.get('bank_enabled', True) else "⏳ 识别中，正在请求 AI..."
        self._insert(hint + "\n\n", "head_none")
        self._insert(text, "q")
        self.answer_text.config(state="disabled")
        self.mini_src.config(text="识别中 · " + time.strftime('%H:%M:%S'))

    def render_stage(self, seq, payload):
        if seq < self.display_seq:
            return
        self.answer_text.config(state="normal")
        self.answer_text.delete("1.0", "end")
        self._insert((payload.get("message") or "处理中...") + "\n\n", "head_llm")
        self._insert(payload.get("query", ""), "q")
        self.answer_text.config(state="disabled")

    # ---------- 事件/消息轮询 ----------
    def _poll(self):
        # 后台结果
        try:
            while True:
                kind, seq, payload = ui_queue.get_nowait()
                if kind == "ocr":
                    self.render_ocr(seq, payload.get("text", ""))
                elif kind == "stage":
                    self.render_stage(seq, payload)
                elif kind == "result":
                    self.render_result(payload)
        except queue.Empty:
            pass
        # 热键事件
        try:
            while True:
                ev = ui_events.get_nowait()
                if ev[0] == "toggle":
                    self.ui_toggle_ocr()
                elif ev[0] == "reselect":
                    self.ui_reselect()
                elif ev[0] == "quit":
                    self.quit_app()
                elif ev[0] == "ocr_state":
                    self._update_ocr_state(ev[1])
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    def _update_ocr_state(self, active):
        self.mini_status.config(text="● 识别中" if active else "○ 已暂停", fg="#4ade80" if active else "#fbbf24")
        self.mini_toggle_btn.config(text="⏸" if active else "▶")

    # ---------- 交互 ----------
    def on_select_region(self, from_hotkey=False):
        was_active = ocr_active
        set_ocr_active(False)
        region = select_region_toplevel()
        if region:
            settings['region'] = region
            global capture_region_coords
            capture_region_coords = region
            save_config()
            r = settings['region']
            self.region_var.set(f"左上角({r['left']}, {r['top']})  尺寸 {r['width']} × {r['height']}")
            if was_active:
                set_ocr_active(True)
        elif was_active:
            set_ocr_active(True)

    def on_bank_file_change(self, _=None):
        idx = self.bank_file_cb.current()
        if 0 <= idx < len(self.file_map):
            settings['bank_file'] = self.file_map[idx]
            settings['bank_name'] = 'all'
            invalidate_bank_cache()
            self._refresh_bank_names()
            self.save_selection()

    def save_selection(self):
        idx = self.bank_name_cb.current()
        if 0 <= idx < len(self.name_map):
            settings['bank_name'] = self.name_map[idx]
        invalidate_bank_cache()
        save_config()

    def collect_settings(self):
        settings['bank_enabled'] = bool(self.bank_enabled.get())
        settings['llm']['enabled'] = bool(self.llm_enabled.get())
        settings['llm']['base_url'] = self.base_url_var.get().strip()
        settings['llm']['api_key'] = self.api_key_var.get().strip()
        settings['llm']['model'] = self.model_var.get().strip()
        invalidate_bank_cache()
        save_config()

    # ---------- 大模型连接测试 ----------
    def on_test_llm(self):
        self.collect_settings()
        llm = settings['llm']
        if not llm['base_url'] or not llm['model']:
            messagebox.showwarning("缺少配置", "请先填写 API 地址和模型名称。", parent=self.root)
            return
        if not llm['api_key']:
            messagebox.showwarning("缺少配置", "请先填写 API Key。", parent=self.root)
            return

        self.test_btn.state(["disabled"])
        self.test_status.set("⏳ 测试中，请稍候...")
        log_line(f"[TEST] 开始测试 {llm['base_url']} model={llm['model']}")

        def worker():
            try:
                answer = ask_llm("连接测试：请直接回复“测试成功”。")
                self.root.after(0, lambda: self._test_done(True, answer))
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self.root.after(0, lambda: self._test_done(False, msg))

        threading.Thread(target=worker, daemon=True).start()

    def _test_done(self, ok, msg):
        self.test_btn.state(["!disabled"])
        if ok:
            log_line(f"[TEST] 成功: {msg[:100]}")
            self.test_status.set("✅ 测试成功")
            messagebox.showinfo("测试成功", f"大模型连接正常。\n\n返回内容：\n{msg[:200]}\n\n详情见 log.txt",
                                parent=self.root)
        else:
            log_line(f"[TEST] 失败: {msg}")
            self.test_status.set("❌ 测试失败")
            messagebox.showerror("测试失败", f"{msg}\n\n常见原因：\n"
                                "· Key 无效 / 未充值（kimi-k3 需真实余额，代金券不可用）\n"
                                "· 模型名称拼写错误\n"
                                "· 超时（深度思考模型较慢）\n\n详情见 log.txt",
                                parent=self.root)

    def on_start(self):
        self.collect_settings()
        if not settings.get('region'):
            messagebox.showwarning("缺少区域", "请先框选题目所在区域。", parent=self.root)
            return
        if not settings.get('bank_enabled', True) and not settings['llm']['enabled']:
            messagebox.showwarning("无可用来源", "题库匹配和 AI 兜底都未启用，无法作答。", parent=self.root)
            return
        if settings['llm']['enabled'] and not settings['llm']['api_key']:
            messagebox.showwarning("缺少 API Key", "已启用 AI 兜底，但 API Key 为空。", parent=self.root)
            return
        self.display_seq = 0
        self.root.withdraw()
        self.mini.deiconify()
        self.mini.attributes("-topmost", True)
        set_ocr_active(True)

    def show_main(self):
        set_ocr_active(False)
        self.mini.withdraw()
        discover_banks()
        self._load_settings_into_ui()
        self.root.deiconify()

    def ui_toggle_ocr(self):
        # 仅在小悬浮窗工作状态时允许热键/按钮切换
        if self.mini.state() != "normal":
            return
        set_ocr_active(not ocr_active)

    def ui_reselect(self):
        # 仅在小窗工作状态下有意义; 主界面时也可直接框选
        self.on_select_region(from_hotkey=True)

    def quit_app(self):
        global running
        try:
            self.collect_settings()   # 退出前保存界面上的大模型等配置
        except Exception:
            pass
        running = False
        self.root.destroy()


# --- Hotkey Callbacks (只投递事件, 由 UI 主线程处理) ---
def hotkey_toggle():
    ui_events.put(("toggle",))


def hotkey_reselect():
    ui_events.put(("reselect",))


def hotkey_quit():
    ui_events.put(("quit",))


# --- Main ---
if __name__ == "__main__":
    if getattr(sys, "frozen", False):  # 打包后: 配置/题库放在 exe 同级目录
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    load_config()
    discover_banks()
    if not banks:
        print("提示: 未发现任何题库 json 文件, 题库匹配不可用 (可仅使用大模型答疑)。")
    elif not settings['bank_file'] or settings['bank_file'] not in banks:
        settings['bank_file'] = sorted(banks.keys())[0]

    print("正在初始化 OCR 引擎 (RapidOCR, 模型内置离线)...")
    try:
        ocr_instance = RapidOCR()
    except Exception as e:
        print(f"OCR 初始化失败: {e}")
        sys.exit(1)

    # 后台线程
    threading.Thread(target=ocr_loop, daemon=True).start()
    threading.Thread(target=question_worker, daemon=True).start()

    # 热键
    keyboard.add_hotkey(TOGGLE_OCR_HOTKEY, hotkey_toggle)
    keyboard.add_hotkey(RESELECT_HOTKEY, hotkey_reselect)
    keyboard.add_hotkey(QUIT_HOTKEY, hotkey_quit)
    print(f"热键: {TOGGLE_OCR_HOTKEY}(暂停/继续) {RESELECT_HOTKEY}(重选) {QUIT_HOTKEY}(退出)")

    root = tk.Tk()
    app = App(root)
    root.mainloop()

    running = False
    print("程序已结束。")
