"""Generate light/dark SVG figures for the context parallelism post.

Usage: python3 scripts/figures/context_parallelism.py <repo_root>

Writes cp-*-{light,dark}.svg to
public/assets/images/post/2026-11-14-Transformer-and-Context-Parallelism/.
The KaTeX font CSS is copied from an existing figure of the LLM 2025-2026 post.
"""

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
REF = ROOT / 'public/assets/images/post/2026-11-01-LLM-in-2025-2026/at-00a-swa-light.svg'
OUT = ROOT / 'public/assets/images/post/2026-11-14-Transformer-and-Context-Parallelism'
FONT_STYLE = re.search(r'<style>.*?</style>', REF.read_text(), re.DOTALL).group(0)

THEMES = {
    'light': {
        'bg': '#efede7',
        'ink': '#2d3142',
        'muted': '#66708a',
        'rule': '#b9bfcc',
        'blue': '#2e5aa8',
        'orange': '#eb6c36',
    },
    'dark': {
        'bg': '#2d3142',
        'ink': '#f5f5f5',
        'muted': '#a3acbf',
        'rule': '#5a6180',
        'blue': '#6a95d8',
        'orange': '#f08a59',
    },
}
SANS = "'Pretendard', system-ui, 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif"
MATH = (
    'font-family="KaTeX_Math, \'Times New Roman\', serif" font-style="italic" font-weight="normal"'
)
MAIN = (
    'font-family="KaTeX_Main, \'Times New Roman\', serif" font-style="normal" font-weight="normal"'
)


def spans(parts, size: float) -> str:
    """Render (kind, text) parts as tspans. Kinds: t, m, r, sub, msub, sup."""
    out, shift = [], 0.0
    for part in parts:
        kind, body = part[0], part[1]
        want = {'sub': size * 0.25, 'msub': size * 0.25, 'msubsup': size * 0.25, 'sup': -size * 0.38}.get(
            kind, 0.0
        )
        dy = want - shift
        shift = want
        attr = f' dy="{dy:.2f}"' if dy else ''
        if kind == 't':
            out.append(f'<tspan{attr}>{body}</tspan>')
        elif kind == 'm':
            out.append(f'<tspan{attr} {MATH}>{body}</tspan>')
        elif kind == 'r':
            out.append(f'<tspan{attr} {MAIN}>{body}</tspan>')
        elif kind == 'msub':
            out.append(f'<tspan{attr} {MATH} font-size="{size * 0.7:.1f}">{body}</tspan>')
        elif kind == 'msubsup':
            # math subscript with an upright superscript stacked above it (w^down_j)
            sub, sup = body, part[2]
            back = size * 0.7 * 0.5 * len(sub)
            out.append(f'<tspan{attr} {MATH} font-size="{size * 0.7:.1f}">{sub}</tspan>')
            out.append(
                f'<tspan dx="{-back:.2f}" dy="{-size * 0.63:.2f}" {MAIN} '
                f'font-size="{size * 0.7:.1f}">{sup}</tspan>'
            )
            shift = want - size * 0.63
        else:
            out.append(f'<tspan{attr} {MAIN} font-size="{size * 0.7:.1f}">{body}</tspan>')
    if shift:
        out.append(f'<tspan dy="{-shift:.2f}">\u200b</tspan>')
    return ''.join(out)


def text(x, y, parts, fill, size=13, anchor='middle', weight=None) -> str:
    if isinstance(parts, str):
        parts = [('t', parts)]
    w = f' font-weight="{weight}"' if weight else ''
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}"{w} xml:space="preserve" '
        f'fill="{fill}">{spans(parts, size)}</text>'
    )


def box(x, y, w, h, stroke, fill='none', opacity=1.0, sw=1.25, rx=8, dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def cell(x, y, w, h, state, c, rx=3) -> str:
    """A small square: 'use' (blue), 'kv' (orange), 'other' (muted fill), 'none' (outline)."""
    if state == 'use':
        return box(x, y, w, h, c['blue'], c['blue'], 0.28, 1.1, rx)
    if state == 'kv':
        return box(x, y, w, h, c['orange'], c['orange'], 0.30, 1.1, rx)
    if state == 'other':
        return box(x, y, w, h, c['muted'], c['muted'], 0.14, 1.0, rx)
    return box(x, y, w, h, c['rule'], 'none', 1.0, 1.0, rx)


def arrow(d, color, key, sw=1.4, dash=None) -> str:
    da = f' stroke-dasharray="{dash}"' if dash else ''
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da} '
        f'marker-end="url(#a-{key})"/>'
    )


def wrap(c, w, h, label, body) -> str:
    markers = ''.join(
        f'<marker id="a-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" '
        f'fill="{c[k]}"/></marker>'
        for k in ('muted', 'blue', 'orange')
    )
    return (
        f'<svg fill="{c["ink"]}" xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'font-family="{SANS}" viewBox="0 0 {w} {h}" role="img" aria-label="{label}" '
        f'font-size="13">\n<rect x="0" y="0" width="{w}" height="{h}" fill="{c["bg"]}"/>'
        f'{FONT_STYLE}<defs>{markers}</defs>\n{body}\n</svg>\n'
    )


def legend(items, y, c, x0=24) -> str:
    """Legend row of (state, label) swatches."""
    p, x = [], x0
    for state, label in items:
        p.append(cell(x, y - 11, 14, 14, state, c))
        p.append(text(x + 22, y, label, c['muted'], 12, 'start'))
        x += 22 + 13 * len(label) * 0.62 + 30
    return ''.join(p)


# --- 0. one transformer layer: attention and FFN share the same inner structure --------------

SLOT_W, SLOT_GAP = 30, 10
ROW_TOK, ROW_Q = 116, 176
ROW_K, ROW_BAR, ROW_V, BAR_H = 228, 310, 322, 44
ROW_SUM = 380


def _lookup_panel(
    c, x0, name, sub, query, slots, weights, color, notes, rows, matrix, out, per, tok
) -> str:
    """One key-value lookup: a query, a row of (key, value) slots, weights, and the sum.

    rows labels the key, coefficient, and value rows; per says what one coefficient
    belongs to; out is the sum in the bottom box; tok = (all_used, note) draws the input
    token row, where only token t is used unless all_used. With
    matrix=True the slots are drawn as the columns of one weight block (index only), not
    as separate per-token cards.
    """
    p = [
        text(x0, 84, name, c['ink'], 14, 'start', 700),
        text(x0, 102, sub, c['muted'], 12, 'start'),
    ]
    xs = [x0 + 6 + i * (SLOT_W + SLOT_GAP) for i in range(len(slots))]
    mid = (xs[0] + xs[-1] + SLOT_W) / 2
    all_used, tok_note = tok
    for k, x in enumerate(xs[:6]):
        cx = x + SLOT_W / 2
        if k == 4:
            p.append(text(cx, ROW_TOK + 16, '⋯', c['ink' if all_used else 'muted'], 14))
            continue
        used = all_used or k == 5
        label = 't' if k == 5 else str(k)
        p.append(
            box(x, ROW_TOK, SLOT_W, 22, c['ink'] if k == 5 else c['muted'] if used else c['rule'],
                c['muted'] if used else 'none', 0.18, 1.3 if k == 5 else 1.0, 4,
                None if used else '3 3')
        )
        p.append(text(cx, ROW_TOK + 16, [('m', label)], c['ink'] if used else c['rule'], 12.5))
    p.append(text(xs[5] + SLOT_W + 18, ROW_TOK + 16, 'token', c['ink'], 12.5, 'start'))
    for k, line in enumerate(tok_note):
        p.append(text(xs[5] + SLOT_W + 18, ROW_TOK + 32 + 14 * k, line, c['muted'], 11, 'start'))
    qx = xs[5] + SLOT_W / 2
    p.append(arrow(f'M{qx} {ROW_TOK + 22}V{ROW_Q - 2}', c['muted'], 'muted', 1.2))
    if all_used:
        # every context token is projected by W_K, W_V into its own key and value
        bx, bw = xs[0] - 4, xs[3] + SLOT_W + 4 - (xs[0] - 4)
        by, bh = ROW_Q - 6, 40
        p.append(box(bx, by, bw, bh, c[color], c[color], 0.12, 1.1, 6))
        p.append(
            f'<line x1="{bx}" y1="{by + bh / 2}" x2="{bx + bw}" y2="{by + bh / 2}" '
            f'stroke="{c[color]}" stroke-width="0.8" stroke-opacity="0.6"/>'
        )
        for k, (wsub, kv) in enumerate((('K', 'k'), ('V', 'v'))):
            p.append(
                text(
                    bx + bw / 2,
                    by + 15 + 20 * k,
                    [('r', '×\u00a0'), ('m', 'W'), ('msub', wsub), ('r', '\u00a0\u00a0→\u00a0\u00a0'), ('m', kv)],
                    c['ink'],
                    12,
                )
            )
        for x in xs[:4]:
            cx = x + SLOT_W / 2
            p.append(arrow(f'M{cx} {ROW_TOK + 22}V{ROW_Q - 8}', c[color], color, 1.0, '3 3'))
            p.append(arrow(f'M{cx} {ROW_Q + 34}V{ROW_K - 2}', c[color], color, 1.0, '3 3'))
    p.append(box(qx - 50, ROW_Q, 100, 28, c['muted'], c['bg']))
    p.append(text(qx, ROW_Q + 19, query, c['ink'], 13))
    pitch = SLOT_W + SLOT_GAP
    for y, lab in zip((ROW_K + 17, ROW_BAR - 12, ROW_V + 17), rows):
        p.append(text(xs[-1] + SLOT_W + 18, y, lab, c['ink'], 13, 'start'))
    p.append(text(xs[-1] + SLOT_W + 18, ROW_BAR + 4, per, c['muted'], 11, 'start'))
    for y in (ROW_K, ROW_V):
        if matrix:
            left, right = xs[0] - SLOT_GAP / 2, xs[-1] + SLOT_W + SLOT_GAP / 2
            p.append(box(left, y, right - left, 24, c[color], c[color], 0.28, 1.2, 4))
            for k in range(1, len(xs)):
                lx = left + k * pitch
                p.append(
                    f'<line x1="{lx}" y1="{y}" x2="{lx}" y2="{y + 24}" '
                    f'stroke="{c[color]}" stroke-width="0.8" stroke-opacity="0.6"/>'
                )
    for x, slot, w in zip(xs, slots, weights):
        cx = x + SLOT_W / 2
        if slot is None:
            p.append(text(cx, ROW_K + 17, '⋯', c['ink' if matrix else 'muted'], 14))
            p.append(text(cx, ROW_V + 17, '⋯', c['ink' if matrix else 'muted'], 14))
            continue
        idx, state = slot
        dashed = state == 'new'
        stroke = c['rule'] if dashed else c[color]
        fill = 'none' if dashed else c[color]
        p.append(
            f'<line x1="{qx}" y1="{ROW_Q + 28}" x2="{cx}" y2="{ROW_K - 2}" stroke="{c["rule"]}" '
            f'stroke-width="1"{" stroke-dasharray=\"3 3\"" if dashed else ""}/>'
        )
        for y, letter in ((ROW_K, 'k'), (ROW_V, 'v')):
            if matrix:
                p.append(text(cx, y + 17, [('m', idx)], c['ink'], 12.5))
                continue
            p.append(box(x, y, SLOT_W, 24, stroke, fill, 0.28, 1.1, 4, '3 3' if dashed else None))
            ink = c['muted'] if dashed else c['ink']
            p.append(text(cx, y + 17, [('m', letter), ('msub', idx)], ink, 13))
        if w is not None:
            h = max(BAR_H * w, 1.5)
            p.append(box(x + 7, ROW_BAR - h - 6, SLOT_W - 14, h, c['muted'], c['muted'], 0.5, 0, 1))
        p.append(
            f'<line x1="{cx}" y1="{ROW_V + 24}" x2="{mid}" y2="{ROW_SUM - 2}" stroke="{c["rule"]}" '
            f'stroke-width="1"{" stroke-dasharray=\"3 3\"" if dashed else ""}/>'
        )
    p.append(box(mid - 60, ROW_SUM, 120, 28, c['muted']))
    p.append(text(mid, ROW_SUM + 19, out, c['ink'], 13))
    for k, note in enumerate(notes):
        p.append(text(x0, ROW_SUM + 58 + 18 * k, note, c['muted' if k else 'ink'], 12, 'start'))
    return ''.join(p)


def fig_structure(c) -> str:
    title = 'attention과 FFN의 key-value 조회'
    p = [
        text(24, 28, title, c['ink'], 15, 'start', 600),
        text(
            24,
            48,
            '둘 다 key와 내적해 score를 내고, 계수로 바꿔 value를 가중합한다. '
            '다른 것은 key와 value가 있는 곳이다',
            c['muted'],
            12,
            'start',
        ),
    ]
    for y, lab in (
        (ROW_TOK + 16, '입력'),
        (ROW_K + 17, '1. score'),
        (ROW_BAR - 12, '2. 계수'),
        (ROW_SUM + 19, '3. 가중합'),
    ):
        p.append(text(24, y, lab, c['muted'], 12, 'start', 600))
    p.append(
        _lookup_panel(
            c,
            96,
            'Attention',
            'context에서 꺼낸다',
            [('m', 'q'), ('t', '\u00a0(token\u00a0'), ('m', 't'), ('t', ')')],
            [('0', 'kv'), ('1', 'kv'), ('2', 'kv'), ('3', 'kv'), None, ('t', 'kv'), ('t+1', 'new')],
            [0.12, 0.08, 0.42, 0.1, None, 0.28, None],
            'orange',
            [
                'context의 token마다 한 쌍, KV cache에 저장',
                [('m', 'T'), ('r', '\u00a0+\u00a0'), ('m', 'P'), ('t', '개, decode마다 하나씩 늘어난다')],
                'softmax: context의 token마다 계수, 합이 1',
            ],
            ([('m', 'K')], [('m', 'a'), ('msub', 'i')], [('m', 'V')]),
            False,
            [('r', 'Σ'), ('msub', 'i'), ('r', '\u00a0'), ('m', 'a'), ('msub', 'i'), ('r', '\u00a0'), ('m', 'v'), ('msub', 'i')],
            'token별',
            (True, ('t 하나를 위해', '앞 token 전부')),
        )
    )
    p.append(
        _lookup_panel(
            c,
            470,
            'FFN',
            'weight에서 꺼낸다',
            [('m', 'x'), ('t', '\u00a0(token\u00a0'), ('m', 't'), ('t', ')')],
            [('1', 'w'), ('2', 'w'), ('3', 'w'), ('4', 'w'), ('5', 'w'), None, ('F', 'w')],
            [0.0, 0.7, 0.0, 0.35, 0.0, None, 0.15],
            'blue',
            [
                '학습된 weight 행렬 그 자체, 고정',
                [('m', 'F'), ('t', '개, context와 무관하다')],
                [('m', 'σ'), ('t', ': 지금 token의 뉴런마다 계수')],
            ],
            ([('m', 'W'), ('sub', 'up')], [('m', 'h'), ('msub', 'j')], [('m', 'W'), ('sub', 'down')]),
            True,
            [('r', 'Σ'), ('msub', 'j'), ('r', '\u00a0'), ('m', 'h'), ('msub', 'j'), ('r', '\u00a0'), ('m', 'w'), ('msubsup', 'j', 'down')],
            '뉴런별',
            (False, ('t 하나만', '다른 token 안 씀')),
        )
    )
    p.append(
        f'<line x1="440" y1="70" x2="440" y2="486" stroke="{c["rule"]}" stroke-width="1"/>'
    )
    return wrap(c, 820, 500, title, '\n'.join(p))


# --- 1. attention vs FFN on two GPUs -------------------------------------------------------


def fig_layer(c) -> str:
    title = 'token을 나눌 때 GPU 경계를 넘는 attention'
    p = [
        text(24, 32, title, c['ink'], 15, 'start', 600),
        text(
            24,
            52,
            'layer 하나를 GPU 2장에 놓고 token 4개를 나눈 예. 두 GPU 모두 attention과 '
            'FFN weight 전체를 갖는다',
            c['muted'],
            12,
            'start',
        ),
    ]
    for g, x0 in enumerate((24, 406)):
        p.append(box(x0, 70, 330, 214, c['rule'], dash='4 3', rx=10))
        p.append(text(x0 + 14, 90, f'GPU {g}', c['ink'], 13, 'start', 600))
        for k, tx in enumerate((x0 + 105, x0 + 225)):
            p.append(box(tx - 45, 100, 90, 28, c['muted']))
            p.append(text(tx, 119, f'token {2 * g + k}', c['ink'], 12.5))
            p.append(arrow(f'M{tx} 128V156', c['muted'], 'muted', 1.2))
            p.append(arrow(f'M{tx} 202V228', c['muted'], 'muted', 1.2))
            p.append(box(tx - 55, 230, 110, 38, c['muted']))
            p.append(text(tx, 254, 'FFN', c['ink'], 12.5))
        hi = g == 1
        p.append(
            box(
                x0 + 15,
                158,
                300,
                42,
                c['blue'] if hi else c['muted'],
                c['blue'] if hi else 'none',
                0.12 if hi else 1.0,
                1.6 if hi else 1.25,
            )
        )
        p.append(text(x0 + 165, 184, 'Attention', c['ink'], 13, weight=600))
    p.append(arrow('M339 179H419', c['blue'], 'blue', 1.8))
    p.append(text(380, 170, [('t', 'K, V')], c['blue'], 12, weight=600))
    p.append(f'<line x1="24" y1="306" x2="52" y2="306" stroke="{c["blue"]}" stroke-width="1.8"/>')
    p.append(
        text(
            60,
            310,
            'token 0, 1의 K, V가 GPU 1의 attention으로 넘어간다. 통신은 여기서만 생긴다',
            c['muted'],
            12,
            'start',
        )
    )
    p.append(f'<line x1="24" y1="328" x2="52" y2="328" stroke="{c["muted"]}" stroke-width="1.2"/>')
    p.append(
        text(
            60,
            332,
            'FFN은 자기 token의 attention 출력만 받는다. GPU를 넘는 선이 없다',
            c['muted'],
            12,
            'start',
        )
    )
    return wrap(c, 760, 350, title, '\n'.join(p))


# --- 2. DP, TP, CP split axes ----------------------------------------------------------------


def _axes_row(c, y, name, desc, tok, head, mlp) -> str:
    p = [text(24, y, name, c['ink'], 13, 'start', 600), text(184, y, desc, c['muted'], 12, 'start')]
    labels = ('이 prompt의 token', 'attention head 4개', 'FFN weight')
    for k, lab in enumerate(labels):
        p.append(text(24, y + 26 + 28 * k, lab, c['muted'], 12, 'start'))
    for g in range(4):
        x = 184 + g * 144
        for t in range(8):
            p.append(cell(x + t * 17, y + 14, 14, 14, tok(g, t), c))
        for h in range(4):
            p.append(cell(x + h * 34, y + 40, 30, 16, head(g, h), c))
        for s in range(4):
            p.append(cell(x + s * 34, y + 68, 30, 12, mlp(g, s), c, 2))
    return ''.join(p)


def fig_axes(c) -> str:
    title = 'DP, TP, CP가 나누는 축'
    p = [
        text(24, 32, title, c['ink'], 15, 'start', 600),
        text(
            24,
            52,
            'layer 하나(attention head 4개, FFN)와 prompt 하나의 token 8개를 GPU 4장에 '
            '놓는 세 가지 방법',
            c['muted'],
            12,
            'start',
        ),
    ]
    for g in range(4):
        p.append(text(184 + g * 144 + 68, 80, f'GPU {g}', c['ink'], 12.5, weight=600))
    rows = [
        (
            'DP: request 분할',
            'GPU 한 장이 prompt 전체와 layer 전체를 맡는다',
            lambda g, t: 'use' if g == 0 else 'none',
            lambda g, h: 'use' if g == 0 else 'other',
            lambda g, s: 'use' if g == 0 else 'other',
        ),
        (
            'TP: weight 분할',
            '모든 GPU가 token 전부를 보지만 head와 FFN은 1/4씩',
            lambda g, t: 'use',
            lambda g, h: 'use' if g == h else 'none',
            lambda g, s: 'use' if g == s else 'none',
        ),
        (
            'CP: token 분할',
            '모든 GPU가 layer 전체를 갖고 token만 1/4씩',
            lambda g, t: 'use' if t // 2 == g else 'none',
            lambda g, h: 'use',
            lambda g, s: 'use',
        ),
    ]
    for k, row in enumerate(rows):
        y = 104 + k * 112
        if k:
            p.append(
                f'<line x1="24" y1="{y - 22}" x2="736" y2="{y - 22}" '
                f'stroke="{c["rule"]}" stroke-width="1"/>'
            )
        p.append(_axes_row(c, y, *row))
    p.append(
        legend(
            [
                ('use', '이 prompt 계산에 쓰인다'),
                ('other', '있지만 다른 request용'),
                ('none', '이 GPU에 없다'),
            ],
            446,
            c,
        )
    )
    return wrap(c, 760, 464, title, '\n'.join(p))


# --- 2b. KV replication under TP vs DCP --------------------------------------------------


def fig_tp_kv(c) -> str:
    title = 'TP8의 MLA KV 복제와 DCP8의 KV 분할'
    p = [
        text(24, 32, title, c['ink'], 15, 'start', 600),
        text(
            24,
            52,
            'Kimi K3, 142K token, MLA 24 layer 기준 GPU당 MLA KV (vLLM MI355X 글)',
            c['muted'],
            12,
            'start',
        ),
    ]
    rows = (
        ('TP8', '약 1.96 GB', lambda g: (0, 8)),
        ([('t', 'TP8'), ('r', ' × '), ('t', 'DCP8')], '약 245 MB', lambda g: (g, g + 1)),
    )
    for k, (name, size, part) in enumerate(rows):
        y = 84 + k * 92
        p.append(text(24, y + 34, name, c['ink'], 13, 'start', 600))
        p.append(text(24, y + 52, size, c['muted'], 12, 'start'))
        for g in range(8):
            x = 140 + g * 75
            p.append(box(x, y, 68, 68, c['rule'], rx=6))
            p.append(text(x + 34, y + 16, f'GPU {g}', c['muted'], 11))
            lo, hi = part(g)
            for s in range(8):
                p.append(
                    cell(
                        x + 6 + (s % 4) * 14,
                        y + 26 + (s // 4) * 18,
                        11,
                        14,
                        'kv' if lo <= s < hi else 'none',
                        c,
                        2,
                    )
                )
    p.append(
        legend([('kv', 'GPU가 저장한 KV의 1/8 조각'), ('none', '저장하지 않은 조각')], 282, c)
    )
    return wrap(c, 760, 300, title, '\n'.join(p))


# --- 3. pass-KV vs pass-Q ----------------------------------------------------------------


def fig_pass(c) -> str:
    title = 'pass-KV와 pass-Q: query와 KV 중 작은 쪽의 이동'
    p = [text(24, 32, title, c['ink'], 15, 'start', 600)]
    panels = (
        (24, 'prefill: 많은 query', 'pass-KV: KV를 query 쪽으로', 150, 64, 'kv'),
        (392, 'decode: token 하나의 query', 'pass-Q: query를 KV 쪽으로', 14, 150, 'q'),
    )
    for x0, head, foot, qw, kw, moving in panels:
        p.append(text(x0, 62, head, c['ink'], 13, 'start', 600))
        for k, (lab, w, col) in enumerate(
            (('GPU A: query', qw, 'blue'), ('GPU B: K, V', kw, 'orange'))
        ):
            bx = x0 + k * 180
            p.append(box(bx, 76, 164, 96, c['rule'], rx=8))
            p.append(text(bx + 82, 96, lab, c['muted'], 12))
            p.append(box(bx + 82 - w / 2, 112, w, 36, c[col], c[col], 0.25, 1.3, 4))
        if moving == 'kv':
            p.append(arrow(f'M{x0 + 262} 172V196H{x0 + 82}V176', c['orange'], 'orange', 1.8))
        else:
            p.append(arrow(f'M{x0 + 82} 172V196H{x0 + 262}V176', c['blue'], 'blue', 1.8))
        p.append(text(x0 + 172, 220, foot, c['ink'], 12.5, weight=600))
    p.append(f'<line x1="24" y1="244" x2="736" y2="244" stroke="{c["rule"]}" stroke-width="1"/>')
    p.append(
        text(
            24,
            272,
            [
                ('t', 'Q가 KV보다 작아지는 조건: '),
                ('m', 'T'),
                ('r', ' / ('),
                ('m', 'T'),
                ('r', ' + '),
                ('m', 'P'),
                ('r', ') ≤ 2'),
                ('m', 'K'),
                ('r', ' / '),
                ('m', 'N'),
                ('t', '   (좌변은 KV cache miss rate)'),
            ],
            c['ink'],
            14,
            'start',
        )
    )
    return wrap(c, 760, 292, title, '\n'.join(p))


# --- 4. PCP load-balanced chunks ------------------------------------------------------------


def fig_chunks(c) -> str:
    title = 'load-balanced sharding의 GPU별 계산량'
    p = [
        text(24, 32, title, c['ink'], 15, 'start', 600),
        text(
            24,
            52,
            [
                ('t', 'GPU 4장('),
                ('m', 'C'),
                ('r', ' = 4'),
                (
                    't',
                    (
                        '), prompt를 8개 chunk로'
                        ' 자른 예. 막대 높이는 chunk의 query가 보는 이전 token 수에 비례한다'
                    ),
                ),
            ],
            c['muted'],
            12,
            'start',
        ),
    ]
    unit, base = 5.5, 196
    for j in range(8):
        x = 64 + j * 84
        col = 'blue' if j < 4 else 'orange'
        owner = j if j < 4 else 7 - j
        h = (2 * j + 1) * unit
        p.append(box(x, base - h, 72, h, c[col], c[col], 0.25, 1.2, 3))
        p.append(text(x + 36, base - h - 6, str(2 * j + 1), c['muted'], 11))
        p.append(text(x + 36, base + 20, [('m', 'C'), ('sub', str(j))], c['ink'], 14))
        p.append(text(x + 36, base + 38, f'GPU {owner}', c[col], 11.5, weight=600))
    p.append(f'<line x1="56" y1="{base}" x2="736" y2="{base}" stroke="{c["rule"]}"/>')
    p.append(f'<line x1="24" y1="252" x2="736" y2="252" stroke="{c["rule"]}" stroke-width="1"/>')
    p.append(text(24, 276, 'GPU별 합', c['ink'], 13, 'start', 600))
    u = 28
    for i in range(4):
        y = 290 + i * 26
        a, b = 2 * i + 1, 2 * (7 - i) + 1
        p.append(text(24, y + 13, f'GPU {i}', c['muted'], 12, 'start'))
        p.append(box(90, y, a * u, 18, c['blue'], c['blue'], 0.25, 1.1, 2))
        p.append(box(90 + a * u, y, b * u, 18, c['orange'], c['orange'], 0.25, 1.1, 2))
        p.append(text(90 + a * u / 2, y + 13, [('m', 'C'), ('sub', str(i))], c['ink'], 11.5))
        p.append(
            text(90 + a * u + b * u / 2, y + 13, [('m', 'C'), ('sub', str(7 - i))], c['ink'], 11.5)
        )
        p.append(
            text(
                90 + 16 * u + 10,
                y + 13,
                [('r', f'{a} + {b} = 16 = 4'), ('m', 'C')],
                c['ink'],
                12,
                'start',
            )
        )
    return wrap(c, 760, 404, title, '\n'.join(p))


# --- 5. DCP decode ---------------------------------------------------------------------------


def fig_dcp_decode(c) -> str:
    title = 'DCP decode에서 KV 쪽으로 가는 query'
    p = [
        text(24, 32, title, c['ink'], 15, 'start', 600),
        text(
            24,
            52,
            [
                ('t', 'GPU 4장. GPU '),
                ('m', 's'),
                ('t', '는 token 번호를 4로 나눈 나머지가 '),
                ('m', 's'),
                ('t', '인 token의 KV만 저장한다'),
            ],
            c['muted'],
            12,
            'start',
        ),
    ]
    cy = 204
    p.append(box(24, cy - 26, 104, 52, c['blue'], c['blue'], 0.12, 1.4))
    p.append(text(76, cy - 4, [('t', 'query '), ('m', 'q')], c['ink'], 13, weight=600))
    p.append(text(76, cy + 14, 'token 1개', c['muted'], 11.5))
    p.append(text(164, 78, '1. AllGather Q', c['blue'], 12, weight=600))
    p.append(text(456, 78, '3. AllGather + ReduceScatter', c['orange'], 12, weight=600))
    for s in range(4):
        y = 88 + s * 62
        p.append(arrow(f'M128 {cy}H164V{y + 23}H200', c['blue'], 'blue', 1.3))
        p.append(box(204, y, 214, 46, c['muted']))
        p.append(
            text(
                214,
                y + 19,
                [('t', f'GPU {s}: '), ('t', f'shard {s}의 KV로 attention')],
                c['ink'],
                12,
                'start',
            )
        )
        p.append(
            text(
                214,
                y + 36,
                [
                    ('t', '2. 출력 '),
                    ('m', 'O'),
                    ('msub', 's'),
                    ('r', ', '),
                    ('t', 'LSE'),
                    ('msub', 's'),
                ],
                c['muted'],
                12,
                'start',
            )
        )
        p.append(arrow(f'M418 {y + 23}H456V{cy}H494', c['orange'], 'orange', 1.3))
    p.append(box(498, cy - 34, 140, 68, c['orange'], c['orange'], 0.12, 1.4))
    p.append(text(568, cy - 6, 'LSE로 가중합', c['ink'], 13, weight=600))
    p.append(text(568, cy + 13, '근사 없는 정확한 값', c['muted'], 11.5))
    p.append(arrow(f'M638 {cy}H672', c['muted'], 'muted', 1.4))
    p.append(box(676, cy - 26, 60, 52, c['muted']))
    p.append(text(706, cy + 6, [('m', 'O')], c['ink'], 15))
    return wrap(c, 760, 340, title, '\n'.join(p))


# --- 6. PCP vs PCP x DCP token layout ---------------------------------------------------


def fig_layout(c) -> str:
    title = 'PCP의 계산 분할과 DCP의 KV 분할'
    p = [
        text(24, 32, title, c['ink'], 15, 'start', 600),
        text(
            24,
            52,
            'prompt 하나의 token 8개(0부터 7), GPU 4장. 칸은 왼쪽부터 token 0, 1, 2, ...',
            c['muted'],
            12,
            'start',
        ),
    ]
    for g in range(4):
        p.append(text(184 + g * 144 + 68, 82, f'GPU {g}', c['ink'], 12.5, weight=600))
    blocks = (
        (
            'PCP만',
            'chunk를 나눠 계산하고, KV는 all-gather해서 4장 모두 저장',
            lambda g, t: t in (g, 7 - g),
            lambda g, t: True,
        ),
        (
            'DCP만',
            '4장 모두 모든 token을 계산하고(head는 TP로 나눔), KV는 mod 4로 나눠 저장',
            lambda g, t: True,
            lambda g, t: t % 4 == g,
        ),
        (
            [('t', 'PCP'), ('r', ' × '), ('t', 'DCP')],
            'chunk를 나눠 계산하고, KV는 token 번호 mod 4인 GPU 한 장만 저장',
            lambda g, t: t in (g, 7 - g),
            lambda g, t: t % 4 == g,
        ),
    )
    for k, (name, desc, comp, store) in enumerate(blocks):
        y = 108 + k * 104
        if k:
            p.append(
                f'<line x1="24" y1="{y - 22}" x2="736" y2="{y - 22}" '
                f'stroke="{c["rule"]}" stroke-width="1"/>'
            )
        p.append(text(24, y, name, c['ink'], 13, 'start', 600))
        p.append(text(184, y, desc, c['muted'], 12, 'start'))
        p.append(text(24, y + 30, '계산하는 token', c['muted'], 12, 'start'))
        p.append(text(24, y + 58, '저장하는 KV', c['muted'], 12, 'start'))
        for g in range(4):
            x = 184 + g * 144
            for t in range(8):
                p.append(cell(x + t * 17, y + 18, 14, 14, 'use' if comp(g, t) else 'none', c))
                p.append(cell(x + t * 17, y + 46, 14, 14, 'kv' if store(g, t) else 'none', c))
    p.append(
        text(
            24,
            422,
            'PCP와 DCP를 함께 쓰면 GPU 0은 token 0과 7을 계산하지만 KV는 token 0과 4의 것을 '
            '저장한다.',
            c['muted'],
            12,
            'start',
        )
    )
    p.append(
        text(
            24,
            440,
            '두 layout이 어긋나므로 layer마다 all-gather가 필요하다.',
            c['muted'],
            12,
            'start',
        )
    )
    p.append(
        legend(
            [('use', '이 GPU가 계산한다'), ('kv', '이 GPU가 KV를 저장한다'), ('none', '해당 없음')],
            470,
            c,
        )
    )
    return wrap(c, 760, 488, title, '\n'.join(p))


# --- 7. GLM-5.3 deployment ------------------------------------------------------------------


def fig_glm(c) -> str:
    title = 'GLM-5.3 prefill engine의 PCP와 DCP'
    p = [
        text(24, 32, title, c['ink'], 15, 'start', 600),
        text(
            24,
            52,
            'H200 16장을 prefill node 하나와 decode node 하나로 나눈 구성 (vLLM GLM-5.3 Part 2 글)',
            c['muted'],
            12,
            'start',
        ),
    ]
    p.append(box(24, 106, 104, 52, c['muted']))
    p.append(text(76, 128, 'request', c['ink'], 13, weight=600))
    p.append(text(76, 146, 'agent turn', c['muted'], 11.5))
    p.append(arrow('M128 132H148', c['muted'], 'muted'))
    p.append(box(152, 106, 104, 52, c['muted']))
    p.append(text(204, 128, 'llm-d router', c['ink'], 12.5, weight=600))
    p.append(text(204, 146, 'prefix-aware', c['muted'], 11.5))
    p.append(arrow('M256 132H276', c['muted'], 'muted'))
    p.append(box(280, 76, 200, 112, c['blue'], c['blue'], 0.10, 1.6))
    p.append(text(380, 98, 'Prefill node (H200 8장)', c['ink'], 13, weight=600))
    p.append(
        text(
            380,
            120,
            [('t', 'PCP8'), ('r', ' × '), ('t', 'DCP8'), ('r', ' × '), ('t', 'EP8')],
            c['blue'],
            12.5,
            weight=600,
        )
    )
    p.append(text(380, 138, 'engine 1개', c['muted'], 11.5))
    p.append(text(380, 160, 'CPU KV offload 512 GiB', c['muted'], 11.5))
    p.append(text(380, 176, '다음 turn이 prefix 재사용', c['muted'], 11.5))
    p.append(arrow('M480 132H528', c['orange'], 'orange', 1.8))
    p.append(text(504, 122, 'KV', c['orange'], 12, weight=600))
    p.append(text(504, 152, 'NIXL', c['orange'], 11.5))
    p.append(box(532, 76, 204, 112, c['muted'], c['muted'], 0.06, 1.4))
    p.append(text(634, 98, 'Decode node (H200 8장)', c['ink'], 13, weight=600))
    p.append(text(634, 120, 'DP rank 8개, EP8', c['ink'], 12.5, weight=600))
    p.append(text(634, 138, 'Hybrid HiSparse', c['muted'], 11.5))
    p.append(text(634, 160, 'rank당 KV 전체를', c['muted'], 11.5))
    p.append(text(634, 176, '자기 paged layout으로', c['muted'], 11.5))
    p.append(
        text(
            24,
            222,
            'prefill 쪽 KV는 token 단위로 8장에 흩어져 있다. connector가 이것을 decode '
            'rank 한 장의 layout으로 다시 모은다.',
            c['muted'],
            12,
            'start',
        )
    )
    return wrap(c, 760, 240, title, '\n'.join(p))


FIGURES = (
    ('cp-00-structure', fig_structure),
    ('cp-01-layer', fig_layer),
    ('cp-02-axes', fig_axes),
    ('cp-02b-tp-kv', fig_tp_kv),
    ('cp-03-pass', fig_pass),
    ('cp-04-chunks', fig_chunks),
    ('cp-05-dcp-decode', fig_dcp_decode),
    ('cp-06-layout', fig_layout),
    ('cp-07-glm', fig_glm),
)

OUT.mkdir(parents=True, exist_ok=True)
for name, fn in FIGURES:
    for theme, colors in THEMES.items():
        (OUT / f'{name}-{theme}.svg').write_text(fn(colors))
