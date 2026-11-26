"""Generate light/dark SVG figures for the MNNVL all-reduce debugging post.

Usage: python3 scripts/figures/mnnvl_allreduce.py <repo_root>

Writes ar-*-{light,dark}.svg to
public/assets/images/post/2026-11-25-vLLM-MNNVL-AllReduce-Debugging/.
"""

import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
OUT = ROOT / 'public/assets/images/post/2026-11-25-vLLM-MNNVL-AllReduce-Debugging'

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
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"
W = 760


def text(x, y, body, fill, size=13, anchor='middle', weight=None, mono=False) -> str:
    w = f' font-weight="{weight}"' if weight else ''
    f = f' font-family="{MONO}"' if mono else ''
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}"{w}{f} '
        f'xml:space="preserve" fill="{fill}">{body}</text>'
    )


def box(x, y, w, h, stroke, fill='none', opacity=1.0, sw=1.25, rx=8, dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def node(x, y, w, h, lines, c, key='muted', dash=None, mono=False) -> str:
    """A labeled box. `key` picks the accent color; 'muted' draws a plain outline."""
    color = c[key]
    fill, opacity = ('none', 1.0) if key == 'muted' else (color, 0.16)
    stroke = c['rule'] if key == 'muted' else color
    if isinstance(lines, str):
        lines = [lines]
    p = [box(x, y, w, h, stroke, fill, opacity, dash=dash)]
    top = y + h / 2 - (len(lines) - 1) * 8 + 4.5
    for i, line in enumerate(lines):
        p.append(text(x + w / 2, top + i * 16, line, c['ink'], 12, mono=mono))
    return ''.join(p)


def arrow(d, color, key, sw=1.4, dash=None) -> str:
    da = f' stroke-dasharray="{dash}"' if dash else ''
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da} '
        f'marker-end="url(#a-{key})"/>'
    )


def wrap(c, h, label, body) -> str:
    markers = ''.join(
        f'<marker id="a-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" '
        f'fill="{c[k]}"/></marker>'
        for k in ('muted', 'blue', 'orange')
    )
    return (
        f'<svg fill="{c["ink"]}" xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" '
        f'font-family="{SANS}" viewBox="0 0 {W} {h}" role="img" aria-label="{label}" '
        f'font-size="13">\n<rect x="0" y="0" width="{W}" height="{h}" fill="{c["bg"]}"/>'
        f'<defs>{markers}</defs>\n{body}\n</svg>\n'
    )


def title(label, c) -> str:
    return text(24, 32, label, c['ink'], 15, 'start', 600)


# --- 1. rank placement: two nodes, NVLink inside, InfiniBand between ------------------------

GPU_W, GPU_GAP = 32, 8


def _host(c, x, name, first_rank) -> str:
    p = [
        box(x, 70, 340, 150, c['rule']),
        text(x + 14, 92, name, c['ink'], 13, 'start', 600),
    ]
    for i in range(8):
        gx = x + 14 + i * (GPU_W + GPU_GAP)
        p.append(box(gx, 104, GPU_W, 36, c['muted'], c['muted'], 0.14, 1.0, 4))
        p.append(text(gx + GPU_W / 2, 127, str(first_rank + i), c['ink'], 12))
        p.append(f'<path d="M{gx + GPU_W / 2} 140V166" stroke="{c["blue"]}" stroke-width="1.2"/>')
    p.append(box(x + 14, 166, 312, 30, c['blue'], c['blue'], 0.16))
    p.append(text(x + 170, 186, 'NVLink (직접 연결 또는 NVSwitch)', c['ink'], 12))
    return ''.join(p)


def fig_topology(c) -> str:
    label = 'multi-node TP의 rank 배치: node 2대와 InfiniBand 연결'
    p = [
        title(label, c),
        box(14, 50, 732, 180, c['muted'], sw=1.0, dash='5 4'),
        text(736, 64, 'TP group (rank 0-15)', c['muted'], 12, 'end'),
        _host(c, 24, 'node 0: GPU 8장', 0),
        _host(c, 396, 'node 1: GPU 8장', 8),
        f'<path d="M194 220V262H566V220" fill="none" stroke="{c["orange"]}" stroke-width="1.6"/>',
        text(380, 282, 'InfiniBand', c['orange'], 13, weight=600),
        text(
            24,
            316,
            'node 안: NVLink로 GPU memory를 서로 직접 읽고 쓸 수 있다.',
            c['muted'],
            12,
            'start',
        ),
        text(
            24,
            336,
            'node 사이: NVLink fabric이 없다. network 전송(NCCL)만 가능하다.',
            c['muted'],
            12,
            'start',
        ),
    ]
    return wrap(c, 356, label, '\n'.join(p))


# --- 1b. reaching a peer GPU's memory: mapping over NVLink vs copying over InfiniBand -------


def _proc(c, x, w, name) -> str:
    return box(x, 74, w, 92, c['rule']) + text(x + 12, 95, name, c['ink'], 12, 'start', 600)


def fig_address_space(c) -> str:
    label = '다른 GPU의 memory에 닿는 두 방식: NVLink의 매핑과 InfiniBand의 복사'
    p = [
        title(label, c),
        text(24, 60, 'NVLink domain', c['ink'], 13, 'start', 600),
        _proc(c, 24, 340, 'rank 0의 process: 가상 주소 공간'),
        node(40, 108, 150, 40, ['내 buffer'], c),
        node(198, 108, 150, 40, ['rank 1의 buffer', '(매핑)'], c, 'blue'),
        node(40, 216, 150, 44, ['GPU 0의 memory'], c),
        node(198, 216, 150, 44, ['GPU 1의 memory'], c, 'blue'),
        arrow('M115 148V214', c['muted'], 'muted'),
        arrow('M273 148V214', c['blue'], 'blue'),
        text(283, 188, 'NVLink', c['blue'], 12, 'start', 600),
        text(
            24,
            292,
            'kernel이 주소로 직접 읽고 쓴다. 보내고 받는 단계가 없다.',
            c['muted'],
            12,
            'start',
        ),
        text(
            24,
            310,
            'MNNVL에서는 다른 node의 GPU도 같은 방식으로 매핑된다.',
            c['muted'],
            12,
            'start',
        ),
        text(396, 60, 'InfiniBand', c['ink'], 13, 'start', 600),
        _proc(c, 396, 160, 'rank 0의 process'),
        _proc(c, 576, 160, 'rank 8의 process'),
        node(408, 108, 136, 40, ['내 buffer'], c),
        node(588, 108, 136, 40, ['내 buffer'], c),
        node(408, 216, 136, 44, ['GPU 0의 memory'], c),
        node(588, 216, 136, 44, ['GPU 8의 memory'], c),
        arrow('M476 148V214', c['muted'], 'muted'),
        arrow('M656 148V214', c['muted'], 'muted'),
        arrow('M546 238H586', c['orange'], 'orange'),
        text(566, 206, '복사', c['orange'], 12, weight=600),
        text(
            396,
            292,
            '상대 memory는 내 주소 공간에 나타나지 않는다.',
            c['muted'],
            12,
            'start',
        ),
        text(396, 310, 'data를 보내고 받아서 옮긴다 (NCCL).', c['muted'], 12, 'start'),
    ]
    return wrap(c, 330, label, '\n'.join(p))


# --- 1c. a sharded matmul: both inputs split on the contracting dimension --------------------


def _rank_row(c, y, name, key, i) -> str:
    return ''.join(
        [
            text(24, y + 32, name, c['ink'], 13, 'start', 600),
            node(92, y, 48, 56, [f'A{i}'], c, key),
            text(156, y + 33, '·', c['ink'], 16),
            node(172, y + 14, 64, 28, [f'B{i}'], c, key),
            arrow(f'M244 {y + 28}H278', c['muted'], 'muted'),
            node(282, y, 64, 56, [f'A{i}·B{i}'], c, key),
            text(356, y + 24, 'C와 같은 크기 [I, K]', c['muted'], 12, 'start'),
            text(356, y + 42, 'J의 절반만 더한 부분합', c['muted'], 12, 'start'),
            arrow(f'M500 {y + 28}H534', c['muted'], 'muted'),
        ]
    )


def fig_partial_sum(c) -> str:
    label = 'sharding된 행렬곱: contracting dimension의 분할과 부분합'
    p = [
        title(label, c),
        text(
            24,
            62,
            '나누기: A[I, J]는 column 방향, B[J, K]는 row 방향',
            c['ink'],
            13,
            'start',
            600,
        ),
        text(88, 82, 'J', c['muted'], 12),
        text(30, 121, 'I', c['muted'], 12),
        node(40, 88, 48, 56, ['A0'], c, 'blue'),
        node(88, 88, 48, 56, ['A1'], c, 'orange'),
        text(156, 121, '·', c['ink'], 16),
        text(218, 82, 'K', c['muted'], 12),
        text(176, 121, 'J', c['muted'], 12),
        node(186, 88, 64, 28, ['B0'], c, 'blue'),
        node(186, 116, 64, 28, ['B1'], c, 'orange'),
        text(
            290,
            104,
            'J는 곱하면서 더해 없어지는 차원(contracting dimension)이다.',
            c['muted'],
            12,
            'start',
        ),
        text(
            290,
            122,
            'A와 B를 모두 J 방향으로 반씩 나눠 rank 0과 rank 1에 준다.',
            c['muted'],
            12,
            'start',
        ),
        text(24, 194, 'rank별 계산', c['ink'], 13, 'start', 600),
        _rank_row(c, 208, 'rank 0', 'blue', 0),
        _rank_row(c, 288, 'rank 1', 'orange', 1),
        node(538, 208, 90, 136, ['all-reduce', '(SUM)'], c),
        arrow('M630 276H660', c['muted'], 'muted'),
        node(664, 248, 64, 56, ['C'], c),
        text(696, 322, '[I, K]', c['muted'], 12),
        text(24, 378, 'C = A0·B0 + A1·B1', c['ink'], 13, 'start', 600),
        text(
            24,
            400,
            'rank 하나의 결과는 크기만 C와 같다. 둘을 더해야 C가 된다.',
            c['muted'],
            12,
            'start',
        ),
    ]
    return wrap(c, 420, label, '\n'.join(p))


# --- 1e. which way ColumnParallelLinear and RowParallelLinear split the weight ---------------


def fig_row_column(c) -> str:
    label = 'ColumnParallelLinear와 RowParallelLinear: weight W의 분할 방향'
    lx, rx = 24, 400
    p = [
        title(label, c),
        text(
            lx,
            62,
            'ColumnParallelLinear: W를 column 방향으로 분할',
            c['ink'],
            13,
            'start',
            600,
        ),
        text(lx + 28, 84, 'X', c['muted'], 12),
        node(lx, 92, 56, 56, ['X'], c),
        text(lx + 70, 125, '·', c['ink'], 16),
        text(lx + 128, 84, 'W', c['muted'], 12),
        node(lx + 84, 92, 44, 56, ['W0'], c, 'blue'),
        node(lx + 128, 92, 44, 56, ['W1'], c, 'orange'),
        arrow(f'M{lx + 180} 120H{lx + 210}', c['muted'], 'muted'),
        text(lx + 258, 84, 'Y', c['muted'], 12),
        node(lx + 214, 92, 44, 56, ['Y0'], c, 'blue'),
        node(lx + 258, 92, 44, 56, ['Y1'], c, 'orange'),
        text(lx, 176, 'W의 column은 출력 차원이다.', c['muted'], 12, 'start'),
        text(
            lx,
            194,
            'rank마다 Y의 column 일부를 온전하게 얻는다.',
            c['muted'],
            12,
            'start',
        ),
        text(lx, 212, '통신이 없다.', c['muted'], 12, 'start'),
        text(
            rx,
            62,
            'RowParallelLinear: W를 row 방향으로 분할',
            c['ink'],
            13,
            'start',
            600,
        ),
        text(rx + 44, 84, 'X', c['muted'], 12),
        node(rx, 92, 44, 56, ['X0'], c, 'blue'),
        node(rx + 44, 92, 44, 56, ['X1'], c, 'orange'),
        text(rx + 102, 125, '·', c['ink'], 16),
        text(rx + 148, 84, 'W', c['muted'], 12),
        node(rx + 116, 92, 64, 28, ['W0'], c, 'blue'),
        node(rx + 116, 120, 64, 28, ['W1'], c, 'orange'),
        arrow(f'M{rx + 188} 120H{rx + 218}', c['muted'], 'muted'),
        text(rx + 283, 84, 'Y의 부분합 둘', c['muted'], 12),
        node(rx + 222, 92, 52, 56, ['X0·W0'], c, 'blue'),
        text(rx + 283, 125, '+', c['ink'], 14),
        node(rx + 292, 92, 52, 56, ['X1·W1'], c, 'orange'),
        text(
            rx,
            176,
            'W의 row는 입력 차원이다. X도 같은 차원으로 나뉘어 온다.',
            c['muted'],
            12,
            'start',
        ),
        text(
            rx,
            194,
            'rank마다 Y와 같은 크기의 부분합을 얻는다.',
            c['muted'],
            12,
            'start',
        ),
        text(rx, 212, 'all-reduce로 더해야 Y가 된다.', c['muted'], 12, 'start'),
        text(
            lx,
            246,
            'Y = XW로 쓰고, W의 shape를 [입력 차원, 출력 차원]으로 본 이름이다.',
            c['ink'],
            12,
            'start',
        ),
    ]
    return wrap(c, 266, label, '\n'.join(p))


# --- 1d/1f. one layer as a chain of matrices: color is the rank, the cut is the split ----------

MAT_H = 60
SHARDS = ('blue', 'orange')
FFN_CHAIN = (
    ('plain', 52, 'X', 'hidden state', '', False),
    ('op', 20, '·', '', '', False),
    ('cols', 88, ('W0', 'W1'), 'gate_up_proj', 'Column: 출력 차원 분할', True),
    ('op', 20, '=', '', '', False),
    ('cols', 148, ('F의 shard 0', 'F의 shard 1'), 'activation', 'rank 안의 계산', False),
    ('op', 20, '·', '', '', False),
    ('rows', 88, ('W0', 'W1'), 'down_proj', 'Row: 입력 차원 분할', True),
    ('op', 20, '=', '', '', False),
    ('blue', 72, '부분합 0', '', '', False),
    ('op', 20, '+', 'all-reduce', '부분합을 더한다', False),
    ('orange', 72, '부분합 1', '', '', False),
    ('op', 20, '=', '', '', False),
    ('plain', 52, 'Y', '출력', '', False),
)


def _expert_grid(c, x, y, w) -> str:
    """Eight experts as small squares: the set is split, not a matrix dimension."""
    cell, gap = 22, 5
    x0 = x + (w - 4 * cell - 3 * gap) / 2
    y0 = y + (MAT_H - 2 * cell - gap) / 2
    p = []
    for k in range(8):
        key = SHARDS[k // 4]
        cx, cy = x0 + (k % 4) * (cell + gap), y0 + (k // 4) * (cell + gap)
        p.append(box(cx, cy, cell, cell, c[key], c[key], 0.16, rx=4))
        p.append(text(cx + cell / 2, cy + cell / 2 + 4, str(k), c['ink'], 11))
    return ''.join(p)


def _chain(c, y, name, items) -> str:
    """One row of matrices. Each item is (kind, width, body, head, sub, mono).

    'plain' is a value every rank holds whole. 'cols' and 'rows' are one matrix cut into two
    shards, side by side or stacked. 'blue' and 'orange' are whole matrices owned by one rank.
    """
    p = [text(24, y - 44, name, c['ink'], 13, 'start', 600)]
    x = 34
    for kind, w, body, head, sub, mono in items:
        cx = x + w / 2
        if head:
            p.append(text(cx, y - 22, head, c['ink'], 12, weight=600, mono=mono))
        if sub:
            p.append(text(cx, y - 7, sub, c['muted'], 11))
        if kind == 'op':
            p.append(text(cx, y + MAT_H / 2 + 6, body, c['ink'], 17, weight=600))
        elif kind == 'plain':
            p.append(node(x, y, w, MAT_H, body, c))
        elif kind == 'cols':
            p.extend(node(x + k * w / 2, y, w / 2, MAT_H, body[k], c, SHARDS[k]) for k in range(2))
        elif kind == 'rows':
            p.extend(
                node(x, y + k * MAT_H / 2, w, MAT_H / 2, body[k], c, SHARDS[k]) for k in range(2)
            )
        elif kind == 'grid':
            p.append(_expert_grid(c, x, y, w))
        else:
            p.append(node(x, y, w, MAT_H, body, c, kind))
        x += w
    return ''.join(p)


def _chain_legend(c, y, extra) -> str:
    p = []
    for x, key, label in (
        (24, 'blue', 'rank 0'),
        (104, 'orange', 'rank 1'),
        (184, 'muted', '모든 rank'),
    ):
        stroke = c['rule'] if key == 'muted' else c[key]
        fill, opacity = ('none', 1.0) if key == 'muted' else (c[key], 0.16)
        p.append(box(x, y - 11, 12, 12, stroke, fill, opacity, rx=3))
        p.append(text(x + 20, y, label, c['muted'], 12, 'start'))
    lines = (
        '좌우로 나뉜 행렬은 column 방향 분할이고, 위아래로 나뉜 행렬은 row 방향 분할이다.',
        *extra,
    )
    p.extend(text(24, y + 22 + k * 20, s, c['muted'], 12, 'start') for k, s in enumerate(lines))
    return ''.join(p)


def fig_layer(c) -> str:
    label = 'transformer layer의 TP: column 방향 분할에서 row 방향 분할까지'
    attention = (
        FFN_CHAIN[:2]
        + (('cols', 88, ('W0', 'W1'), 'qkv_proj', 'Column: 출력 차원 분할', True),)
        + FFN_CHAIN[3:4]
        + (
            (
                'cols',
                148,
                ('head 0, 1', 'head 2, 3'),
                'attention',
                'rank 안의 계산',
                False,
            ),
        )
        + FFN_CHAIN[5:6]
        + (('rows', 88, ('W0', 'W1'), 'o_proj', 'Row: 입력 차원 분할', True),)
        + FFN_CHAIN[7:]
    )
    p = [
        title(label, c),
        _chain(c, 104, 'attention', attention),
        _chain(c, 228, 'FFN', FFN_CHAIN),
        _chain_legend(c, 322, ()),
    ]
    return wrap(c, 362, label, '\n'.join(p))


def fig_moe(c) -> str:
    label = 'Kimi K3 MoE layer의 TP와 EP: expert 분할과 row 방향 분할'
    routed = (
        ('plain', 60, 'X', 'hidden state', '', False),
        ('op', 20, '·', '', '', False),
        ('plain', 72, 'W', 'down projection', '복제', False),
        ('op', 24, '→', '', '', False),
        ('grid', 116, None, 'expert', 'EP: expert를 나눈다', False),
        ('op', 20, '=', '', '', False),
        ('blue', 72, '부분합 0', '', '', False),
        ('op', 20, '+', 'all-reduce', '그 뒤에 RMSNorm', False),
        ('orange', 72, '부분합 1', '', '', False),
        ('op', 20, '·', '', '', False),
        ('plain', 72, 'W', 'up projection', '복제', False),
        ('op', 20, '=', '', '', False),
        ('plain', 64, 'Y', '출력', '', False),
    )
    p = [
        title(label, c),
        _chain(c, 104, 'routed expert 경로 (EP)', routed),
        _chain(c, 228, 'shared expert 경로 (TP)', FFN_CHAIN),
        _chain_legend(
            c,
            322,
            (
                '위쪽의 all-reduce는 RMSNorm이 뒤따르므로 FlashInfer fusion kernel의 대상이다.',
                '두 경로의 출력을 더한 것이 MoE layer의 출력이다.',
            ),
        ),
    ]
    return wrap(c, 402, label, '\n'.join(p))


# --- 2. two paths that compute the same sum ------------------------------------------------


def fig_two_paths(c) -> str:
    label = '같은 합을 계산하는 두 경로'
    ya, yb, h = 72, 196, 44
    yc = (ya + yb + h) / 2
    steps = ['all-reduce', '+ residual', 'RMSNorm']
    p = [
        title(label, c),
        node(24, yc - h / 2, 110, h, ['rank별 부분합'], c),
        node(626, yc - h / 2, 110, h, ['같은 결과'], c),
        text(180, ya - 12, '(a) 일반 경로: kernel 3번', c['ink'], 13, 'start', 600),
        text(180, yb + h + 20, '(b) fusion 경로: kernel 1번', c['ink'], 13, 'start', 600),
        box(180, yb, 400, h, c['orange'], c['orange'], 0.16),
    ]
    for i, step in enumerate(steps):
        x = 180 + i * 140
        key = 'blue' if i == 0 else 'muted'
        lines = [step, 'backend: NCCL'] if i == 0 else [step]
        p.append(node(x, ya, 120, h, lines, c, key))
        p.append(text(180 + 400 / 6 + i * 400 / 3, yb + 26.5, step, c['ink'], 12))
        if i:
            p.append(arrow(f'M{x - 20} {ya + 22}H{x - 2}', c['muted'], 'muted'))
            xd = 180 + i * 400 / 3
            p.append(
                f'<path d="M{xd:.1f} {yb + 8}V{yb + h - 8}" stroke="{c["orange"]}" '
                f'stroke-width="1" stroke-opacity="0.6"/>'
            )
    for y in (ya, yb):
        p.append(arrow(f'M134 {yc}H157V{y + 22}H178', c['muted'], 'muted'))
        p.append(arrow(f'M580 {y + 22}H603V{yc}H624', c['muted'], 'muted'))
    p += [
        arrow(f'M240 {yb}V{ya + h + 2}', c['muted'], 'muted', dash='5 4'),
        text(252, yc + 2, 'workspace가 None이면 (a)로 fallback', c['muted'], 12, 'start'),
        node(
            320,
            yb + 84,
            260,
            36,
            ['workspace: rank 사이 공유 memory'],
            c,
            'orange',
            dash='5 4',
        ),
        arrow(f'M450 {yb + 84}V{yb + h + 2}', c['orange'], 'orange'),
    ]
    return wrap(c, yb + 144, label, '\n'.join(p))


# --- 3. two ways to hand a GPU memory handle to a peer -------------------------------------


def _rank(c, x, y, name, key='muted') -> str:
    return node(x, y, 84, 34, [name], c, key)


def _handle_panel(c, x, head, sub) -> str:
    return ''.join(
        [
            text(x, 66, head, c['ink'], 13, 'start', 600),
            text(x, 84, sub, c['muted'], 12, 'start'),
            box(x, 98, 150, 150, c['rule']),
            text(x + 12, 118, 'node 0', c['muted'], 12, 'start'),
            box(x + 194, 98, 150, 150, c['rule']),
            text(x + 206, 118, 'node 1', c['muted'], 12, 'start'),
        ]
    )


def fig_handles(c) -> str:
    label = 'GPU memory handle의 두 전달 방식'
    lx, rx = 24, 392
    p = [
        title(label, c),
        _handle_panel(c, lx, 'POSIX fd', 'Unix socket의 SCM_RIGHTS로 전달'),
        _rank(c, lx + 33, 132, 'rank 0'),
        _rank(c, lx + 33, 198, 'rank 1'),
        _rank(c, lx + 227, 132, 'rank 8'),
        arrow(f'M{lx + 75} 166V196', c['blue'], 'blue'),
        text(lx + 84, 186, '전달됨', c['blue'], 12, 'start'),
        arrow(f'M{lx + 117} 149H{lx + 225}', c['orange'], 'orange', dash='5 4'),
        text(lx + 172, 141, '연결 불가', c['orange'], 12, weight=600),
        text(lx, 270, '다른 host의 fd.sock 경로는 열 수 없다.', c['muted'], 12, 'start'),
        text(lx, 288, '30초 재시도 뒤 timeout.', c['muted'], 12, 'start'),
        _handle_panel(c, rx, 'FABRIC handle', 'byte 열이라 allgather로 전달'),
        _rank(c, rx + 33, 132, 'rank 0'),
        _rank(c, rx + 33, 198, 'rank 1'),
        _rank(c, rx + 227, 132, 'rank 8'),
        arrow(f'M{rx + 75} 166V196', c['blue'], 'blue'),
        arrow(f'M{rx + 117} 149H{rx + 225}', c['blue'], 'blue'),
        text(rx + 172, 141, '전달됨', c['blue'], 12),
        text(rx, 270, '조건: 모든 rank가 같은 NVLink fabric에', c['muted'], 12, 'start'),
        text(rx, 288, '속해야 한다 (같은 clusterUuid).', c['muted'], 12, 'start'),
    ]
    return wrap(c, 308, label, '\n'.join(p))


# --- 4. workspace creation order before and after the fix ----------------------------------


def _skipped(c, x, y, w, h, lines) -> str:
    """A step that never runs: dashed outline, muted text."""
    p = [box(x, y, w, h, c['rule'], dash='5 4')]
    top = y + h / 2 - (len(lines) - 1) * 8 + 4.5
    for i, line in enumerate(lines):
        p.append(text(x + w / 2, top + i * 16, line, c['muted'], 12))
    return ''.join(p)


def _fix_row(c, y, head, steps, result, note) -> str:
    """One run on the bug cluster: backend, two steps in order, and what comes out."""
    xs, ws, h = (24, 184, 394, 604), (130, 180, 180, 132), 52
    p = [
        text(24, y - 12, head, c['ink'], 13, 'start', 600),
        node(xs[0], y, ws[0], h, ['backend 결정', 'auto → mnnvl'], c),
    ]
    for i, (kind, lines) in enumerate(steps, start=1):
        if kind == 'skipped':
            p.append(_skipped(c, xs[i], y, ws[i], h, lines))
        else:
            p.append(node(xs[i], y, ws[i], h, lines, c, kind))
    p.append(node(xs[3], y, ws[3], h, result[1], c, result[0]))
    for i in range(3):
        dash = '5 4' if steps[min(i, 1)][0] == 'skipped' and i >= 1 else None
        d = f'M{xs[i] + ws[i]} {y + h / 2}H{xs[i + 1] - 2}'
        p.append(arrow(d, c['muted'], 'muted', dash=dash))
    p.append(text(xs[1], y + h + 20, note, c['muted'], 12, 'start'))
    return '\n'.join(p)


def fig_fix(c) -> str:
    label = 'workspace의 생성과 검사 순서: 수정 전과 후'
    ya, yb = 76, 202
    p = [
        title(label, c),
        _fix_row(
            c,
            ya,
            '수정 전 (v0.27.1): 생성이 먼저',
            (
                ('orange', ['workspace 생성', '30초 멈춘 뒤 예외']),
                ('skipped', ['검사: multicast 주소', '도달하지 못함']),
            ),
            ('orange', ['None', 'GPU memory 잔류']),
            '생성이 실패해야 쓸 수 없다는 것을 안다.',
        ),
        _fix_row(
            c,
            yb,
            '수정 후 (#53377): 검사가 먼저',
            (
                ('blue', ['검사: NVLink fabric', 'rank 16개 모두 False']),
                ('skipped', ['workspace 생성', '실행하지 않음']),
            ),
            ('muted', ['None', '멈춤과 잔류 없음']),
            '모든 rank가 답을 모으고, 하나라도 False면 생성하지 않는다.',
        ),
        text(
            24,
            yb + 108,
            '두 경우 모두 None을 받은 호출자는 일반 all-reduce와 RMSNorm으로 돌아간다.',
            c['muted'],
            12,
            'start',
        ),
    ]
    return wrap(c, yb + 128, label, '\n'.join(p))


# --- 5. the capability vote: every rank answers, every rank gets the same AND ----------------

VOTE_X0, VOTE_PITCH, VOTE_GAP, VOTE_H = 140, 36, 16, 24
VOTE_FAILED = 11


def _vote_x(i) -> float:
    return VOTE_X0 + i * VOTE_PITCH + (VOTE_GAP if i >= 8 else 0)


def _vote_cells(c, y, name, values) -> str:
    """One row of 16 rank cells. A value is 'T', 'F', a rank number, or None for no answer."""
    p = [text(24, y + 16.5, name, c['ink'], 12, 'start', 600)]
    for i, v in enumerate(values):
        x = _vote_x(i)
        if v is None:
            p.append(box(x, y, 34, VOTE_H, c['rule'], rx=5, dash='4 3'))
        elif v in ('T', 'F'):
            key = 'blue' if v == 'T' else 'orange'
            p.append(box(x, y, 34, VOTE_H, c[key], c[key], 0.2, rx=5))
            p.append(text(x + 17, y + 16.5, v, c['ink'], 12, weight=600))
        else:
            p.append(text(x + 17, y + 16.5, v, c['muted'], 11))
    return ''.join(p)


def _vote_band(c, y, name, body, dash=None) -> str:
    x, w = VOTE_X0, _vote_x(15) + 34 - VOTE_X0
    return ''.join(
        [
            text(24, y + 16.5, name, c['ink'], 12, 'start', 600),
            box(x, y, w, VOTE_H, c['rule'], rx=5, dash=dash),
            text(x + w / 2, y + 16.5, body, c['ink'], 12),
        ]
    )


def fig_vote(c) -> str:
    label = 'capability 투표: probe, allgather, AND'
    probe = ['F' if i == VOTE_FAILED else 'T' for i in range(16)]
    ya, yb = 96, 318
    p = [
        title(label, c),
        text(24, ya - 34, '(a) 모든 rank가 투표할 때', c['ink'], 13, 'start', 600),
        text(_vote_x(0), ya - 10, 'node 0', c['muted'], 11, 'start'),
        text(_vote_x(8), ya - 10, 'node 1', c['muted'], 11, 'start'),
        _vote_cells(c, ya, 'rank', [str(i) for i in range(16)]),
        _vote_cells(c, ya + 30, '1. probe', probe),
        _vote_band(c, ya + 66, '2. allgather', '모든 rank가 답 16개의 같은 목록을 받는다'),
        _vote_cells(c, ya + 102, '3. AND', ['F'] * 16),
        text(
            24,
            ya + 150,
            'rank 11은 probe가 예외를 던졌지만 False로 투표한다. 모든 rank가 같은 결과 False를 얻는다.',
            c['muted'],
            12,
            'start',
        ),
        text(24, yb - 34, '(b) rank 11이 투표하지 않고 빠질 때', c['ink'], 13, 'start', 600),
        _vote_cells(c, yb - 20, '1. probe', [None if v == 'F' else v for v in probe]),
        _vote_band(c, yb + 16, '2. allgather', 'rank 11의 답을 기다린다', dash='5 4'),
        text(
            24,
            yb + 64,
            'allgather는 모든 rank가 불러야 끝난다. 나머지 rank 15개가 여기서 멈춘다.',
            c['muted'],
            12,
            'start',
        ),
    ]
    return wrap(c, yb + 84, label, '\n'.join(p))


FIGURES = (
    ('ar-01-topology', fig_topology),
    ('ar-01b-address-space', fig_address_space),
    ('ar-01c-partial-sum', fig_partial_sum),
    ('ar-01e-row-column', fig_row_column),
    ('ar-01d-layer', fig_layer),
    ('ar-01f-moe', fig_moe),
    ('ar-02-two-paths', fig_two_paths),
    ('ar-03-handles', fig_handles),
    ('ar-04-fix', fig_fix),
    ('ar-05-vote', fig_vote),
)

OUT.mkdir(parents=True, exist_ok=True)
for name, fn in FIGURES:
    for theme, colors in THEMES.items():
        (OUT / f'{name}-{theme}.svg').write_text(fn(colors))
