"""Generate light/dark SVG figures for the matrix gradient post.

Usage: python3 scripts/figures/matrix_gradient.py <repo_root>

Writes mg-*-{light,dark}.svg to public/assets/images/post/2026-11-14-Matrix-Gradient/.
The KaTeX font CSS is copied from an existing figure of the LLM 2025-2026 post.
"""

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
REF = ROOT / 'public/assets/images/post/2026-11-01-LLM-in-2025-2026/at-00a-swa-light.svg'
OUT = ROOT / 'public/assets/images/post/2026-11-14-Matrix-Gradient'
FONT_STYLE = re.search(r'<style>.*?</style>', REF.read_text(), re.S).group(0)

THEMES = {
    'light': {'bg': '#f5f5f5', 'ink': '#2d3142', 'muted': '#66708a', 'rule': '#cfd3dc',
              'blue': '#2e5aa8', 'orange': '#eb6c36'},
    'dark': {'bg': '#2d3142', 'ink': '#f5f5f5', 'muted': '#a3acbf', 'rule': '#4a5068',
             'blue': '#6a95d8', 'orange': '#f08a59'},
}
SANS = "'Pretendard', system-ui, 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif"
MATH = 'font-family="KaTeX_Math, \'Times New Roman\', serif" font-style="italic" font-weight="normal"'
MAIN = 'font-family="KaTeX_Main, \'Times New Roman\', serif" font-style="normal" font-weight="normal"'


def spans(parts: list[tuple[str, str]], size: float) -> str:
    """Render (kind, text) parts as tspans. Kinds: t, m, r, sub, sup."""
    out, shift = [], 0.0
    for part in parts:
        kind, text = part[0], part[1]
        want = {'sub': size * 0.25, 'msub': size * 0.25, 'sup': -size * 0.38}.get(kind, 0.0)
        dy = want - shift
        shift = want
        attr = f' dy="{dy:.2f}"' if dy else ''
        if len(part) > 2:
            attr += f' fill="{part[2]}"'
        if kind == 't':
            out.append(f'<tspan{attr}>{text}</tspan>')
        elif kind == 'm':
            out.append(f'<tspan{attr} {MATH}>{text}</tspan>')
        elif kind == 'r':
            out.append(f'<tspan{attr} {MAIN}>{text}</tspan>')
        elif kind == 'msub':
            out.append(f'<tspan{attr} {MATH} font-size="{size * 0.7:.1f}">{text}</tspan>')
        else:
            out.append(f'<tspan{attr} {MAIN} font-size="{size * 0.7:.1f}">{text}</tspan>')
    if shift:
        out.append(f'<tspan dy="{-shift:.2f}">​</tspan>')
    return ''.join(out)


def text(x, y, parts, fill, size=13, anchor='middle', weight=None) -> str:
    w = f' font-weight="{weight}"' if weight else ''
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}"{w} '
            f'fill="{fill}">{spans(parts, size)}</text>')


def box(x, y, w, h, stroke, fill='none', opacity=1.0, sw=1.25) -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
            f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"/>')


def bar(cx, y, color, half=4.5) -> str:
    return (f'<line x1="{cx - half}" y1="{y}" x2="{cx + half}" y2="{y}" stroke="{color}" '
            f'stroke-width="1.1"/>')


def wrap(c, w, h, label, body) -> str:
    markers = ''.join(
        f'<marker id="a-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" '
        f'fill="{c[k]}"/></marker>' for k in ('muted', 'blue', 'orange'))
    return (f'<svg fill="{c["ink"]}" xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'font-family="{SANS}" viewBox="0 0 {w} {h}" role="img" aria-label="{label}" '
            f'font-size="13">\n<rect x="0" y="0" width="{w}" height="{h}" fill="{c["bg"]}"/>'
            f'{FONT_STYLE}<defs>{markers}</defs>\n{body}\n</svg>\n')


def fig_dual(c) -> str:
    title = 'derivative와 gradient는 같은 변화량을 낸다'
    p = []
    p.append(text(24, 34, [('t', title)], c['ink'], 15, 'start', 600))
    p.append(text(24, 54, [('t', '예: '), ('m', 'f'), ('r', '('), ('m', 'x'), ('sub', '1'),
                           ('r', ', '), ('m', 'x'), ('sub', '2'), ('r', ') = '), ('m', 'x'),
                           ('sub', '1'), ('sup', '2'), ('r', ' + 3'), ('m', 'x'), ('sub', '2')],
                  c['muted'], 12, 'start'))
    line = f'fill="none" stroke="{c["muted"]}" stroke-width="1.25" marker-end="url(#a-muted)"'
    for d in ('M196 182H228V112H258', 'M196 182H228V252H258',
              'M500 112H528V176H554', 'M500 252H528V188H554'):
        p.append(f'<path d="{d}" {line}/>')
    p.append(f'<line x1="380" y1="146" x2="380" y2="218" stroke="{c["orange"]}" '
             f'stroke-width="1.25" marker-start="url(#a-orange)" marker-end="url(#a-orange)"/>')
    p.append(text(392, 180, [('t', '같은 정보')], c['ink'], 12, 'start', 600))
    p.append(text(392, 196, [('t', 'inner product로 대응')], c['muted'], 11.5, 'start'))
    # input
    p.append(box(24, 154, 172, 56, c['muted']))
    p.append(text(110, 178, [('t', '방향 '), ('m', 'dx')], c['ink'], 13.5, weight=600))
    p.append(text(110, 197, [('r', '('), ('m', 'dx'), ('sub', '1'), ('r', ', '), ('m', 'dx'),
                             ('sub', '2'), ('r', ')')], c['muted'], 13))
    # derivative
    p.append(box(260, 84, 240, 56, c['blue'], c['blue'], 0.10))
    p.append(text(380, 108, [('t', 'derivative '), ('m', 'df'), ('msub', 'x'),
                             ('t', ' : linear function')], c['ink'], 13.5, weight=600))
    p.append(text(380, 127, [('m', 'dx'), ('r', ' ↦ 2'), ('m', 'x'), ('sub', '1'),
                             ('m', 'dx'), ('sub', '1'), ('r', ' + 3'), ('m', 'dx'),
                             ('sub', '2')], c['muted'], 13))
    # gradient
    p.append(box(260, 224, 240, 56, c['blue'], c['blue'], 0.10))
    p.append(text(380, 248, [('t', 'gradient '), ('r', '∇'), ('m', 'f'), ('r', '('),
                             ('m', 'x'), ('r', ') = (2'), ('m', 'x'), ('sub', '1'),
                             ('r', ', 3)')], c['ink'], 13.5, weight=600))
    p.append(text(380, 267, [('t', 'inner product '), ('r', '⟨∇'), ('m', 'f'),
                             ('r', '('), ('m', 'x'), ('r', '), '), ('m', 'dx'),
                             ('r', '⟩')], c['muted'], 13))
    # output
    p.append(box(556, 154, 180, 56, c['orange'], c['orange'], 0.14, 2))
    p.append(text(646, 178, [('t', '변화량 '), ('m', 'df')], c['ink'], 13.5, weight=600))
    p.append(text(646, 197, [('t', 'scalar 하나')], c['muted'], 12.5))
    return wrap(c, 760, 304, title, '\n'.join(p))


def pairing(x0, y, a, b, c) -> str:
    """<a-bar, b> laid out by hand so the bar sits on the first symbol."""
    s = 16
    return ''.join([
        text(x0, y, [('r', '⟨')], c['ink'], s),
        text(x0 + 11, y, [('m', a)], c['ink'], s),
        bar(x0 + 11, y - 12.5, c['ink'], 5),
        text(x0 + 20, y, [('r', ',')], c['ink'], s),
        text(x0 + 37, y, [('m', b)], c['ink'], s),
        text(x0 + 52, y, [('r', '⟩')], c['ink'], s),
    ])


def fig_vjp(c) -> str:
    title = '변화량은 앞으로, gradient는 뒤로 흐른다'
    p = []
    p.append(text(24, 34, [('t', title)], c['ink'], 15, 'start', 600))
    p.append(text(24, 54, [('t', 'layer '), ('m', 'y'), ('r', ' = '), ('m', 'h'), ('r', '('),
                           ('m', 'x'), ('r', ')'), ('t', ' 하나를 지날 때. '), ('m', 'J'),
                           ('t', '는 '), ('m', 'h'), ('t', '의 Jacobian')],
                  c['muted'], 12, 'start'))
    lx, rx, w, h = 70, 510, 180, 58
    lcx, rcx = lx + w / 2, rx + w / 2
    # forward row
    p.append(box(lx, 82, w, h, c['blue'], c['blue'], 0.10))
    p.append(text(lcx, 107, [('m', 'dx')], c['ink'], 17))
    p.append(text(lcx, 127, [('t', '입력의 변화량')], c['muted'], 12))
    p.append(box(rx, 82, w, h, c['blue'], c['blue'], 0.10))
    p.append(text(rcx, 107, [('m', 'dy'), ('r', ' = '), ('m', 'J'), ('r', ' '), ('m', 'dx')],
                  c['ink'], 17))
    p.append(text(rcx, 127, [('t', '출력의 변화량')], c['muted'], 12))
    p.append(f'<line x1="{lx + w + 6}" y1="111" x2="{rx - 8}" y2="111" stroke="{c["blue"]}" '
             f'stroke-width="1.5" marker-end="url(#a-blue)"/>')
    p.append(text(380, 101, [('t', 'JVP: '), ('m', 'J'), ('r', ' '), ('m', 'v')], c['blue'], 14,
                  weight=600))
    p.append(text(380, 129, [('t', 'forward. Jacobian 오른쪽에 vector')], c['muted'], 12))
    # backward row
    p.append(box(lx, 248, w, h, c['orange'], c['orange'], 0.14))
    p.append(text(lcx - 40, 273, [('m', 'x'), ('sup', '⊤')], c['ink'], 17, 'start'))
    p.append(bar(lcx - 35, 260, c['ink'], 5))
    p.append(text(lcx - 14, 273, [('r', '=')], c['ink'], 17, 'start'))
    p.append(text(lcx + 6, 273, [('m', 'y'), ('sup', '⊤'), ('r', ' '), ('m', 'J')],
                  c['ink'], 17, 'start'))
    p.append(bar(lcx + 10.5, 260, c['ink'], 5))
    p.append(text(lcx, 293, [('t', '입력의 gradient')], c['muted'], 12))
    p.append(box(rx, 248, w, h, c['orange'], c['orange'], 0.14))
    p.append(text(rcx, 273, [('m', 'y')], c['ink'], 17))
    p.append(bar(rcx + 0.5, 260, c['ink'], 5))
    p.append(text(rcx, 293, [('t', '출력의 gradient (loss 쪽에서 받음)')], c['muted'], 12))
    p.append(f'<line x1="{rx - 6}" y1="277" x2="{lx + w + 8}" y2="277" stroke="{c["orange"]}" '
             f'stroke-width="1.5" marker-end="url(#a-orange)"/>')
    p.append(text(380, 267, [('t', 'VJP: '), ('m', 'v'), ('sup', '⊤'), ('r', ' '), ('m', 'J')],
                  c['orange'], 14, weight=600))
    p.append(text(380, 295, [('t', 'backward. Jacobian 왼쪽에 vector')], c['muted'], 12))
    p.append(text(lcx, 326, [('t', '양변을 transpose하면 '), ('m', 'J'), ('sup', '⊤'),
                             ('t', '를 곱한 꼴')], c['muted'], 11.5))
    # pairing row
    dash = f'stroke="{c["muted"]}" stroke-width="1" stroke-dasharray="3 3"'
    for cx in (lcx, rcx):
        p.append(f'<line x1="{cx}" y1="144" x2="{cx}" y2="170" {dash}/>')
        p.append(f'<line x1="{cx}" y1="218" x2="{cx}" y2="244" {dash}/>')
    p.append(pairing(lcx - 26, 200, 'x', 'dx', c))
    p.append(pairing(rcx - 26, 200, 'y', 'dy', c))
    p.append(f'<line x1="{lcx + 44}" y1="195" x2="262" y2="195" {dash}/>')
    p.append(f'<line x1="498" y1="195" x2="{rcx - 44}" y2="195" {dash}/>')
    p.append(text(380, 192, [('t', '둘 다 같은 '), ('m', 'dL')], c['ink'], 13, weight=600))
    p.append(text(380, 209, [('t', '같은 움직임을 '), ('m', 'x'), ('t', ' 쪽과 '), ('m', 'y'),
                             ('t', ' 쪽에서 잰 loss 변화')], c['muted'], 11.5))
    return wrap(c, 760, 344, title, '\n'.join(p))


def mat(x, cy, cols, rows, col, label, c, dashed=False, u=16) -> str:
    """Matrix block of cols x rows cells, vertically centered on cy."""
    w, h = cols * u, rows * u
    y = cy - h / 2
    dash = ' stroke-dasharray="4 3"' if dashed else ''
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{col}" '
           f'fill-opacity="0.16" stroke="{col}" stroke-width="1.5"{dash}/>']
    for i in range(1, cols):
        out.append(f'<line x1="{x + i * u}" y1="{y}" x2="{x + i * u}" y2="{y + h}" '
                   f'stroke="{col}" stroke-opacity="0.3" stroke-width="0.8"/>')
    for j in range(1, rows):
        out.append(f'<line x1="{x}" y1="{y + j * u}" x2="{x + w}" y2="{y + j * u}" '
                   f'stroke="{col}" stroke-opacity="0.3" stroke-width="0.8"/>')
    out.append(f'<rect x="{x + w / 2 - 15}" y="{cy - 11}" width="30" height="21" rx="4" '
               f'fill="{c["bg"]}" fill-opacity="0.85"/>')
    out.append(text(x + w / 2, cy + 5.5, label, c['ink'], 15))
    return ''.join(out)


def fig_delta(c) -> str:
    title = '첫째. 미분 전체를 tensor로 적지 않고 변화량만 적는다'
    g, bl, o = c['muted'], c['blue'], c['orange']
    p = []
    p.append(text(24, 34, [('t', title)], c['ink'], 15, 'start', 600))
    p.append(text(24, 54, [('m', 'W'), ('t', '를 '), ('m', 'dW'), ('t', '만큼 움직이면 '),
                           ('m', 'Y'), ('t', '는 '), ('m', 'X'), ('r', ' '), ('m', 'dW'),
                           ('t', '만큼 움직인다. 블록 크기는 shape, 예: '), ('m', 'B'), ('r', ' = 4, '),
                           ('m', 'D'), ('r', ' = 3, '), ('m', 'F'), ('r', ' = 2')],
                  c['muted'], 12, 'start'))
    xX, xW, xdW, xY, xdY = 196, 290, 354, 452, 516
    rows = [(118, '움직이기 전'), (222, '움직인 후'), (326, '새로 생긴 부분')]
    p.append(f'<rect x="16" y="{326 - 46}" width="728" height="92" rx="8" fill="{c["rule"]}" '
             f'fill-opacity="0.35"/>')
    for cy, lab in rows:
        p.append(text(24, cy + 5, [('t', lab)], c['ink'], 13, 'start', 600))
    op = lambda x, cy, ch, size=18: text(x, cy + 6, [('r', ch)], c['ink'], size)
    X = lambda cy: mat(xX, cy, 3, 4, g, [('m', 'X')], c)
    # row 1
    cy = 118
    p += [X(cy), op(260, cy, '·'), mat(xW, cy, 2, 3, bl, [('m', 'W')], c),
          op(428, cy, '='), mat(xY, cy, 2, 4, o, [('m', 'Y')], c)]
    for x, w, h, lab in ((xX, 48, 64, 'B'), (xW, 32, 48, 'D'), (xY, 32, 64, 'B')):
        sub = {'B': [('m', 'B'), ('r', ' × '), ('m', 'D')] if x == xX else
               [('m', 'B'), ('r', ' × '), ('m', 'F')],
               'D': [('m', 'D'), ('r', ' × '), ('m', 'F')]}[lab]
        p.append(text(x + w / 2, cy + h / 2 + 17, sub, c['muted'], 11.5))
    p.append(text(578, cy + 5, [('t', 'forward 계산')], c['muted'], 12, 'start'))
    # row 2
    cy = 222
    p += [X(cy), op(260, cy, '·'), op(278, cy, '(', 30),
          mat(xW, cy, 2, 3, bl, [('m', 'W')], c), op(338, cy, '+'),
          mat(xdW, cy, 2, 3, bl, [('m', 'dW')], c, True), op(398, cy, ')', 30),
          op(428, cy, '='), mat(xY, cy, 2, 4, o, [('m', 'Y')], c), op(500, cy, '+'),
          mat(xdY, cy, 2, 4, o, [('m', 'dY')], c, True)]
    p.append(text(578, cy - 3, [('m', 'W'), ('t', '를 조금 움직이면')], c['muted'], 12, 'start'))
    p.append(text(578, cy + 14, [('m', 'Y'), ('t', '도 조금 움직인다')], c['muted'], 12,
                  'start'))
    # row 3
    cy = 326
    p += [X(cy), op(260, cy, '·'), mat(xdW, cy, 2, 3, bl, [('m', 'dW')], c, True),
          op(428, cy, '='), mat(xdY, cy, 2, 4, o, [('m', 'dY')], c, True)]
    p.append(text(xdW + 16, cy + 24 + 17, [('m', 'D'), ('r', ' × '), ('m', 'F')],
                  c['muted'], 11.5))
    p.append(text(578, cy - 3, [('t', '둘째 줄에서 첫째 줄을 빼면')], c['muted'], 12, 'start'))
    p.append(text(578, cy + 14, [('m', 'X'), ('t', '는 그대로, '), ('m', 'W'), ('t', ' 자리만 '),
                                 ('m', 'dW')], c['ink'], 12, 'start', 600))
    return wrap(c, 760, 388, title, '\n'.join(p))


def fig_pair(c) -> str:
    title = '둘째. gradient는 dW의 원소에 곱해서 dL을 만드는 수들이다'
    o, bl = c['orange'], c['blue']
    p = []
    p.append(text(24, 34, [('t', '둘째. gradient는 '), ('m', 'dW'), ('t', '의 원소에 곱해서 '),
                           ('m', 'dL'), ('t', '을 만드는 수들이다')],
                  c['ink'], 15, 'start', 600))
    p.append(text(24, 54, [('t', '예: 첫째 그림의 '), ('m', 'W'), ('r', ' ('), ('m', 'D'),
                           ('r', ' × '), ('m', 'F'), ('r', ' = 3 × 2)'),
                           ('t', '. 숫자는 임의로 고른 값')], c['muted'], 12, 'start'))
    cw, ch = 64, 40
    grads = [['2', '−1'], ['0', '3'], ['1', '−2']]
    gx, dx_, gy = 70, 322, 100
    p.append(text(gx + cw, gy - 12, [('t', 'gradient')], o, 13, weight=600))
    p.append(text(dx_ + cw, gy - 12, [('t', '변화량 '), ('m', 'dW')], bl, 13, weight=600))
    for i in range(3):
        for j in range(2):
            x, y = gx + j * cw, gy + i * ch
            p.append(f'<rect x="{x}" y="{y}" width="{cw - 4}" height="{ch - 4}" rx="6" '
                     f'fill="{o}" fill-opacity="0.14" stroke="{o}" stroke-width="1.25"/>')
            p.append(text(x + cw / 2 - 2, y + 24, [('r', grads[i][j])], c['ink'], 17))
            x = dx_ + j * cw
            p.append(f'<rect x="{x}" y="{y}" width="{cw - 4}" height="{ch - 4}" rx="6" '
                     f'fill="{bl}" fill-opacity="0.10" stroke="{bl}" stroke-width="1.25" '
                     f'stroke-dasharray="4 3"/>')
            p.append(text(x + cw / 2 - 2, y + 24, [('m', 'dW'), ('sub', f'{i + 1}{j + 1}')],
                          c['ink'], 15))
    mid = gy + 1.5 * ch
    p.append(text(gx + cw, gy + 3 * ch + 14, [('m', 'W'), ('t', '와 같은 shape')],
                  c['muted'], 12))
    p.append(text(dx_ + cw, gy + 3 * ch + 14, [('t', '첫째 그림의 '), ('m', 'dW')],
                  c['muted'], 12))
    p.append(text(258, mid + 4, [('r', '×')], c['ink'], 22))
    p.append(text(258, mid + 24, [('t', '같은 위치끼리')], c['muted'], 11.5))
    p.append(f'<line x1="462" y1="{mid - 2}" x2="548" y2="{mid - 2}" '
             f'stroke="{c["muted"]}" stroke-width="1.25" marker-end="url(#a-muted)"/>')
    p.append(text(505, mid - 12, [('t', '모두 더한다')], c['muted'], 11.5))
    p.append(box(556, mid - 32, 170, 60, c['ink'], 'none', 1, 1.5))
    p.append(text(641, mid - 6, [('m', 'dL')], c['ink'], 18))
    p.append(text(641, mid + 14, [('t', 'loss의 변화량, scalar 하나')], c['muted'], 12))
    ly = gy + 3 * ch + 34
    p.append(f'<line x1="24" y1="{ly}" x2="736" y2="{ly}" stroke="{c["rule"]}" stroke-width="1"/>')
    f = [('m', 'dL'), ('r', ' = ')]
    coefs = ['2', '(−1)', '0', '3', '1', '(−2)']
    idxs = ['11', '12', '21', '22', '31', '32']
    for k, (coef, idx) in enumerate(zip(coefs, idxs)):
        if k:
            f.append(('r', ' + '))
        f += [('r', coef, o), ('r', '·'), ('m', 'dW', bl), ('sub', idx, bl)]
    p.append(text(380, ly + 32, f, c['ink'], 15))
    p.append(text(380, ly + 56, [('t', '곱해지는 수 6개를 '), ('m', 'W'),
                                 ('t', '와 같은 shape로 모으면 gradient')], c['muted'], 12))
    return wrap(c, 760, ly + 76, title, '\n'.join(p))


def grid(x, y, cols, rows, col, hot, c, u=28) -> str:
    """Matrix of cells; cells in `hot` are filled strongly."""
    out = []
    for i in range(rows):
        for j in range(cols):
            on = (i, j) in hot
            out.append(f'<rect x="{x + j * u}" y="{y + i * u}" width="{u}" height="{u}" '
                       f'fill="{col}" fill-opacity="{0.75 if on else 0.08}" '
                       f'stroke="{col}" stroke-opacity="0.45" stroke-width="0.8"/>')
    out.append(f'<rect x="{x}" y="{y}" width="{cols * u}" height="{rows * u}" rx="2" '
               f'fill="none" stroke="{col}" stroke-width="1.5"/>')
    return ''.join(out)


def fig_cell(c) -> str:
    title = 'W_ab 한 칸만 움직이면 Y는 b번째 열만 움직인다'
    g, bl, o, ink, mu = c['muted'], c['blue'], c['orange'], c['ink'], c['muted']
    u = 28
    p = []
    p.append(text(24, 34, [('m', 'W'), ('msub', 'ab'), ('t', ' 한 칸만 움직이면 '), ('m', 'Y'),
                           ('t', '는 '), ('m', 'b'), ('t', '번째 열만 움직인다')],
                  ink, 15, 'start', 600))
    p.append(text(24, 54, [('t', '예: '), ('m', 'a'), ('r', ' = 2, '), ('m', 'b'),
                           ('r', ' = 1'), ('t', '. 진한 칸이 관여하는 원소')], mu, 12, 'start'))
    top = 108
    xX, xW, xY = 48, 188, 300
    p.append(grid(xX, top, 3, 4, g, {(i, 1) for i in range(4)}, c))
    p.append(grid(xW, top + u / 2, 2, 3, bl, {(1, 0)}, c))
    p.append(grid(xY, top, 2, 4, o, {(i, 0) for i in range(4)}, c))
    p.append(text(xX + 1.5 * u, top + 4 * u + 22, [('m', 'X')], ink, 16))
    p.append(text(xW + u, top + 3.5 * u + 22, [('m', 'W')], ink, 16))
    p.append(text(xY + u, top + 4 * u + 22, [('m', 'Y')], ink, 16))
    p.append(text(xX + 4 * u - 12, top + 2 * u + 6, [('r', '·')], ink, 18))
    p.append(text(xW + 2 * u + 28, top + 2 * u + 6, [('r', '=')], ink, 18))
    # index tags
    p.append(text(xX + 1.5 * u, top - 10, [('m', 'a')], ink, 14))
    p.append(text(xW + 0.5 * u, top + u / 2 - 10, [('m', 'b')], ink, 14))
    p.append(text(xW - 12, top + u / 2 + 1.5 * u + 5, [('m', 'a')], ink, 14))
    p.append(text(xY + 0.5 * u, top - 10, [('m', 'b')], ink, 14))
    # explanation
    tx = 412
    lines = [
        [('t', '1. '), ('m', 'W'), ('msub', 'ab'), ('t', '는 '), ('m', 'W'), ('t', '의 '),
         ('m', 'b'), ('t', '번째 열에 있다')],
        [('t', '2. '), ('m', 'Y'), ('t', '의 '), ('m', 'b'), ('t', '번째 열은 '), ('m', 'X'),
         ('t', '와 '), ('m', 'W'), ('t', '의 '), ('m', 'b'), ('t', '번째 열로만 만든다')],
        [('t', '3. 그래서 '), ('m', 'Y'), ('t', '의 '), ('m', 'b'),
         ('t', '번째 열만 움직이고, 다른 열은 그대로다')],
        [('t', '4. '), ('m', 'Y'), ('msub', 'ib'), ('t', '에서 '), ('m', 'W'), ('msub', 'ab'),
         ('t', '와 곱해지는 것은 '), ('m', 'X'), ('msub', 'ia'), ('t', ' 하나다')],
    ]
    for k, ln in enumerate(lines):
        p.append(text(tx, 112 + k * 28, ln, ink, 13, 'start'))
    p.append(box(tx, 232, 324, 50, o, o, 0.12, 1.5))
    p.append(text(tx + 162, 263, [('m', 'dY'), ('msub', 'ib'), ('r', ' = '), ('m', 'X'),
                                  ('msub', 'ia'), ('r', ' '), ('m', 'dW'), ('msub', 'ab')],
                  ink, 18))
    p.append(text(tx + 162, 302, [('t', '다른 열 '), ('m', 'j'), ('r', ' ≠ '), ('m', 'b'),
                                  ('t', '에서는 '), ('m', 'dY'), ('msub', 'ij'), ('r', ' = 0')],
                  mu, 12.5))
    return wrap(c, 760, 330, title, '\n'.join(p))


def arrow(x1, y1, x2, y2, col, key, w=2.0, dash='') -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
            f'stroke-width="{w}" marker-end="url(#a-{key})"{d}/>')


def fig_terrain(c) -> str:
    title = '지형으로 본 derivative와 gradient'
    ink, mu, bl, o, rule = c['ink'], c['muted'], c['blue'], c['orange'], c['rule']
    p = []
    p.append(text(24, 34, [('t', title)], ink, 15, 'start', 600))
    p.append(text(24, 54, [('t', '위에서 내려다본 지도. 선은 높이가 같은 곳을 이은 등고선')],
                  mu, 12, 'start'))
    mx, my, mw, mh = 24, 72, 400, 340
    cx, cy, u = 214, 246, 34
    p.append(f'<defs><clipPath id="map"><rect x="{mx}" y="{my}" width="{mw}" height="{mh}" '
             f'rx="8"/></clipPath></defs>')
    p.append(f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="8" fill="none" '
             f'stroke="{rule}" stroke-width="1.25"/>')
    g = []
    norm = 13 ** 0.5
    for k in range(-5, 6):
        lev = 3 * k
        # point on line (svg): center + lev/13 * (2, -3) * u
        px, py = cx + lev / 13 * 2 * u, cy - lev / 13 * 3 * u
        dx, dy = 3 / norm * 400, 2 / norm * 400
        g.append(f'<line x1="{px - dx:.1f}" y1="{py - dy:.1f}" x2="{px + dx:.1f}" '
                 f'y2="{py + dy:.1f}" stroke="{mu if lev == 0 else rule}" '
                 f'stroke-width="{1.2 if lev == 0 else 1}"/>')
    p.append(f'<g clip-path="url(#map)">{"".join(g)}</g>')
    halo = lambda x: x.replace('<text ', f'<text stroke="{c["bg"]}" stroke-width="5" paint-order="stroke" stroke-linejoin="round" ', 1)
    p.append(halo(text(mx + mw - 14, my + 24, [('t', '↗ 높은 쪽')], mu, 11.5, 'end')))
    p.append(halo(text(mx + 44, my + mh - 14, [('t', '↙ 낮은 쪽')], mu, 11.5, 'start')))
    # compass
    p.append(arrow(mx + 22, my + mh - 38, mx + 22, my + mh - 66, mu, 'muted', 1.25))
    p.append(halo(text(mx + 22, my + mh - 24, [('t', '북')], mu, 11)))
    # arrows
    gx, gy = cx + 2 * u, cy - 3 * u
    sw = 2.25
    dash = f'stroke="{bl}" stroke-width="1.25" stroke-dasharray="3 3"'
    p.append(f'<line x1="{cx + 2 * u}" y1="{cy}" x2="{gx}" y2="{gy}" {dash}/>')
    p.append(f'<line x1="{cx}" y1="{cy - 3 * u}" x2="{gx}" y2="{gy}" {dash}/>')
    p.append(arrow(cx, cy, cx - 2 * u * 0.9, cy + 3 * u * 0.9, o, 'orange', sw, '5 4'))
    p.append(arrow(cx, cy, cx + 3 / norm * 78, cy + 2 / norm * 78, mu, 'muted', sw))
    p.append(arrow(cx, cy, cx + 2 * u, cy, bl, 'blue', sw))
    p.append(arrow(cx, cy, cx, cy - 3 * u, bl, 'blue', sw))
    p.append(arrow(cx, cy, gx, gy, o, 'orange', sw))
    p.append(f'<circle cx="{cx}" cy="{cy}" r="4.5" fill="{ink}"/>')
    p.append(halo(text(cx - 12, cy + 5, [('t', '지금 위치')], ink, 11.5, 'end')))
    p.append(halo(text(cx + 2 * u + 8, cy + 5, [('t', '동쪽 성분 2')], bl, 12, 'start', 600)))
    p.append(halo(text(cx + 2 * u + 8, cy + 21, [('t', '(동 1걸음에 +2m)')], bl, 11, 'start')))
    p.append(halo(text(cx - 8, cy - 3 * u + 10, [('t', '북쪽 성분 3')], bl, 12, 'end', 600)))
    p.append(halo(text(cx - 8, cy - 3 * u + 26, [('t', '(북 1걸음에 +3m)')], bl, 11, 'end')))
    p.append(halo(text(gx + 8, gy - 4, [('t', 'gradient '), ('r', '(2, 3)')], o, 12.5, 'start', 600)))
    p.append(halo(text(gx + 8, gy + 12, [('t', '가장 많이 오르는 방향')], o, 11.5, 'start')))
    p.append(halo(text(cx + 3 / norm * 78 - 4, cy + 2 / norm * 78 + 22, [('t', '등고선을 따라가면 0m')],
                  mu, 11.5, 'start')))
    ex, ey = cx - 2 * u * 0.9, cy + 3 * u * 0.9
    p.append(halo(text(ex + 8, ey + 4, [('t', 'gradient descent')], o, 11.5, 'start', 600)))
    p.append(halo(text(ex + 8, ey + 19, [('t', '가장 많이 내려가는 방향')], o, 11.5, 'start')))
    # right panel
    tx = 448
    df = [('m', 'df'), ('msub', 'x')]
    p.append(text(tx, 92, [('t', 'derivative')], bl, 13.5, 'start', 600))
    p.append(text(tx + 82, 92, [('t', '방향을 넣으면 고도 변화를 내는 함수')], mu, 12, 'start'))
    p.append(text(tx + 8, 118, df + [('r', '('), ('t', '동 1걸음'), ('r', ') = 2')], ink, 13,
                  'start'))
    p.append(text(tx + 8, 140, df + [('r', '('), ('t', '북 1걸음'), ('r', ') = 3')], ink, 13,
                  'start'))
    p.append(box(tx, 154, 288, 44, bl, bl, 0.10, 1.5))
    p.append(text(tx + 144, 182, df + [('r', '('), ('m', 'dx'), ('r', ') = 2'), ('m', 'dx'),
                                       ('sub', '1'), ('r', ' + 3'), ('m', 'dx'), ('sub', '2')],
                  ink, 15))
    p.append(text(tx + 144, 218, [('t', '함수가 하는 일은 2와 3을 곱해서 더하는 것뿐')], mu, 12))
    p.append(text(tx, 256, [('t', 'gradient')], o, 13.5, 'start', 600))
    p.append(text(tx + 70, 256, [('t', '그 두 수를 모아 화살표 하나로 적은 것')], mu, 12, 'start'))
    p.append(box(tx, 270, 288, 76, o, o, 0.12, 1.5))
    p.append(text(tx + 144, 298, [('r', '∇'), ('m', 'f'), ('r', '('), ('m', 'x'),
                                  ('r', ') = (2, 3)')], ink, 15))
    p.append(text(tx + 144, 328, df + [('r', '('), ('m', 'dx'), ('r', ') = ⟨(2, 3), '),
                                       ('m', 'dx'), ('r', '⟩')], ink, 15))
    p.append(text(tx, 378, [('t', '방향이 같을수록 inner product가 크므로')], mu, 12, 'start'))
    p.append(text(tx, 396, [('t', '화살표 방향이 가장 가파른 오르막이다')], mu, 12, 'start'))
    return wrap(c, 760, 432, title, '\n'.join(p))


def fig_chain(c) -> str:
    title = 'backprop에서 layer 하나가 하는 일'
    ink, mu, bl, o, rule = c['ink'], c['muted'], c['blue'], c['orange'], c['rule']
    p = []
    p.append(text(24, 34, [('t', title)], ink, 15, 'start', 600))
    p.append(text(24, 54, [('t', 'forward는 입력에서 loss로, backward는 loss에서 입력으로 흐른다')],
                  mu, 12, 'start'))
    top, bh = 112, 44
    cy = top + bh / 2
    # regions
    p.append(f'<rect x="388" y="76" width="350" height="176" rx="10" fill="{o}" '
             f'fill-opacity="0.08" stroke="{o}" stroke-opacity="0.5" stroke-dasharray="4 3"/>')
    p.append(text(563, 96, [('m', 'g'), ('t', ': backprop이 이미 지나온 부분')], o, 12.5,
                  weight=600))
    p.append(f'<rect x="22" y="76" width="122" height="176" rx="10" fill="{mu}" '
             f'fill-opacity="0.06" stroke="{mu}" stroke-opacity="0.4" stroke-dasharray="4 3"/>')
    p.append(text(83, 96, [('t', '아직 안 온 부분')], mu, 12))
    # boxes
    boxes = [(40, 'layer', mu, 0.0), (220, None, bl, 0.14), (410, 'layer', mu, 0.0),
             (560, 'loss', mu, 0.0)]
    for x, lab, col, op in boxes:
        p.append(box(x, top, 84 if lab else 120, bh, col, col, op, 1.5 if lab is None else 1.25))
        if lab:
            p.append(text(x + 42, cy + 5, [('t', lab)], ink, 13))
    p.append(text(280, cy + 5, [('t', 'layer '), ('m', 'h')], ink, 13.5, weight=600))
    p.append(text(280, top - 8, [('t', '지금 볼 layer')], bl, 11.5, weight=600))
    # forward arrows and values
    fw = f'stroke="{mu}" stroke-width="1.5" marker-end="url(#a-muted)"'
    for x1, x2, lab in ((124, 216, [('m', 'x')]), (340, 406, [('m', 'y')]),
                        (494, 556, [('r', '···')]), (644, 700, [('m', 'L')])):
        p.append(f'<line x1="{x1}" y1="{cy}" x2="{x2}" y2="{cy}" {fw}/>')
        if lab:
            p.append(text((x1 + x2) / 2, cy - 8, lab, ink, 16))
    # backward arrows
    by = 196
    bw = f'stroke="{o}" stroke-width="1.75" marker-end="url(#a-orange)"'
    p.append(f'<line x1="700" y1="{by}" x2="414" y2="{by}" {bw}/>')
    p.append(f'<line x1="406" y1="{by}" x2="344" y2="{by}" {bw}/>')
    p.append(f'<line x1="216" y1="{by}" x2="128" y2="{by}" {bw}/>')
    p.append(f'<line x1="120" y1="{by}" x2="40" y2="{by}" stroke="{o}" stroke-width="1.75" '
             f'stroke-dasharray="4 4" marker-end="url(#a-orange)"/>')
    p.append(text(375, by + 22, [('r', '∇'), ('m', 'g'), ('r', '('), ('m', 'y'), ('r', ')')],
                  ink, 15))
    p.append(text(375, by + 40, [('t', '받는 것')], o, 11.5, weight=600))
    p.append(text(170, by + 22, [('r', '∇'), ('m', 'f'), ('r', '('), ('m', 'x'), ('r', ')')],
                  ink, 15))
    p.append(text(170, by + 40, [('t', '만들 것')], o, 11.5, weight=600))
    p.append(text(280, by + 4, [('m', 'J'), ('sup', '⊤'), ('t', '를 곱한다')], o, 12.5,
                  weight=600))
    # legend
    p.append(f'<line x1="24" y1="292" x2="52" y2="292" {fw}/>')
    p.append(text(60, 296, [('t', 'forward: 값이 흐르는 방향')], mu, 12, 'start'))
    p.append(f'<line x1="300" y1="292" x2="272" y2="292" {bw}/>')
    p.append(text(308, 296, [('t', 'backward: gradient가 흐르는 방향')], mu, 12, 'start'))
    return wrap(c, 760, 316, title, '\n'.join(p))


OUT.mkdir(parents=True, exist_ok=True)
for name, fn in (('mg-02a-chain', fig_chain), ('mg-01b-terrain', fig_terrain), ('mg-00a-delta', fig_delta), ('mg-00b-pair', fig_pair), ('mg-01-dual', fig_dual), ('mg-02-vjp', fig_vjp)):
    for theme, colors in THEMES.items():
        (OUT / f'{name}-{theme}.svg').write_text(fn(colors))
