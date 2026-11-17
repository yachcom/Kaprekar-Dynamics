# -*- coding: utf-8 -*-
"""
Kaprekar Dynamics — Streamlit app for the paper
«Процедура Капрекара: линейная структура образа,
 симметричные суммы и перепись аттракторов».

Deployed via GitHub → Streamlit Cloud / Render.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations_with_replacement
from math import comb, factorial
from collections import defaultdict

# ============================================================
# PAGE CONFIG + CSS
# ============================================================

st.set_page_config(
    layout="wide",
    page_title="Kaprekar Dynamics",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Horizontal scroll for wide tables on mobile */
    .kp-wrap {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        margin: 6px 0;
        max-width: 100%;
    }
    .kp-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 13px;
    }
    .kp-table th, .kp-table td {
        padding: 4px 8px;
        border: 1px solid #d0d0d0;
        text-align: center;
        white-space: nowrap;
    }
    .kp-table th {
        background: #f0f0f0;
        font-weight: 600;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    .kp-table tr:hover { background: #fafafa; }
    .kp-table .anomaly { background: #ffe0e0; font-weight: 600; }
    .kp-table .good   { background: #e6f4ea; }
    .kp-table caption {
        caption-side: top;
        font-weight: 600;
        padding: 6px;
        text-align: left;
    }
    /* Compact on mobile */
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem !important; }
        .kp-table { font-size: 11px; }
        .kp-table th, .kp-table td { padding: 2px 4px; }
    }
    /* Keep math from auto-translation */
    .notranslate { translate: no; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CORE
# ============================================================

def K(n, k):
    """Kaprekar step: desc(n) − asc(n), on k-digit strings."""
    s = str(int(n)).zfill(k)
    return int(''.join(sorted(s, reverse=True))) - int(''.join(sorted(s)))

def digits_of(n, k):
    return np.array([int(c) for c in str(int(n)).zfill(k)])

def d_vec(n, k):
    """Symmetric differences d_i = x_i − x_{k+1−i} on sorted-desc digits."""
    s = sorted(str(int(n)).zfill(k), reverse=True)
    return tuple(int(s[i]) - int(s[k - 1 - i]) for i in range(k // 2))

def z_from_d(d):
    """Number of trailing zeros in d-vector."""
    z = 0
    for v in reversed(d):
        if v == 0:
            z += 1
        else:
            break
    return z

def z_of(n, k):
    return z_from_d(d_vec(n, k))

def weights(k):
    """w_i = 10^(k−i) − 10^(i−1), i = 1..m."""
    return [10 ** (k - 1 - i) - 10 ** i for i in range(k // 2)]

def encode_K(d, k):
    w = weights(k)
    return sum(int(d[i]) * w[i] for i in range(len(d)))

def decode_K(K_val, k):
    """Greedy decoding by Lemma 2 (weights dominate)."""
    w = weights(k)
    d, r = [], int(K_val)
    for w_i in w:
        q = r // w_i
        d.append(int(q))
        r -= q * w_i
    return tuple(d)

def trajectory(n, k, max_steps=5000):
    path, seen = [int(n)], {int(n): 0}
    while len(path) < max_steps:
        nxt = K(path[-1], k)
        if nxt in seen:
            return path, seen[nxt], path[seen[nxt]:]
        seen[nxt] = len(path)
        path.append(nxt)
    return path, len(path) - 1, [path[-1]]

def S_of_raw(n, k):
    """Raw symmetric sums, WITH 18 (Theorem 3). No normalization."""
    a = digits_of(n, k)
    m = k // 2
    if k % 2 == 0:
        return tuple(int(a[m - 1 - i] + a[m + i]) for i in range(m))
    return tuple(int(a[m - i] + a[m + i]) for i in range(1, m + 1))

def dr(x):
    return 0 if x == 0 else 1 + (x - 1) % 9

def _half_offset(k):
    return 0 if k % 2 == 0 else 1

def SP2(n, k):
    """Digitwise sum of halves (raw, no reduction)."""
    a = digits_of(n, k)
    m, off = k // 2, _half_offset(k)
    return tuple(int(a[i] + a[m + off + i]) for i in range(m))

def SP3(n, k):
    """Left − right."""
    a = digits_of(n, k)
    m, off = k // 2, _half_offset(k)
    return tuple(int(a[i] - a[m + off + i]) for i in range(m))

def SP4(n, k):
    """Reversed left − right."""
    a = digits_of(n, k)
    m = k // 2
    if k % 2 == 0:
        return tuple(int(a[m - 1 - i] - a[m + i]) for i in range(m))
    return tuple(int(a[m - i] - a[m + i]) for i in range(m))

def fmt(n, k):
    return str(int(n)).zfill(k)

def multinomial(digits):
    cnt = defaultdict(int)
    for d in digits:
        cnt[int(d)] += 1
    total = factorial(sum(cnt.values()))
    for c in cnt.values():
        total //= factorial(c)
    return total

def structure_from_d(d, k):
    """Theorem 2: predicted digits of K from d."""
    m = k // 2
    z = z_from_d(d)
    if z == m:
        return tuple([0] * k)
    eps = 1 if k % 2 else 0
    if z == m - 1:
        return tuple([d[0] - 1] + [9] * (2 * m - 2 + eps) + [10 - d[0]])
    left = list(d[: m - z - 1]) + [d[m - z - 1] - 1]
    middle = [9] * (2 * z + eps)
    right = [9 - d[j] for j in range(m - z - 1, 0, -1)] + [10 - d[0]]
    return tuple(left + middle + right)

def expected_S(k, z):
    """Theorem 3: expected S for given k, z."""
    m = k // 2
    if z == m:
        return [0] * m
    if z == m - 1:
        return [18] * (m - 1) + [9]
    return [18] * z + [8] + [9] * (m - z - 2) + [10]

def check_sp3_pattern(sp3, k):
    """Proposition 1: SP3 = (p, q, …, q, p−2)."""
    m = k // 2
    if m < 2:
        return True
    if sp3[-1] != sp3[0] - 2:
        return False
    for i in range(1, m - 1):
        if sp3[i] != sp3[m - 1 - i]:
            return False
    return True

# ============================================================
# ENUMERATION (cached)
# ============================================================

@st.cache_data(show_spinner="Перебираю классы эквивалентности…")
def enumerate_classes_cached(k):
    """dict: canon_cycle_tuple -> {'cycle': [...], 'count': classes, 'numbers': k-perms}."""
    cycles = {}
    for combo in combinations_with_replacement(range(10), k):
        if len(set(combo)) == 1:
            continue
        n = int(''.join(str(x) for x in sorted(combo, reverse=True)))
        path, ci, cyc = trajectory(n, k)
        canon = tuple(sorted(cyc))[0]
        if canon not in cycles:
            cycles[canon] = {'cycle': list(cyc), 'count': 0, 'numbers': 0}
        cycles[canon]['count'] += 1
        cycles[canon]['numbers'] += multinomial(combo)
    # Trivial zero (all-same digits → K = 0)
    zero = tuple([0] * k)
    cycles[zero] = {'cycle': [0], 'count': 10, 'numbers': 10}
    return cycles

@st.cache_data(show_spinner=False)
def d_basins_cached(k):
    """Basins in d-space: number of d-vectors leading to each attractor."""
    m = k // 2
    result = defaultdict(int)
    for d in combinations_with_replacement(range(9, -1, -1), m):
        kv = encode_K(d, k)
        _, _, cyc = trajectory(kv, k)
        canon = tuple(sorted(cyc))[0]
        result[canon] += 1
    return dict(result)

@st.cache_data(show_spinner=False)
def d_fiber_count_cached(d, k):
    """Number of multisets (classes) giving this d."""
    count = 0
    for combo in combinations_with_replacement(range(10), k):
        s = tuple(sorted(combo, reverse=True))
        dd = tuple(int(s[i]) - int(s[k - 1 - i]) for i in range(k // 2))
        if dd == d:
            count += 1
    return count

def cut_by_layer(k, d1v, d2v, cycles):
    """§5.3: remove d₁=d1v, d₂=d2v layer; report survived/cut cycles."""
    m = k // 2
    if m < 2:
        return {'survived': list(cycles.values()), 'cut': [],
                'note': f'k={k}: m={m}<2, d₂ не определён'}
    survived, cut = [], []
    for canon, info in cycles.items():
        has = any(
            len(d_vec(n, k)) >= 2
            and d_vec(n, k)[0] == d1v
            and d_vec(n, k)[1] == d2v
            for n in info['cycle']
        )
        (cut if has else survived).append(info)
    return {'survived': survived, 'cut': cut}

# ============================================================
# HTML TABLES
# ============================================================

def html_table(headers, rows, caption=None):
    html = '<div class="kp-wrap"><table class="kp-table">'
    if caption:
        html += f'<caption>{caption}</caption>'
    html += '<thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>'
    for row in rows:
        row_class = row.get('_class', '') if isinstance(row, dict) else ''
        cells = row['cells'] if isinstance(row, dict) else row
        html += f'<tr class="{row_class}">'
        html += ''.join(f'<td>{c}</td>' for c in cells)
        html += '</tr>'
    return html + '</tbody></table></div>'

def trajectory_table(path, k, ci, cyc):
    headers = ['Итер.', 'Число', 'd', 'z', 'S (сырые)', 'S (корень)',
               'ΣK', 'E', 'SP2', 'SP3', 'SP3=p..p−2?']
    rows = []
    for t, n in enumerate(path):
        s_raw = S_of_raw(n, k)
        s_dr = tuple(dr(x) for x in s_raw)
        sum_K = sum(int(c) for c in fmt(n, k))
        a = digits_of(n, k)
        if k >= 4:
            E_val = (10 * int(a[0]) + int(a[1])) + (10 * int(a[-2]) + int(a[-1]))
            E_str = str(E_val)
        else:
            E_str = '—'
        sp2 = SP2(n, k)
        sp3 = SP3(n, k)
        sp3_ok = check_sp3_pattern(sp3, k)
        z = z_of(n, k)
        row_class = 'anomaly' if z > 0 else ('good' if t >= ci and len(cyc) > 1 else '')
        label = f"{t} ▶" if (t == ci and len(cyc) > 1) else str(t)
        rows.append({
            '_class': row_class,
            'cells': [
                label, fmt(n, k),
                ', '.join(map(str, d_vec(n, k))),
                str(z),
                ', '.join(map(str, s_raw)),
                ', '.join(map(str, s_dr)),
                str(sum_K), E_str,
                ', '.join(f"{v}({dr(v)})" for v in sp2),
                ', '.join(f"{v:+d}" for v in sp3),
                '✓' if sp3_ok else '✗',
            ]
        })
    return html_table(headers, rows, caption=f"Траектория k={k}")

def antisym_pair(a_num, b_num, k):
    a, b = digits_of(a_num, k), digits_of(b_num, k)
    diffs = [int(a[i]) - int(b[i]) for i in range(k)]
    anti = all(diffs[i] == -diffs[k - 1 - i] for i in range(k // 2))
    return diffs, anti

# ============================================================
# VERIFICATION
# ============================================================

def run_verification():
    results = []
    def _chk(name, fn):
        try:
            fn()
            results.append((name, '✅', ''))
        except AssertionError as e:
            results.append((name, '❌', str(e)[:200]))
        except Exception as e:
            results.append((name, '⚠️', str(e)[:200]))

    def c1():
        for n, k in ((495, 3), (6174, 4), (864197532, 9),
                     (9753086421, 10), (554999445, 9), (549945, 6),
                     (555499994445, 12)):
            assert K(n, k) == n, f"{n} not fixed for k={k}"
    _chk("Неподвижные точки (включая 555499994445, z=2)", c1)

    def c2():
        for k in range(3, 11):
            m = k // 2
            vals = {encode_K(d, k)
                    for d in combinations_with_replacement(range(9, -1, -1), m)}
            assert len(vals) == comb(m + 9, 9), f"k={k}: {len(vals)}"
    _chk("Теорема 1: |im K| = C(m+9,9) для k=3..10", c2)

    def c3():
        for k in range(3, 10):
            m = k // 2
            for d in combinations_with_replacement(range(9, -1, -1), m):
                kv = encode_K(d, k)
                assert tuple(int(c) for c in fmt(kv, k)) == structure_from_d(d, k), \
                    f"structure k={k} d={d}"
                z = z_from_d(d)
                assert S_of_raw(kv, k) == tuple(expected_S(k, z)), f"S k={k} d={d}"
                tot = sum(int(c) for c in fmt(kv, k))
                exp = 0 if z == m else 9 * ((k + 1) // 2 + z)
                assert tot == exp, f"sum k={k} d={d}"
    _chk("Теоремы 2–4: структура, S, сумма цифр (k=3..9)", c3)

    def c4():
        k, base = 8, 98626311
        base_S = S_of_raw(base, k)
        for other in (65299644, 87426522, 65408544, 83208762):
            other_S = S_of_raw(other, k)
            _, anti = antisym_pair(base, other, k)
            assert anti == (base_S == other_S), f"pair {base}, {other}"
    _chk("Теорема 5: антисимметрия ⇔ равны S", c4)

    def c5():
        for n in (98626311, 65408544, 82108872):
            a = digits_of(n, 8)
            E = (10 * int(a[0]) + int(a[1])) + (10 * int(a[-2]) + int(a[-1]))
            assert E == 100 + 9 * (int(a[0]) - int(a[1])), f"n={n}"
    _chk("§5.2: E = 100 + 9(a₁−a₂)", c5)

    def c6():
        cnt = d_fiber_count_cached((9, 8, 5, 3), 8)
        assert cnt == 24, f"fiber={cnt}"
    _chk("Пример 2: |fiber(9,8,5,3)| = 24 при k=8", c6)

    return results

# ============================================================
# UI
# ============================================================

st.title("Динамика процедуры Капрекара")
st.caption("Интерактивная визуализация статьи «Процедура Капрекара: "
           "линейная структура образа, симметричные суммы и перепись аттракторов»")

tabs = st.tabs([
    "Обзор", "Траектория", "d ↔ K", "S и z, SP3/SP4",
    "Перепись", "Контейнеры", "Проверка",
])

# ===== TAB 1: ОБЗОР =====
with tabs[0]:
    st.header("Обзор результатов")
    st.markdown("""
    Этот инструмент визуализирует основные результаты статьи.
    Каждая вкладка соответствует разделу статьи:

    | Вкладка | Что показывает | Раздел статьи |
    |---|---|---|
    | **Траектория** | Путь числа, цикл, S, SP2/SP3, аномалия z>0 | §3, §4, примеры 4–8 |
    | **d ↔ K** | Линейная запись K = Σdᵢwᵢ, биекция, размер образа C(m+9,9), волокно | §2, теорема 1 |
    | **S и z** | Классификация S, формула суммы цифр, обратная z, SP3/SP4 | §3.3–3.4, §4.2 |
    | **Перепись** | Аттракторы k=3..12, бассейны в трёх пространствах, неподвижные по z | §7, прил. A |
    | **Контейнеры** | Разрез слоя d₁=8, d₂=6 | §5 |
    | **Проверка** | Запуск assert-проверок (как verify_kaprekar.py) | прил. B |
    """)

    st.subheader("Размер образа |im K| = C(m+9, 9)")
    st.caption("Для k=10: из 10¹⁰ входов получается лишь 2002 различных значения K.")
    size_data = [(k, k // 2, comb(k // 2 + 9, 9)) for k in range(3, 13)]
    st.dataframe(
        pd.DataFrame(size_data, columns=['k', 'm = ⌊k/2⌋', '|im K| = C(m+9,9)']),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Ключевые факты")
    st.markdown("""
    - **K линейно по d:** K(n) = Σᵢ dᵢ wᵢ, где wᵢ = 10^(k−i) − 10^(i−1), dᵢ = xᵢ − x_{k+1−i}.
    - **Биекция:** d ↔ K взаимно однозначно, |im K| = C(m+9, 9).
    - **Симметричные суммы S** имеют вид (8, 9, …, 9, 10) при z=0 и сдвигаются с ростом z
      (внутренние пары дают 18, а не 9).
    - **Сумма цифр:** sum(K) = 9(⌈k/2⌉ + z); обратно z = sum(K)/9 − ⌈k/2⌉ (при z < m).
    - **Антисимметрия:** Δ антисимметрична ⇔ S(X) = S(Y). Это не свойство итераций,
      а равенство S.
    - **Гипотеза:** при k ≥ 6 все циклы длины > 1 лежат в z = 0; z > 0 — только неподвижные точки.
    """)

# ===== TAB 2: ТРАЕКТОРИЯ =====
with tabs[1]:
    st.header("Траектория и аномалии")

    n_input = st.number_input("Число n:", min_value=0, value=12899850, step=1, key='n_traj')
    k_input = st.number_input("k:", min_value=2, max_value=12,
                              value=max(2, len(str(int(n_input)))), step=1, key='k_traj')

    if n_input >= 10 ** k_input:
        st.error(f"Число не помещается в {k_input} цифр")
    else:
        path, ci, cyc = trajectory(n_input, k_input)
        c1, c2, c3 = st.columns(3)
        c1.metric("Длина пути", len(path))
        c2.metric("Вход в цикл", ci)
        c3.metric("Длина цикла", len(cyc))

        if len(cyc) > 1:
            st.code(" → ".join(fmt(x, k_input) for x in cyc) + " → " + fmt(cyc[0], k_input))
        else:
            st.code(f"Неподвижная точка: {fmt(cyc[0], k_input)}")

        st.subheader("Траектория")
        st.caption("Красным — состояния с z > 0 (аномалия). Зелёным — цикл. "
                   "S показаны сырыми (с 18), цифровой корень — отдельной колонкой. "
                   "E = 100 + 9(a₁−a₂); на итерации z=1 значение выбивается из ряда 109/118/…")
        st.markdown(trajectory_table(path, k_input, ci, cyc), unsafe_allow_html=True)

        st.subheader("Антисимметрия: сравнение любых двух итераций")
        st.caption("Теорема 5: разность антисимметрична ⇔ S(X) = S(Y). "
                   "Поэтому все пары с итерацией z=1 нарушают антисимметрию, "
                   "а пары линейных итераций — нет.")
        if len(path) > 1:
            cc1, cc2 = st.columns(2)
            with cc1:
                i_a = st.selectbox("Итерация A:", range(len(path)), index=0, key='iA')
            with cc2:
                i_b = st.selectbox("Итерация B:", range(len(path)),
                                   index=min(6, len(path) - 1), key='iB')
            a_n, b_n = path[i_a], path[i_b]
            diffs, anti = antisym_pair(a_n, b_n, k_input)
            S_a = S_of_raw(a_n, k_input)
            S_b = S_of_raw(b_n, k_input)
            colA, colB = st.columns(2)
            with colA:
                st.write(f"S({fmt(a_n, k_input)}) = {S_a}")
            with colB:
                st.write(f"S({fmt(b_n, k_input)}) = {S_b}")
            if anti:
                st.success(f"✓ Антисимметрична (S равны: {S_a == S_b})")
            else:
                st.warning(f"✗ Не антисимметрична (S равны: {S_a == S_b})")
            st.write("Разности: " + ", ".join(f"{d:+d}" for d in diffs))

        with st.expander("Экспериментально: 3D-динамика (не из статьи)"):
            if len(path) > 2:
                Z_diffs = np.array([
                    digits_of(path[t + 1], k_input) - digits_of(path[t], k_input)
                    for t in range(len(path) - 1)
                ])
                fig = go.Figure(go.Surface(z=Z_diffs, colorscale='RdBu', cmid=0))
                fig.update_layout(
                    scene=dict(xaxis_title='Позиция цифры',
                               yaxis_title='Итерация',
                               zaxis_title='Δ'),
                    height=500, margin=dict(l=0, r=0, t=20, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

# ===== TAB 3: d ↔ K =====
with tabs[2]:
    st.header("Биекция d ↔ K")
    st.markdown(r"$K(n) = \sum_{i=1}^{m} d_i\,w_i$, "
                r"$w_i = 10^{k-i} - 10^{i-1}$, "
                r"$d_i = x_i - x_{k+1-i}$, $m = \lfloor k/2 \rfloor$.")
    st.caption("Теорема 1: K инъективно на d-симплексе и сюръективно на образ; "
               "|im K| = C(m+9, 9).")

    col1, col2 = st.columns(2)
    with col1:
        k_bij = st.number_input("k:", 3, 12, 8, key='k_bij')
    with col2:
        st.metric("|im K| = C(m+9,9)", comb(k_bij // 2 + 9, 9))

    m_bij = k_bij // 2
    st.subheader("Кодирование: d → K")
    default_d = "9, 8, 5, 3" if k_bij == 8 else ", ".join(["9"] * m_bij)
    d_str = st.text_input("d-вектор (через запятую):", default_d, key='d_bij')

    try:
        d_parsed = tuple(int(x.strip()) for x in d_str.split(',') if x.strip())
    except ValueError:
        d_parsed = ()
        st.error("Не удалось разобрать d — введите целые числа через запятую.")

    if d_parsed:
        if len(d_parsed) != m_bij:
            st.error(f"Ожидалось {m_bij} компонент, получено {len(d_parsed)}")
        elif any(x < 0 or x > 9 for x in d_parsed):
            st.error("Все компоненты должны быть в [0, 9]")
        else:
            if any(d_parsed[i] < d_parsed[i + 1] for i in range(len(d_parsed) - 1)):
                st.warning("d не невозрастающий — это не d-симплекс, "
                           "но K для такого d всё равно определён.")

            K_val = encode_K(d_parsed, k_bij)
            d_back = decode_K(K_val, k_bij)

            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**K** = `{K_val}`")
                st.caption(f"Цифры K: `{fmt(K_val, k_bij)}`")
            with colB:
                st.markdown(f"**Декодировано:** `{d_back}`")
                if d_back == d_parsed:
                    st.success("✓ d ↔ K взаимно однозначно")
                else:
                    st.error("✗ Декодирование не совпадает")

            st.subheader("Волокно: сколько классов дают этот d?")
            st.caption("Пример 2 статьи: для d=(9,8,5,3), k=8 — ровно 24 класса.")
            if st.button("Посчитать волокно", key='count_fiber'):
                with st.spinner("Считаю…"):
                    cnt = d_fiber_count_cached(d_parsed, k_bij)
                st.metric("Классов с этим d", cnt)
                if cnt > 0:
                    st.caption(f"{cnt} различных мультимножеств цифр дают "
                               f"один и тот же K = {K_val}.")

            st.subheader("Структура цифр K (теорема 2)")
            z_val = z_from_d(d_parsed)
            st.write(f"z = {z_val}")
            predicted = structure_from_d(d_parsed, k_bij)
            actual = tuple(int(c) for c in fmt(K_val, k_bij))
            st.markdown(
                f"**Предсказано:** `{''.join(map(str, predicted))}`  \n"
                f"**Фактически:** `{''.join(map(str, actual))}`"
            )
            if predicted == actual:
                st.success("✓ Теорема 2 работает")

# ===== TAB 4: S и z =====
with tabs[3]:
    st.header("Симметричные суммы S и параметр z")
    st.markdown("""
    **Теорема 3.** Для 0 ≤ z ≤ m−2:
    S = (18, …, 18 [z раз], 8, 9, …, 9 [m−z−2 раз], 10).
    Для z = m−1: S = (18, …, 18 [m−1 раз], 9).
    При z = m все суммы равны 0.

    **Теорема 4.** sum(K) = 9(⌈k/2⌉ + z); обратно z = sum(K)/9 − ⌈k/2⌉ (при z < m).
    """)

    k_sz = st.number_input("k:", 3, 12, 8, key='k_sz')
    m_sz = k_sz // 2

    st.subheader("S(z) для заданного k (теорема 3)")
    z_tab = []
    for z in range(m_sz + 1):
        S = expected_S(k_sz, z)
        sumK = 0 if z == m_sz else 9 * ((k_sz + 1) // 2 + z)
        z_tab.append({'z': z, 'S(z)': str(tuple(S)), 'sum(K)': sumK})
    st.dataframe(pd.DataFrame(z_tab), use_container_width=True, hide_index=True)

    st.subheader("Проверка SP3/SP4 (предложение 1)")
    default_n = 98626311 if k_sz == 8 else 495
    n_sp = st.number_input("Число n:", min_value=0, value=default_n, step=1, key='n_sp')
    if n_sp < 10 ** k_sz:
        a = digits_of(n_sp, k_sz)
        sp3 = SP3(n_sp, k_sz)
        sp4 = SP4(n_sp, k_sz)
        sp3_ok = check_sp3_pattern(sp3, k_sz)
        z_n = z_of(n_sp, k_sz)
        st.write(f"**SP3** = `{sp3}`  (паттерн (p, q, …, q, p−2): "
                 f"{'✓' if sp3_ok else '✗'})")
        st.write(f"**SP4** = `{sp4}`")
        st.write(f"**z** = {z_n}")
        if z_n == 0 and not sp3_ok:
            st.error("Для z=0 паттерн должен выполняться — это ошибка.")
        elif z_n > 0 and not sp3_ok:
            st.info("При z>0 паттерн нарушается — это ожидаемо (исключение z=1).")
        st.markdown(r"$\mathrm{SP4} - \mathrm{SP3} = L^R - L$ (следствие 4).")
        st.code("SP4 − SP3 = " + str(tuple(int(sp4[i] - sp3[i]) for i in range(m_sz))))
        st.code("L^R − L     = " + str(tuple(int(a[m_sz - 1 - i] - a[i]) for i in range(m_sz))))
    else:
        st.error(f"Число должно быть < 10^{k_sz}")

# ===== TAB 5: ПЕРЕПИСЬ =====
with tabs[4]:
    st.header("Перепись аттракторов")
    st.caption("k=3..9 — перебор быстрый. k=10–12 — по кнопке (первый запуск медленный).")

    k_cen = st.selectbox("k:", list(range(3, 13)), index=7, key='k_cen')
    cycles = None

    if k_cen <= 9:
        cycles = enumerate_classes_cached(k_cen)
    else:
        est = comb(k_cen + 9, 9)
        time_hint = {10: "~30 сек", 11: "~1–2 мин", 12: "~3–5 мин"}.get(k_cen, "несколько минут")
        st.warning(f"k={k_cen}: {est:,} классов. Первый запуск {time_hint}. "
                   f"Результат кэшируется.")
        if st.button(f"Загрузить k={k_cen}", key=f'load_{k_cen}'):
            st.session_state[f'loaded_{k_cen}'] = True
            st.rerun()
        if st.session_state.get(f'loaded_{k_cen}'):
            cycles = enumerate_classes_cached(k_cen)

    if cycles is not None:
        d_basins = d_basins_cached(k_cen)
        total_c = sum(info['count'] for info in cycles.values())
        total_n = sum(info['numbers'] for info in cycles.values())
        total_d = sum(d_basins.values())

        rows = []
        for canon, info in sorted(cycles.items(), key=lambda x: -x[1]['count']):
            cyc = info['cycle']
            if len(cyc) == 1 and cyc[0] == 0:
                name = "0 (неподвижная)"
            elif len(cyc) == 1:
                name = f"{fmt(cyc[0], k_cen)} (неподв., z={z_of(cyc[0], k_cen)})"
            else:
                name = " → ".join(fmt(x, k_cen) for x in cyc[:3])
                if len(cyc) > 3:
                    name += " → …"
            d_count = d_basins.get(canon, 0)
            rows.append({
                'Аттрактор': name,
                'Длина': len(cyc),
                'Классов': info['count'],
                'Классов %': f"{info['count']/total_c*100:.2f}",
                'd-сост.': d_count,
                'd-сост. %': f"{d_count/total_d*100:.2f}",
                'Чисел': f"{info['numbers']:,}",
                'Чисел %': f"{info['numbers']/total_n*100:.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Всего классов: {total_c} = C({k_cen}+9,9) = {comb(k_cen+9, 9)}. "
                   f"d-состояний: {total_d} = C({k_cen//2}+9,9). "
                   f"Чисел: {total_n:,} = {10**k_cen:,}.")

    st.subheader("Неподвижные точки по z (k=3..12)")
    st.markdown("""
    | k | Неподвижные точки (z) |
    |---|---|
    | 3 | 495 (0) |
    | 4 | 6174 (0) |
    | 6 | 631764 (0), 549945 (1) |
    | 8 | 97508421 (0), 63317664 (0) |
    | 9 | 864197532 (0), 554999445 (1) |
    | 10 | 9753086421 (0), 6333176664 (0), 9975084201 (0) |
    | 11 | 86431976532 (0) |
    | 12 | 975330866421, 997530864201, 633331766664, 999750842001 (0); 555499994445 (2) |
    """)

# ===== TAB 6: КОНТЕЙНЕРЫ =====
with tabs[5]:
    st.header("Разрез слоя d₁ = 8, d₂ = 6 (контейнер 86__32)")
    st.markdown("""
    Контейнер **86__32** означает a₁ = 8, a₂ = 6, a_{k−1} = 3, a_k = 2;
    по теореме 2 это равносильно d₁ = 8, d₂ = 6 (автоматически z ≤ m−3).

    §5.3 статьи: если удалить из d-графа слой d₁=8, d₂=6, то
    при k = 6…9 не остаётся циклов длины > 1 (только неподвижные),
    а при k = 10 выживают ровно два 3-цикла без 86__32.
    """)

    k_cut = st.selectbox("k:", list(range(4, 11)), index=6, key='k_cut')
    col1, col2 = st.columns(2)
    with col1:
        d1v = st.number_input("d₁:", 0, 9, 8, key='d1v')
    with col2:
        d2v = st.number_input("d₂:", 0, 9, 6, key='d2v')

    if st.button("Показать разрез", key='cut_btn'):
        with st.spinner("Перебираю классы и режу слой…"):
            cyc_cut = enumerate_classes_cached(k_cut)
            result = cut_by_layer(k_cut, d1v, d2v, cyc_cut)

        st.write(f"**Проверено циклов:** {len(cyc_cut)}")
        st.write(f"**Разрезано:** {len(result['cut'])}")
        st.write(f"**Выжило:** {len(result['survived'])}")

        if result.get('note'):
            st.info(result['note'])

        st.subheader("Выжившие аттракторы")
        for info in result['survived']:
            cyc = info['cycle']
            st.write(f"- {' → '.join(fmt(x, k_cut) for x in cyc)} "
                     f"({len(cyc)}-цикл, {info['count']} классов)")

# ===== TAB 7: ПРОВЕРКА =====
with tabs[6]:
    st.header("Проверка утверждений")
    st.caption("Assert-проверки, аналогичные verify_kaprekar.py: известные константы, "
               "теоремы 1–5, формула E, волокно 24.")

    if st.button("Запустить все проверки", type="primary", key='run_verify'):
        with st.spinner("Проверяю…"):
            results = run_verification()
        df = pd.DataFrame(results, columns=['Проверка', 'Статус', 'Детали'])
        st.dataframe(df, use_container_width=True, hide_index=True)
        n_ok = sum(1 for _, s, _ in results if s == '✅')
        st.metric(f"Пройдено {n_ok}/{len(results)}", "")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Deployed via GitHub → Streamlit / Render. "
           "Based on: «Процедура Капрекара: линейная структура образа, "
           "симметричные суммы и перепись аттракторов» (2026).")