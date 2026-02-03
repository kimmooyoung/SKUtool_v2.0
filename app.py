import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==================================================
# Page Config
# ==================================================
st.set_page_config(page_title="SKU Automation Eligibility Analyzer v 2.0", layout="wide")

st.title("📦 SKU Automation Eligibility Analyzer")
st.caption("SKU Master 기반 자동화 전환 · 물리적 치수 기반 Divider 매칭 분석")

# ==================================================
# Sidebar – Container & Constraints
# ==================================================
st.sidebar.header("1️⃣ 자동화 설비 용기 사이즈")
container_l = st.sidebar.number_input("용기 가로 (mm)", min_value=1, value=600, step=10)
container_w = st.sidebar.number_input("용기 세로 (mm)", min_value=1, value=400, step=10)
container_h = st.sidebar.number_input("용기 높이 (mm)", min_value=1, value=300, step=10)

st.sidebar.header("2️⃣ 중량 제한")
weight_unit = st.sidebar.selectbox("SKU 중량 단위", ["kg", "g", "lb"])
max_weight_input = st.sidebar.number_input("용기 최대 허용 중량", min_value=0.1, value=20.0)

unit_factor = {"kg": 1.0, "g": 0.001, "lb": 0.453592}
max_weight = max_weight_input * unit_factor[weight_unit]

st.sidebar.header("🎨 3D 시각화 스타일")
color_eligible = st.sidebar.color_picker("자동화 가능 SKU", "#1f77b4")
color_size_fail = st.sidebar.color_picker("사이즈 탈락 SKU", "#ff7f0e")
color_weight_fail = st.sidebar.color_picker("중량 탈락 SKU", "#9467bd")
color_both_fail = st.sidebar.color_picker("사이즈+중량 탈락 SKU", "#7f7f7f")
color_container = st.sidebar.color_picker("자동화 용기 테두리", "#d62728")

opacity_eligible = st.sidebar.slider("Eligible 투명도", 0.1, 1.0, 1.0, 0.1)
opacity_size = st.sidebar.slider("Size Fail 투명도", 0.1, 1.0, 0.6, 0.1)
opacity_weight = st.sidebar.slider("Weight Fail 투명도", 0.1, 1.0, 0.6, 0.1)
opacity_both = st.sidebar.slider("Size+Weight Fail 투명도", 0.1, 1.0, 0.4, 0.1)

# ==================================================
# Upload & Preprocessing
# ==================================================
st.sidebar.header("📂 SKU Master 업로드")
file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
if file is None:
    st.info("좌측에서 SKU Master CSV 파일을 업로드하세요")
    st.stop()

df_raw = pd.read_csv(file)
df = df_raw.copy()

required_cols = ["SKU_ID", "LENGTH", "WIDTH", "HEIGHT", "WEIGHT", "SHIP_QTY"]
missing = set(required_cols) - set(df.columns)
if missing:
    st.error(f"누락된 컬럼: {missing}")
    st.stop()

numeric_cols = ["LENGTH", "WIDTH", "HEIGHT", "WEIGHT", "SHIP_QTY"]

# ==================================================
# Missing Value Handling & Category Refinement
# ==================================================
st.sidebar.header("🧹 결측치 처리")
nan_strategy = st.sidebar.radio("결측치 처리 방식", ["결측치 그대로 유지", "전체 평균", "카테고리 평균", "결측 SKU 제외"])

# MODIFIED: 수치형 데이터의 0값을 결측치(NaN)로 간주
for c in numeric_cols:
    df[c] = df[c].replace(0, np.nan)

category_col = None
cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != "SKU_ID"]

if nan_strategy == "카테고리 평균":
    if cat_cols:
        category_col = st.sidebar.selectbox("카테고리 기준 컬럼", cat_cols)
    else:
        nan_strategy = "전체 평균"

if nan_strategy == "결측 SKU 제외":
    df = df.dropna(subset=numeric_cols) 
elif nan_strategy == "전체 평균":
    for c in numeric_cols: df[c] = df[c].fillna(df[c].mean()) 
elif nan_strategy == "카테고리 평균" and category_col:
    # MODIFIED: 그룹별 평균 적용 및 전체 평균으로 2차 보완
    for c in numeric_cols:
        df[c] = df.groupby(category_col)[c].transform(lambda x: x.fillna(x.mean()))
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].mean())

# MODIFIED: 모델링을 위한 카테고리 인코딩 데이터프레임 생성
df_model = pd.get_dummies(df.drop(columns=['SKU_ID']), columns=cat_cols) if cat_cols else df.drop(columns=['SKU_ID'])

# ==================================================
# 물리적 적입 판정 로직 (Physical Fit Logic)
# ==================================================
def check_physical_fit(l, w, h, target_l, target_w, target_h):
    if h > target_h: return False
    fit_normal = (l <= target_l) and (w <= target_w)
    fit_rotate = (w <= target_l) and (l <= target_w)
    return fit_normal or fit_rotate

df["WEIGHT_KG"] = df["WEIGHT"] * unit_factor[weight_unit]
df["SIZE_FIT"] = df.apply(lambda r: check_physical_fit(r['LENGTH'], r['WIDTH'], r['HEIGHT'], container_l, container_w, container_h), axis=1)
df["WEIGHT_FIT"] = df["WEIGHT_KG"] <= max_weight

df["FAIL_REASON"] = "SIZE_WEIGHT_FAIL"
df.loc[df["SIZE_FIT"] & df["WEIGHT_FIT"], "FAIL_REASON"] = "ELIGIBLE"
df.loc[~df["SIZE_FIT"] & df["WEIGHT_FIT"], "FAIL_REASON"] = "SIZE_FAIL"
df.loc[df["SIZE_FIT"] & ~df["WEIGHT_FIT"], "FAIL_REASON"] = "WEIGHT_FAIL"

def get_max_physical_divider(l, w, h, cl, cw, ch):
    if not check_physical_fit(l, w, h, cl, cw, ch): return 1
    if check_physical_fit(l, w, h, cl/2, cw/2, ch): return 4
    if check_physical_fit(l, w, h, cl/2, cw, ch): return 2
    return 1

df["PHYSICAL_MAX_DIVIDER"] = df.apply(lambda r: get_max_physical_divider(r['LENGTH'], r['WIDTH'], r['HEIGHT'], container_l, container_w, container_h), axis=1)

# ==================================================
# Tabs Definition
# ==================================================
tab1, tab2, tab3 = st.tabs([
    "① 자동화 전환 KPI",
    "② 입수 · 결측치",
    "③ Divider & Clustering 통합 분석"
])

# --------------------------------------------------
# TAB 1: KPI & 3D Distribution
# --------------------------------------------------
with tab1:
    st.subheader("📊 자동화 전환 KPI")
    total_sku = len(df)
    eligible_sku = (df.FAIL_REASON == "ELIGIBLE").sum()
    sku_ratio = eligible_sku / total_sku if total_sku else 0
    total_ship = df.SHIP_QTY.sum()
    eligible_ship = df.loc[df.FAIL_REASON == "ELIGIBLE", "SHIP_QTY"].sum()
    ship_ratio = eligible_ship / total_ship if total_ship else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("대상 SKU 건수", f"{total_sku:,}건")
    k2.metric("자동화 가능 SKU", f"{eligible_sku:,}건", f"{sku_ratio:.1%} 커버")
    k3.metric("총 출하량", f"{int(total_ship):,}")
    k4.metric("출하 커버리지(Hit)", f"{int(eligible_ship):,}", f"{ship_ratio:.1%} 전환")

    st.markdown("---")
    st.subheader("🧊 SKU 3D Dimension Distribution")
    fig = go.Figure()

    def plot_group(reason, color, name, opacity):
        d = df[df["FAIL_REASON"] == reason]
        if len(d) == 0: return
        fig.add_trace(go.Scatter3d(
            x=d["LENGTH"], y=d["WIDTH"], z=d["HEIGHT"],
            mode="markers", marker=dict(size=3, color=color, opacity=opacity), name=name
        ))

    plot_group("ELIGIBLE", color_eligible, "Eligible", opacity_eligible)
    plot_group("SIZE_FAIL", color_size_fail, "Size Fail", opacity_size)
    plot_group("WEIGHT_FAIL", color_weight_fail, "Weight Fail", opacity_weight)
    plot_group("SIZE_WEIGHT_FAIL", color_both_fail, "Size + Weight Fail", opacity_both)

    cx = [0, container_l, container_l, 0, 0, container_l, container_l, 0]
    cy = [0, 0, container_w, container_w, 0, 0, container_w, container_w]
    cz = [0, 0, 0, 0, container_h, container_h, container_h, container_h]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[cx[i], cx[j]], y=[cy[i], cy[j]], z=[cz[i], cz[j]],
            mode="lines", line=dict(color=color_container, width=6), showlegend=False
        ))
    fig.update_layout(height=750, margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# TAB 2: Infeed Analysis
# --------------------------------------------------
with tab2:
    st.subheader("📦 용기 입수 분석 및 모델링 데이터")
    df2 = df.copy()
    # MODIFIED: 부피 변수 생성
    df2["VOLUME"] = df2["LENGTH"] * df2["WIDTH"] * df2["HEIGHT"]
    
    df2["MAX_QTY_SIZE"] = np.maximum(
        np.floor(container_l / df2["LENGTH"]) * np.floor(container_w / df2["WIDTH"]),
        np.floor(container_l / df2["WIDTH"]) * np.floor(container_w / df2["LENGTH"])
    )
    df2["MAX_QTY_WEIGHT"] = np.where(df2["WEIGHT_KG"] > 0, np.floor(max_weight / df2["WEIGHT_KG"]), 0)
    df2["MAX_QTY_CONTAINER"] = np.minimum(df2["MAX_QTY_SIZE"], df2["MAX_QTY_WEIGHT"])
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("📊 분석 결과 데이터 (일부)")
        st.dataframe(df2[["SKU_ID", "VOLUME", "MAX_QTY_CONTAINER", "FAIL_REASON"]], use_container_width=True)
    with col_b:
        st.write("🤖 인코딩 완료 데이터 (학습용)")
        st.dataframe(df_model.head(100), use_container_width=True)

# --------------------------------------------------
# TAB 3: Divider & Clustering 통합 분석
# --------------------------------------------------
with tab3:
    st.header("🧩 Divider 전략 및 시뮬레이션")
    
    with st.expander("🛠️ 전략 파라미터 및 디바이더 설정", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### **[분배 전략]**")
            a_ratio = st.slider("A급 SKU 기준 (출하량 상위 %)", 5, 50, 20, 5)
            apply_ratio = st.slider("B/C급 SKU 중 디바이더 적용 비율 (%)", 0, 100, 60, 10)
            # MODIFIED: 클러스터링 범위 선택 옵션 추가
            clustering_scope = st.radio("클러스터링 범위 설정", ["전체 대상", "카테고리 내 (선택 시 상단 결측치 카테고리 기준)"])
        with c2:
            st.markdown("##### **[물리 제약]**")
            vol_util_ratio = st.slider("📦 Bin 용적률 제한 (%)", 10, 100, 80, 5)
            weight_limit = st.slider("⚖️ 실시간 중량 제한 조정 (kg)", 0.1, float(max_weight), float(max_weight), 0.5)
        with c3:
            st.markdown("##### **[디바이더 가용성]**")
            use_div_0 = st.checkbox("0분할 (원물통) 사용", value=True, disabled=True)
            use_div_2 = st.checkbox("2분할 디바이더 사용", value=True)
            use_div_4 = st.checkbox("4분할 디바이더 사용", value=True)

    # --- 데이터 랭킹 및 등급 부여 ---
    df = df.sort_values("SHIP_QTY", ascending=False).reset_index(drop=True)
    df["SHIP_CUM_RATIO"] = df["SHIP_QTY"].cumsum() / df["SHIP_QTY"].sum()
    df["SHIP_GRADE"] = np.where(df["SHIP_CUM_RATIO"] <= a_ratio / 100, "A", "B/C")
    
    # --- 시뮬레이션용 디바이더 할당 로직 ---
    np.random.seed(42)
    def calculate_sim_divider(row):
        if row["SHIP_GRADE"] == "A": return 1
        if np.random.rand() >= apply_ratio/100: return 1
        
        phys_max = row["PHYSICAL_MAX_DIVIDER"]
        if phys_max == 4 and use_div_4: return 4
        if phys_max >= 2 and use_div_2: return 2
        return 1

    df["SIM_DIVIDER"] = df.apply(calculate_sim_divider, axis=1)
    df["BIN_REQ"] = 1 / df["SIM_DIVIDER"]

    # --- 매칭 엔진 함수 개선 ---
    def run_physical_clustering(input_df, cl, cw, ch, limit_w, util_ratio, cat_col): 
        eligible = input_df[(input_df["FAIL_REASON"] == "ELIGIBLE") & (input_df["SIM_DIVIDER"] > 1)].copy() 
        matched = []
        used_ids = set()
        bin_vol = cl * cw * ch

        div_types = []
        if use_div_4: div_types.append(4)
        if use_div_2: div_types.append(2)

        # MODIFIED: 클러스터링 범위 설정에 따른 그룹 분기 처리
        if clustering_scope == "카테고리 내 (선택 시 상단 결측치 카테고리 기준)" and cat_col and cat_col in eligible.columns:
            groups = eligible.groupby(cat_col)
        else:
            groups = [("All", eligible)]

        for _, group_df in groups: 
            for div in div_types:
                pool = group_df[(~group_df["SKU_ID"].isin(used_ids)) & (group_df["PHYSICAL_MAX_DIVIDER"] >= div)].to_dict('records')
                
                for i in range(0, len(pool) - (len(pool) % div), div):
                    group = pool[i:i+div]
                    group_weight = sum(g["WEIGHT_KG"] for g in group)
                    group_vol = sum(g["LENGTH"] * g["WIDTH"] * g["HEIGHT"] for g in group)
                    
                    if group_weight <= limit_w and group_vol <= (bin_vol * (util_ratio/100)):
                        matched.append({
                            "type": f"{div}-Divider", 
                            "sku_ids": [g["SKU_ID"] for g in group], 
                            "weight": group_weight,
                            "vol_usage": (group_vol / bin_vol) * 100
                        })
                        for g in group: used_ids.add(g["SKU_ID"])
        return matched

    # --- 매칭 엔진 실행 ---
    matches = run_physical_clustering(df, container_l, container_w, container_h, weight_limit, vol_util_ratio, category_col) 

    # --- 시뮬레이션 및 클러스터링 결과 KPI ---
    st.subheader("📊 시뮬레이션 및 클러스터링 결과")
    k1, k2, k3, k4 = st.columns(4)
    
    total_sku_count = len(df)
    matched_sku_ids = [sku_id for m in matches for sku_id in m["sku_ids"]] 
    matched_sku_count = len(matched_sku_ids) 
    unmatched_sku_count = total_sku_count - matched_sku_count 
    
    total_actual_bins = len(matches) + unmatched_sku_count 
    bins_saved_actual = total_sku_count - total_actual_bins 
    save_ratio_actual = (bins_saved_actual / total_sku_count) if total_sku_count > 0 else 0 

    k1.metric("전체 대상 SKU", f"{total_sku_count:,}건")
    k2.metric("필요 Bin 수 (실제)", f"{int(total_actual_bins):,}개", 
              delta=f"기존 대비 {int(total_actual_bins/total_sku_count*100) if total_sku_count > 0 else 0}%", delta_color="inverse")
    k3.metric("절감 Bin 수", f"{int(bins_saved_actual):,}개", delta=f"절감률 {save_ratio_actual:.1%}") 
    k4.metric("디바이더 적용 SKU", f"{matched_sku_count:,}건", 
              delta=f"전체 대비 {(matched_sku_count / total_sku_count * 100) if total_sku_count > 0 else 0:.1%}%") 

    # --- 시나리오 바 차트 ---
    scenario_res = []
    # MODIFIED: 시나리오 계산 시 재현성을 위해 시드 고정 및 간단한 로직 유지
    scenario_config = {
        "Best": {"A_RATIO": a_ratio * 0.7, "DIVIDER_APPLY": min(100, apply_ratio + 20)},
        "Base": {"A_RATIO": a_ratio, "DIVIDER_APPLY": apply_ratio},
        "Worst": {"A_RATIO": min(50, a_ratio * 1.3), "DIVIDER_APPLY": max(0, apply_ratio - 30)}
    }
    for name, cfg in scenario_config.items():
        temp = df.copy()
        temp["SCENARIO_A"] = (temp["SHIP_QTY"].cumsum() / temp["SHIP_QTY"].sum()) <= cfg["A_RATIO"]/100
        temp["DIV"] = np.where(temp["SCENARIO_A"], 1, 
                               np.where(np.random.rand(len(temp)) < cfg["DIVIDER_APPLY"]/100, 
                                        temp["PHYSICAL_MAX_DIVIDER"], 1))
        scenario_res.append({"Scenario": name, "Total Bin": (1/temp["DIV"]).sum()})
    st.bar_chart(pd.DataFrame(scenario_res).set_index("Scenario"))

    st.divider()
    # --- 상세 매칭 리스트 및 시각화 ---
    if matches:
        rec_df = pd.DataFrame([{ 
            "Bin_ID": f"BIN-{i+1:03d}", 
            "Type": m["type"], 
            "SKUs": ", ".join(map(str, m["sku_ids"])), 
            "Weight(kg)": round(m["weight"], 2),
            "Vol_Usage(%)": round(m["vol_usage"], 1)
        } for i, m in enumerate(matches)])
        
        col_list, col_viz = st.columns([1, 1.2])
        
        with col_list:
            st.write("📋 실시간 클러스터링 매칭 리스트")
            sum1, sum2, sum3 = st.columns(3)
            counts = rec_df["Type"].value_counts()
            sum1.metric("생성 Bin", f"{len(rec_df)}개")
            sum2.metric("4-Div", f"{counts.get('4-Divider', 0)}개")
            sum3.metric("2-Div", f"{counts.get('2-Divider', 0)}개")
            
            sel_bin = st.selectbox("🔍 3D로 확인할 Bin 선택", rec_df["Bin_ID"].tolist())
            st.dataframe(rec_df, use_container_width=True, height=400)
        
        with col_viz:
            target_idx = int(sel_bin.split("-")[1]) - 1
            target = matches[target_idx]
            fig_bin = go.Figure()

            if target["type"] == "4-Divider":
                fig_bin.add_trace(go.Scatter3d(x=[container_l/2]*2, y=[0, container_w], z=[0,0], mode='lines', line=dict(color='black', width=6), showlegend=False))
                fig_bin.add_trace(go.Scatter3d(x=[0, container_l], y=[container_w/2]*2, z=[0,0], mode='lines', line=dict(color='black', width=6), showlegend=False))
                regions = [(0,0,container_l/2,container_w/2), (container_l/2,0,container_l/2,container_w/2), 
                           (0,container_w/2,container_l/2,container_w/2), (container_l/2,container_w/2,container_l/2,container_w/2)]
            else:
                fig_bin.add_trace(go.Scatter3d(x=[container_l/2]*2, y=[0, container_w], z=[0,0], mode='lines', line=dict(color='black', width=6), showlegend=False))
                regions = [(0,0,container_l/2,container_w), (container_l/2,0,container_l/2,container_w)]

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
            for idx, sid in enumerate(target["sku_ids"]):
                s_data = df[df["SKU_ID"] == sid].iloc[0]
                rx, ry, rw, rh = regions[idx]
                sl, sw, sh = s_data["LENGTH"], s_data["WIDTH"], s_data["HEIGHT"]
                if sl > rw: sl, sw = sw, sl
                cx, cy = rx + (rw-sl)/2, ry + (rh-sw)/2
                
                fig_bin.add_trace(go.Mesh3d(
                    x=[cx, cx+sl, cx+sl, cx, cx, cx+sl, cx+sl, cx],
                    y=[cy, cy, cy+sw, cy+sw, cy, cy, cy+sw, cy+sw],
                    z=[0, 0, 0, 0, sh, sh, sh, sh],
                    i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], 
                    j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], 
                    k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                    color=colors[idx % 4], opacity=0.7, name=f"SKU {sid}"
                ))

            fig_bin.update_layout(
                title=f"{sel_bin} ({target['type']}) 내부 적입 현황",
                scene=dict(xaxis_title='L', yaxis_title='W', zaxis_title='H', aspectmode='data'),
                margin=dict(l=0, r=0, b=0, t=40), height=600
            )
            st.plotly_chart(fig_bin, use_container_width=True)
    else:
        st.warning("설정된 제약 조건 내에서 매칭되는 SKU 조합을 찾을 수 없습니다.")

st.sidebar.markdown("---")
st.sidebar.success(f"v2.0 Config: {container_l}x{container_w} Bin Ready")