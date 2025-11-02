import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title='卒中预测Demo', layout='centered')

# ---------- 1. 载入模型 ----------
@st.cache_resource
def load_model():
    # 返回任意能 predict/predict_proba 的对象
    return joblib.load('tabpfn_model.pkl')

model = load_model()

# ---------- 2. 特征清单 ----------
CONT_FEATURES = ['onset', '到院收缩压', 'BMI', '溶栓剂量', '溶栓前braden',
                 '溶栓前NIHSS', '溶栓结束时NIHSS', '溶前中性粒细胞比例',
                 '溶前红细胞压积', '溶前谷丙转氨酶', '尿素', '血糖',
                 '尿酸', '低密度脂蛋白']

DISC_FEATURES = ['高血压', '脑卒中']

# ---------- 3. 侧边栏输入 ----------
st.title('ENI预测')

with st.sidebar:
    st.header('连续特征')
    cont = {f: st.text_input(f, placeholder='直接回车即空') for f in CONT_FEATURES}

    st.header('离散特征')
    disc = {f: st.text_input(f, placeholder='0 或 1') for f in DISC_FEATURES}

# ---------- 4. 预测 ----------
if st.sidebar.button('Predict'):
    record = {}
    # 连续值：能转 float 就写，否则跳过
    for f, v in cont.items():
        try:
            record[f] = float(v)
        except ValueError:
            pass
    # 离散值：能转 int 就写，否则跳过
    for f, v in disc.items():
        try:
            record[f] = int(v)
        except ValueError:
            pass

    if not record:
        st.error('至少填一个特征！')
        st.stop()

    X = pd.DataFrame([record])          # 只含用户填的列
    proba = model.predict_proba(X)[1]
    #label = '高风险 ⚠️' if proba >= 0.5 else '低风险 ✅'

    st.metric('Probability of ENI', f'{proba:.2%}')
    #st.subheader(label)