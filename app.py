import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="인공신경망 데이터 예측 프로그램", layout="wide")
st.title("🧠 인공신경망(ANN) 기반 데이터 예측 프로그램")
st.write("수업 시간에 배운 **입력층, 은닉층, 가중치, 편향, 활성화 함수(Sigmoid), 그리고 경사하강법(Backpropagation)**을 활용하여 CSV 데이터를 학습하고 예측하는 인공신경망 웹 앱입니다.")

st.sidebar.header("1. 데이터 파일 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 업로드된 데이터 미리보기 (상위 5행)")
        st.dataframe(df.head())
        
        columns = df.columns.tolist()
        
        st.sidebar.header("2. 변수(Feature & Target) 선택")
        x_cols = st.sidebar.multiselect("X축 입력 변수 (Input Features, 다중 선택 가능)", columns, default=[columns[0]] if len(columns)>0 else [])
        y_col = st.sidebar.selectbox("Y축 출력 변수 (Target)", columns, index=len(columns)-1 if len(columns)>1 else 0)
        
        st.sidebar.header("3. 인공신경망 하이퍼파라미터")
        hidden_units = st.sidebar.slider("은닉층 뉴런 수 (Hidden Neurons)", 1, 10, 4, step=1)
        learning_rate = st.sidebar.number_input("학습률 (Learning Rate)", value=0.01, format="%.4f")
        epochs = st.sidebar.slider("학습 횟수 (Epochs)", 100, 10000, 2000, step=100)
        
        if len(x_cols) > 0 and y_col:
            # 데이터 정제 및 숫자로 변환
            sub_cols = x_cols + [y_col]
            temp_df = df[sub_cols].copy()
            for col in sub_cols:
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
            temp_df = temp_df.dropna()
            
            if len(temp_df) > 2:
                X_raw = temp_df[x_cols].values
                y_raw = temp_df[y_col].values.reshape(-1, 1)
                
                # 신경망 학습 안정을 위한 표준화(Standardization) 스케일링
                X_mean, X_std = X_raw.mean(axis=0), X_raw.std(axis=0)
                X_std[X_std == 0] = 1.0
                X = (X_raw - X_mean) / X_std
                
                y_mean, y_std = y_raw.mean(), y_raw.std()
                if y_std == 0: y_std = 1.0
                y = (y_raw - y_mean) / y_std
                
                N, D = X.shape
                H = hidden_units
                
                # 가중치 및 편향 초기화
                np.random.seed(42)
                W1 = np.random.randn(D, H) * 0.1
                b1 = np.zeros((1, H))
                W2 = np.random.randn(H, 1) * 0.1
                b2 = np.zeros((1, 1))
                
                # 활성화 함수: Sigmoid (수업 자료 반영)
                def sigmoid(z):
                    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                
                def sigmoid_derivative(a):
                    return a * (1 - a)
                
                loss_history = []
                
                # 경사하강법 및 역전파 학습 루프
                for epoch in range(epochs):
                    # 1. 순전파 (Forward Propagation)
                    Z1 = np.dot(X, W1) + b1
                    A1 = sigmoid(Z1)
                    Z2 = np.dot(A1, W2) + b2
                    Y_pred_norm = Z2
                    
                    # 2. 손실 계산 (MSE)
                    error = Y_pred_norm - y
                    loss = np.mean(error ** 2)
                    loss_history.append(loss)
                    
                    # 3. 역전파 및 가중치 갱신 (Backward Propagation & Gradient Descent)
                    dY = 2 * error / N
                    dW2 = np.dot(A1.T, dY)
                    db2 = np.sum(dY, axis=0, keepdims=True)
                    
                    dA1 = np.dot(dY, W2.T)
                    dZ1 = dA1 * sigmoid_derivative(A1)
                    dW1 = np.dot(X.T, dZ1)
                    db1 = np.sum(dZ1, axis=0, keepdims=True)
                    
                    W1 -= learning_rate * dW1
                    b1 -= learning_rate * db1
                    W2 -= learning_rate * dW2
                    b2 -= learning_rate * db2
                
                # 최종 예측값 복원
                Z1_final = np.dot(X, W1) + b1
                A1_final = sigmoid(Z1_final)
                Y_pred_scaled = np.dot(A1_final, W2) + b2
                Y_pred = Y_pred_scaled * y_std + y_mean
                Y_true_orig = y_raw
                
                # 시각화 및 결과 출력
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📈 인공신경망 학습 결과 시각화")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    if D == 1:
                        sort_idx = np.argsort(X_raw[:, 0])
                        ax1.scatter(X_raw[:, 0], Y_true_orig, color='crimson', label='실제 데이터 (True)', zorder=5)
                        ax1.plot(X_raw[sort_idx, 0], Y_pred[sort_idx], color='navy', linewidth=2.5, label='인공신경망 예측 (ANN)', zorder=6)
                        ax1.set_xlabel(x_cols[0])
                        ax1.set_ylabel(y_col)
                    else:
                        ax1.scatter(Y_true_orig, Y_pred, color='teal', alpha=0.8)
                        ax1.plot([Y_true_orig.min(), Y_true_orig.max()], [Y_true_orig.min(), Y_true_orig.max()], 'r--', label='이상적 예측선')
                        ax1.set_xlabel("실제값 (Actual)")
                        ax1.set_ylabel("예측값 (Predicted)")
                    
                    ax1.set_title("실제값 vs 인공신경망 예측값")
                    ax1.legend()
                    ax1.grid(True, linestyle='--', alpha=0.5)
                    
                    ax2.plot(loss_history, color='purple', linewidth=1.5)
                    ax2.set_title("학습 에폭별 손실 함수 (MSE Loss)")
                    ax2.set_xlabel("Epochs")
                    ax2.set_ylabel("Loss")
                    ax2.grid(True, linestyle='--', alpha=0.5)
                    
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("📌 인공신경망 학습 요약")
                    st.success(f"**네트워크 구조:**\n- 입력층 뉴런: {D}개\n- 은닉층 뉴런: {H}개 (Sigmoid)\n- 출력층 뉴런: 1개")
                    st.metric(label="최종 손실 (MSE Loss)", value=f"{loss_history[-1]:.6f}")
                    st.metric(label="학습 데이터 수", value=f"{N}개")
                    
                    result_df = pd.DataFrame(X_raw, columns=x_cols)
                    result_df['Actual'] = Y_true_orig.flatten()
                    result_df['Predicted'] = Y_pred.flatten()
                    st.subheader("🔍 예측 결과 비교표")
                    st.dataframe(result_df.head(10))
            else:
                st.warning("⚠️ 유효한 숫자 데이터가 3개 이상 필요합니다.")
        else:
            st.warning("⚠️ X축 입력 변수와 Y축 출력 변수를 각각 선택해 주세요.")
    except Exception as e:
        st.error(f"⚠️ 데이터를 처리하는 중 오류가 발생했습니다: {e}")
else:
    st.info("👈 왼쪽 사이드바에서 분석할 **CSV 형식의 데이터 파일**을 업로드해 주세요!")
