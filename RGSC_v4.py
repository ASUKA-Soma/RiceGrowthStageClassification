import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import pandas as pd
import torch
import os
import plotly.express as px
import io
import matplotlib.pyplot as plt

# モデルをロード
def load_model():
    use_model = st.sidebar.selectbox("モデル", ['YOLO11m', 'YOLO11n'])
    if use_model == 'YOLO11m':
        model = YOLO('yolo11m.pt')
        conf = 0.568
    else:
        model = YOLO('yolo11n.pt')
        conf = 0.664
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, conf, use_model, device

# UIを初期化
def initialize_ui(device):
    st.title('稲成長段階分類AI')
    st.text("ver.4")
    st.text(f"{device}を使用して推論を行います．")

# 動画アップロード処理
def upload_video():
    return st.file_uploader("動画アップロード", type='mp4')

# CSVアップロード処理
def upload_csv():
    return st.file_uploader("CSVアップロード", type='csv')

# 動画を処理する関数（推論＋保存）
def process_video(upload_file, model, conf, use_model, batch_size=8, input_size=640):
    if upload_file is not None:
        if st.button(f'{use_model}で推論を開始する'):
            status_text = st.empty()
            progress_bar = st.progress(0)

            # 一時ファイルを作成
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(upload_file.read())
                temp_file_path = temp_file.name

            cap = cv2.VideoCapture(temp_file_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            filename = f"{os.path.splitext(upload_file.name)[0]}_{use_model}_result.mp4"
            writer = cv2.VideoWriter(filename,
                                    cv2.VideoWriter_fourcc(*'MP4V'),
                                    fps,
                                    (width, height))

            num = 0
            nums, leafs, leaf3, leaf4, leaf5, leaf6 = [], [], [], [], [], []
            batch_imgs = []  # バッチ用の画像リスト
            batch_originals = []  # 元のサイズの画像リスト

            with torch.no_grad():  # メモリ節約のため
                while cap.isOpened():
                    ret, img = cap.read()
                    if not ret:
                        break

                    # **📌 ① YOLO の要求するサイズにリサイズ**
                    img_resized = cv2.resize(img, (input_size, input_size))  # 640x640にリサイズ
                    batch_imgs.append(img_resized)
                    batch_originals.append(img)  # 元のサイズの画像も保持
                    num += 1

                    # バッチが満たされたら推論
                    if len(batch_imgs) == batch_size or num == count:
                        batch_tensor = [torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0 for im in batch_imgs]
                        batch_tensor = torch.stack(batch_tensor).to(model.device)

                        # **🚀 ② バッチ推論**
                        batch_results = model(batch_tensor, conf=conf)

                        # 各フレームの結果を処理
                        for i, img_result in enumerate(batch_results):
                            img_original = batch_originals[i]  # **元のサイズの画像**
                            img_annotated = img_result.plot(labels=True, conf=True)

                            # **📌 ③ 予測結果を元のサイズにリサイズして保存**
                            img_annotated = cv2.resize(img_annotated, (width, height))
                            writer.write(img_annotated)

                            categories = img_result.boxes.cls
                            leaf_num = len(categories)
                            leaf3_num = torch.sum(torch.eq(img_result.boxes.cls, 0)).item()
                            leaf4_num = torch.sum(torch.eq(img_result.boxes.cls, 1)).item()
                            leaf5_num = torch.sum(torch.eq(img_result.boxes.cls, 2)).item()
                            leaf6_num = torch.sum(torch.eq(img_result.boxes.cls, 3)).item()

                            nums.append(num - len(batch_imgs) + i + 1)
                            leafs.append(leaf_num)
                            leaf3.append(leaf3_num)
                            leaf4.append(leaf4_num)
                            leaf5.append(leaf5_num)
                            leaf6.append(leaf6_num)

                        batch_imgs = []  # バッチクリア
                        batch_originals = []

                    # 進捗表示
                    if num % 30 == 0:
                        progress = int(100 * num / count)
                        progress_bar.progress(min(progress, 100))
                        status_text.text(f"処理中... {progress}% ({num}/{count})")

            cap.release()
            writer.release()
            os.remove(temp_file_path)  # 一時ファイル削除

            progress_bar.progress(100)
            status_text.text("処理完了！")

            # データフレーム作成
            leaf_data = pd.DataFrame({'frame': nums, 'leafs': leafs, 'leaf3': leaf3, 'leaf4': leaf4, 'leaf5': leaf5, 'leaf6': leaf6})
            leaf_data['sec'] = leaf_data['frame'] / fps
            leaf_data = leaf_data[['sec', 'leafs', 'leaf3', 'leaf4', 'leaf5', 'leaf6']]

            return leaf_data, leaf3, leaf4, leaf5, leaf6

    return None, None, None, None, None

# CSV表示
def generate_csv(upload_file):
    if upload_file is not None:
        data = pd.read_csv(upload_file)
        data = data.dropna()
        #st.dataframe(data)
        return data


def generate_map_timeline(df):
    if df is not None:
        # グラフを作成
        fig, ax = plt.subplots(figsize=(8, 6))  # fig, ax を明示的に取得
        scatter = ax.scatter(df["longitude"], df["latitude"], c=range(len(df)), cmap='viridis', marker='o', edgecolor='k')
    
        # カラーバーを追加
        cbar = plt.colorbar(scatter, ax=ax, label="Time Step")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Position Over Time")

        # Streamlit で表示
        st.pyplot(fig)


def aggregate_variable_blocks(df, target_rows):
    """
    指定した target_rows 行に圧縮するため、データを変則的なブロックサイズで集約する。
    """
    df = df.copy()  # 元データを変更しないようにコピー
    total_rows = len(df)
    
    step = total_rows / target_rows  # 1ブロックあたりの目標行数（小数値）
    
    # 各ブロックの範囲を決定
    block_ranges = [int(step * (i + 1)) for i in range(target_rows)]
    block_sizes = np.diff([0] + block_ranges)  # 各ブロックの行数
    
    # 各ブロックごとにデータを集約
    grouped_data = []
    start = 0
    for block_size in block_sizes:
        if block_size <= 0:  # 念のため異常な値を回避
            continue
        
        block_df = df.iloc[start:start + block_size]
        start += block_size
        
        aggregated = {
            'sec': block_df['sec'].mean(),  # sec は平均
            'leafs': block_df['leafs'].sum(),  # 葉の総数
            'leaf3': block_df['leaf3'].sum(),
            'leaf4': block_df['leaf4'].sum(),
            'leaf5': block_df['leaf5'].sum(),
            'leaf6': block_df['leaf6'].sum()
        }

        # leaf_ave の再計算
        if aggregated['leafs'] == 0:
            aggregated['leaf_ave'] = 0  # ゼロ除算防止
        else:
            aggregated['leaf_ave'] = (
                aggregated['leaf3'] * 3 +
                aggregated['leaf4'] * 4 +
                aggregated['leaf5'] * 5 +
                aggregated['leaf6'] * 6
            ) / aggregated['leafs']
        
        grouped_data.append(aggregated)

    # データフレームに変換
    grouped_df = pd.DataFrame(grouped_data)
    
    return grouped_df


# 結果を可視化
def generate_charts(leaf_data, leaf3, leaf4, leaf5, leaf6):
    if leaf_data is not None:
        data = pd.DataFrame({
            "葉齢": ['3葉期', '4葉期', '5葉期', '6葉期'],
            "値": [sum(leaf3), sum(leaf4), sum(leaf5), sum(leaf6)]
        })

        # 円グラフを描画
        fig = px.pie(data, values='値', names='葉齢', title="各葉齢の割合", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig)

        # 時系列データをプロット
        st.markdown("### 🌟 時刻ごとの葉齢の推移")
        st.line_chart(leaf_data, x="sec")

        # 割合を表示
        total_leafs = sum(leaf3) + sum(leaf4) + sum(leaf5) + sum(leaf6)
        if total_leafs > 0:
            st.text(f'3葉率：{sum(leaf3) / total_leafs * 100:.2f}%，'
                    f'4葉率：{sum(leaf4) / total_leafs * 100:.2f}%，'
                    f'5葉率：{sum(leaf5) / total_leafs * 100:.2f}%，'
                    f'6葉率：{sum(leaf6) / total_leafs * 100:.2f}%')

        # データフレームを表示
        #st.dataframe(leaf_data)

def merge_dataframes(df1, df2, join_type="outer"):
    merged_df = pd.concat([df1, df2], axis=1, join=join_type)
    #st.dataframe(merged_df)
    
    # グラフを作成
    fig, ax = plt.subplots(figsize=(8, 6))  # fig, ax を明示的に取得
    ax.set_facecolor('lightblue')
    #scatter = ax.scatter(merged_df["longitude"], merged_df["latitude"], 
    # c=merged_df["leaf_ave"], cmap='viridis', marker='o', edgecolor='k')
    
    # 0 ではないデータのマーカーは 'o'
    mask_3 = merged_df["leaf_ave"] > 2.5
    scatter = ax.scatter(merged_df["longitude"][mask_3], merged_df["latitude"][mask_3], 
               c='green', marker='o', edgecolor='k', vmin=3, vmax=6, label="leaf3")
    mask_4 = merged_df["leaf_ave"] > 3.5
    scatter = ax.scatter(merged_df["longitude"][mask_4], merged_df["latitude"][mask_4], 
               c='yellow', marker='o', edgecolor='k', vmin=3, vmax=6, label="leaf4")
    mask_5 = merged_df["leaf_ave"] > 4.5
    scatter = ax.scatter(merged_df["longitude"][mask_5], merged_df["latitude"][mask_5], 
               c='orange', marker='o', edgecolor='k', vmin=3, vmax=6, label="leaf5")
    mask_6 = merged_df["leaf_ave"] > 5.5
    scatter = ax.scatter(merged_df["longitude"][mask_6], merged_df["latitude"][mask_6], 
               c='red', marker='o', edgecolor='k', vmin=3, vmax=6, label="leaf6")
    
    # 0 のデータのマーカーは 'x'
    #mask_zero = merged_df["leaf_ave"] == 0
    #ax.scatter(merged_df["longitude"][mask_zero], merged_df["latitude"][mask_zero], 
    #           c='black', marker='x', edgecolor='k')

    # カラーバーを追加
    #cbar = plt.colorbar(scatter, ax=ax, label="Leaf Stage")
    ax.legend(title="Leaf Stage")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Position Leaf Stage")

    # Streamlit で表示
    st.pyplot(fig)



# メイン処理
def main():
    model, conf, use_model, device = load_model()
    initialize_ui(device)
    upload_file = upload_video()
    csv = upload_csv()
    
    df_gps = generate_csv(csv)
    leaf_data, leaf3, leaf4, leaf5, leaf6 = process_video(upload_file, model, conf, use_model)
    
    generate_charts(leaf_data, leaf3, leaf4, leaf5, leaf6)
    #generate_map_timeline(df_gps)

    if leaf_data is not None:
        df_leaf = aggregate_variable_blocks(leaf_data, len(df_gps))
        #st.dataframe(df_leaf)
        #st.dataframe(df_gps)
        merge_dataframes(df_leaf, df_gps)
    


    

if __name__ == "__main__":
    main()
