import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def predict_boiling_point(carbon_number):
    """
    CnHn分子（アルカン）の沸点を予測する関数
    
    Parameters:
    carbon_number (int): 炭素数
    
    Returns:
    float: 予測沸点（摂氏）
    """
    # 既知のデータ（炭素数と対応する沸点（摂氏））
    # メタン(C1)からデカン(C10)までのデータ
    carbon_numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
    boiling_points = np.array([-161.5, -88.6, -42.1, -0.5, 36.1, 68.7, 98.4, 125.7, 150.8, 174.0])
    
    # 線形回帰モデルの作成
    model = LinearRegression()
    model.fit(carbon_numbers, boiling_points)
    
    # 入力された炭素数での沸点予測
    predicted_bp = model.predict(np.array([[carbon_number]]))[0]
    
    return predicted_bp

def plot_boiling_points(max_carbon=20):
    """
    炭素数と沸点の関係をプロットする関数
    
    Parameters:
    max_carbon (int): プロットする最大炭素数
    """
    carbon_range = range(1, max_carbon + 1)
    boiling_points = [predict_boiling_point(c) for c in carbon_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(carbon_range, boiling_points, 'o-', linewidth=2)
    plt.xlabel('炭素数')
    plt.ylabel('沸点 (°C)')
    plt.title('アルカン(CnH2n+2)の炭素数と沸点の関係')
    plt.grid(True)
    plt.savefig('alkane_boiling_points.png')
    plt.show()

if __name__ == "__main__":
    # ユーザー入力
    while True:
        try:
            carbon_input = input("予測したいアルカンの炭素数を入力してください（終了するには'q'）: ")
            if carbon_input.lower() == 'q':
                break
            
            carbon_number = int(carbon_input)
            if carbon_number <= 0:
                print("正の整数を入力してください")
                continue
                
            bp = predict_boiling_point(carbon_number)
            print(f"C{carbon_number}H{2*carbon_number+2}の予測沸点: {bp:.1f}°C")
            
            # グラフを表示するかどうか
            show_graph = input("炭素数と沸点の関係をグラフ表示しますか？ (y/n): ")
            if show_graph.lower() == 'y':
                max_c = int(input("最大炭素数を入力してください: "))
                plot_boiling_points(max_c)
                
        except ValueError:
            print("有効な数値を入力してください")
        
        print()  # 空行を挿入
