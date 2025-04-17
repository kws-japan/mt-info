import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments

def predict_tg(smiles):
    """
    高分子モノマーのSMILES表記からガラス転移温度(Tg)を予測する関数
    
    Args:
        smiles (str): モノマーのSMILES表記
    
    Returns:
        float: 予測されたTg値（ケルビン）
    """
    # モノマーの分子構造を読み込み
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 基本的な構造特徴量を計算
    mol_weight = Descriptors.MolWt(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    
    # 官能基のカウント
    aromatic_rings = Chem.GetSSSR(mol)
    aromatic_count = 0
    for ring in aromatic_rings:
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            aromatic_count += 1
    
    # 特定の官能基のカウント
    amide_count = Fragments.fr_amide(mol)
    ester_count = Fragments.fr_ester(mol)
    ether_count = Fragments.fr_ether(mol)
    alcohol_count = Fragments.fr_alcohol(mol)
    
    # 各特徴量に対する寄与度（これらの値は文献や実験データから調整が必要）
    # 以下は簡略化したモデルでの仮の係数
    base_tg = 150  # 基準Tg（K）
    mw_contrib = 0.1 * mol_weight
    flex_contrib = -7 * rotatable_bonds  # 柔軟性は一般的にTgを下げる
    hbond_contrib = 10 * (h_donors + 0.5 * h_acceptors)  # 水素結合はTgを上げる
    
    # 官能基の寄与
    aromatic_contrib = 30 * aromatic_count  # 芳香族環は剛直性を上げTgを上昇
    amide_contrib = 50 * amide_count  # アミド結合は強い分子間力によりTgを上昇
    ester_contrib = 20 * ester_count
    ether_contrib = 5 * ether_count
    alcohol_contrib = 25 * alcohol_count
    
    # 最終的なTg予測値（K）
    predicted_tg = (base_tg + mw_contrib + flex_contrib + hbond_contrib +
                   aromatic_contrib + amide_contrib + ester_contrib +
                   ether_contrib + alcohol_contrib)
    
    return max(0, predicted_tg)  # 物理的に意味のある範囲に制限

def main():
    # テスト用のモノマー例（SMILES表記）
    test_monomers = {
        "ポリスチレン モノマー": "C=CC1=CC=CC=C1",
        "メチルメタクリレート": "C=C(C)C(=O)OC",
        "ビニルアルコール": "C=CO",
        "アクリルアミド": "C=CC(=O)N"
    }
    
    print("モノマーのガラス転移温度(Tg)予測結果:")
    print("-" * 50)
    for name, smiles in test_monomers.items():
        tg_k = predict_tg(smiles)
        tg_c = tg_k - 273.15  # ケルビンから摂氏に変換
        print(f"{name} (SMILES: {smiles}):")
        print(f"  予測Tg = {tg_k:.1f} K ({tg_c:.1f} °C)")
        print("-" * 50)
    
    # ユーザー入力
    print("\nSMILES形式でモノマーを入力してください（終了するには'q'を入力）:")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'q':
            break
        
        tg_k = predict_tg(user_input)
        if tg_k is None:
            print("無効なSMILES形式です。再度入力してください。")
        else:
            tg_c = tg_k - 273.15
            print(f"予測Tg = {tg_k:.1f} K ({tg_c:.1f} °C)")

if __name__ == "__main__":
    main()