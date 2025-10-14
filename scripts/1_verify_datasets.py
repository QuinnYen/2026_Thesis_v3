"""
資料集驗證腳本
用於快速驗證 SemEval 資料集是否正確放置和可讀取
"""

import os
import sys

# 確保可以導入專案模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import SemEvalDatasetLoader


def check_file_existence():
    """檢查所有必要的資料檔案是否存在"""
    print("\n" + "="*60)
    print("  資料檔案存在性檢查")
    print("="*60 + "\n")
    
    base_path = "data/raw"
    
    # SemEval-2014 檔案
    semeval_2014_files = {
        'Laptop': [
            'Laptop_Train_v2.xml',
            'Laptops_Test_Data_PhaseA.xml',
            'Laptops_Test_Data_phaseB.xml'
        ],
        'Restaurant': [
            'Restaurants_Train_v2.xml',
            'Restaurants_Test_Data_PhaseA.xml',
            'Restaurants_Test_Data_phaseB.xml'
        ]
    }
    
    # SemEval-2016 檔案
    semeval_2016_files = {
        'Laptop': [
            'Laptops_Train_sb1.xml',
            'laptops_test_sb1.xml'
        ],
        'Restaurant': [
            'restaurants_train_sb1.xml',
            'restaurants_test_sb1.xml'
        ]
    }
    
    all_exist = True
    
    # 檢查 SemEval-2014
    print("SemEval-2014:")
    for domain, files in semeval_2014_files.items():
        print(f"  {domain}:")
        for filename in files:
            filepath = os.path.join(base_path, "SemEval-2014", filename)
            exists = os.path.exists(filepath)
            status = "✓" if exists else "✗"
            print(f"    {status} {filename}")
            if not exists:
                all_exist = False
    
    # 檢查 SemEval-2016
    print("\nSemEval-2016:")
    for domain, files in semeval_2016_files.items():
        print(f"  {domain}:")
        for filename in files:
            filepath = os.path.join(base_path, "SemEval-2016", filename)
            exists = os.path.exists(filepath)
            status = "✓" if exists else "✗"
            print(f"    {status} {filename}")
            if not exists:
                all_exist = False
    
    print("\n" + "="*60)
    if all_exist:
        print("✓ 所有資料檔案都存在！")
    else:
        print("✗ 部分資料檔案缺失，請參考 DATA_PREPARATION_GUIDE.md")
    print("="*60 + "\n")
    
    return all_exist


def verify_datasets():
    """驗證資料集並顯示統計資訊"""
    print("\n開始驗證資料集...\n")
    
    # 先檢查檔案是否存在
    files_exist = check_file_existence()
    
    if not files_exist:
        print("\n⚠ 警告: 部分資料檔案缺失")
        print("請先下載並放置資料集檔案")
        print("詳細說明請參考: DATA_PREPARATION_GUIDE.md\n")
        return False
    
    # 載入資料集
    try:
        loader = SemEvalDatasetLoader(base_path="data/raw")
        
        print("正在載入 SemEval-2014 資料集...")
        loader.load_semeval_2014(domain='both')
        
        print("正在載入 SemEval-2016 資料集...")
        loader.load_semeval_2016(domain='both')
        
        # 顯示統計資訊
        loader.print_all_statistics()
        
        # 驗證資料完整性
        print("\n" + "="*60)
        print("  資料完整性驗證")
        print("="*60 + "\n")
        
        all_valid = True
        for name, dataset in loader.datasets.items():
            stats = dataset.get_statistics()
            is_valid = (
                stats['total_sentences'] > 0 and
                stats['total_aspects'] >= 0
            )
            status = "✓" if is_valid else "✗"
            print(f"{status} {name}: {stats['total_sentences']} 句, {stats['total_aspects']} 面向詞")
            if not is_valid:
                all_valid = False
        
        print("\n" + "="*60)
        if all_valid:
            print("✓ 所有資料集驗證通過！")
            print("\n下一步: 執行 dataset_statistics.py 生成詳細報告和視覺化")
        else:
            print("✗ 部分資料集驗證失敗")
        print("="*60 + "\n")
        
        return all_valid
        
    except Exception as e:
        print(f"\n✗ 驗證過程中發生錯誤: {e}\n")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" "*15 + "SemEval 資料集驗證工具")
    print("="*60)
    
    success = verify_datasets()
    
    if success:
        print("\n✅ 資料集準備完成！可以開始進行下一步工作。\n")
        sys.exit(0)
    else:
        print("\n⚠️  資料集準備未完成，請檢查上述訊息。\n")
        sys.exit(1)

