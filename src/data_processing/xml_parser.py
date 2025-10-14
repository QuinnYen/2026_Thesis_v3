"""
SemEval 資料集 XML 解析器
支援 SemEval-2014 和 SemEval-2016 資料集格式
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
from collections import defaultdict
import os


class SemEvalDataset:
    """SemEval 資料集解析類別"""
    
    def __init__(self, xml_path: str):
        """
        初始化資料集解析器
        
        Args:
            xml_path: XML 檔案路徑
        """
        self.xml_path = xml_path
        self.sentences = []
        self.aspects = []
        self.parse_xml()
    
    def parse_xml(self):
        """解析 XML 檔案"""
        if not os.path.exists(self.xml_path):
            print(f"警告: 檔案不存在 - {self.xml_path}")
            return
        
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            
            # 解析句子和面向詞
            for sentence in root.findall('.//sentence'):
                sent_id = sentence.get('id')
                text_elem = sentence.find('text')
                text = text_elem.text if text_elem is not None else ""
                
                # 解析面向詞和情感
                aspect_terms = sentence.find('aspectTerms')
                sent_aspects = []
                
                if aspect_terms is not None:
                    for aspect_term in aspect_terms.findall('aspectTerm'):
                        aspect_data = {
                            'term': aspect_term.get('term'),
                            'polarity': aspect_term.get('polarity', 'unknown'),  # 預設為 'unknown'
                            'from': int(aspect_term.get('from', 0)),
                            'to': int(aspect_term.get('to', 0))
                        }
                        sent_aspects.append(aspect_data)
                        self.aspects.append(aspect_data)
                
                self.sentences.append({
                    'id': sent_id,
                    'text': text,
                    'aspects': sent_aspects
                })
        
        except Exception as e:
            print(f"解析錯誤: {e}")
    
    def get_statistics(self) -> Dict:
        """
        獲取資料集統計資訊
        
        Returns:
            統計資訊字典
        """
        stats = {
            'total_sentences': len(self.sentences),
            'total_aspects': len(self.aspects),
            'sentences_with_aspects': sum(1 for s in self.sentences if s['aspects']),
            'sentences_without_aspects': sum(1 for s in self.sentences if not s['aspects']),
            'sentiment_distribution': defaultdict(int),
            'avg_aspects_per_sentence': 0
        }
        
        # 計算情感分佈
        for aspect in self.aspects:
            polarity = aspect['polarity']
            stats['sentiment_distribution'][polarity] += 1
        
        # 計算平均面向詞數
        if stats['total_sentences'] > 0:
            stats['avg_aspects_per_sentence'] = stats['total_aspects'] / stats['total_sentences']
        
        return stats
    
    def print_statistics(self):
        """列印統計資訊"""
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"資料集: {os.path.basename(self.xml_path)}")
        print(f"{'='*60}")
        print(f"總句子數: {stats['total_sentences']}")
        print(f"  - 有面向詞的句子: {stats['sentences_with_aspects']}")
        print(f"  - 無面向詞的句子: {stats['sentences_without_aspects']}")
        print(f"\n總面向詞數: {stats['total_aspects']}")
        print(f"平均每句面向詞數: {stats['avg_aspects_per_sentence']:.2f}")
        
        print(f"\n情感分佈:")
        if stats['sentiment_distribution']:
            for sentiment, count in sorted(stats['sentiment_distribution'].items()):
                percentage = (count / stats['total_aspects'] * 100) if stats['total_aspects'] > 0 else 0
                sentiment_str = str(sentiment) if sentiment is not None else 'unknown'
                print(f"  - {sentiment_str:15s}: {count:5d} ({percentage:5.2f}%)")
        else:
            print(f"  (無情感標籤資料)")
        print(f"{'='*60}\n")


class SemEvalDatasetLoader:
    """SemEval 資料集載入器 - 統一管理所有資料集"""
    
    def __init__(self, base_path: str = "data/raw"):
        """
        初始化資料集載入器
        
        Args:
            base_path: 資料集根目錄
        """
        self.base_path = base_path
        self.datasets = {}
    
    def load_semeval_2014(self, domain: str = "both"):
        """
        載入 SemEval-2014 資料集
        
        Args:
            domain: 'laptop', 'restaurant', 或 'both'
        """
        semeval_2014_path = os.path.join(self.base_path, "SemEval-2014")
        
        files = {
            'laptop': {
                'train': 'Laptop_Train_v2.xml',
                'test_a': 'Laptops_Test_Data_PhaseA.xml',
                'test_b': 'Laptops_Test_Data_phaseB.xml'
            },
            'restaurant': {
                'train': 'Restaurants_Train_v2.xml',
                'test_a': 'Restaurants_Test_Data_PhaseA.xml',
                'test_b': 'Restaurants_Test_Data_phaseB.xml'
            }
        }
        
        domains_to_load = ['laptop', 'restaurant'] if domain == 'both' else [domain]
        
        for dom in domains_to_load:
            for split, filename in files[dom].items():
                key = f"semeval2014_{dom}_{split}"
                filepath = os.path.join(semeval_2014_path, filename)
                self.datasets[key] = SemEvalDataset(filepath)
    
    def load_semeval_2016(self, domain: str = "both"):
        """
        載入 SemEval-2016 資料集
        
        Args:
            domain: 'laptop', 'restaurant', 或 'both'
        """
        semeval_2016_path = os.path.join(self.base_path, "SemEval-2016")
        
        files = {
            'laptop': {
                'train': 'Laptops_Train_sb1.xml',
                'test': 'laptops_test_sb1.xml'
            },
            'restaurant': {
                'train': 'restaurants_train_sb1.xml',
                'test': 'restaurants_test_sb1.xml'
            }
        }
        
        domains_to_load = ['laptop', 'restaurant'] if domain == 'both' else [domain]
        
        for dom in domains_to_load:
            for split, filename in files[dom].items():
                key = f"semeval2016_{dom}_{split}"
                filepath = os.path.join(semeval_2016_path, filename)
                self.datasets[key] = SemEvalDataset(filepath)
    
    def print_all_statistics(self):
        """列印所有已載入資料集的統計資訊"""
        print("\n" + "="*60)
        print(" "*15 + "SemEval 資料集統計報告")
        print("="*60)
        
        for name, dataset in sorted(self.datasets.items()):
            dataset.print_statistics()
    
    def get_dataset(self, name: str) -> SemEvalDataset:
        """
        獲取特定資料集
        
        Args:
            name: 資料集名稱
            
        Returns:
            SemEvalDataset 實例
        """
        return self.datasets.get(name)


if __name__ == "__main__":
    # 使用範例
    loader = SemEvalDatasetLoader()
    
    print("正在載入 SemEval-2014 資料集...")
    loader.load_semeval_2014(domain='both')
    
    print("正在載入 SemEval-2016 資料集...")
    loader.load_semeval_2016(domain='both')
    
    # 列印所有統計資訊
    loader.print_all_statistics()

