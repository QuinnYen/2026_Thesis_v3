"""
多維度實驗調度器

統一管理和執行五個維度的實驗組合：
1. 基礎架構
2. 詞向量提升
3. 上下文編碼器
4. 句法信息
5. 多種子穩定性

功能：
- 自動生成實驗計劃
- 斷點續傳機制
- 實驗進度追蹤
- 結果自動彙總
- 平行執行支援

使用範例：
    # 執行基礎實驗（維度1）
    python scripts/run_multi_dimensional_experiments.py --dimensions 1

    # 執行詞向量實驗（維度1+2）
    python scripts/run_multi_dimensional_experiments.py --dimensions 1 2

    # 執行完整實驗（所有維度）
    python scripts/run_multi_dimensional_experiments.py --dimensions 1 2 3 4 5

    # 使用預定義組合
    python scripts/run_multi_dimensional_experiments.py --combination basic
    python scripts/run_multi_dimensional_experiments.py --combination full

    # 斷點續傳
    python scripts/run_multi_dimensional_experiments.py --resume checkpoint_20231029_123456
"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import pickle

# 加入專案路徑
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "configs"))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from experiment_dimensions import (
    generate_experiment_plan,
    print_experiment_summary,
    get_dimension_config,
    get_combination_config,
    DIMENSION_COMBINATIONS,
    EXECUTION_CONFIG
)
from experiment_executor import ConfigParser, TrainingExecutor

# ============================================================================
# 實驗追蹤器
# ============================================================================

class ExperimentTracker:
    """實驗進度追蹤器"""

    def __init__(self, checkpoint_dir: Path, log_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.checkpoint_dir / f"session_{self.session_id}.json"
        self.log_file = self.log_dir / f"experiment_log_{self.session_id}.txt"

        self.state = {
            "session_id": self.session_id,
            "start_time": time.time(),
            "total_experiments": 0,
            "completed_experiments": 0,
            "failed_experiments": 0,
            "skipped_experiments": 0,
            "experiments": {},
            "results": []
        }

    def initialize(self, plan: Dict[str, Any]):
        """初始化追蹤器"""
        self.state["total_experiments"] = len(plan["experiments"])
        self.state["plan"] = plan
        self.save_checkpoint()

        # 寫入日誌
        self.log(f"實驗會話開始: {self.session_id}")
        self.log(f"總實驗數: {self.state['total_experiments']}")

    def mark_completed(self, exp_id: str, result: Dict[str, Any]):
        """標記實驗完成"""
        self.state["completed_experiments"] += 1
        # 清理結果，只保留可序列化的數據
        clean_result = self._clean_result(result)
        self.state["experiments"][exp_id] = {
            "status": "completed",
            "result": clean_result,
            "completed_time": time.time()
        }
        self.state["results"].append(clean_result)
        self.save_checkpoint()

        self.log(f"[OK] 實驗完成 [{self.state['completed_experiments']}/{self.state['total_experiments']}]: {exp_id}")

    def mark_failed(self, exp_id: str, error: str):
        """標記實驗失敗"""
        self.state["failed_experiments"] += 1
        self.state["experiments"][exp_id] = {
            "status": "failed",
            "error": error,
            "failed_time": time.time()
        }
        self.save_checkpoint()

        self.log(f"[FAIL] 實驗失敗: {exp_id} - {error}")

    def mark_skipped(self, exp_id: str, reason: str):
        """標記實驗跳過"""
        self.state["skipped_experiments"] += 1
        self.state["experiments"][exp_id] = {
            "status": "skipped",
            "reason": reason
        }
        self.save_checkpoint()

        self.log(f"[SKIP] 實驗跳過: {exp_id} - {reason}")

    def is_completed(self, exp_id: str) -> bool:
        """檢查實驗是否已完成"""
        return exp_id in self.state["experiments"] and self.state["experiments"][exp_id]["status"] == "completed"

    def _clean_result(self, result: Any) -> Any:
        """
        清理結果數據，移除無法序列化的對象和循環引用

        Args:
            result: 原始結果

        Returns:
            清理後的可序列化結果
        """
        if result is None:
            return None

        if isinstance(result, (str, int, float, bool)):
            return result

        if isinstance(result, (list, tuple)):
            return [self._clean_result(item) for item in result]

        if isinstance(result, dict):
            clean_dict = {}
            for key, value in result.items():
                # 跳過某些可能造成循環引用的鍵
                if key in ['model', 'optimizer', 'train_loader', 'val_loader', 'config']:
                    continue
                try:
                    clean_dict[key] = self._clean_result(value)
                except (TypeError, ValueError):
                    # 如果無法序列化，轉為字符串
                    clean_dict[key] = str(value)
            return clean_dict

        # 其他類型轉為字符串
        return str(result)

    def save_checkpoint(self):
        """儲存檢查點"""
        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False, default=str)

    def load_checkpoint(self, session_id: str):
        """載入檢查點"""
        checkpoint_file = self.checkpoint_dir / f"session_{session_id}.json"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"找不到檢查點: {checkpoint_file}")

        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            self.state = json.load(f)

        self.session_id = session_id
        self.session_file = checkpoint_file
        self.log_file = self.log_dir / f"experiment_log_{session_id}.txt"

        self.log(f"從檢查點恢復: {session_id}")

    def log(self, message: str):
        """寫入日誌"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        # 印出到控制台
        print(log_message)

        # 寫入日誌文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")

    def get_progress(self) -> Dict[str, Any]:
        """獲取進度資訊"""
        elapsed_time = time.time() - self.state["start_time"]
        completed = self.state["completed_experiments"]
        total = self.state["total_experiments"]

        progress = {
            "completed": completed,
            "total": total,
            "percentage": (completed / total * 100) if total > 0 else 0,
            "failed": self.state["failed_experiments"],
            "skipped": self.state["skipped_experiments"],
            "elapsed_time_seconds": elapsed_time,
            "estimated_remaining_seconds": (elapsed_time / completed * (total - completed)) if completed > 0 else None
        }

        return progress

    def print_progress(self):
        """印出進度"""
        progress = self.get_progress()

        print("\n" + "=" * 80)
        print("實驗進度")
        print("=" * 80)
        print(f"完成: {progress['completed']}/{progress['total']} ({progress['percentage']:.1f}%)")
        print(f"失敗: {progress['failed']}")
        print(f"跳過: {progress['skipped']}")
        print(f"已用時間: {progress['elapsed_time_seconds']/60:.1f} 分鐘")

        if progress['estimated_remaining_seconds']:
            print(f"預估剩餘時間: {progress['estimated_remaining_seconds']/60:.1f} 分鐘")

        print("=" * 80 + "\n")

    def finalize(self):
        """完成實驗"""
        self.state["end_time"] = time.time()
        self.state["total_time"] = self.state["end_time"] - self.state["start_time"]
        self.save_checkpoint()

        # 生成最終報告
        self.generate_final_report()

    def generate_final_report(self):
        """生成最終報告"""
        report_file = self.log_dir / f"final_report_{self.session_id}.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("多維度實驗最終報告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"會話ID: {self.session_id}\n")
            f.write(f"開始時間: {datetime.fromtimestamp(self.state['start_time'])}\n")
            f.write(f"結束時間: {datetime.fromtimestamp(self.state['end_time'])}\n")
            f.write(f"總耗時: {self.state['total_time']/3600:.2f} 小時\n\n")

            f.write(f"總實驗數: {self.state['total_experiments']}\n")
            f.write(f"完成: {self.state['completed_experiments']}\n")
            f.write(f"失敗: {self.state['failed_experiments']}\n")
            f.write(f"跳過: {self.state['skipped_experiments']}\n\n")

            # 結果摘要
            if self.state["results"]:
                f.write("=" * 80 + "\n")
                f.write("實驗結果摘要\n")
                f.write("=" * 80 + "\n\n")

                for result in self.state["results"]:
                    f.write(f"實驗: {result.get('experiment_id', 'N/A')}\n")
                    f.write(f"  數據集: {result.get('dataset', 'N/A')}\n")
                    f.write(f"  模型: {result.get('model_type', 'N/A')}\n")
                    f.write(f"  F1: {result.get('final_metrics', {}).get('macro_f1', 'N/A'):.4f}\n")
                    f.write(f"  Accuracy: {result.get('final_metrics', {}).get('accuracy', 'N/A'):.4f}\n\n")

        self.log(f"最終報告已儲存: {report_file}")


# ============================================================================
# 實驗執行器
# ============================================================================

class ExperimentExecutor:
    """實驗執行器"""

    def __init__(self, tracker: ExperimentTracker, dry_run: bool = False):
        self.tracker = tracker
        self.dry_run = dry_run
        self.training_executor = TrainingExecutor() if not dry_run else None

    def execute_experiment(self, experiment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        執行單個實驗

        Args:
            experiment: 實驗配置

        Returns:
            實驗結果（如果成功）
        """
        exp_id = self._generate_experiment_id(experiment)

        # 檢查是否已完成
        if self.tracker.is_completed(exp_id):
            self.tracker.log(f"實驗已完成，跳過: {exp_id}")
            return None

        self.tracker.log(f"開始執行實驗: {exp_id}")

        if self.dry_run:
            self.tracker.log(f"[模擬模式] 模擬執行: {exp_id}")
            time.sleep(1)  # 模擬執行時間
            result = {
                "experiment_id": exp_id,
                "status": "simulated",
                "dataset": experiment["dataset"],
                "model_type": experiment["config"].get("model_type", "unknown"),
                "final_metrics": {
                    "macro_f1": 0.75,
                    "accuracy": 0.80
                }
            }
            self.tracker.mark_completed(exp_id, result)
            return result

        try:
            # 解析實驗配置
            train_config = ConfigParser.parse_dimension_config(experiment)

            # 執行訓練
            result = self.training_executor.execute(train_config)

            if result:
                self.tracker.mark_completed(exp_id, result)
                return result
            else:
                self.tracker.mark_failed(exp_id, "訓練返回空結果")
                return None

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.tracker.mark_failed(exp_id, error_msg)
            return None

    def _generate_experiment_id(self, experiment: Dict[str, Any]) -> str:
        """生成實驗ID"""
        parts = [
            f"dim{experiment['dimension']}",
            experiment['group'],
            experiment['dataset']
        ]

        # 添加層數信息（如果有）
        if 'layers' in experiment['config']:
            layers = experiment['config']['layers']
            if isinstance(layers, list):
                parts.append(f"layers{min(layers)}-{max(layers)}")
            else:
                parts.append(f"layers{layers}")

        return "_".join(parts)


# ============================================================================
# 主程式
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='多維度實驗調度器')

    # 實驗維度選擇
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dimensions', type=int, nargs='+',
                      choices=[1, 2, 3, 4, 5],
                      help='指定要執行的維度（1-5）')
    group.add_argument('--combination', type=str,
                      choices=list(DIMENSION_COMBINATIONS.keys()),
                      help='使用預定義的維度組合')

    # 數據集選擇
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='指定數據集（預設：使用所有數據集）')

    # 執行模式
    parser.add_argument('--dry-run', action='store_true',
                       help='模擬模式（不實際執行訓練）')
    parser.add_argument('--resume', type=str,
                       help='從檢查點恢復（提供 session_id）')

    # 輸出配置
    parser.add_argument('--checkpoint-dir', type=str,
                       default='outputs/experiment_checkpoints',
                       help='檢查點儲存目錄')
    parser.add_argument('--log-dir', type=str,
                       default='outputs/experiment_logs',
                       help='日誌儲存目錄')

    args = parser.parse_args()

    # 確定維度
    if args.combination:
        combination_config = get_combination_config(args.combination)
        dimensions = combination_config["dimensions"]
        print(f"\n使用預定義組合: {args.combination}")
        print(f"描述: {combination_config['description']}")
    else:
        dimensions = args.dimensions

    print("\n" + "=" * 80)
    print("多維度實驗調度器")
    print("=" * 80)

    # 初始化追蹤器
    checkpoint_dir = BASE_DIR / args.checkpoint_dir
    log_dir = BASE_DIR / args.log_dir
    tracker = ExperimentTracker(checkpoint_dir, log_dir)

    # 恢復或新建會話
    if args.resume:
        tracker.log(f"恢復會話: {args.resume}")
        tracker.load_checkpoint(args.resume)
        plan = tracker.state.get("plan")
        if not plan:
            print("[錯誤] 檢查點中沒有實驗計劃")
            return
    else:
        # 生成實驗計劃
        plan = generate_experiment_plan(dimensions, args.datasets)
        print_experiment_summary(plan)

        # 確認執行
        if not args.dry_run:
            response = input("\n是否開始執行？(y/n): ")
            if response.lower() != 'y':
                print("已取消執行")
                return

        tracker.initialize(plan)

    # 建立執行器
    executor = ExperimentExecutor(tracker, dry_run=args.dry_run)

    # 執行實驗
    tracker.log("\n開始執行實驗...")

    for i, experiment in enumerate(plan["experiments"], 1):
        tracker.log(f"\n--- 實驗 {i}/{len(plan['experiments'])} ---")

        result = executor.execute_experiment(experiment)

        # 定期印出進度
        if i % 5 == 0 or i == len(plan["experiments"]):
            tracker.print_progress()

    # 完成
    tracker.finalize()
    tracker.print_progress()

    print("\n" + "=" * 80)
    print("所有實驗完成！")
    print("=" * 80)
    print(f"\n檢查點: {tracker.session_file}")
    print(f"日誌: {tracker.log_file}")


if __name__ == "__main__":
    main()
