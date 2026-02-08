#!/usr/bin/env python3
import json, time, math, argparse, os, sys
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    os.system(f"{sys.executable} -m pip install openai -q")
    from openai import OpenAI

API_BASE = "http://localhost:8000/v1"
VAL_DATA_PATH = "/home/gx10/train/data/buddhist_val_alpaca.json"
OUTPUT_DIR = "/home/gx10/train/eval/results_dual_node"


def get_client():
    return OpenAI(base_url=API_BASE, api_key="dummy")


def get_model_name(client):
    models = client.models.list()
    name = models.data[0].id
    print(f"Detected model: {name}")
    return name


def load_val_data(path, max_samples=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:max_samples] if max_samples else data


def test_sanity(client, model):
    print("\n" + "=" * 60)
    print("阶段 0: 双机推理健康检查")
    print("=" * 60)
    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "你好，请用一句话介绍自己。"}],
        max_tokens=64,
        temperature=0.0,
    )
    elapsed = time.time() - start
    text = resp.choices[0].message.content
    print(f"[{model}] ({elapsed:.2f}s) => {text[:200]}")
    print("健康检查通过")
    return True


def test_buddhist_qa(client, model, val_data, max_samples=30):
    print("\n" + "=" * 60)
    print(f"阶段 1: 佛经问答测试 ({max_samples} 样本)")
    print("=" * 60)
    samples = val_data[:max_samples]
    results = []
    total_time = 0

    for i, item in enumerate(samples):
        question = item["instruction"]
        if item.get("input"):
            question += "\n" + item["input"]

        start = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            max_tokens=300,
            temperature=0.0,
        )
        elapsed = time.time() - start
        total_time += elapsed
        answer = resp.choices[0].message.content

        results.append(
            {
                "id": i,
                "question": question[:200],
                "expected": item["output"],
                "response": answer,
                "time": round(elapsed, 2),
                "source": item.get("source", ""),
            }
        )
        if i < 3 or i % 10 == 0:
            print(f"  [{i + 1}/{max_samples}] {elapsed:.1f}s | {question[:50]}...")

    avg = total_time / len(samples)
    print(f"  avg {avg:.1f}s/sample, total {total_time:.0f}s")
    return results, {
        "avg_time": round(avg, 2),
        "total_time": round(total_time, 1),
        "samples": len(samples),
    }


def test_perplexity(client, model, val_data, max_samples=50):
    print("\n" + "=" * 60)
    print(f"阶段 2: 困惑度测试 ({max_samples} 样本)")
    print("=" * 60)
    samples = val_data[:max_samples]
    total_logprob = 0.0
    total_tokens = 0

    for i, item in enumerate(samples):
        text = item["instruction"]
        if item.get("input"):
            text += "\n" + item["input"]
        text += "\n" + item["output"]

        try:
            resp = client.completions.create(
                model=model,
                prompt=text,
                max_tokens=1,
                logprobs=1,
                echo=True,
                temperature=0.0,
            )
            choice = resp.choices[0]
            if choice.logprobs and choice.logprobs.token_logprobs:
                lps = [lp for lp in choice.logprobs.token_logprobs if lp is not None]
                total_logprob += sum(lps)
                total_tokens += len(lps)
        except Exception as e:
            if i == 0:
                print(f"  [WARN] completions API logprobs not supported: {e}")
                return {
                    "perplexity": "N/A",
                    "note": "API does not support echo logprobs",
                }
        if i % 10 == 0:
            print(f"  [{i + 1}/{max_samples}]", end=" ", flush=True)

    if total_tokens > 0:
        avg_lp = total_logprob / total_tokens
        ppl = math.exp(-avg_lp)
        print(f"\n  PPL: {ppl:.2f}, tokens: {total_tokens}")
        return {
            "perplexity": round(ppl, 2),
            "avg_logprob": round(avg_lp, 4),
            "total_tokens": total_tokens,
        }
    return {"perplexity": "N/A"}


def test_general_knowledge(client, model):
    print("\n" + "=" * 60)
    print("阶段 3: 通用能力保持测试")
    print("=" * 60)
    questions = [
        {"q": "请简要解释量子力学中的测不准原理。", "cat": "物理"},
        {"q": "Python中装饰器的作用是什么？请举例说明。", "cat": "编程"},
        {"q": "第二次世界大战的转折点是哪场战役？为什么？", "cat": "历史"},
        {"q": "请用简洁的语言解释什么是机器学习中的过拟合。", "cat": "AI/ML"},
        {"q": "《心经》的核心思想是什么？", "cat": "佛学"},
        {"q": "什么是四圣谛？请逐一解释。", "cat": "佛学"},
        {"q": "唯识学的'三性'是什么？", "cat": "佛学(进阶)"},
        {"q": "请解释'缘起性空'的含义。", "cat": "佛学"},
    ]
    results = []
    for item in questions:
        start = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": item["q"]}],
            max_tokens=200,
            temperature=0.0,
        )
        elapsed = time.time() - start
        answer = resp.choices[0].message.content
        results.append(
            {
                "category": item["cat"],
                "question": item["q"],
                "response": answer,
                "time": round(elapsed, 2),
            }
        )
        print(f"  [{item['cat']}] {elapsed:.1f}s | {answer[:80]}...")
    return results


def test_throughput(client, model, batch_sizes=(1, 5, 10)):
    print("\n" + "=" * 60)
    print("阶段 4: 吞吐量测试")
    print("=" * 60)
    prompt = "请解释佛教中'缘起性空'的含义。"
    results = {}
    for bs in batch_sizes:
        start = time.time()
        for _ in range(bs):
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0,
            )
        elapsed = time.time() - start
        qps = bs / elapsed
        results[f"batch_{bs}"] = {
            "total_time": round(elapsed, 2),
            "qps": round(qps, 3),
            "avg_latency": round(elapsed / bs, 2),
        }
        print(
            f"  batch={bs}: {elapsed:.1f}s total, {qps:.2f} qps, {elapsed / bs:.1f}s/req"
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--val-data", default=VAL_DATA_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--tag", default="", help="Tag for output file (e.g. 'finetuned' or 'base')"
    )
    parser.add_argument("--qa-samples", type=int, default=30)
    parser.add_argument("--ppl-samples", type=int, default=50)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    args = parser.parse_args()

    client = OpenAI(base_url=args.api_base, api_key="dummy")
    model = get_model_name(client)
    tag = args.tag or model

    print("=" * 60)
    print(f"DGX Spark 双机 PP=2 测试 — {tag}")
    print(f"API: {args.api_base} | Model: {model}")
    print(f"时间: {datetime.now().isoformat()}")
    print("=" * 60)

    results = {
        "meta": {"timestamp": datetime.now().isoformat(), "model": model, "tag": tag}
    }

    test_sanity(client, model)
    val_data = load_val_data(args.val_data, max(args.qa_samples, args.ppl_samples))

    def save_results(results, tag, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        outfile = (
            f"{output_dir}/test_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {outfile}")
        return outfile

    try:
        qa_results, qa_metrics = test_buddhist_qa(
            client, model, val_data, args.qa_samples
        )
        results["buddhist_qa"] = {"results": qa_results, "metrics": qa_metrics}
    except Exception as e:
        print(f"  [ERROR] QA测试失败: {e}")
        results["buddhist_qa"] = {"error": str(e)}

    if not args.skip_ppl:
        try:
            results["perplexity"] = test_perplexity(
                client, model, val_data, args.ppl_samples
            )
        except Exception as e:
            print(f"  [ERROR] PPL测试失败: {e}")
            results["perplexity"] = {"error": str(e)}

    try:
        results["general_knowledge"] = test_general_knowledge(client, model)
    except Exception as e:
        print(f"  [ERROR] 通用能力测试失败: {e}")
        results["general_knowledge"] = {"error": str(e)}

    if not args.skip_throughput:
        try:
            results["throughput"] = test_throughput(client, model)
        except Exception as e:
            print(f"  [ERROR] 吞吐量测试失败: {e}")
            results["throughput"] = {"error": str(e)}

    outfile = save_results(results, tag, args.output_dir)
    return outfile


if __name__ == "__main__":
    main()
