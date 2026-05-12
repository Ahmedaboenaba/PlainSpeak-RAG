import time
import logging
from retrieval_engine import PlainSpeakRetrieval

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def test_suite():
    engine = PlainSpeakRetrieval()
    
    test_cases = [
        {
            "query": "The patient exhibits symptoms of myocardial infarction.",
            "target": 5.0,
            "expected_domain": "medical"
        },
        {
            "query": "The defendant is liable for breach of contract.",
            "target": 8.0,
            "expected_domain": "legal"
        },
        {
            "query": "Mac OS X 10.5 Leopard was released on October 26, 2007.",
            "target": 5.0,
            "expected_domain": "general"
        },
        {
            "query": "The surgery was performed to remove a malignant tumor.",
            "target": 5.0,
            "expected_domain": "medical"
        },
        {
            "query": "The judge issued a subpoena for the testimony.",
            "target": 8.0,
            "expected_domain": "legal"
        }
    ]
    
    log.info("=" * 60)
    log.info("RETRIEVAL ENGINE VERIFICATION")
    log.info("=" * 60)
    
    latencies = []
    domain_correct = 0
    
    for case in test_cases:
        log.info(f"Testing Query: '{case['query']}'")
        res = engine.retrieve(case['query'], case['target'])
        
        # Domain check
        is_domain_correct = res['domain'] == case['expected_domain']
        if is_domain_correct: domain_correct += 1
        
        log.info(f"  Detected Domain: {res['domain']} ({'[PASS]' if is_domain_correct else '[FAIL]'})")
        log.info(f"  Best Match FK:   {res['best_match']['fk_grade']} (Target: {case['target']})")
        log.info(f"  Latency:         {res['latency_ms']:.2f}ms")
        log.info(f"  Result:          {res['best_match']['simple'][:100]}...")
        log.info("-" * 40)
        
        latencies.append(res['latency_ms'])
        
    avg_lat = sum(latencies) / len(latencies)
    domain_acc = (domain_correct / len(test_cases)) * 100
    
    log.info("=" * 60)
    log.info("FINAL VERIFICATION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Average Latency: {avg_lat:.2f} ms")
    log.info(f"  Domain Accuracy: {domain_acc:.1f}% ({domain_correct}/{len(test_cases)})")
    
    if avg_lat < 200:
        log.info("  [PASS] Latency within budget (< 200ms)")
    else:
        log.warning("  [FAIL] Latency exceeds budget (> 200ms)")
        
    if domain_acc >= 80:
        log.info("  [PASS] Domain accuracy >= 80%")
    else:
        log.warning("  [FAIL] Domain accuracy < 80%")
    log.info("=" * 60)

if __name__ == "__main__":
    test_suite()
