import time
import math

def long_computation(n):
    """Simulate a heavy computation."""
    results = []
    for i in range(1, n + 1):
        # Just some math to take time
        val = math.sqrt(i) ** 1.5 + math.log(i + 1)
        results.append(val)

        # Print progress every 5%
        if i % (n // 20) == 0:
            print(f"Progress: {i / n * 100:.0f}%")
    return results

def main():
    print("Starting long computation...")
    start_time = time.time()

    results = long_computation(10_000_000)  # Adjust for longer/shorter runs

    elapsed = time.time() - start_time
    print(f"Computation finished in {elapsed:.2f} seconds.")
    print(f"First 5 results: {results[:5]}")

if __name__ == "__main__":
    main()