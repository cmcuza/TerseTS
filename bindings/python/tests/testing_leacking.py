import time
import numpy as np
from tersets import compress, decompress, Method # Make sure to import your functions

def run_test():
    """Calls the compress function repeatedly to test for memory leaks."""
    print("Starting memory leak test. Watch this process in your system monitor.")
    
    # Create some sample data once to avoid measuring numpy allocation time
    data_to_compress = np.random.rand(1000).astype(np.float64)

    for i in range(1, 50001):
        # Call the function that might be leaking
        # print("Compressing", data_to_compress.shape)
        data = compress(data_to_compress, Method.SwingFilter, 0.01)
        decompress(data)

        if i % 1000 == 0:
            print(f"Completed {i} iterations...")
            # A tiny sleep makes the graph in the monitor easier to read
            time.sleep(0.1)

    print("Test finished. Check the final memory usage.")
    # Keep the script alive for a bit so you can see the final state
    time.sleep(10)

if __name__ == "__main__":
    run_test()