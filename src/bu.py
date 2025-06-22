import numpy as np

def quantize_time_series(x, epsilon):
    """
    Quantizes a time series such that each quantized value is within epsilon of the original.

    Parameters:
        x (List[float]): input time series
        epsilon (float): maximum absolute error allowed

    Returns:
        List[float]: quantized time series
    """
    quantized = []
    for value in x:
        # Round to the center of the quantization bin
        bin_index = round(value / (2 * epsilon))
        q_value = bin_index * (2 * epsilon)
        quantized.append(q_value)
    return quantized

def quantize(x, epsilon):
    """
    Quantizes a time series such that each quantized value is within epsilon of the original.

    Parameters:
        x (List[float]): input time series
        epsilon (float): maximum absolute error allowed

    Returns:
        List[float]: quantized time series
    """
    quantized = []
    for value in x:
        # Round to the center of the quantization bin
        bin_index = np.floor(value / (1.99 * epsilon) + 0.5)
        q_value = bin_index * (1.99 * epsilon)
        quantized.append(q_value)
    return quantized


ts = np.round(np.random.uniform(1, 100, 1000), 1).tolist()
epsilon = np.round(np.random.uniform(0.1, 10.0))

quantized_ts = quantize_time_series(ts, epsilon)
np.testing.assert_allclose(
    quantized_ts,
    ts,
    atol=epsilon,
    err_msg="Quantized values are not within the specified error bound."
)

print(epsilon)
print(ts[:10])
print(quantized_ts[:10])

print("\nUsing the quantize function...\n")

quantized_ts = quantize(ts, epsilon)
np.testing.assert_allclose(
    quantized_ts,
    ts,
    atol=epsilon,
    err_msg="Quantized values are not within the specified error bound."
)

print(quantized_ts[:10])