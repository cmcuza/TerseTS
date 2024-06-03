#include <stdint.h>

// A pointer to uncompressed values and the number of values.
struct UncompressedValues {
  double const * const data;
  uintptr_t const len;
};

// A pointer to compressed values and the number of bytes.
struct CompressedValues {
  uint8_t const * const data;
  uintptr_t const len;
};

// Configuration to use for compression and/or decompression.
struct Configuration {
  uint8_t const method;
  float const error_bound;
};

// Compress uncompressed_values to compressed_values according to
// configuration. The following non-zero values are returned on errors:
// - 1) Unsupported compression method.
int32_t compress(struct UncompressedValues const uncompressed_values,
                 struct CompressedValues *const compressed_values,
                 struct Configuration const configuration);

// Decompress compressed_values to uncompressed_values according to
// configuration. The following non-zero values are returned on errors:
// - 1) Unsupported decompression method.
int32_t decompress(struct CompressedValues const compressed_values,
                   struct UncompressedValues const * const uncompressed_values);
