#pragma once
#include <stddef.h>   
#include <stdint.h>  

#ifdef __cplusplus
extern "C" {
#endif

// Mirror the compression methods provided by TerseTS.
// An uint8_t is used instead of this enum to simplify FFI.
enum Method {
  Uncompressed                  = 0,
  PoorMansCompressionMidrange   = 1,
  PoorMansCompressionMean       = 2,
  SwingFilter                   = 3,
  SwingFilterDisconnected       = 4,
  SlideFilter                   = 5,
  SimPiece                      = 6,
  PiecewiseConstantHistogram    = 7,
  PiecewiseLinearHistogram      = 8,
  ABCLinearApproximation        = 9,
  VisvalingamWhyatt             = 10,
  SlidingWindow                 = 11,
  BottomUp                      = 12,
  MixPiece                      = 13,
  BitPackedQuantization         = 14,
  RunLengthEncoding             = 15,
  NonLinearApproximation        = 16,
  SerfQT                        = 17,
  BitPackedBUFF                 = 18,
  Chimp64                       = 19,
  Chimp128                      = 20,
  BitPackedDeltaEncoding        = 21,
  DiscreteFourierTransform      = 22,
  MacaqueS                      = 23,
  MacaqueV                      = 24,
  Shrink                        = 25,
};

// A pointer to uncompressed values and the number of values.
struct UncompressedValues {
  const double *data;
  size_t        len;
};

// A pointer to compressed values and the number of bytes.
struct CompressedValues {
  uint8_t *data;   
  size_t   len;    
};

// A pointer to indices and the number of values.
struct Indices {
  uint64_t *data;
  size_t   len;
};

// A pointer to coefficients and the number of values.
struct Coefficients {
  double *data;
  size_t   len;
};

// Compress uncompressed_values to compressed_values according to configuration.
// Returns 0 on success, non-zero on error (e.g., 1 = unsupported method).
int32_t compress(struct UncompressedValues uncompressed_values,
                 struct CompressedValues *compressed_values,
		 uint8_t method_index,
                 const char *configuration);

// Decompress compressed_values to uncompressed_values according to configuration.
// Returns 0 on success, non-zero on error (e.g., 1 = unsupported method).
int32_t decompress(struct CompressedValues compressed_values,
                   struct UncompressedValues *uncompressed_values);

// Extract indices and coefficients from compressed_values.
// Returns 0 on success, non-zero on error (e.g., 1 = unsupported method).
int32_t extract(struct CompressedValues compressed_values,
                struct Indices *indices,
                struct Coefficients *coefficients);

// Rebuild compressed_values from indices and coefficients.
// Returns 0 on success, non-zero on error (e.g., 1 = unsupported method).
int32_t rebuild(struct Indices indices,
		struct Coefficients coefficients,
                struct CompressedValues *compressed_values,
		uint8_t method_index);


// Free functions.
void freeCompressedValues(struct CompressedValues *compressed_values);
void freeUncompressedValues(struct UncompressedValues *uncompressed_values);
void freeIndices(struct Indices *indices);
void freeCoefficients(struct Coefficients *coefficients);

#ifdef __cplusplus
}
#endif
